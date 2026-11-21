"""
Portable, cluster-ready DECODE training script powered by the trained Conv-Beta-VAE background generator.

Saves all outputs (overwritten latest checkpoint, best model weights, matched param_run_used.yaml,
and training_log.csv) into `localisation_workflow/decode_model/`.

Examples:
  # Local Mac / CPU quick smoke test:
  python localisation_workflow/train_decode.py --device cpu --epochs 2 --pseudo-ds-size 128 --test-size 32 --batch-size 16

  # Full GPU cluster run (auto-selects CUDA if available):
  python localisation_workflow/train_decode.py --device auto --epochs 100 --batch-size 64 --num-workers 4

  # Resume from overwritten checkpoint if interrupted:
  python localisation_workflow/train_decode.py --device auto --resume
"""

import argparse
import copy
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import decode.evaluation
import decode.neuralfitter
import decode.neuralfitter.train.live_engine as live_engine
import decode.neuralfitter.train_val_impl as train_val_impl
import decode.neuralfitter.utils.logger as decode_logger
import decode.simulation.background
import decode.utils
from decode.neuralfitter.train.random_simulation import setup_random_simulation


class StableGaussianMMLoss(decode.neuralfitter.loss.GaussianMMLoss):
    """DECODE mixture loss with higher-precision categorical normalization."""

    def _compute_gmm_loss(self, p, pxyz_mu, pxyz_sig, pxyz_tar, mask):
        if not torch.isfinite(p).all() or (p < 0).any() or (p.sum(dim=(-2, -1)) <= 0).any():
            raise ValueError("Non-finite or invalid detection probabilities in model output")

        batch_size = p.shape[0]
        p_mean = p.sum(dim=(-2, -1))
        p_var = (p - p.square()).sum(dim=(-2, -1))
        count = mask.sum(-1)
        count_log_prob = torch.distributions.Normal(p_mean, p_var.sqrt()).log_prob(count) * count

        # Categorical's float32 simplex validation can fail for long probability
        # vectors even after normalization. Keep its normalization in float64.
        mix = torch.distributions.Categorical(probs=p.reshape(batch_size, -1).double())
        indices = tuple((p + 1).nonzero(as_tuple=False).transpose(1, 0))
        means = pxyz_mu[indices[0], :, indices[1], indices[2]]
        means[:, 1] += self._offset2coord.bin_ctr_x[indices[1]].to(means.device)
        means[:, 2] += self._offset2coord.bin_ctr_y[indices[2]].to(means.device)
        means = means.reshape(batch_size, -1, 4)
        sigmas = pxyz_sig[indices[0], :, indices[1], indices[2]].reshape(batch_size, -1, 4)
        components = torch.distributions.Independent(torch.distributions.Normal(means, sigmas), 1)
        mixture = torch.distributions.MixtureSameFamily(mix, components)

        if mask.sum():
            log_prob = mixture.log_prob(pxyz_tar.transpose(0, 1)).transpose(0, 1)
            count_log_prob = count_log_prob + (log_prob * mask).sum(-1)
        return -count_log_prob


# ------------------------------------------------------------------------------
# 1. Frozen Conv-Beta-VAE Architecture & DECODE Background Wrapper
# ------------------------------------------------------------------------------
class SmoothBgBetaVAE(nn.Module):
    """Matches the 8x8-bottleneck Conv-Beta-VAE trained in 04_vae_background.ipynb."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='replicate'),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.GroupNorm(8, 64), nn.SiLU(),
        )
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, padding_mode='replicate'),
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc_dec(z)).view(-1, 64, 8, 8)
        return self.dec(h)


class VAEBackground(decode.simulation.background.Background):
    """
    DECODE Background sampler powered by the frozen Conv-Beta-VAE decoder.
    Generates normalised 2D background shapes from z ~ N(0, I) and scales them
    to empirical (mu, sigma) physical photon units from noise_maps/manifest.json.
    """

    def __init__(self, vae_model: SmoothBgBetaVAE, manifest_list: list, forward_return: str = None):
        super().__init__(forward_return=forward_return)
        self.vae = vae_model.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.roi_means = np.array([m['roi_mean'] for m in manifest_list], dtype=np.float32)
        self.roi_stds = np.array([m['roi_std'] for m in manifest_list], dtype=np.float32)

    @torch.no_grad()
    def sample(self, size: torch.Size, device=torch.device('cpu')) -> torch.Tensor:
        target_h, target_w = size[-2], size[-1]
        batch_shape = size[:-2]
        n_samples = int(np.prod(batch_shape)) if len(batch_shape) > 0 else 1

        vae_dev = next(self.vae.parameters()).device
        z = torch.randn(n_samples, self.vae.latent_dim, device=vae_dev)
        b_hat = self.vae.decode(z)
        if b_hat.shape[-2:] != (target_h, target_w):
            b_hat = F.interpolate(b_hat, size=(target_h, target_w), mode='bilinear', align_corners=False)
        b_hat = (b_hat - b_hat.mean(dim=(2, 3), keepdim=True)) / (b_hat.std(dim=(2, 3), keepdim=True) + 1e-6)

        idx = np.random.randint(0, len(self.roi_means), size=n_samples)
        mu_t = torch.from_numpy(self.roi_means[idx]).to(vae_dev).view(n_samples, 1, 1, 1)
        sig_scale = np.random.uniform(0.5, 1.5, size=n_samples).astype(np.float32)
        sig_t = torch.from_numpy(self.roi_stds[idx] * sig_scale).to(vae_dev).view(n_samples, 1, 1, 1)

        b_phys = torch.clamp(mu_t + sig_t * b_hat, min=1.0).squeeze(1)
        return b_phys.reshape(size).to(device)


class VariableDensityEmitterSampler:
    """
    Lightweight proxy around DECODE's EmitterSamplerBlinking that varies emitter
    density *within* every sampled acquisition (intra-epoch). It samples at
    em_av_max and thins emitter tracks across temporal chunks to
    Uniform(em_av_min, em_av_max), preserving multi-frame blinking continuity
    for 3-channel temporal context while giving every epoch and validation set
    a balanced mix of sparse and crowded frames.
    """

    def __init__(self, base_sampler, em_av_min: float, em_av_max: float, chunk_frames: int = 32):
        self._base = base_sampler
        self.em_av_min = float(em_av_min)
        self.em_av_max = float(em_av_max)
        self.chunk_frames = int(max(1, chunk_frames))

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __call__(self):
        return self.sample()

    def sample(self):
        lam_max = self.em_av_max
        self._base._em_avg = lam_max
        self._base._emitter_av_total = (
            lam_max * self._base._num_frames_plus / (self._base.lifetime_avg + 1.0)
        )
        em = self._base.sample()
        if len(em) == 0 or self.em_av_min >= lam_max:
            return em

        frame_ix = em.frame_ix.long()
        f_min = int(frame_ix.min().item())
        f_max = int(frame_ix.max().item())
        n_frames = max(1, f_max - f_min + 1)
        n_chunks = max(1, (n_frames + self.chunk_frames - 1) // self.chunk_frames)

        # Draw one target density per temporal chunk ~ Uniform(em_av_min, em_av_max)
        chunk_lams = np.random.uniform(self.em_av_min, lam_max, size=n_chunks)
        chunk_probs = torch.as_tensor(chunk_lams / lam_max, dtype=torch.float32, device=frame_ix.device)
        emitter_chunks = torch.clamp((frame_ix - f_min) // self.chunk_frames, 0, n_chunks - 1)

        # Thin by unique emitter track ID when available so blinking tracks stay intact across t-1, t, t+1
        em_id = em.id.long()
        uniq_ids, inverse_ix = torch.unique(em_id, return_inverse=True)
        if len(uniq_ids) > 1:
            track_chunk = torch.zeros(len(uniq_ids), dtype=torch.long, device=frame_ix.device)
            track_chunk[inverse_ix] = emitter_chunks
            keep_track = torch.rand(len(uniq_ids), device=frame_ix.device) < chunk_probs[track_chunk]
            keep_mask = keep_track[inverse_ix]
        else:
            keep_mask = torch.rand(len(em), device=frame_ix.device) < chunk_probs[emitter_chunks]

        return em[keep_mask]


# ------------------------------------------------------------------------------
# 2. Helpers: Device Resolution, Parameter Configuration, Checkpointing & Metrics
# ------------------------------------------------------------------------------
def resolve_device(device_arg: str) -> str:
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda:0'
        return 'cpu'
    if device_arg.startswith('cuda') and not torch.cuda.is_available():
        print('[WARNING] CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.', flush=True)
        return 'cpu'
    return device_arg


def build_decode_params(project_dir: Path, out_dir: Path, args, patch_size: int):
    param_path = project_dir / args.param_rel
    cal_path = project_dir / args.cal_rel
    if not param_path.exists():
        raise FileNotFoundError(f'Base DECODE param file not found: {param_path}')
    if not cal_path.exists():
        raise FileNotFoundError(f'3D PSF calibration file not found: {cal_path}')

    param = decode.utils.param_io.ParamHandling().load_params(str(param_path))
    param.InOut.calibration_file = str(cal_path)
    param.InOut.experiment_out = str(out_dir)

    param.Hardware.device = args.device_resolved
    param.Hardware.device_simulation = args.device_resolved
    param.Hardware.num_worker_train = args.num_workers
    param.Hardware.torch_threads = args.torch_threads

    # ONI Nanoimager sCMOS camera settings (from 01_psf_fitting / 03_background_gen / 05_simulator)
    param.Camera.baseline = 400.0
    param.Camera.e_per_adu = 0.46
    param.Camera.em_gain = None
    param.Camera.qe = 0.8
    param.Camera.read_sigma = 1.6
    param.Camera.spur_noise = 0.0
    param.Camera.px_size = [117.0, 117.0]
    param.Camera.convert2photons = True

    # Simulation settings (64x64 patches, 3D z-range, experimental photon distribution)
    param.Simulation.mode = 'acquisition'
    param.Simulation.img_size = [patch_size, patch_size]
    param.Simulation.psf_extent = [[-0.5, patch_size - 0.5], [-0.5, patch_size - 0.5], None]
    param.Simulation.emitter_extent = [[-0.5, patch_size - 0.5], [-0.5, patch_size - 0.5], [-200.0, 200.0]]
    param.Simulation.emitter_av = 0.5 * (args.em_av_min + args.em_av_max)
    param.Simulation.intensity_mu_sig = [1800.0, 750.0]
    param.Simulation.lifetime_avg = args.lifetime_avg

    # Scaling matched to ONI Nanoimager photon / background / axial ranges
    param.Scaling.bg_max = 260.0
    param.Scaling.input_offset = 130.0
    param.Scaling.input_scale = 140.0
    param.Scaling.phot_max = 6000.0
    param.Scaling.z_max = 250.0

    # Hyperparameters
    param.HyperParameter.channels_in = args.channels_in
    param.HyperParameter.epochs = args.epochs
    param.HyperParameter.batch_size = args.batch_size
    param.HyperParameter.pseudo_ds_size = args.pseudo_ds_size
    param.HyperParameter.opt_param.lr = args.lr
    param.HyperParameter.max_number_targets = max(250, int(args.em_av_max * 3))
    param.HyperParameter.learning_rate_scheduler = 'ReduceLROnPlateau'
    param.HyperParameter.learning_rate_scheduler_param = {
        'mode': 'min',
        'factor': 0.5,
        'patience': 10,
        'threshold': 1e-4,
        'cooldown': 3,
        'verbose': False,
    }

    # Validation / TestSet settings (simulated on-the-fly with VAEBackground)
    param.TestSet.test_size = args.test_size
    param.TestSet.frame_extent = param.Simulation.psf_extent
    param.TestSet.img_size = [patch_size, patch_size]
    param.TestSet.mode = 'simulat'

    return param


def save_overwritten_checkpoint(ckpt_file: Path, model, optimizer, lr_scheduler, epoch: int, best_val: float, args):
    tmp_file = ckpt_file.with_suffix('.tmp')
    torch.save({
        'epoch': epoch,
        'best_val_loss': best_val,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': lr_scheduler.state_dict(),
        'args': vars(args),
    }, tmp_file)
    tmp_file.replace(ckpt_file)


def compute_val_metrics(test_out, em_tar, post_processor, matcher):
    """Computes Jaccard, Precision, Recall, and lateral/axial RMSE on the simulated validation set."""
    try:
        em_out = post_processor.forward(test_out.y_out)
        tp, fp, fn, tp_match = matcher.forward(em_out, em_tar)
        n_tp = float(len(tp))
        n_fp = float(len(fp))
        n_fn = float(len(fn))
        prec = n_tp / max(1.0, n_tp + n_fp)
        rec = n_tp / max(1.0, n_tp + n_fn)
        jac = n_tp / max(1.0, n_tp + n_fp + n_fn)
        if len(tp) > 0:
            diff_nm = (tp.xyz_nm - tp_match.xyz_nm).detach().cpu().numpy()
            rmse_lat = float(np.sqrt(np.mean(diff_nm[:, 0] ** 2 + diff_nm[:, 1] ** 2)))
            rmse_ax = float(np.sqrt(np.mean(diff_nm[:, 2] ** 2)))
        else:
            rmse_lat, rmse_ax = float('nan'), float('nan')
        return jac, prec, rec, rmse_lat, rmse_ax
    except Exception:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')


# ------------------------------------------------------------------------------
# 3. Main Training Routine
# ------------------------------------------------------------------------------
def parse_args():
    default_project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Train DECODE with Frozen VAE Background Generator')
    parser.add_argument('--project-dir', type=str, default=str(default_project), help='Repository root path')
    parser.add_argument('--out-dir', type=str, default='localisation_workflow/decode_model', help='Output folder relative to project-dir')
    parser.add_argument('--vae-ckpt', type=str, default='localisation_workflow/vae_background/bg_vae_64x64.pt')
    parser.add_argument('--manifest', type=str, default='localisation_workflow/vae_background/noise_maps/manifest.json')
    parser.add_argument('--cal-rel', type=str, default='data/bead_calibration/Beadzstack_3DLens_3dcal.mat')
    parser.add_argument('--param-rel', type=str, default='localisation_workflow/param_run.yaml')

    parser.add_argument('--device', type=str, default='auto', help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers (use 4 on Linux GPU cluster, 0 on macOS)')
    parser.add_argument('--torch-threads', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--channels-in', type=int, choices=[1, 3], default=3, help='Temporal input frames: 3 (t-1,t,t+1) or 1 (single frame)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--pseudo-ds-size', type=int, default=16384, help='Simulated frames per training epoch')
    parser.add_argument('--test-size', type=int, default=512, help='Simulated frames for validation')
    parser.add_argument('--lr', type=float, default=6e-4)

    parser.add_argument('--em-av-min', type=float, default=12.0, help='Minimum emitters/frame (64x64)')
    parser.add_argument('--em-av-max', type=float, default=120.0, help='Maximum emitters/frame (64x64)')
    parser.add_argument('--lifetime-avg', type=float, default=1.0, help='Mean blink on-time in frames')

    parser.add_argument('--resume', action='store_true', help='Resume from ckpt_latest.pt if present')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    project_dir = Path(args.project_dir).resolve()
    out_dir = (project_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    args.device_resolved = resolve_device(args.device)
    torch_dev = torch.device(args.device_resolved)
    torch.set_num_threads(args.torch_threads)

    print('=' * 76, flush=True)
    print(f'DECODE + VAE Training Pipeline | Device: {args.device_resolved}', flush=True)
    print(f'Project Root : {project_dir}', flush=True)
    print(f'Output Folder: {out_dir}', flush=True)
    print(f'Channels In  : {args.channels_in} ({"3-frame temporal context" if args.channels_in == 3 else "1-frame mode"})', flush=True)
    print(f'Emitter Range: Uniform({args.em_av_min}, {args.em_av_max}) per patch', flush=True)
    print('=' * 76, flush=True)

    # Load frozen VAE background model & manifest
    vae_ckpt_path = project_dir / args.vae_ckpt
    manifest_path = project_dir / args.manifest
    if not vae_ckpt_path.exists():
        raise FileNotFoundError(f'Trained VAE checkpoint not found: {vae_ckpt_path}')
    if not manifest_path.exists():
        raise FileNotFoundError(f'Noise-map manifest not found: {manifest_path}')

    vae_ckpt = torch.load(vae_ckpt_path, map_location=torch_dev)
    patch_size = int(vae_ckpt.get('patch_size', 64))
    vae_model = SmoothBgBetaVAE(latent_dim=int(vae_ckpt['latent_dim'])).to(torch_dev)
    vae_model.load_state_dict(vae_ckpt['model_state'], strict=True)
    vae_model.eval()
    manifest_list = json.loads(manifest_path.read_text())
    print(f'[OK] Loaded frozen VAE (d={vae_model.latent_dim}, patch={patch_size}x{patch_size}) on {args.device_resolved}', flush=True)

    # Configure DECODE parameters & save matched param_run_used.yaml
    param = build_decode_params(project_dir, out_dir, args, patch_size)
    param_out_path = out_dir / 'param_run_used.yaml'
    decode.utils.param_io.ParamHandling().write_params(param_out_path, param)
    print(f'[OK] Saved matched DECODE parameters to {param_out_path}', flush=True)

    # Build train & validation simulators and inject VAEBackground + VariableDensityEmitterSampler
    sim_train, sim_test = setup_random_simulation(param)
    sim_train.background = VAEBackground(vae_model, manifest_list)
    sim_test.background = VAEBackground(vae_model, manifest_list)
    sim_train.em_sampler = VariableDensityEmitterSampler(sim_train.em_sampler, args.em_av_min, args.em_av_max)
    sim_test.em_sampler = VariableDensityEmitterSampler(sim_test.em_sampler, args.em_av_min, args.em_av_max)

    ckpt_latest_path = out_dir / 'ckpt_latest.pt'
    model_best_path = out_dir / 'model_best.pt'
    model_final_path = out_dir / 'model_final.pt'
    log_csv_path = out_dir / 'training_log.csv'

    print('[INFO] Initialising DECODE SigmaMUNet, GaussianMMLoss, and initial simulated validation set...', flush=True)
    nolog = decode_logger.NoLog()
    ds_train, ds_test, model, model_ls, optimizer, criterion, lr_scheduler, grad_mod, post_processor, matcher, _ = \
        live_engine.setup_trainer(sim_train, sim_test, nolog, model_final_path, ckpt_latest_path, args.device_resolved, param)
    criterion = StableGaussianMMLoss(
        xextent=param.Simulation.psf_extent[0], yextent=param.Simulation.psf_extent[1],
        img_shape=param.Simulation.img_size, device=args.device_resolved,
        chweight_stat=param.HyperParameter.chweight_stat,
    )
    dl_train, dl_test = live_engine.setup_dataloader(param, ds_train, ds_test)

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and ckpt_latest_path.exists():
        print(f'[RESUME] Loading checkpoint from {ckpt_latest_path}...', flush=True)
        ckpt_data = torch.load(ckpt_latest_path, map_location=torch_dev)
        model.load_state_dict(ckpt_data['model_state'])
        optimizer.load_state_dict(ckpt_data['optimizer_state'])
        lr_scheduler.load_state_dict(ckpt_data['scheduler_state'])
        start_epoch = int(ckpt_data['epoch']) + 1
        best_val_loss = float(ckpt_data.get('best_val_loss', float('inf')))
        print(f'[RESUME] Resuming from epoch {start_epoch} (best_val_loss={best_val_loss:.4f})', flush=True)

    if not log_csv_path.exists() or not args.resume:
        with open(log_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'jaccard', 'precision', 'recall', 'rmse_lat_nm', 'rmse_ax_nm', 'epoch_sec'])

    t0_all = time.time()
    current_epoch = start_epoch

    try:
        for epoch in range(start_epoch, args.epochs):
            current_epoch = epoch
            t0_ep = time.time()
            lr_now = optimizer.param_groups[0]['lr']

            # Training step (Epoch 0 in DECODE can be baseline eval, or train from epoch 0 directly)
            train_loss = float(train_val_impl.train(
                model=model,
                optimizer=optimizer,
                loss=criterion,
                dataloader=dl_train,
                grad_rescale=param.HyperParameter.moeller_gradient_rescale,
                grad_mod=grad_mod,
                epoch=epoch,
                device=torch_dev,
                logger=nolog,
            ))

            if not np.isfinite(train_loss):
                raise RuntimeError(f'Non-finite training loss ({train_loss}) encountered at epoch {epoch}!')

            # Validation step on held-out simulated VAE dataset
            val_loss, test_out = train_val_impl.test(
                model=model,
                loss=criterion,
                dataloader=dl_test,
                epoch=epoch,
                device=torch_dev,
            )
            val_loss = float(val_loss)
            jac, prec, rec, rmse_lat, rmse_ax = compute_val_metrics(test_out, ds_test.emitter, post_processor, matcher)

            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_loss)
            else:
                lr_scheduler.step()

            # Overwrite single latest checkpoint + save best model weights
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_best_path)

            save_overwritten_checkpoint(ckpt_latest_path, model, optimizer, lr_scheduler, epoch, best_val_loss, args)

            ep_sec = time.time() - t0_ep
            eta_min = (ep_sec * (args.epochs - epoch - 1)) / 60.0
            best_tag = ' *BEST*' if is_best else ''

            print(
                f'Epoch {epoch+1:>3}/{args.epochs} | lr={lr_now:.2e} | '
                f'Train={train_loss:.4f} | Val={val_loss:.4f}{best_tag} | '
                f'Jac={jac:.3f} (P={prec:.3f}, R={rec:.3f}) | '
                f'RMSE_xy={rmse_lat:.1f}nm, z={rmse_ax:.1f}nm | '
                f'{ep_sec:.1f}s (ETA {eta_min:.1f}m)',
                flush=True
            )

            with open(log_csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch + 1, f'{lr_now:.6e}', f'{train_loss:.6f}', f'{val_loss:.6f}',
                    f'{jac:.4f}', f'{prec:.4f}', f'{rec:.4f}', f'{rmse_lat:.2f}', f'{rmse_ax:.2f}', f'{ep_sec:.2f}'
                ])

            # Re-simulate fresh training frames for next epoch (with new VAE backgrounds & random densities)
            if epoch + 1 < args.epochs:
                ds_train.sample(True)

        torch.save(model.state_dict(), model_final_path)
        total_min = (time.time() - t0_all) / 60.0
        print('=' * 76, flush=True)
        print(f'[DONE] Training finished in {total_min:.2f} min. Best Val Loss = {best_val_loss:.4f}', flush=True)
        print(f'  Best model : {model_best_path}', flush=True)
        print(f'  Final model: {model_final_path}', flush=True)
        print(f'  Checkpoint : {ckpt_latest_path} (overwritten)', flush=True)
        print('=' * 76, flush=True)

    except KeyboardInterrupt:
        print(f'\n[INTERRUPTED] Epoch {current_epoch+1} was incomplete. '
              f'The latest completed checkpoint is {ckpt_latest_path}.', flush=True)
        sys.exit(130)

    except Exception as exc:
        print(f'\n[ERROR] Training failed at epoch {current_epoch+1}: {exc}', flush=True)
        traceback.print_exc()
        print(f'[INFO] The latest completed checkpoint remains at {ckpt_latest_path}.', flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

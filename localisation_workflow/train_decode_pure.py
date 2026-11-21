"""
Zero-DECODE, zero-spline pure-PyTorch DECODE training script powered by `decode_pure`.
Runs natively on Apple Silicon MPS, CUDA GPUs, and CPU.

Saves checkpoints (`ckpt_latest.pt`, `model_best.pt`, `model_final.pt`),
`param_run_used.yaml`, and `training_log.csv` compatible with
`decode_pure/infer.py`.

Examples:
  # Local Mac (auto-selects MPS if available):
  python localisation_workflow/train_decode_pure.py --device auto --epochs 50 --batch-size 32

  # Quick Mac MPS/CPU smoke test:
  python localisation_workflow/train_decode_pure.py --device auto --epochs 2 --pseudo-ds-size 128 --test-size 32 --batch-size 16

  # Resume from overwritten checkpoint:
  python localisation_workflow/train_decode_pure.py --device auto --resume
"""

import argparse
import csv
import math
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

# Ensure repo root is importable when invoked as `python localisation_workflow/train_decode_pure.py`
import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from decode_pure.infer import decode_nms_postprocess, resolve_infer_device
from decode_pure.model import model_from_params
from decode_pure.simulator import (
    EmitterSetPure, SimulationPure, build_vae_simulator, _torch_load_compat,
)


# ------------------------------------------------------------------------------
# 1. Pure-PyTorch DECODE GaussianMMLoss (MPS / CUDA / CPU float32-safe)
# ------------------------------------------------------------------------------
class GaussianMMLossPure(nn.Module):
    """
    Pure-PyTorch implementation of DECODE's Gaussian Mixture Model loss:
      1. Count loss (Gaussian approximation to Poisson-binomial over pixel probs p)
      2. 4D GMM negative log-likelihood over (x_px, y_px, z_norm, phot_norm)
      3. Background MSE loss
    Uses logsumexp and emitter chunking for numerical stability and low VRAM on MPS/CUDA.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int] = (64, 64),
        phot_max: float = 6000.0,
        z_max: float = 250.0,
        bg_max: float = 260.0,
        chweight_em: float = 1.0,
        chweight_bg: float = 1.0,
        eps: float = 1e-6,
        xy_origin: Tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__()
        self.img_shape = tuple(img_shape)
        self.phot_max = float(phot_max)
        self.z_max = float(z_max)
        self.bg_max = float(bg_max)
        self.chweight_em = float(chweight_em)
        self.chweight_bg = float(chweight_bg)
        self.eps = float(eps)

        H, W = self.img_shape
        rows = torch.arange(H, dtype=torch.float32).view(1, H, 1).expand(1, H, W).reshape(1, -1)
        cols = torch.arange(W, dtype=torch.float32).view(1, 1, W).expand(1, H, W).reshape(1, -1)
        self.register_buffer("grid_x", rows + xy_origin[0])  # (1, HW)
        self.register_buffer("grid_y", cols + xy_origin[1])  # (1, HW)

    def forward(
        self,
        y_out: torch.Tensor,
        tar_xyz_px: torch.Tensor,
        tar_phot: torch.Tensor,
        tar_mask: torch.Tensor,
        tar_bg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y_out:      (B, 10, H, W) network output
            tar_xyz_px: (B, K_max, 3) target emitter (x_px, y_px, z_nm)
            tar_phot:   (B, K_max) target emitter photons
            tar_mask:   (B, K_max) bool mask of active emitters per frame
            tar_bg:     (B, H, W) target background map in photons
        """
        B, _, H, W = y_out.shape
        HW = H * W

        # Flatten spatial dimensions: (B, 10, HW)
        flat = y_out.reshape(B, 10, HW)
        p = flat[:, 0]  # (B, HW)
        mu_phot = flat[:, 1]
        mu_x = flat[:, 2] + self.grid_x
        mu_y = flat[:, 3] + self.grid_y
        mu_z = flat[:, 4]

        sig_phot = torch.clamp(flat[:, 5], min=1e-3)
        sig_x = torch.clamp(flat[:, 6], min=1e-3)
        sig_y = torch.clamp(flat[:, 7], min=1e-3)
        sig_z = torch.clamp(flat[:, 8], min=1e-3)
        bg_pred = flat[:, 9]

        # 1. Count loss
        k_true = tar_mask.float().sum(dim=1)  # (B,)
        mu_p = p.sum(dim=1)  # (B,)
        var_p = (p - p.square()).sum(dim=1).clamp_min(self.eps)
        loss_count = 0.5 * k_true * ((k_true - mu_p).square() / var_p + torch.log(2 * math.pi * var_p))

        # 2. 4D Gaussian Mixture Model NLL (chunked over K_max to bound memory)
        # Match Categorical(probs) and MixtureSameFamily, including the
        # probability floor used when converting normalized weights to logits.
        probs = p / mu_p.unsqueeze(1)
        log_w = torch.distributions.Categorical(probs=probs).logits.log_softmax(dim=1)
        log_norm_const = (
            -0.5 * torch.log(2.0 * math.pi * (sig_x ** 2))
            - 0.5 * torch.log(2.0 * math.pi * (sig_y ** 2))
            - 0.5 * torch.log(2.0 * math.pi * (sig_z ** 2))
            - 0.5 * torch.log(2.0 * math.pi * (sig_phot ** 2))
        )
        base_log = log_w + log_norm_const  # (B, HW)

        inv_2sx2 = -0.5 / (sig_x ** 2)
        inv_2sy2 = -0.5 / (sig_y ** 2)
        inv_2sz2 = -0.5 / (sig_z ** 2)
        inv_2sp2 = -0.5 / (sig_phot ** 2)

        tx = tar_xyz_px[:, :, 0]
        ty = tar_xyz_px[:, :, 1]
        tz = tar_xyz_px[:, :, 2] / self.z_max
        tp = tar_phot / self.phot_max

        loss_gmm = torch.zeros(B, dtype=y_out.dtype, device=y_out.device)
        k_max = tar_xyz_px.shape[1]
        chunk_k = 64
        for k0 in range(0, k_max, chunk_k):
            k1 = min(k_max, k0 + chunk_k)
            m_chunk = tar_mask[:, k0:k1]
            if not m_chunk.any():
                continue
            dx = tx[:, k0:k1].unsqueeze(2) - mu_x.unsqueeze(1)      # (B, kc, HW)
            dy = ty[:, k0:k1].unsqueeze(2) - mu_y.unsqueeze(1)
            dz = tz[:, k0:k1].unsqueeze(2) - mu_z.unsqueeze(1)
            dp = tp[:, k0:k1].unsqueeze(2) - mu_phot.unsqueeze(1)

            ll = (
                base_log.unsqueeze(1)
                + (dx * dx) * inv_2sx2.unsqueeze(1)
                + (dy * dy) * inv_2sy2.unsqueeze(1)
                + (dz * dz) * inv_2sz2.unsqueeze(1)
                + (dp * dp) * inv_2sp2.unsqueeze(1)
            )
            log_prob = torch.logsumexp(ll, dim=2)  # (B, kc)
            loss_gmm = loss_gmm - (log_prob * m_chunk.float()).sum(dim=1)

        # 3. Background MSE loss (summed over pixels, matching DECODE)
        bg_tar_norm = tar_bg.reshape(B, HW) / self.bg_max
        loss_bg = ((bg_pred - bg_tar_norm) ** 2).sum(dim=1)

        total = self.chweight_em * (loss_count + loss_gmm) + self.chweight_bg * loss_bg
        return total.mean()


# ------------------------------------------------------------------------------
# 2. Intra-Epoch Variable-Density Simulation & Batch Builder
# ------------------------------------------------------------------------------
def simulate_variable_density_epoch(
    sim: SimulationPure,
    n_frames: int,
    channels_in: int,
    em_av_min: float,
    em_av_max: float,
    chunk_frames: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, EmitterSetPure]:
    """
    Simulates the inclusive acquisition [0, n_frames] used by DECODE
    with intra-epoch randomized density `lam ~ Uniform(em_av_min, em_av_max)` per
    temporal chunk, preserving multi-frame blinking tracks.
    Returns:
      frames: (n_total_frames, H, W) float32 tensor on sim.psf.device
      bg_maps: (n_total_frames, H, W) float32 tensor on sim.psf.device
      em_set: EmitterSetPure across [0, n_total_frames - 1]
    """
    # setup_random_simulation uses the inclusive acquisition range [0, n_frames].
    total_frames = n_frames + 1
    sim.frame_range = (0, total_frames - 1)
    sim.em_sampler.frame_range = (0, total_frames - 1)

    # Sample at em_av_max and thin tracks per chunk to Uniform(em_av_min, em_av_max)
    sim.em_sampler.emitter_av = float(em_av_max)
    em = sim.em_sampler.sample()

    if len(em) > 0 and em_av_min < em_av_max:
        frame_ix = em.frame_ix.cpu()
        f_min = int(frame_ix.min())
        n_chunks = (int(frame_ix.max()) - f_min) // chunk_frames + 1
        probs = torch.tensor(np.random.uniform(em_av_min, em_av_max, n_chunks) / em_av_max,
                             dtype=torch.float32)
        ids, inverse = torch.unique(em.id.cpu(), return_inverse=True)
        chunks = torch.div(frame_ix - f_min, chunk_frames, rounding_mode="floor")
        if len(ids) > 1:
            track_chunk = torch.zeros_like(ids)
            track_chunk[inverse] = chunks
            keep_mask = (torch.rand(len(ids)) < probs[track_chunk])[inverse]
        else:
            keep_mask = torch.rand(len(em)) < probs[chunks]
        keep_mask = keep_mask.to(em.frame_ix.device)
        em = EmitterSetPure(
            xyz_px=em.xyz_px[keep_mask],
            phot=em.phot[keep_mask],
            frame_ix=em.frame_ix[keep_mask],
            id=em.id[keep_mask],
        )

    frames, bg_maps = sim.forward(em)
    return frames, bg_maps, em


def prepare_padded_targets(
    em_set: EmitterSetPure,
    n_total_frames: int,
    channels_in: int,
    device: torch.device,
    phot_min: float = 100.0,
    extent=None,
    n_max=None,
    disabled_attributes=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Packs variable-length per-frame targets into padded tensors for fast batched training.
    Filters emitters with `phot < phot_min` (matching DECODE's `emitter_label_photon_min`).
    Center frame indices are [1 .. n_total_frames - 2] for channels_in == 3,
    or [0 .. n_total_frames - 1] for channels_in == 1.
    """
    offset = 1 if channels_in == 3 else 0
    n_samples = n_total_frames - 2 * offset

    # Pack on CPU (avoids MPS aten::bincount NotImplementedError), then move to target device
    f_ix = em_set.frame_ix.detach().cpu().long() - offset
    phot_cpu = em_set.phot.detach().cpu().float()
    xyz_cpu = em_set.xyz_px.detach().cpu().float()
    valid = (f_ix >= 0) & (f_ix < n_samples)
    if phot_min is not None:
        valid &= phot_cpu >= phot_min
    if extent is not None:
        for axis in range(2):
            low, high = extent[axis]
            valid &= (xyz_cpu[:, axis] >= low) & (xyz_cpu[:, axis] < high)
    f_ix = f_ix[valid]
    xyz = xyz_cpu[valid]
    phot = phot_cpu[valid]

    if len(f_ix) == 0:
        return (
            torch.zeros((n_samples, 1, 3), dtype=torch.float32, device=device),
            torch.zeros((n_samples, 1), dtype=torch.float32, device=device),
            torch.zeros((n_samples, 1), dtype=torch.bool, device=device),
        )

    counts = torch.bincount(f_ix, minlength=n_samples)
    k_max = max(1, int(counts.max().item()))

    if n_max is not None and k_max > n_max:
        raise ValueError("Number of actual emitters exceeds number of max. emitters.")
    tar_xyz = torch.zeros((n_samples, k_max, 3), dtype=torch.float32)
    tar_phot = torch.zeros((n_samples, k_max), dtype=torch.float32)
    tar_mask = torch.zeros((n_samples, k_max), dtype=torch.bool)

    order = torch.from_numpy(np.argsort(f_ix.numpy(), kind="stable"))
    f_sorted = f_ix[order]
    xyz_sorted = xyz[order]
    phot_sorted = phot[order]

    # Compute within-frame slot index 0..count-1
    starts = torch.zeros(n_samples, dtype=torch.long)
    if n_samples > 1:
        starts[1:] = torch.cumsum(counts[:-1], dim=0)
    slot_ix = torch.arange(len(f_sorted), dtype=torch.long) - starts[f_sorted]

    tar_xyz[f_sorted, slot_ix] = xyz_sorted
    tar_phot[f_sorted, slot_ix] = phot_sorted
    tar_mask[f_sorted, slot_ix] = True
    if disabled_attributes is not None:
        for attr in np.atleast_1d(disabled_attributes):
            if attr == 0:
                tar_phot.zero_()
            else:
                tar_xyz[:, :, attr - 1] = 0
    return tar_xyz.to(device), tar_phot.to(device), tar_mask.to(device)


def build_input_batch(
    frames: torch.Tensor,
    idx: torch.Tensor,
    channels_in: int,
    input_offset: float,
    input_scale: float,
) -> torch.Tensor:
    if channels_in == 3:
        x = torch.stack([frames[idx], frames[idx + 1], frames[idx + 2]], dim=1)
    else:
        x = frames[idx].unsqueeze(1)
    return (x - input_offset) / input_scale


# ------------------------------------------------------------------------------
# 3. Pure-PyTorch 3D Validation Matcher (Jaccard, Precision, Recall, RMSE)
# ------------------------------------------------------------------------------
def compute_val_metrics_pure(
    y_out_all: torch.Tensor,
    em_tar: EmitterSetPure,
    channels_in: int,
    phot_max: float,
    z_max: float,
    px_size: Tuple[float, float],
    raw_th: float = 0.5,
    dist_lat_nm: float = 250.0,
    dist_ax_nm: float = 500.0,
    phot_min: float = 100.0,
    dist_vol_nm=None,
    match_dims=3,
    xy_origin=(0.0, 0.0),
):
    preds = decode_nms_postprocess(
        y_out_all,
        raw_th=raw_th,
        phot_max=phot_max,
        z_max=z_max,
        px_size=px_size, xy_origin=xy_origin,
    )
    offset = 1 if channels_in == 3 else 0
    n_eval = y_out_all.shape[0]

    tar_f = (em_tar.frame_ix.long() - offset).cpu().numpy()
    tar_xyz = em_tar.xyz_px.cpu().numpy()
    tar_phot = em_tar.phot.cpu().numpy()
    valid_tar = (tar_f >= 0) & (tar_f < n_eval)
    if phot_min is not None:
        valid_tar &= tar_phot >= phot_min
    tar_f = tar_f[valid_tar]
    tar_x_nm = tar_xyz[valid_tar, 0] * px_size[0]
    tar_y_nm = tar_xyz[valid_tar, 1] * px_size[1]
    tar_z_nm = tar_xyz[valid_tar, 2]

    pred_f = preds["frame_ix"]
    pred_x_nm = preds["x_nm"]
    pred_y_nm = preds["y_nm"]
    pred_z_nm = preds["z_nm"]

    n_tp = 0
    n_fp = 0
    n_fn = 0
    sq_lat = []
    sq_ax = []

    for f in range(n_eval):
        pm = pred_f == f
        tm = tar_f == f
        np_f = int(pm.sum())
        nt_f = int(tm.sum())
        if np_f == 0 and nt_f == 0:
            continue
        if np_f == 0:
            n_fn += nt_f
            continue
        if nt_f == 0:
            n_fp += np_f
            continue

        px_c = np.stack([pred_x_nm[pm], pred_y_nm[pm], pred_z_nm[pm]], axis=1)
        tx_c = np.stack([tar_x_nm[tm], tar_y_nm[tm], tar_z_nm[tm]], axis=1)
        # torch.cdist switches algorithms on larger sets. Use the same
        # distances as the reference for gating and tie-breaking in dense frames.
        pred = torch.from_numpy(px_c)
        target = torch.from_numpy(tx_c)
        cost = torch.cdist(pred[:, :match_dims], target[:, :match_dims])
        for axes, limit in ((slice(2), dist_lat_nm), (slice(2, 3), dist_ax_nm), (slice(3), dist_vol_nm)):
            if limit is not None:
                cost[torch.cdist(pred[:, axes], target[:, axes]) > limit] = torch.inf
        cost = cost.numpy()

        # DECODE's GreedyHungarianMatching consumes the nearest remaining pair.
        tp_f = 0
        while np.isfinite(cost).any():
            row, col = np.unravel_index(cost.argmin(), cost.shape)
            squared = np.square(px_c[row] - tx_c[col])
            sq_lat.append(squared[0] + squared[1])
            sq_ax.append(squared[2])
            tp_f += 1
            cost[row, :] = np.inf
            cost[:, col] = np.inf
        n_tp += tp_f
        n_fp += np_f - tp_f
        n_fn += nt_f - tp_f

    prec = n_tp / max(1, n_tp + n_fp)
    rec = n_tp / max(1, n_tp + n_fn)
    jac = n_tp / max(1, n_tp + n_fp + n_fn)
    rmse_lat = float(np.sqrt(np.mean(sq_lat))) if sq_lat else float("nan")
    rmse_ax = float(np.sqrt(np.mean(sq_ax))) if sq_ax else float("nan")
    return jac, prec, rec, rmse_lat, rmse_ax


# ------------------------------------------------------------------------------
# 4. Main Training Routine
# ------------------------------------------------------------------------------
def save_overwritten_checkpoint(ckpt_file: Path, model, optimizer, lr_scheduler, epoch: int, best_val: float, args, wandb_run=None):
    tmp_file = ckpt_file.with_suffix(".tmp")
    torch.save({
        "epoch": epoch,
        "best_val_loss": best_val,
        "loss_version": 2,
        "wandb_run": wandb_run,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": lr_scheduler.state_dict(),
        "args": vars(args),
    }, tmp_file)
    tmp_file.replace(ckpt_file)


def parse_args():
    default_project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Pure-PyTorch DECODE Training (MPS / CUDA / CPU)")
    parser.add_argument("--project-dir", type=str, default=str(default_project))
    parser.add_argument("--out-dir", type=str, default="localisation_workflow/decode_model_pure")
    parser.add_argument("--vae-ckpt", type=str, default="localisation_workflow/vae_background/bg_vae_64x64.pt")
    parser.add_argument("--manifest", type=str, default="localisation_workflow/vae_background/noise_maps/manifest.json")
    parser.add_argument("--cal-rel", type=str, default="data/bead_calibration/Beadzstack_3DLens_3dcal.mat")
    parser.add_argument("--param-rel", type=str, default="localisation_workflow/param_run.yaml")

    parser.add_argument("--device", type=str, default="auto", help="auto (CUDA->MPS->CPU), mps, cuda, cpu")
    parser.add_argument("--torch-threads", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--channels-in", type=int, choices=[1, 3], default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pseudo-ds-size", type=int, default=4096, help="Simulated frames per training epoch")
    parser.add_argument("--test-size", type=int, default=256, help="Simulated frames for validation")
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient norm clip (<=0 disables)")

    parser.add_argument("--em-av-min", type=float, default=12.0)
    parser.add_argument("--em-av-max", type=float, default=120.0)
    parser.add_argument("--lifetime-avg", type=float, default=1.0)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    tracking = parser.add_argument_group("Weights & Biases")
    tracking.add_argument("--wandb", action="store_true", help="Enable remote training metrics")
    tracking.add_argument("--wandb-project", help="W&B project (default: mphys-localisation)")
    tracking.add_argument("--wandb-entity", help="W&B user or team")
    tracking.add_argument("--wandb-name", help="Display name for the run")
    tracking.add_argument("--wandb-api-key-file", metavar="PATH",
                          help="Read the API key from a file (default: WANDB_API_KEY or wandb login)")
    return parser.parse_args()


def main():
    with ExitStack() as stack:
        train(parse_args(), stack)


def train(args, stack):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    project_dir = Path(args.project_dir).resolve()
    out_dir = (project_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_str = resolve_infer_device(args.device)
    torch_dev = torch.device(dev_str)
    if dev_str == "cpu":
        torch.set_num_threads(args.torch_threads)

    print("=" * 76, flush=True)
    print(f"Pure-PyTorch DECODE + VAE Training | Device: {dev_str}", flush=True)
    print(f"Project Root : {project_dir}", flush=True)
    print(f"Output Folder: {out_dir}", flush=True)
    print(f"Channels In  : {args.channels_in} | Batch: {args.batch_size} | LR: {args.lr:.1e}", flush=True)
    print(f"Emitter Range: Uniform({args.em_av_min}, {args.em_av_max}) per 32-frame chunk", flush=True)
    print("=" * 76, flush=True)

    param = yaml.safe_load((project_dir / args.param_rel).read_text())
    cam = param["Camera"]
    scaling = param["Scaling"]
    hp = param["HyperParameter"]
    ev = param.get("Evaluation", {})

    input_offset = float(scaling.get("input_offset", 130.0))
    input_scale = float(scaling.get("input_scale", 140.0))
    phot_max = float(scaling.get("phot_max", 6000.0))
    z_max = float(scaling.get("z_max", 250.0))
    bg_max = float(scaling.get("bg_max", 260.0))
    px_size = (float(cam["px_size"][0]), float(cam["px_size"][1]))
    raw_th = float(param.get("PostProcessingParam", {}).get("raw_th", 0.5))
    phot_min = hp["emitter_label_photon_min"]

    # Update param dict and save param_run_used.yaml for decode_pure.infer compatibility
    hp["channels_in"] = int(args.channels_in)
    hp["epochs"] = int(args.epochs)
    hp["batch_size"] = int(args.batch_size)
    hp["pseudo_ds_size"] = int(args.pseudo_ds_size)
    if args.grad_clip is None:
        args.grad_clip = 0.03 if hp["grad_mod"] else 0.0
    hp["opt_param"].update(lr=args.lr, weight_decay=args.weight_decay)
    hp["grad_mod"] = bool(args.grad_clip > 0)
    hp["max_number_targets"] = max(250, int(args.em_av_max * 3))
    sim_param = param["Simulation"]
    sim_param.update(lifetime_avg=args.lifetime_avg, emitter_av=(args.em_av_min + args.em_av_max) / 2)
    param["TestSet"].update(test_size=args.test_size, frame_extent=sim_param["psf_extent"],
                            img_size=sim_param["img_size"])
    param["Hardware"].update(device=dev_str, device_simulation=dev_str, torch_threads=args.torch_threads)
    xy_origin = tuple(axis[0] + 0.5 for axis in sim_param["psf_extent"][:2])
    target_options = dict(phot_min=phot_min, extent=sim_param["psf_extent"],
                          n_max=hp["max_number_targets"], disabled_attributes=hp["disabled_attributes"])
    param["InOut"]["calibration_file"] = str(project_dir / args.cal_rel)
    param["InOut"]["experiment_out"] = str(out_dir)
    (out_dir / "param_run_used.yaml").write_text(yaml.safe_dump(param, sort_keys=False))

    sim = build_vae_simulator(project_dir, args.vae_ckpt, args.manifest, args.cal_rel, args.param_rel,
                              lifetime_avg=args.lifetime_avg, device=dev_str)
    model = model_from_params(param).to(torch_dev)
    chweight = hp["chweight_stat"] or (1.0, 1.0)
    criterion = GaussianMMLossPure(
        img_shape=sim.psf.img_shape, xy_origin=xy_origin,
        phot_max=phot_max,
        z_max=z_max,
        bg_max=bg_max,
        chweight_em=chweight[0], chweight_bg=chweight[1],
    ).to(torch_dev)

    optim_cls = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[hp["optimizer"]]
    optimizer = optim_cls(model.parameters(), **hp["opt_param"])
    scheduler_cls = {"ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
                     "StepLR": torch.optim.lr_scheduler.StepLR}[hp["learning_rate_scheduler"]]
    schedule = hp["learning_rate_scheduler_param"].copy()
    schedule.pop("verbose", None)  # Removed from modern PyTorch.
    lr_scheduler = scheduler_cls(optimizer, **schedule)

    ckpt_latest_path = out_dir / "ckpt_latest.pt"
    model_best_path = out_dir / "model_best.pt"
    model_final_path = out_dir / "model_final.pt"
    log_csv_path = out_dir / "training_log.csv"

    start_epoch = 0
    best_val_loss = float("inf")
    wandb_ref = {}
    if args.resume and ckpt_latest_path.exists():
        # Keep legacy AdamW step counters on CPU when resuming with PyTorch 1.12.
        ckpt_data = _torch_load_compat(ckpt_latest_path, map_location="cpu")
        wandb_ref = ckpt_data.get("wandb_run") or {}
        model.load_state_dict(ckpt_data["model_state"])
        optimizer.load_state_dict(ckpt_data["optimizer_state"])
        start_epoch = int(ckpt_data["epoch"]) + 1
        if ckpt_data.get("loss_version") == 2:
            lr_scheduler.load_state_dict(ckpt_data["scheduler_state"])
            best_val_loss = float(ckpt_data["best_val_loss"])
        else:
            print("[RESUME] Starting a new validation-loss history for the corrected objective.", flush=True)
        print(f"[RESUME] Resuming from epoch {start_epoch + 1} (best_val_loss={best_val_loss:.4f})", flush=True)

    run = None
    if args.wandb:
        try:
            import wandb
        except ModuleNotFoundError as exc:
            if exc.name != "wandb":
                raise
            raise SystemExit("--wandb requires the SDK. Install it with: python -m pip install wandb") from exc

        if args.wandb_api_key_file:
            key = Path(args.wandb_api_key_file).expanduser().read_text().strip()
            if not key:
                raise ValueError("The W&B API key file is empty")
            os.environ["WANDB_API_KEY"] = key
        run = stack.enter_context(wandb.init(
            project=args.wandb_project or wandb_ref.get("project") or os.getenv("WANDB_PROJECT") or "mphys-localisation",
            entity=args.wandb_entity or wandb_ref.get("entity") or None,
            name=args.wandb_name, id=wandb_ref.get("id"), resume="allow" if wandb_ref else None,
            dir=str(out_dir),
            config={"hyperparameters": hp, "simulation": sim_param, "device": dev_str,
                    "seed": args.seed, "em_av_min": args.em_av_min, "em_av_max": args.em_av_max},
        ))
        wandb_ref = {"id": run.id, "entity": run.entity, "project": run.project}
        run.define_metric("epoch/*", step_metric="epoch")
        run.define_metric("train/batch_loss", step_metric="train/epoch_progress")
        if run.url:
            print(f"[W&B] {run.url}", flush=True)

    if not log_csv_path.exists() or not args.resume:
        with open(log_csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "lr", "train_loss", "val_loss",
                "jaccard", "precision", "recall", "rmse_lat_nm", "rmse_ax_nm", "epoch_sec",
            ])

    # Fixed held-out validation set with balanced intra-epoch density
    print(f"[INFO] Simulating {args.test_size} held-out validation frames...", flush=True)
    t_val_sim = time.time()
    val_frames, val_bg, val_em = simulate_variable_density_epoch(
        sim, args.test_size, args.channels_in, args.em_av_min, args.em_av_max
    )
    val_xyz, val_phot, val_mask = prepare_padded_targets(
        val_em, val_frames.shape[0], args.channels_in, torch_dev, **target_options
    )
    val_offset = 1 if args.channels_in == 3 else 0
    val_bg_center = val_bg[val_offset : val_frames.shape[0] - val_offset]
    print(f"[OK] Validation set ready ({time.time() - t_val_sim:.1f}s).", flush=True)

    t0_all = time.time()
    current_epoch = start_epoch

    try:
        for epoch in range(start_epoch, args.epochs):
            current_epoch = epoch
            t0_ep = time.time()
            lr_now = optimizer.param_groups[0]["lr"]

            # Give each epoch its own reproducible acquisition, including after resume.
            epoch_seed = (args.seed + epoch + 1) % (2**32)
            torch.manual_seed(epoch_seed)
            np.random.seed(epoch_seed)
            # Simulate fresh training acquisition each epoch
            print(f"\rEpoch {epoch + 1:>3}/{args.epochs} | simulating {args.pseudo_ds_size} frames...", end="", flush=True)
            tr_frames, tr_bg, tr_em = simulate_variable_density_epoch(
                sim, args.pseudo_ds_size, args.channels_in, args.em_av_min, args.em_av_max
            )
            tr_xyz, tr_phot, tr_mask = prepare_padded_targets(
                tr_em, tr_frames.shape[0], args.channels_in, torch_dev, **target_options
            )
            tr_offset = 1 if args.channels_in == 3 else 0
            tr_bg_center = tr_bg[tr_offset : tr_frames.shape[0] - tr_offset]

            # Train loop
            model.train()
            n_tr = tr_xyz.shape[0]
            total_batches = n_tr // args.batch_size
            if total_batches == 0:
                raise ValueError("Training acquisition must contain at least one full batch of windows")
            perm = torch.randperm(n_tr, device=torch_dev)
            tr_loss_sum = 0.0
            n_tr_batches = 0

            for b0 in range(0, total_batches * args.batch_size, args.batch_size):
                b_ix = perm[b0 : b0 + args.batch_size]
                x_b = build_input_batch(tr_frames, b_ix, args.channels_in, input_offset, input_scale)
                optimizer.zero_grad(set_to_none=True)
                y_b = model(x_b)
                loss = criterion(y_b, tr_xyz[b_ix], tr_phot[b_ix], tr_mask[b_ix], tr_bg_center[b_ix])
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                loss_val = float(loss.item())
                tr_loss_sum += loss_val
                n_tr_batches += 1

                pct = n_tr_batches / max(1, total_batches)
                if run is not None and (n_tr_batches == 1 or n_tr_batches % 10 == 0 or n_tr_batches == total_batches):
                    run.log({"train/batch_loss": loss_val, "train/epoch_progress": epoch + pct})
                bar_len = 20
                filled = int(bar_len * pct)
                bar = "=" * filled + "-" * (bar_len - filled)
                print(
                    f"\rEpoch {epoch + 1:>3}/{args.epochs} [{bar}] {pct*100:5.1f}% "
                    f"({n_tr_batches}/{total_batches}) | L={loss_val:.2e} | {time.time() - t0_ep:.1f}s",
                    end="",
                    flush=True,
                )

            print("\r" + " " * 96 + "\r", end="", flush=True)
            train_loss = tr_loss_sum / max(1, n_tr_batches)
            if not math.isfinite(train_loss):
                raise RuntimeError(f"Non-finite training loss ({train_loss}) at epoch {epoch + 1}")

            # Validation loop
            model.eval()
            val_loss_sum = 0.0
            val_outs = []
            with torch.no_grad():
                for b0 in range(0, val_xyz.shape[0], args.batch_size):
                    b_ix = torch.arange(b0, min(val_xyz.shape[0], b0 + args.batch_size), device=torch_dev)
                    x_b = build_input_batch(val_frames, b_ix, args.channels_in, input_offset, input_scale)
                    y_b = model(x_b)
                    v_loss = criterion(y_b, val_xyz[b_ix], val_phot[b_ix], val_mask[b_ix], val_bg_center[b_ix])
                    val_loss_sum += float(v_loss.item()) * len(b_ix)
                    val_outs.append(y_b)

            val_loss = val_loss_sum / len(val_xyz)
            y_val_all = torch.cat(val_outs, dim=0)
            jac, prec, rec, rmse_lat, rmse_ax = compute_val_metrics_pure(
                y_val_all, val_em, args.channels_in, phot_max, z_max, px_size,
                raw_th=raw_th,
                dist_lat_nm=ev.get("dist_lat"),
                dist_ax_nm=ev.get("dist_ax"), dist_vol_nm=ev.get("dist_vol"),
                match_dims=ev["match_dims"], xy_origin=xy_origin,
                phot_min=phot_min,
            )

            if not math.isfinite(val_loss):
                raise RuntimeError(f"Non-finite validation loss at epoch {epoch + 1}")
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_loss)
            else:
                lr_scheduler.step()
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_best_path)

            save_overwritten_checkpoint(ckpt_latest_path, model, optimizer, lr_scheduler, epoch, best_val_loss, args, wandb_ref)

            ep_sec = time.time() - t0_ep
            eta_min = (ep_sec * (args.epochs - epoch - 1)) / 60.0
            best_tag = " *BEST*" if is_best else ""
            print(
                f"Epoch {epoch + 1:>3}/{args.epochs} | lr={lr_now:.2e} | "
                f"Train={train_loss:.4f} | Val={val_loss:.4f}{best_tag} | "
                f"Jac={jac:.3f} (P={prec:.3f}, R={rec:.3f}) | "
                f"RMSE_xy={rmse_lat:.1f}nm, z={rmse_ax:.1f}nm | "
                f"{ep_sec:.1f}s (ETA {eta_min:.1f}m)",
                flush=True,
            )

            with open(log_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1, f"{lr_now:.6e}", f"{train_loss:.6f}", f"{val_loss:.6f}",
                    f"{jac:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{rmse_lat:.2f}", f"{rmse_ax:.2f}", f"{ep_sec:.2f}",
                ])

            if run is not None:
                run.log({"epoch": epoch + 1, "epoch/lr": lr_now, "epoch/train_loss": train_loss,
                         "epoch/val_loss": val_loss, "epoch/best_val_loss": best_val_loss,
                         "epoch/jaccard": jac, "epoch/precision": prec, "epoch/recall": rec,
                         "epoch/rmse_lat_nm": rmse_lat, "epoch/rmse_ax_nm": rmse_ax,
                         "epoch/duration_sec": ep_sec})

        torch.save(model.state_dict(), model_final_path)
        total_min = (time.time() - t0_all) / 60.0
        print("=" * 76, flush=True)
        print(f"[DONE] Finished in {total_min:.2f} min. Best Val Loss = {best_val_loss:.4f}", flush=True)
        print(f"  Best model : {model_best_path}", flush=True)
        print(f"  Final model: {model_final_path}", flush=True)
        print("=" * 76, flush=True)

    except KeyboardInterrupt:
        print(f"Interrupted during epoch {current_epoch + 1}. Resume from the last completed epoch.", flush=True)
        sys.exit(130)


if __name__ == "__main__":
    main()

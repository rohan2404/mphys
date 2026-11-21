"""
Pure-PyTorch 3D SMLM Simulator with Frozen Conv-Beta-VAE Background.
Zero dependency on `decode` or `spline`; runs on CPU, CUDA, and Apple Silicon MPS.

Mirrors the DECODE + VAE simulation pipeline from `05_simulator.ipynb` and `train_decode.py`:
  1. EmitterSamplerPure: Poisson/blinking 3D emitters with normally distributed photon flux
  2. CubicSplinePSFPure: 3D tricubic spline PSF rendering from `.mat` calibration
  3. VAEBackgroundPure: Frozen Conv-Beta-VAE decoder + empirical manifest scaling
  4. CameraNoisePure: ONI Nanoimager sCMOS Poisson shot noise + Gaussian read noise
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .model import resolve_infer_device
from .spline_psf import CubicSplinePSFPure


def _torch_load_compat(path: Union[str, Path], map_location: Union[str, torch.device]):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


class SmoothBgBetaVAE(nn.Module):
    """Matches the 8x8-bottleneck Conv-Beta-VAE trained in 04_vae_background.ipynb."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, 64), nn.SiLU(),
        )
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, padding_mode="replicate"),
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc_dec(z)).view(-1, 64, 8, 8)
        return self.dec(h)


class VAEBackgroundPure:
    """
    Pure-PyTorch background sampler powered by the frozen Conv-Beta-VAE decoder.
    Samples z ~ N(0, I), standardizes per patch, and rescales to empirical
    (mu_ROI, sigma_ROI) physical photon units from noise_maps/manifest.json.
    """

    def __init__(self, vae_model: SmoothBgBetaVAE, manifest_list: list):
        self.vae = vae_model.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.roi_means = np.array([m["roi_mean"] for m in manifest_list], dtype=np.float32)
        self.roi_stds = np.array([m["roi_std"] for m in manifest_list], dtype=np.float32)

    @torch.no_grad()
    def sample(self, size: Tuple[int, ...], device: Union[str, torch.device] = "cpu", chunk_size: int = 256) -> torch.Tensor:
        target_h, target_w = size[-2], size[-1]
        batch_shape = size[:-2]
        n_samples = int(np.prod(batch_shape)) if len(batch_shape) > 0 else 1

        vae_dev = next(self.vae.parameters()).device
        out_chunks = []
        for start in range(0, n_samples, chunk_size):
            n_c = min(chunk_size, n_samples - start)
            z = torch.randn(n_c, self.vae.latent_dim, device=vae_dev)
            b_hat = self.vae.decode(z)
            if b_hat.shape[-2:] != (target_h, target_w):
                b_hat = F.interpolate(b_hat, size=(target_h, target_w), mode="bilinear", align_corners=False)
            b_hat = (b_hat - b_hat.mean(dim=(2, 3), keepdim=True)) / (b_hat.std(dim=(2, 3), keepdim=True) + 1e-6)

            idx = np.random.randint(0, len(self.roi_means), size=n_c)
            mu_t = torch.from_numpy(self.roi_means[idx]).to(vae_dev).view(n_c, 1, 1, 1)
            sig_scale = np.random.uniform(0.5, 1.5, size=n_c).astype(np.float32)
            sig_t = torch.from_numpy(self.roi_stds[idx] * sig_scale).to(vae_dev).view(n_c, 1, 1, 1)

            b_phys = torch.clamp(mu_t + sig_t * b_hat, min=1.0).squeeze(1)
            out_chunks.append(b_phys.to(device))

        return torch.cat(out_chunks, dim=0).reshape(size)


@dataclass
class EmitterSetPure:
    """Lightweight pure-PyTorch container matching DECODE's EmitterSet interface."""

    xyz_px: torch.Tensor
    phot: torch.Tensor
    frame_ix: torch.Tensor
    id: torch.Tensor

    def __len__(self) -> int:
        return int(self.xyz_px.shape[0])

    def get_subset_frame(self, f_start: int, f_end: Optional[int] = None) -> "EmitterSetPure":
        if f_end is None:
            f_end = f_start
        mask = (self.frame_ix >= f_start) & (self.frame_ix <= f_end)
        return EmitterSetPure(
            xyz_px=self.xyz_px[mask],
            phot=self.phot[mask],
            frame_ix=self.frame_ix[mask],
            id=self.id[mask],
        )

    def to(self, device: Union[str, torch.device]) -> "EmitterSetPure":
        return EmitterSetPure(
            xyz_px=self.xyz_px.to(device),
            phot=self.phot.to(device),
            frame_ix=self.frame_ix.to(device),
            id=self.id.to(device),
        )


class EmitterSamplerPure:
    """
    Pure-PyTorch emitter sampler matching DECODE's blinking/Poisson emitter sampling.
    """

    def __init__(
        self,
        emitter_av: float = 24.0,
        emitter_extent: Sequence[Sequence[float]] = ((-0.5, 63.5), (-0.5, 63.5), (-200.0, 200.0)),
        intensity_mu_sig: Tuple[float, float] = (1800.0, 750.0),
        lifetime_avg: float = 1.0,
        frame_range: Tuple[int, int] = (0, 0),
        device: Union[str, torch.device] = "cpu",
        intensity_th: Optional[float] = None,
    ):
        self.emitter_av = float(emitter_av)
        self.emitter_extent = tuple((float(a), float(b)) for a, b in emitter_extent)
        self.intensity_mu_sig = (float(intensity_mu_sig[0]), float(intensity_mu_sig[1]))
        self.lifetime_avg = float(lifetime_avg)
        self.frame_range = (int(frame_range[0]), int(frame_range[1]))
        self.device = torch.device(device)

        self.intensity_th = 1e-8 if intensity_th is None else float(intensity_th)
        if self.lifetime_avg <= 0:
            raise ValueError("lifetime_avg must be positive")

    def sample(self) -> EmitterSetPure:
        f0, f1 = self.frame_range
        start = f0 - 3 * self.lifetime_avg
        stop = f1 + 3 * self.lifetime_avg
        n = np.random.poisson(self.emitter_av * (stop - start + 1) / (self.lifetime_avg + 1))

        # Draw each track on CPU as in DECODE, then transfer its visible frames.
        extent = torch.tensor(self.emitter_extent, dtype=torch.float32)
        xyz = torch.rand(n, 3) * (extent[:, 1] - extent[:, 0]) + extent[:, 0]
        mu, sigma = self.intensity_mu_sig
        flux = torch.normal(mu, sigma, size=(n,)).clamp_min(self.intensity_th)
        t0 = start + torch.rand(n) * (stop - start)
        duration = torch.distributions.Exponential(1 / self.lifetime_avg).rsample((n,))
        te = t0 + duration
        first, last = t0.floor().long(), te.floor().long()
        ids = torch.arange(n)

        # LooseEmitterSet emits interior frames first, then first and last
        # partial exposures. Single-frame acquisitions use the same blink law.
        counts = (last - first - 1).clamp_min(0)
        inner_ids = ids.repeat_interleave(counts)
        offsets = torch.arange(len(inner_ids)) - (counts.cumsum(0) - counts)[inner_ids]
        has_last = last > first
        frame = torch.cat((first[inner_ids] + 1 + offsets, first, last[has_last]))
        track = torch.cat((inner_ids, ids, ids[has_last]))
        first_dt = torch.minimum(te - t0, first + 1 - t0)
        last_dt = torch.minimum(te - t0, te - last)
        phot = torch.cat((flux[inner_ids], flux * first_dt, (flux * last_dt)[has_last]))
        keep = (frame >= f0) & (frame <= f1)
        return EmitterSetPure(xyz[track[keep]], phot[keep], frame[keep], track[keep]).to(self.device)


class CameraNoisePure:
    """
    ONI Nanoimager sCMOS camera noise model:
      Photon shot noise and electron read noise are applied before ADU quantisation.
    """

    def __init__(
        self,
        baseline: float = 400.0,
        e_per_adu: float = 0.46,
        qe: float = 0.8,
        read_sigma: float = 1.6,
        convert2photons: bool = True,
        spur_noise: float = 0.0,
        em_gain: Optional[float] = None,
    ):
        self.baseline = float(baseline)
        self.e_per_adu = float(e_per_adu)
        self.qe = float(qe)
        self.read_sigma = float(read_sigma)
        self.convert2photons = bool(convert2photons)
        self.spur_noise = float(spur_noise)
        self.em_gain = em_gain

    @torch.no_grad()
    def forward(self, phot_frames: torch.Tensor) -> torch.Tensor:
        rate = torch.clamp(phot_frames, min=0.0) * self.qe + self.spur_noise
        try:
            electrons = torch.poisson(rate)
        except (NotImplementedError, RuntimeError):
            electrons = torch.poisson(rate.cpu()).to(phot_frames.device)

        if self.em_gain is not None:
            electrons = torch.distributions.Gamma(electrons, 1 / self.em_gain, validate_args=False).sample()
        electrons = electrons + torch.randn_like(electrons) * self.read_sigma
        adu = ((electrons / self.e_per_adu).floor() + self.baseline).clamp_min(0)
        if self.convert2photons:
            return self.backward(adu)
        return adu


    def backward(self, adu):
        electrons = (adu - self.baseline) * self.e_per_adu
        if self.em_gain is not None:
            electrons = electrons / self.em_gain
        return (electrons - self.spur_noise) / self.qe


class SimulationPure:
    """
    Pure-PyTorch SMLM frame simulator combining PSF, EmitterSamplerPure,
    VAEBackgroundPure, and CameraNoisePure.
    """

    def __init__(
        self,
        psf: CubicSplinePSFPure,
        em_sampler: EmitterSamplerPure,
        background: VAEBackgroundPure,
        noise: CameraNoisePure,
        frame_range: Tuple[int, int] = (0, 0),
    ):
        self.psf = psf
        self.em_sampler = em_sampler
        self.background = background
        self.noise = noise
        self.frame_range = (int(frame_range[0]), int(frame_range[1]))

    @torch.no_grad()
    def forward(self, em_set: EmitterSetPure) -> Tuple[torch.Tensor, torch.Tensor]:
        f0, f1 = self.frame_range
        n_frames = f1 - f0 + 1
        H, W = self.psf.img_shape
        dev = self.psf.device

        rel_frame_ix = em_set.frame_ix - f0
        fg_frames = self.psf.forward(em_set.xyz_px, em_set.phot, frame_ix=rel_frame_ix, n_frames=n_frames)
        bg_maps = self.background.sample((n_frames, H, W), device=dev)
        noisy_frames = self.noise.forward(fg_frames + bg_maps)
        return noisy_frames, bg_maps

    @torch.no_grad()
    def sample(self) -> Tuple[EmitterSetPure, torch.Tensor, torch.Tensor]:
        em_set = self.em_sampler.sample()
        noisy_frames, bg_maps = self.forward(em_set)
        return em_set, noisy_frames, bg_maps


def build_vae_simulator(
    project_dir: Union[str, Path] = ".",
    ckpt_rel: str = "localisation_workflow/vae_background/bg_vae_64x64.pt",
    manifest_rel: str = "localisation_workflow/vae_background/noise_maps/manifest.json",
    cal_rel: str = "data/bead_calibration/Beadzstack_3DLens_3dcal.mat",
    param_rel: str = "localisation_workflow/param_run.yaml",
    z_range_nm: Optional[Tuple[float, float]] = None,
    intensity_mu_sig: Optional[Tuple[float, float]] = None,
    lifetime_avg: Optional[float] = None,
    frame_range: Tuple[int, int] = (0, 0),
    roi_size: Tuple[int, int] = None,
    device: str = "auto",
) -> SimulationPure:
    """
    Zero-DECODE portable factory that builds a complete SimulationPure powered
    by the frozen Conv-Beta-VAE background and 3D cubic spline PSF calibration.
    """
    dev = resolve_infer_device(device)
    project_dir = Path(project_dir).resolve()

    ckpt = _torch_load_compat(project_dir / ckpt_rel, map_location=dev)
    vae_m = SmoothBgBetaVAE(latent_dim=int(ckpt["latent_dim"])).to(dev)
    vae_m.load_state_dict(ckpt["model_state"], strict=True)
    vae_m.eval()

    manifest_data = json.loads((project_dir / manifest_rel).read_text())
    param = yaml.safe_load((project_dir / param_rel).read_text())
    cam = param["Camera"]
    sim_param = param["Simulation"]

    psf = CubicSplinePSFPure.from_mat(
        mat_path=project_dir / cal_rel,
        img_shape=tuple(sim_param["img_size"]),
        roi_size=sim_param["roi_size"] if roi_size is None else roi_size,
        roi_auto_center=sim_param["roi_auto_center"],
        extent_origin=tuple(axis[0] for axis in sim_param["psf_extent"][:2]),
        device=dev,
    )
    em_sampler = EmitterSamplerPure(
        emitter_av=sim_param["emitter_av"],
        emitter_extent=(*sim_param["emitter_extent"][:2], z_range_nm or sim_param["emitter_extent"][2]),
        intensity_mu_sig=intensity_mu_sig or sim_param["intensity_mu_sig"],
        intensity_th=sim_param.get("intensity_th"),
        lifetime_avg=sim_param["lifetime_avg"] if lifetime_avg is None else lifetime_avg,
        frame_range=frame_range,
        device=dev,
    )
    bg = VAEBackgroundPure(vae_m, manifest_data)
    noise = CameraNoisePure(
        baseline=float(cam["baseline"]),
        e_per_adu=float(cam["e_per_adu"]),
        qe=float(cam["qe"]),
        read_sigma=float(cam["read_sigma"]),
        convert2photons=bool(cam.get("convert2photons", True)),
        spur_noise=cam.get("spur_noise", 0.0), em_gain=cam.get("em_gain"),
    )
    return SimulationPure(psf=psf, em_sampler=em_sampler, background=bg, noise=noise, frame_range=frame_range)


def simulation(
    sim_obj: Optional[SimulationPure] = None,
    emitter_av_bounds: Tuple[float, float] = (12.0, 120.0),
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, EmitterSetPure, float]:
    """
    Portable pure-PyTorch simulation() call matching 05_simulator.ipynb:
      - Samples a random target emitter count lambda ~ Uniform(*emitter_av_bounds)
      - Samples a fresh VAE background patch and simulates 3D PSF + sCMOS Poisson/read noise
    Returns: (frames_tensor, bg_tensor, emitter_set, sampled_em_av)
    """
    if sim_obj is None:
        sim_obj = build_vae_simulator()
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    target_em_av = float(np.random.uniform(*emitter_av_bounds))
    sim_obj.em_sampler.emitter_av = target_em_av
    em_set, frames, bg_maps = sim_obj.sample()
    return frames, bg_maps, em_set, target_em_av


if __name__ == "__main__":
    sim = build_vae_simulator(project_dir=Path(__file__).resolve().parents[1])
    frames, bg_maps, em_set, lam_av = simulation(sim, seed=2026)
    print(
        f"Simulated {tuple(frames.shape)} | lam={lam_av:.1f} | "
        f"emitters={len(em_set)} | bg_mean={bg_maps.mean().item():.1f} ph | "
        f"frame_range=[{frames.min().item():.1f}, {frames.max().item():.1f}] ph"
    )

"""
Pure-PyTorch DECODE Inference + SpatialIntegration NMS Post-Processing.
Works on CPU, CUDA, and Apple Silicon MPS with zero dependency on `decode` or `spline`.

Usage:
  python -m decode_pure.infer \
    --movie data/low_density_examples/RegistrationCheck_LowDensity_GapT_Exp1_GapT_SealG_F3.tif \
    --weights localisation_workflow/decode_model/model_best.pt \
    --param localisation_workflow/decode_model/param_run_used.yaml \
    --out localisation_workflow/decode_model/locs_SealG_F3.csv \
    --right-channel --device auto
"""

import argparse
import csv
from pathlib import Path
from typing import Union
import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from .model import load_sigma_munet, resolve_infer_device
from .simulator import CameraNoisePure


def decode_nms_postprocess(
    y_out: torch.Tensor,
    raw_th: float = 0.5,
    split_th: float = 0.6,
    phot_max: float = 6000.0,
    z_max: float = 250.0,
    px_size: tuple = (117.0, 117.0),
    bg_max: float = 260.0,
    xy_origin: tuple = (0.0, 0.0),
):
    """
    Exact pure-PyTorch mirror of decode.neuralfitter.post_processing.SpatialIntegration (norm_sum)
    + ParameterListRescale inverse scaling.

    Returns dict of 1D numpy arrays:
      frame_ix, x_px, y_px, x_nm, y_nm, z_nm, phot, prob, sigma_x_nm, sigma_y_nm, sigma_z_nm, sigma_phot, bg
    """
    with torch.no_grad():
        p = y_out[:, 0].clone()
        p_clip = torch.where(p > raw_th, p, torch.zeros_like(p))[:, None]
        pool = F.max_pool2d(p_clip, kernel_size=3, stride=1, padding=1)
        max_mask1 = torch.eq(p[:, None], pool).float()

        filt = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=p.dtype,
            device=p.device,
        ).view(1, 1, 3, 3)
        conv = F.conv2d(p[:, None], filt, padding=1)
        p_ps1 = max_mask1 * conv

        p_copy = p * (1.0 - max_mask1[:, 0])
        max_mask2 = torch.where(p_copy > split_th, torch.ones_like(p_copy), torch.zeros_like(p_copy))[:, None]
        p_ps2 = max_mask2 * conv

        p_nms = torch.clamp(p_ps1 + p_ps2, 0.0, 1.0).squeeze(1)
        active = p_nms >= raw_th

        coords = active.nonzero(as_tuple=False)  # (M, 3): [batch_frame, row, col]
        if coords.shape[0] == 0:
            empty = np.zeros((0,), dtype=np.float32)
            out_dict = {k: empty.copy() for k in [
                "x_px", "y_px", "x_nm", "y_nm", "z_nm", "phot", "prob",
                "sigma_x_nm", "sigma_y_nm", "sigma_z_nm", "sigma_phot", "bg",
            ]}
            out_dict["frame_ix"] = np.zeros((0,), dtype=np.int64)
            return out_dict

        frame_ix = coords[:, 0].cpu().numpy().astype(np.int64)
        row = coords[:, 1].float()
        col = coords[:, 2].float()

        prob = p_nms[active].cpu().numpy().astype(np.float32)
        feat = y_out.permute(1, 0, 2, 3)[:, active].cpu()  # (10, M)

        phot = (feat[1] * phot_max).numpy().astype(np.float32)
        x_px = ((row.cpu() + xy_origin[0]) + feat[2]).numpy().astype(np.float32)
        y_px = ((col.cpu() + xy_origin[1]) + feat[3]).numpy().astype(np.float32)
        z_nm = (feat[4] * z_max).numpy().astype(np.float32)

        sigma_phot = (feat[5] * phot_max).numpy().astype(np.float32)
        sigma_x_nm = (feat[6] * px_size[0]).numpy().astype(np.float32)
        sigma_y_nm = (feat[7] * px_size[1]).numpy().astype(np.float32)
        sigma_z_nm = (feat[8] * z_max).numpy().astype(np.float32)

        return {
            "frame_ix": frame_ix,
            "x_px": x_px,
            "y_px": y_px,
            "x_nm": x_px * float(px_size[0]),
            "y_nm": y_px * float(px_size[1]),
            "z_nm": z_nm,
            "phot": phot,
            "prob": prob,
            "sigma_x_nm": sigma_x_nm,
            "sigma_y_nm": sigma_y_nm,
            "sigma_z_nm": sigma_z_nm,
            "sigma_phot": sigma_phot,
            "bg": (feat[9] * bg_max).numpy(),
        }


def infer_movie(
    movie_path: Union[str, Path],
    param_yaml: Union[str, Path],
    weights_pt: Union[str, Path],
    out_csv: Union[str, Path] = None,
    right_channel_only: bool = True,
    batch_size: int = 16,
    raw_th: float = None,
    device: str = "auto",
):
    """Run sliding-window DECODE inference on a TIFF stack in pure PyTorch."""
    dev = resolve_infer_device(device)
    model, param = load_sigma_munet(param_yaml, weights_pt, device=dev)

    cam = param["Camera"]
    sc = param["Scaling"]
    ch_in = int(param["HyperParameter"]["channels_in"])
    if raw_th is None:
        raw_th = float(param.get("PostProcessingParam", {}).get("raw_th", 0.5))

    camera = CameraNoisePure(**{k: v for k, v in cam.items() if k != "px_size"})
    xy_origin = tuple(axis[0] + 0.5 for axis in param["TestSet"]["frame_extent"][:2])
    offset = float(sc["input_offset"])
    scale = float(sc["input_scale"])
    phot_max = float(sc["phot_max"])
    z_max = float(sc["z_max"])
    px_size = tuple(float(v) for v in cam["px_size"])

    ap = param["HyperParameter"]["arch_param"]
    stride = 2 ** max(int(ap["depth_shared"]), int(ap["depth_union"]))
    all_chunks = {k: [] for k in [
        "frame_ix", "x_px", "y_px", "x_nm", "y_nm", "z_nm", "phot", "prob",
        "sigma_x_nm", "sigma_y_nm", "sigma_z_nm", "sigma_phot", "bg",
    ]}

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    with tifffile.TiffFile(str(movie_path)) as tif, torch.no_grad():
        n_frames = len(tif.pages)
        height, width = tif.pages[0].shape
        x_start = width // 2 if right_channel_only else 0
        H, W = height, width - x_start
        pad_h, pad_w = -H % stride, -W % stride
        for start in range(0, n_frames, batch_size):
            batch_f = np.arange(start, min(start + batch_size, n_frames))
            # InferenceDataset repeats the nearest frame at both movie ends.
            windows = np.clip(batch_f[:, None] + np.arange(ch_in) - ch_in // 2, 0, n_frames - 1)
            first, last = int(windows.min()), int(windows.max())
            pages = np.stack([tif.pages[f].asarray()[:, x_start:] for f in range(first, last + 1)])
            frames = torch.as_tensor(pages.astype(np.float32), device=dev)
            if camera.convert2photons:
                frames = camera.backward(frames)
            frames = (frames - offset) / scale
            x_t = frames[torch.as_tensor(windows - first, device=dev)]
            if pad_h > 0 or pad_w > 0:
                x_t = F.pad(x_t, (0, pad_w, 0, pad_h), mode="replicate")
            y_out = model(x_t)
            if pad_h > 0 or pad_w > 0:
                y_out = y_out[:, :, :H, :W]
            res = decode_nms_postprocess(
                y_out, raw_th=raw_th, phot_max=phot_max, z_max=z_max, px_size=px_size,
                bg_max=sc["bg_max"], xy_origin=xy_origin,
            )
            res["frame_ix"] = np.array(batch_f, dtype=np.int64)[res["frame_ix"]]
            for k, v in res.items():
                all_chunks[k].append(v)

    merged = {k: (np.concatenate(v) if len(v) > 0 else np.zeros((0,))) for k, v in all_chunks.items()}

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(merged.keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            n_det = len(merged["frame_ix"])
            for row_i in range(n_det):
                writer.writerow([merged[k][row_i] for k in keys])
        print(f"[DONE] Saved {len(merged['frame_ix']):,} localisations to {out_path}", flush=True)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Pure-PyTorch DECODE Inference")
    parser.add_argument("--movie", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--param", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--right-channel", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--raw-th", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    infer_movie(
        movie_path=args.movie,
        param_yaml=args.param,
        weights_pt=args.weights,
        out_csv=args.out,
        right_channel_only=args.right_channel,
        batch_size=args.batch_size,
        raw_th=args.raw_th,
        device=args.device,
    )


if __name__ == "__main__":
    main()

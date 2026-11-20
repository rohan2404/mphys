"""
Pure-PyTorch 3D Cubic-Spline PSF evaluator (zero C++/CUDA `spline` extension required).
Evaluates the 64-coefficient tricubic polynomial per voxel from a calibrated `.mat` file
(e.g. `Beadzstack_3DLens_3dcal.mat`) on CPU, CUDA 11/12, or Apple Silicon MPS.
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import scipy.io
import torch


def load_calibration_mat(mat_path: Union[str, Path]):
    """Load `coeff`, `vx_size` (dz, dx, dy or dx, dy, dz), and `ref0` from a DECODE/SMAP 3dcal.mat file."""
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "SXY" in mat:
        sxy = mat["SXY"]
        spline_obj = sxy.cspline
        coeff = np.asarray(spline_obj.coeff, dtype=np.float32)
        vx_size = np.array([1.0, 1.0, float(spline_obj.dz)], dtype=np.float32)
        ref0 = np.array([
            float(spline_obj.x0) - 1.0,
            float(spline_obj.x0) - 1.0,
            float(spline_obj.z0),
        ], dtype=np.float32)
        return coeff, vx_size, ref0

    # Fallback for direct struct/dict formats
    coeff = np.asarray(mat["coeff"], dtype=np.float32)
    vx_size = np.asarray(mat.get("vx_size", [117.0, 117.0, 10.0]), dtype=np.float32)
    ref0 = np.asarray(mat.get("ref0", [coeff.shape[0] / 2.0, coeff.shape[1] / 2.0, coeff.shape[2] / 2.0]), dtype=np.float32)
    return coeff, vx_size, ref0


class CubicSplinePSFPure:
    """
    Pure-PyTorch 3D cubic spline PSF renderer matching decode.simulation.psf_kernel.CubicSplinePSF.
    """

    def __init__(
        self,
        coeff: np.ndarray,
        vx_size: Tuple[float, float, float],
        ref0: Tuple[float, float, float],
        img_shape: Tuple[int, int],
        roi_size: Tuple[int, int] = (17, 17),
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        # coeff shape: (X_spl, Y_spl, Z_spl, 64)
        self.coeff = torch.from_numpy(np.asarray(coeff, dtype=np.float32)).to(self.device)
        self.vx_size = torch.tensor(vx_size, dtype=torch.float32, device=self.device)
        self.ref0 = torch.tensor(ref0, dtype=torch.float32, device=self.device)
        self.img_shape = tuple(img_shape)
        self.roi_size = tuple(roi_size)

    @classmethod
    def from_mat(cls, mat_path: Union[str, Path], img_shape: Tuple[int, int], roi_size: Tuple[int, int] = (17, 17), device: str = "cpu"):
        coeff, vx_size, ref0 = load_calibration_mat(mat_path)
        return cls(coeff=coeff, vx_size=vx_size, ref0=ref0, img_shape=img_shape, roi_size=roi_size, device=device)

    def eval_rois(self, xyz_impl: torch.Tensor, phot: torch.Tensor) -> torch.Tensor:
        """
        Evaluate tricubic spline ROIs for N emitters at continuous spline-grid coordinates `xyz_impl` (N, 3).
        In DECODE/SMAP, the 64 coefficients per voxel follow c[i + 4*j + 16*k] * tx^i * ty^j * tz^k.
        """
        n_em = xyz_impl.shape[0]
        rx, ry = self.roi_size
        xs_max, ys_max, zs_max, _ = self.coeff.shape

        # Base grid offsets across ROI (0 .. rx-1, matching spline.PSF.forward_rois)
        ox = torch.arange(rx, device=self.device, dtype=torch.float32)
        oy = torch.arange(ry, device=self.device, dtype=torch.float32)
        gx, gy = torch.meshgrid(ox, oy, indexing="ij")  # (rx, ry)

        # Continuous spline coordinates per ROI pixel: (N, rx, ry)
        cx = xyz_impl[:, 0].view(n_em, 1, 1) + gx.unsqueeze(0)
        cy = xyz_impl[:, 1].view(n_em, 1, 1) + gy.unsqueeze(0)
        cz = xyz_impl[:, 2].view(n_em, 1, 1).expand(n_em, rx, ry)

        ix = torch.floor(cx).long()
        iy = torch.floor(cy).long()
        iz = torch.floor(cz).long()

        valid = (ix >= 0) & (ix < xs_max) & (iy >= 0) & (iy < ys_max) & (iz >= 0) & (iz < zs_max)
        ix_c = ix.clamp(0, xs_max - 1)
        iy_c = iy.clamp(0, ys_max - 1)
        iz_c = iz.clamp(0, zs_max - 1)

        tx = cx - ix_c.float()
        ty = cy - iy_c.float()
        tz = cz - iz_c.float()

        # Polynomial basis powers: 1, t, t^2, t^3
        px = torch.stack([torch.ones_like(tx), tx, tx * tx, tx * tx * tx], dim=-1)  # (N, rx, ry, 4)
        py = torch.stack([torch.ones_like(ty), ty, ty * ty, ty * ty * ty], dim=-1)
        pz = torch.stack([torch.ones_like(tz), tz, tz * tz, tz * tz * tz], dim=-1)

        # Outer product basis (N, rx, ry, 4, 4, 4) -> (N, rx, ry, 64) matching k*16 + j*4 + i
        basis = (pz.unsqueeze(-1).unsqueeze(-1) * py.unsqueeze(-2).unsqueeze(-1) * px.unsqueeze(-2).unsqueeze(-2)).reshape(n_em, rx, ry, 64)
        c_vox = self.coeff[ix_c, iy_c, iz_c]  # (N, rx, ry, 64)
        rois = (c_vox * basis).sum(dim=-1)
        rois = torch.where(valid, torch.clamp(rois, min=0.0), torch.zeros_like(rois))
        return rois * phot.view(n_em, 1, 1)

    def forward(
        self,
        xyz: torch.Tensor,
        phot: torch.Tensor,
        frame_ix: torch.Tensor = None,
        n_frames: int = 1,
    ) -> torch.Tensor:
        """
        Render N emitters at (x_px, y_px, z_nm) with photon counts `phot` onto a
        (n_frames, H, W) image stack.
        """
        H, W = self.img_shape
        frames = torch.zeros((n_frames, H, W), dtype=torch.float32, device=self.device)
        n_em = xyz.shape[0]
        if n_em == 0:
            return frames

        xyz = xyz.to(device=self.device, dtype=torch.float32)
        phot = phot.to(device=self.device, dtype=torch.float32)
        if frame_ix is None:
            frame_ix = torch.zeros((n_em,), dtype=torch.long, device=self.device)
        else:
            frame_ix = frame_ix.to(device=self.device, dtype=torch.long)

        rx, ry = self.roi_size
        half_x, half_y = rx // 2, ry // 2

        ix0 = torch.round(xyz[:, 0]).long()
        iy0 = torch.round(xyz[:, 1]).long()
        dx = xyz[:, 0] - ix0.float()
        dy = xyz[:, 1] - iy0.float()

        x_impl = self.ref0[0] - half_x - dx
        y_impl = self.ref0[1] - half_y - dy
        z_impl = self.ref0[2] + xyz[:, 2] / self.vx_size[2]
        xyz_impl = torch.stack([x_impl, y_impl, z_impl], dim=-1)

        rois = self.eval_rois(xyz_impl, phot)  # (N, rx, ry)

        ox = torch.arange(rx, device=self.device, dtype=torch.long) - half_x
        oy = torch.arange(ry, device=self.device, dtype=torch.long) - half_y
        gx, gy = torch.meshgrid(ox, oy, indexing="ij")  # (rx, ry)

        img_x = ix0.view(n_em, 1, 1) + gx.unsqueeze(0)
        img_y = iy0.view(n_em, 1, 1) + gy.unsqueeze(0)
        img_f = frame_ix.view(n_em, 1, 1).expand(n_em, rx, ry)

        in_bounds = (
            (img_f >= 0) & (img_f < n_frames)
            & (img_x >= 0) & (img_x < H)
            & (img_y >= 0) & (img_y < W)
        )
        flat_idx = (img_f * (H * W) + img_x * W + img_y)[in_bounds]
        vals = rois[in_bounds]
        frames.view(-1).index_add_(0, flat_idx, vals)
        return frames

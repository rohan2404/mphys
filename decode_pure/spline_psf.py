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
    """Load `coeff`, `vx_size` (dx, dy, dz), and `ref0` from a DECODE/SMAP 3dcal.mat file."""
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
        roi_size: Tuple[int, int] = None,
        device: str = "cpu",
        roi_auto_center: bool = False,
        extent_origin: Tuple[float, float] = (-0.5, -0.5),
    ):
        self.device = torch.device(device)
        # coeff shape: (X_spl, Y_spl, Z_spl, 64)
        self.coeff = torch.from_numpy(np.asarray(coeff, dtype=np.float32)).to(self.device)
        self.vx_size = torch.tensor(vx_size, dtype=torch.float32, device=self.device)
        self.ref0 = torch.tensor(ref0, dtype=torch.float32, device=self.device)
        self.img_shape = tuple(img_shape)
        self.roi_size = tuple(self.coeff.shape[:2] if roi_size is None else roi_size)
        if roi_auto_center and any(size % 2 == 0 for size in self.roi_size):
            raise ValueError("Auto-centering requires odd ROI dimensions")
        self.roi_ref = (self.ref0.new_tensor([(size - 1) // 2 for size in self.roi_size])
                        if roi_auto_center else self.ref0[:2])
        self.origin = self.ref0.new_tensor(extent_origin) + 0.5

    @classmethod
    def from_mat(cls, mat_path: Union[str, Path], img_shape: Tuple[int, int], roi_size: Tuple[int, int] = None, device: str = "cpu",
                 roi_auto_center: bool = False, extent_origin=(-0.5, -0.5)):
        coeff, vx_size, ref0 = load_calibration_mat(mat_path)
        return cls(coeff=coeff, vx_size=vx_size, ref0=ref0, img_shape=img_shape, roi_size=roi_size, device=device,
                   roi_auto_center=roi_auto_center, extent_origin=extent_origin)

    def eval_rois(self, xyz_impl: torch.Tensor, phot: torch.Tensor) -> torch.Tensor:
        """
        Evaluate tricubic spline ROIs for N emitters at continuous spline-grid coordinates `xyz_impl` (N, 3).
        In DECODE/SMAP, the 64 coefficients per voxel follow c[i + 4*j + 16*k] * tx^i * ty^j * tz^k.
        """
        n_em = xyz_impl.shape[0]
        rx, ry = self.roi_size
        xs_max, ys_max, zs_max, _ = self.coeff.shape

        # The kernel computes the fractional offset once for the entire ROI.
        xyz_impl = xyz_impl.to(self.coeff)
        base = xyz_impl.floor()
        tx, ty, tz = (xyz_impl - base).unbind(1)
        ix, iy, iz = base.long().unbind(1)
        ix = ix[:, None, None] + torch.arange(rx, device=self.device)[None, :, None]
        iy = iy[:, None, None] + torch.arange(ry, device=self.device)[None, None, :]
        valid = (ix >= 0) & (ix < xs_max) & (iy >= 0) & (iy < ys_max)
        # SMAP clamps the axial plane but retains the original fractional z.
        coeff = self.coeff[ix.clamp(0, xs_max - 1), iy.clamp(0, ys_max - 1),
                           iz.clamp(0, zs_max - 1)[:, None, None]]

        def powers(t):
            return torch.stack((torch.ones_like(t), t, t * t, t * t * t), dim=1)

        basis = (powers(tz)[:, :, None, None] * powers(ty)[:, None, :, None]
                 * powers(tx)[:, None, None, :]).reshape(n_em, 1, 1, 64)
        values = (coeff * basis).sum(dim=-1) * phot.to(self.coeff).reshape(n_em, 1, 1)
        return torch.where(valid, values, 1e-10)

    def forward(
        self,
        xyz: torch.Tensor,
        phot: torch.Tensor,
        frame_ix: torch.Tensor = None,
        n_frames: int = 1,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """
        Render N emitters at (x_px, y_px, z_nm) with photon counts `phot` onto a
        (n_frames, H, W) image stack in memory-bounded emitter chunks.
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
        ox = torch.arange(rx, device=self.device)
        oy = torch.arange(ry, device=self.device)
        gx, gy = torch.meshgrid(ox, oy, indexing="ij")  # (rx, ry)
        frames_flat = frames.view(-1)

        for start in range(0, n_em, chunk_size):
            end = min(n_em, start + chunk_size)
            xyz_c = xyz[start:end]
            phot_c = phot[start:end]
            f_c = frame_ix[start:end]
            nc = end - start

            # Match CubicSplinePSF.frame2roi_coord and coord2impl. The public
            # simulator takes lateral pixels and axial nanometres.
            xy = xyz_c[:, :2]
            ix0, iy0 = (xy - self.roi_ref).floor().long().unbind(1)
            xy_relative = (xy.remainder(1) + self.ref0[:2]) * self.vx_size[:2]
            xy_impl = (-xy_relative / self.vx_size[:2] + self.origin + self.ref0[:2]
                       - (self.roi_ref - self.ref0[:2]))
            z_impl = self.ref0[2] - xyz_c[:, 2] / self.vx_size[2]
            xyz_impl = torch.cat((xy_impl, z_impl[:, None]), dim=1)

            rois = self.eval_rois(xyz_impl, phot_c)  # (nc, rx, ry)

            img_x = ix0.view(nc, 1, 1) + gx.unsqueeze(0)
            img_y = iy0.view(nc, 1, 1) + gy.unsqueeze(0)
            img_f = f_c.view(nc, 1, 1).expand(nc, rx, ry)

            in_bounds = (
                (img_f >= 0) & (img_f < n_frames)
                & (img_x >= 0) & (img_x < H)
                & (img_y >= 0) & (img_y < W)
            )
            flat_idx = (img_f * (H * W) + img_x * W + img_y)[in_bounds]
            vals = rois[in_bounds]
            frames_flat.index_add_(0, flat_idx, vals)

        return frames

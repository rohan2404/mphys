"""
Zero-DECODE, zero-spline pure-PyTorch implementation of DECODE's SigmaMUNet,
NMS post-processing, and 3D cubic-spline PSF evaluator.

Compatible with modern Python (3.8 - 3.13+), PyTorch (1.12 - 2.x+),
CPU, CUDA 11/12, and Apple Silicon MPS.
"""

from .model import SigmaMUNet, load_sigma_munet
from .spline_psf import CubicSplinePSFPure
from .infer import infer_movie, decode_nms_postprocess
from .simulator import (
    SmoothBgBetaVAE,
    VAEBackgroundPure,
    EmitterSetPure,
    EmitterSamplerPure,
    CameraNoisePure,
    SimulationPure,
    build_vae_simulator,
    simulation,
)

__all__ = [
    "SigmaMUNet",
    "load_sigma_munet",
    "CubicSplinePSFPure",
    "infer_movie",
    "decode_nms_postprocess",
    "SmoothBgBetaVAE",
    "VAEBackgroundPure",
    "EmitterSetPure",
    "EmitterSamplerPure",
    "CameraNoisePure",
    "SimulationPure",
    "build_vae_simulator",
    "simulation",
]

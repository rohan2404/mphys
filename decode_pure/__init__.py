"""Pure-PyTorch DECODE model, post-processing, and cubic-spline simulation."""

from importlib import import_module

from .model import SigmaMUNet, load_sigma_munet
from .spline_psf import CubicSplinePSFPure

# The CLI modules are also public Python APIs. Load them on demand so that
# running python -m decode_pure.infer or .simulator executes the module once.
_exports = {
    "infer_movie": "infer",
    "decode_nms_postprocess": "infer",
    "SmoothBgBetaVAE": "simulator",
    "VAEBackgroundPure": "simulator",
    "EmitterSetPure": "simulator",
    "EmitterSamplerPure": "simulator",
    "CameraNoisePure": "simulator",
    "SimulationPure": "simulator",
    "build_vae_simulator": "simulator",
    "simulation": "simulator",
}

__all__ = ["SigmaMUNet", "load_sigma_munet", "CubicSplinePSFPure", *_exports]


def __getattr__(name):
    if name not in _exports:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_exports[name]}", __name__), name)
    globals()[name] = value
    return value

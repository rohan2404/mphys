# `decode_pure` — Dependency-Free Pure-PyTorch DECODE Module

`decode_pure` removes all legacy `decode==0.10.2`, `spline==0.10.0`, `python=3.8`, and `pytorch=1.12.0` constraints. It runs on **any modern Python (`3.8`–`3.13+`) and PyTorch (`1.12`–`2.x+`)** across **CPU, CUDA 11/12 (including H100), and Apple Silicon MPS**.

## Minimal `pip` Install (No Conda Needed)
```bash
pip install torch numpy scipy pyyaml tifffile
```

## Components
- [`model.py`](file:///Users/rohan/Documents/projects/mphys/localisation_workflow/decode_pure/model.py): Exact pure-PyTorch `SigmaMUNet` (`load_sigma_munet`) matching `decode==0.10.2` state_dict keys bit-for-bit (`0.00e+00` forward diff verified against `decode==0.10.2`). Loads `model_best.pt` with `strict=True`.
- [`infer.py`](file:///Users/rohan/Documents/projects/mphys/localisation_workflow/decode_pure/infer.py): Sliding-window 3-frame (or 1-frame) movie inference + DECODE `SpatialIntegration` NMS post-processing, exporting localisations to CSV.
- [`spline_psf.py`](file:///Users/rohan/Documents/projects/mphys/localisation_workflow/decode_pure/spline_psf.py): Pure-PyTorch 3D cubic-spline PSF evaluator (`CubicSplinePSFPure`) reading `.mat` calibration files directly (max diff `0.019` photons vs. `spline==0.10.0` C++ extension), including `forward()` frame rendering.
- [`simulator.py`](file:///Users/rohan/Documents/projects/mphys/localisation_workflow/decode_pure/simulator.py): Zero-DECODE 3D SMLM simulator (`build_vae_simulator`, `simulation`, `SimulationPure`, `VAEBackgroundPure`, `EmitterSamplerPure`, `CameraNoisePure`) matching `05_simulator.ipynb` and `param_run.yaml`.

## Running the Pure-PyTorch Simulator
```bash
python -m localisation_workflow.decode_pure.simulator
```
Or in Python:
```python
from localisation_workflow.decode_pure import build_vae_simulator, simulation

sim = build_vae_simulator(project_dir=".")
frames, bg_maps, em_set, lam_av = simulation(sim, emitter_av_bounds=(12.0, 120.0), seed=2026)
```

## Running Inference Once `model_best.pt` Finishes Training
```bash
python -m localisation_workflow.decode_pure.infer \
  --movie data/low_density_examples/RegistrationCheck_LowDensity_GapT_Exp1_GapT_SealG_F3.tif \
  --weights localisation_workflow/decode_model/model_best.pt \
  --param localisation_workflow/decode_model/param_run_used.yaml \
  --out localisation_workflow/decode_model/locs_SealG_F3.csv \
  --right-channel --device auto
```

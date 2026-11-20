# DECODE + VAE Training Handover (`train_decode.py`)

This guide explains how to set up the Conda environment and run [`train_decode.py`](file:///Users/rohan/Documents/projects/mphys/localisation_workflow/train_decode.py) on a local machine or a Linux GPU cluster.

---

## 1. Environment & Conda Installation

`decode==0.10.2` requires **Python 3.8** and **PyTorch 1.12** (along with the compiled `spline` PSF package from `turagalab`).

### Option A: Linux / Windows GPU Cluster (CUDA)
```bash
conda create -n decode_gpu python=3.8.20 -y
conda activate decode_gpu

# Install PyTorch 1.12 with CUDA (adjust cudatoolkit version to match your cluster GPU driver, e.g. 11.3 or 11.6)
conda install pytorch=1.12.1 torchvision cudatoolkit=11.3 -c pytorch -c conda-forge -y

# Install DECODE 0.10.2 + compiled 3D spline library
conda install decode=0.10.2 spline -c turagalab -c conda-forge -y

# Supporting scientific packages
conda install numpy=1.19.5 scipy=1.8.0 pyyaml tqdm -y
```

### Option B: CPU-Only (Local / Mac)
```bash
conda create -n decode_cpu python=3.8.20 -y
conda activate decode_cpu

conda install pytorch=1.12.0 cpuonly -c pytorch -c conda-forge -y
conda install decode=0.10.2 spline -c turagalab -c conda-forge -y
conda install numpy=1.19.5 scipy=1.8.0 pyyaml tqdm -y
```

*(Verified local environment versions: `python 3.8.20`, `torch 1.12.0.post2`, `decode 0.10.2`, `numpy 1.19.5`, `scipy 1.8.0`.)*

---

## 2. Files Needed on Another Computer / Cluster

Keep the repository folder structure intact (or copy these 5 files with their relative paths):
1. `localisation_workflow/train_decode.py` — training script
2. `localisation_workflow/vae_background/bg_vae_64x64.pt` — trained Conv-$\beta$-VAE background weights
3. `localisation_workflow/vae_background/noise_maps/manifest.json` — empirical $(\mu_{\mathrm{ROI}}, \sigma_{\mathrm{ROI}})$ photon scaling distribution
4. `data/bead_calibration/Beadzstack_3DLens_3dcal.mat` — calibrated 3D cubic spline PSF
5. `localisation_workflow/param_run.yaml` — your ONI Nanoimager DECODE configuration

*(You do **not** need to copy the raw multi-GB `.tif` movies or the individual `.npy` noise maps to train DECODE, since `VAEBackground` generates backgrounds purely from `bg_vae_64x64.pt` + `manifest.json`.)*

---

## 3. Running `train_decode.py`

Run from the **repository root** (`/path/to/mphys`):

### Full GPU Cluster Run (Default 3-Frame Temporal Mode, `--channels-in 3`)
```bash
python localisation_workflow/train_decode.py \
  --device auto \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 4
```

### Fast Single-Frame Mode (`--channels-in 1`)
If you want ~3× lower memory/compute per batch without temporal $(t-1, t, t+1)$ context:
```bash
python localisation_workflow/train_decode.py --device auto --channels-in 1 --epochs 100 --batch-size 64 --num-workers 4
```

### Local Quick Smoke Test (CPU)
```bash
python localisation_workflow/train_decode.py \
  --device cpu \
  --epochs 2 \
  --pseudo-ds-size 64 \
  --test-size 16 \
  --batch-size 16 \
  --num-workers 0
```

### Resume After Preemption / Crash
```bash
python localisation_workflow/train_decode.py --device auto --resume
```

### Useful Optional Flags
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--device` | `auto` | `auto` (picks `cuda:0` if available, else `cpu`), `cpu`, `cuda`, `cuda:0` |
| `--channels-in` | `3` | `3` for 3-frame temporal context $(t-1, t, t+1)$; `1` for single-frame |
| `--em-av-min` | `12.0` | Minimum Poisson emitter average per $64 \times 64$ patch |
| `--em-av-max` | `120.0` | Maximum Poisson emitter average per $64 \times 64$ patch |
| `--lifetime-avg` | `1.0` | Mean fluorophore on-time (frames) |
| `--pseudo-ds-size` | `16384` | Newly simulated training frames per epoch |
| `--test-size` | `512` | Fixed simulated validation frames |
| `--num-workers` | `0` | DataLoader workers (`4` on Linux GPU cluster, `0` on macOS) |

---

## 4. Outputs Saved in `localisation_workflow/decode_model/`

To keep disk usage minimal, checkpoints are overwritten in place:
- `ckpt_latest.pt` — overwritten every epoch (and on interrupt/crash) for `--resume`
- `model_best.pt` — overwritten whenever simulated validation loss improves (`*BEST*`)
- `model_final.pt` — final `SigmaMUNet` weights after the last epoch
- `param_run_used.yaml` — matched DECODE config required alongside `model_best.pt` for inference
- `training_log.csv` — per-epoch `train_loss`, `val_loss`, `jaccard`, `precision`, `recall`, `rmse_lat_nm`, `rmse_ax_nm`, `lr`, and runtime

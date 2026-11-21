# Train DECODE with a VAE background

Run commands from the repository root. The [pure-PyTorch trainer](train_decode_pure.py) uses the top-level [decode_pure package](../decode_pure/) to simulate microscope frames, train SigmaMUNet, and validate localisations against simulated emitters. Its default input is three consecutive frames, with the middle frame supplying the target.

## Environment and training inputs

Activate a Python environment with a PyTorch build for the target device, then install the workflow dependencies. On this Mac, the shared environment is available with `source /Users/rohan/venvs/sharedenv/bin/activate`.

```bash
python -m pip install torch numpy scipy pyyaml tifffile
```

For a cluster run, copy `decode_pure/`, `localisation_workflow/train_decode_pure.py`, and the following files with their relative paths intact.

| Path from the repository root | Used for |
| --- | --- |
| `localisation_workflow/param_run.yaml` | Model, simulation, camera, and scaling settings |
| `data/bead_calibration/Beadzstack_3DLens_3dcal.mat` | Calibrated 3D spline PSF |
| `localisation_workflow/vae_background/bg_vae_64x64.pt` | Frozen background VAE |
| `localisation_workflow/vae_background/noise_maps/manifest.json` | Measured background means and standard deviations |

Training generates frames from these inputs. Raw TIFFs are needed for movie inference.

## Train and resume

For a quick CPU check, use a small acquisition and a separate output directory.

```bash
python localisation_workflow/train_decode_pure.py --device cpu --epochs 1 \
  --pseudo-ds-size 17 --test-size 9 --batch-size 4 \
  --out-dir localisation_workflow/decode_smoke
```

For a full training run, choose a fresh directory. The example below keeps the new run separate from the weights already in `localisation_workflow/decode_model_pure/`.

```bash
python localisation_workflow/train_decode_pure.py --device auto --epochs 100 \
  --out-dir localisation_workflow/decode_model_run
```

To resume, repeat the same command with `--resume`. The trainer restores `ckpt_latest.pt` from the selected directory and continues after the last completed epoch. Keep the original architecture, simulation settings, and seed. `--epochs` is the total epoch count for the run.

| Option | Default | Meaning |
| --- | --- | --- |
| `--device` | `auto` | Selects CUDA, then Apple MPS, then CPU. An explicit device such as `cuda:0` also works. |
| `--channels-in` | `3` | Use `1` for single-frame training or `3` for temporal context. |
| `--batch-size` | `32` | Training windows per batch |
| `--pseudo-ds-size` / `--test-size` | `4096` / `256` | Final frame index of each simulated acquisition. Frame numbering starts at zero. Training data is sampled each epoch and validation data is fixed. |
| `--em-av-min` / `--em-av-max` | `12` / `120` | Bounds for the mean emitter count per frame in a temporal chunk |
| `--lifetime-avg` | `1.0` | Mean emitter on-time in frames |
| `--lr` | `0.0006` | Initial learning rate |
| `--out-dir` | `localisation_workflow/decode_model_pure` | Directory for checkpoints, run settings, and metrics |

For example, `--pseudo-ds-size 17` generates frames 0 through 17. With three input channels this gives 16 training windows, enough for four batches of four.

## Monitor training with Weights & Biases

Tracking is off by default. To enable it, install the optional SDK in the training environment and authenticate on the machine running training.

```bash
python -m pip install wandb
wandb login
python localisation_workflow/train_decode_pure.py --device auto --epochs 100 \
  --out-dir localisation_workflow/decode_model_run \
  --wandb --wandb-project mphys-localisation --wandb-name decode-3frame
```

Open the W&B URL printed by the trainer. Batch loss updates on the first, every tenth, and final batch of each epoch. The epoch charts include training and validation loss, best validation loss, learning rate, Jaccard index, precision, recall, lateral and axial RMSE in nm, and elapsed seconds.

For a cluster job, supply `WANDB_API_KEY` through the job environment or add `--wandb-api-key-file /path/to/secret` to read a file containing the key. Use `--wandb-entity TEAM` to select a team. To continue the same online dashboard, repeat the training command with `--resume --wandb` and the same output directory. The checkpoint stores the W&B run ID, project, and entity. An older checkpoint without a run ID starts a new dashboard.

## Outputs and movie inference

| File in the run directory | Contents |
| --- | --- |
| `model_best.pt` | Weights from the epoch with the lowest validation loss |
| `model_final.pt` | Weights at the end of training |
| `ckpt_latest.pt` | Latest completed epoch, weights, optimiser, scheduler, and best validation loss |
| `param_run_used.yaml` | Model architecture, camera parameters, and scaling to use with these weights |
| `training_log.csv` | Per-epoch loss, learning rate, Jaccard index, precision, recall, lateral and axial RMSE, and runtime |

Use weights and YAML from the same run. For the full-run example above, the following command writes right-channel localisations to CSV. Coordinates are relative to the cropped channel and frame indices start at zero.

```bash
python -m decode_pure.infer \
  --movie data/low_density_examples/RegistrationCheck_LowDensity_GapT_Exp1_GapT_SealG_F3.tif \
  --weights localisation_workflow/decode_model_run/model_best.pt \
  --param localisation_workflow/decode_model_run/param_run_used.yaml \
  --out localisation_workflow/decode_model_run/locs_SealG_F3.csv \
  --right-channel --device auto
```

## Original DECODE trainer

The [original trainer](train_decode.py) uses DECODE 0.10.2 and compiled spline 0.10.0 in the existing `decode_intel` Conda environment. Its default output directory is `localisation_workflow/decode_model/`. This command runs a short local reference job.

```bash
conda run -n decode_intel python localisation_workflow/train_decode.py \
  --device cpu --epochs 1 --pseudo-ds-size 17 --test-size 9 --batch-size 4 \
  --out-dir localisation_workflow/decode_legacy_smoke
```

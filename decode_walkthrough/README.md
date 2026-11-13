# DECODE lab walkthrough

Open the numbered notebooks in order:

1. `01_simulation.ipynb` — run DECODE's example spline simulator; explore signal, background and noise, and compare the left and right halves of a raw frame.
2. `02_psf_calibration.ipynb` — understand and inspect a bead z-stack and follow the external SMAP spline-calibration workflow. This notebook does **not** itself fit a PSF.
3. `03_train_scratch.ipynb` — prepare a short example run from random initialization.
4. `04_fine_tune.ipynb` — prepare an example warm start from the bundled weights, using `finetune.py`.
5. `05_predict.ipynb` — fit a small number of example frames, then set up a cautious own-data test.

## Configuration

Edit `config.yaml` and rerun the notebook from its first cell. Paths are relative to the project root unless absolute. The `paths` section points at DECODE’s example model and its original parameter files; `example_simulation` and `example_training` control the example demonstrations. For an own-data prediction, set `prediction.use_example: false`, choose a channel, and fill its matching model, parameter file, camera settings and mirroring. Values under `own_data.pixel_size_nm` and channel `calibration_file` are documentation placeholders; they do **not** change example training or a saved model. Your own training requires a separately adapted DECODE parameter file.

## Bring to the lab PC

Copy this `decode_walkthrough` folder **including `finetune.py`**, and the complete `experimental_data_workflow` folder, into the same project folder. Bring your raw TIFFs and corresponding acquisition metadata if permitted. The example TIFF and bead stacks are large; allow time and disk space. The pre-existing folder is DECODE's example, not your experiment. You do not need to copy `.venv` or a Conda environment between machines. The notebooks set `ROOT` to the current directory or its parent; edit the first code cell if the files are elsewhere.

Ask the data collector for channel-specific bead z-stacks or an existing compatible SMAP `3dcal.mat`, channel/orientation, z step, camera offset, sensitivity (electrons per ADU), read noise, exposure and other camera settings. The two checked low-density recordings report 117 nm per pixel; check metadata for each dataset. Without suitable calibration, the example can demonstrate DECODE but cannot establish a valid Nanoimager model.

## Install DECODE on the lab computer

With Conda installed, use the [official instructions](https://decode.readthedocs.io/en/release-0.10/installation.html). For a Windows or Linux PC with a compatible NVIDIA GPU, their documented command is:

```
conda create -n decode_env -c turagalab -c pytorch -c conda-forge decode=0.10.2 cudatoolkit=11.3 jupyterlab ipykernel
conda activate decode_env
python -c "import decode, spline, torch; print('CUDA:', torch.cuda.is_available()); print('spline CUDA:', spline.cuda_is_available())"
python -m ipykernel install --user --name decode_env --display-name "Python (DECODE)"
```

For CPU only, omit `cudatoolkit=11.3`. Installation and CUDA depend on the computer and driver; these commands have not been tested on the lab PC. A CUDA-enabled PyTorch installation alone does not guarantee spline CUDA support. Select `Python (DECODE)` in VS Code/Jupyter and open the notebooks. Training can be slow on CPU; `epochs = 2` is a smoke test, not a research model. If the Conda solver fails, keep the full error. DECODE's [official Colab notebooks](https://github.com/TuragaLab/DECODE#-getting-started) are another option.

## Safety and interpretation

Start with the example files and use a small subset. The `example_training.write_config` and `prediction.save_output` settings in `config.yaml` default to `false`. Training is started separately from a terminal so Windows multiprocessing is more reliable; training makes files in `decode_work`. Example camera settings describe an EMCCD, while the Nanoimager camera is sCMOS. Do not use example weights, PSF or background/camera settings as validated values for your experiment. DECODE predicts locations and uncertainty, not automatically verified ground truth. Validate with simulations and suitable real data, using appropriate registration/drift methods and checking border crops, orientation, units and frame indices. Output from the prediction notebook is in processed coordinates; the notebook does not automatically undo crops/mirroring or map between channels.

The warm-start helper is written for the checked-out 0.10 implementation: it loads a state dict with strict architecture matching, starts a fresh optimizer, and does not resume a checkpoint. It must be tested with the version installed at the lab. Never overwrite the supplied model or the example files.

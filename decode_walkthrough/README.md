# Start here: DECODE walkthrough

Open these notebooks in VS Code and select **Python (DECODE Intel)** as the kernel. Run the cells from top to bottom with Shift+Enter.

1. **[Look at your data and simulate a frame](01_simulation.ipynb).** See the left and right halves of a raw TIFF, then generate example frames with adjustable emitter density and background.
2. **[Understand PSF calibration](02_psf_calibration.ipynb).** See an example bead z-stack and learn what to ask the data collector for. The actual spline fit is done in SMAP, not this notebook.
3. **[Run the supplied example network](05_predict.ipynb).** See predicted molecule positions on a small crop of DECODE's example data. This does not train.
4. **[Train from scratch](03_train_scratch.ipynb)** and **[warm-start training](04_fine_tune.ipynb)** are optional once the example makes sense. Both prepare short *example* runs; they are not ready-to-use models for your recordings. Training on this Mac uses the CPU and can be slow.

## Changing settings

Notebook 1 has its movie path and simulation settings directly in its cells; edit them there and rerun the relevant cells. The other notebooks still use [config.yaml](config.yaml): edit it, save it, and rerun those notebooks from the first cell. Relative paths start at the project root (`mphys`).

The example model, calibration and camera settings come from DECODE's bundled experiment. They are not measurements of your Nanoimager. The `own_data` entries for pixel size and calibration are notes; they do not automatically adapt the example training. For your own data you need a matching PSF, camera parameters, a suitable trained model and its parameter file. Check orientation, crop offsets, registration and drift before combining channels. The prediction notebook does not automatically map localisations back to the original two-channel frame.

The supplied example training defaults to two epochs (one is initial evaluation). This is only a workflow check. The save and training-config flags are off by default; original data and example weights are not overwritten. The custom fine-tuning helper uses DECODE internals. A tiny example run was tested, but results still require scientific validation.

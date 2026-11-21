# Pure-PyTorch DECODE

`decode_pure` provides the SigmaMUNet model, emitter and camera simulation, calibrated cubic-spline PSF rendering, and localisation post-processing used by the localisation workflow. It loads weights from both the original DECODE trainer and the pure-PyTorch trainer.

From the repository root, run the simulator with the project calibration and frozen VAE background.

```bash
python -m decode_pure.simulator
```

The Python API also returns the background maps and ground-truth emitters.

```python
from decode_pure import build_vae_simulator, simulation

sim = build_vae_simulator(project_dir=".")
frames, backgrounds, emitters, mean_emitter_count = simulation(sim, seed=42)
```

For dependencies, training, checkpoint resume, and TIFF-to-CSV inference, see the [training guide](../localisation_workflow/train_decode.md).

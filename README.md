# TIFF stack viewer

Browse TIFFs in `data/low_density_examples`, `data/high_density_examples`, and `data/preprocessed_data` with the project environment:

```sh
.venv\Scripts\python.exe view_tif.py
```

The script opens a local browser page. Select a TIFF, drag the frame slider or use the arrow keys, and use Play (or the space bar) to animate. Auto contrast sets black and white levels from the current frame; edit those values to hold a consistent scale across frames. Stop the server with Ctrl+C in the terminal. The viewer reads frames on demand and keeps only a small cache in memory.

To open another directory or a single TIFF, pass its path:

```sh
.venv\Scripts\python.exe view_tif.py data/preprocessed_data
```

Requires Python, NumPy and Pillow. `--no-browser` prints the local URL without opening it; `--port 8000` selects a fixed port. Playback speed depends on disk and rendering speed. Files are shown separately: for example, the low-density SealG recording is split between the main TIFF (1,826 frames) and `_1.tif` (174 frames). Frame counts reflect the actual TIFF pages, which can differ from acquisition metadata.

For registration and drift inspection, run the viewer with the project environment (`.venv\Scripts\python.exe view_tif.py`), select a recording and click **Prepare correction**. This fits the frame-0 channel homography and estimates Picasso RCC drift from every tenth left-channel frame. When preparation finishes, choose Raw, Registered, Drift corrected, or Corrected overlay (left green, right magenta), then scrub or play. Preparation is per file and held in memory; the source TIFF is never changed. This exploratory correction requires OpenCV, SciPy, pandas and Picasso, and uses the notebook's detection and placeholder camera settings. Short files may have too few frames for RCC; inspect calibration matches and drift before relying on the result.

Choose **Localisations (raw)** to see Picasso-fitted centres for the current frame (green circles on the left, red on the right). Change **Min. net gradient** to adjust detection; the selected value is also used when you click **Prepare correction**. If you change it after preparation, prepare again to apply the new value. Localisation and correction require Picasso in the selected Python environment.

TIFFs inside `data/preprocessed_data` are labelled **preprocessed** and shown as saved; localisation and correction controls are disabled for those files. The preprocessing output places the left and registered right channels side by side. Display contrast only changes the visualisation.

The exploratory notebook is [explore_registration_and_drift.ipynb](preprocessing/explore_registration_and_drift.ipynb). Run it from the project root or the `preprocessing` folder; its final cell writes to `data/preprocessed_data/corrected.tif` and will refuse to overwrite an existing file. For DECODE setup and example notebooks, see [decode_walkthrough](decode_walkthrough/README.md).

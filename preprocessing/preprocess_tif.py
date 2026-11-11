"""Apply the viewer's channel registration and RCC drift correction to a TIFF."""
from pathlib import Path
import os
import tempfile

import numpy as np
from PIL import Image
import tifffile

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os

def preprocess_frame(frame, homography, drift_xy=(0.0, 0.0)):
    """Return left and registered right frames, drift-corrected in left pixels.

    Does not mutate the source. Homography maps right-half to left-half
    coordinates; drift_xy is measured in left coordinates for this frame.
    """
    import cv2
    from scipy.ndimage import shift

    if frame.ndim != 2 or frame.shape[1] % 2:
        raise ValueError("Expected a grayscale frame with two equal-width halves")
    height, width = frame.shape
    midpoint = width // 2
    left = frame[:, :midpoint].astype(np.float32)
    right = frame[:, midpoint:].astype(np.float32)
    right = cv2.warpPerspective(right, homography, (midpoint, height),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    dx, dy = drift_xy
    if dx or dy:
        left = shift(left, (-dy, -dx), mode='constant', cval=0.0)
        right = shift(right, (-dy, -dx), mode='constant', cval=0.0)
    return left, right

def _process_single_frame(raw, homography, drift_vec, need_copies):
    """Worker function: transforms the frame and concatenates before returning."""
    left, right = preprocess_frame(raw, homography, drift_vec)
    combined = np.concatenate((left, right), axis=1)
    # Only return `right` if the main thread needs it for plotting
    return combined, (right.copy() if need_copies else None)



def localise_frames(frames, min_gradient, start_frame=0):
    """Use the same Picasso detection and fitting settings throughout the viewer."""
    from picasso.localize import localize_frames

    if not np.isfinite(min_gradient) or min_gradient < 0:
        raise ValueError('Min. net gradient must be a non-negative finite number')
    return localize_frames(
        frames, info=None,
        params={'Box Size': 7, 'Min. Net Gradient': min_gradient},
        camera_info={'Baseline': 0, 'Sensitivity': 1.0, 'Gain': 1, 'Pixelsize': 117},
        start_frame=int(start_frame), fitting_method='gausslq', threaded=False,
    )


def prepare_correction(path, progress, frame_stride=10, segment_frames=300, batch_size=16,
                       min_gradient=3000):
    """Estimate the notebook's frame-0 homography and sampled Picasso RCC drift."""
    import cv2
    import pandas as pd
    from picasso.postprocess import undrift

    if frame_stride < 1 or batch_size < 1:
        raise ValueError('frame_stride and batch_size must be positive')

    def localise(image, frame_index):
        return localise_frames(image[np.newaxis], min_gradient, frame_index)

    with Image.open(path) as movie:
        n_frames = movie.n_frames # type: ignore
        width, height = movie.size
        if width % 2:
            raise ValueError('Expected two equal-width channels')
        midpoint = width // 2
        first = np.array(movie)
        progress('Fitting calibration spots in frame 0…')
        reference = localise(first[:, :midpoint], 0)[['x', 'y']].to_numpy(dtype=np.float32)
        target = localise(first[:, midpoint:], 0)[['x', 'y']].to_numpy(dtype=np.float32)
        if not len(reference) or not len(target):
            raise ValueError('No calibration localisations in one channel')
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = sorted(matcher.match(reference, target), key=lambda m: m.distance)
        if len(matches) < 4:
            raise ValueError(f'Only {len(matches)} matched calibration spots; at least four needed to calculate a valid "transformation matrix" (homography).')
        ref = np.float32([reference[m.queryIdx] for m in matches]) # type: ignore
        tgt = np.float32([target[m.trainIdx] for m in matches]) # type: ignore
        H, mask = cv2.findHomography(tgt, ref, cv2.RANSAC) # type: ignore
        if H is None or mask is None or not np.isfinite(H).all():
            raise ValueError('Unable to estimate a valid homography')
        sample = np.arange(0, n_frames, frame_stride)
        batches = []
        for start in range(0, len(sample), batch_size):
            frame_numbers = sample[start:start + batch_size]
            # python for loop
            images = []
            for frame_number in frame_numbers:
                movie.seek(int(frame_number))
                images.append(np.asarray(movie)[:, :midpoint].copy())
            ##
            locs = localise_frames(np.stack(images), min_gradient)
            if not locs.empty:
                # Picasso reports indices within this stack. The input frames
                # are sampled, so an offset alone cannot recover TIFF indices.
                locs['frame'] = frame_numbers[locs['frame'].to_numpy(dtype=int)]
                batches.append(locs)
            progress(f'Localising sampled frames: {start + len(frame_numbers)}/{len(sample)}')
    if not batches:
        raise ValueError('No localisations found for drift estimation')
    points = pd.concat(batches, ignore_index=True)
    n_segments = int(np.round(n_frames / segment_frames))
    if n_segments < 4:
        raise ValueError('RCC needs at least four segments; this file is too short for the current settings')
    bounds = np.linspace(0, n_frames - 1, n_segments + 1, dtype=np.uint32)
    counts = [((points['frame'] >= bounds[i]) & (points['frame'] < bounds[i+1])).sum()
              for i in range(n_segments)]
    if min(counts) == 0:
        raise ValueError('An RCC segment has no localisations; sample more frames or adjust detection settings')
    progress('Estimating RCC drift…')
    info = [{'Frames': n_frames, 'Height': height, 'Width': midpoint,
             'Pixelsize': 117}, {}]
    drift, _ = undrift(points, info, segmentation=segment_frames, display=False)
    values = drift[['x', 'y']].to_numpy(dtype=float)
    if values.shape != (n_frames, 2) or not np.isfinite(values).all():
        raise ValueError('Picasso returned invalid drift values')
    values -= values[0].copy()
    return {'homography': H, 'drift': values, 'matches': len(matches),
            'inliers': int(mask.sum()), 'counts': counts}


def _calibration_matches(frame, homography):
    """Repeat the viewer's frame-0 matching for optional diagnostics."""
    import cv2

    mid = frame.shape[1] // 2
    ref = localise_frames(frame[None, :, :mid], 3000)[['x', 'y']].to_numpy(np.float32)
    target = localise_frames(frame[None, :, mid:], 3000)[['x', 'y']].to_numpy(np.float32)
    matches = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True).match(ref, target)
    ref = np.asarray([ref[m.queryIdx] for m in matches], dtype=np.float32)
    target = np.asarray([target[m.trainIdx] for m in matches], dtype=np.float32)
    projected = cv2.perspectiveTransform(target[:, None, :], homography)[:, 0, :]
    return ref, target, projected, np.linalg.norm(projected - ref, axis=1)


def _plot_diagnostics(first_raw, first_right, last_right, H, plot_localisations,
                      plot_drift, box_radius):
    import matplotlib.pyplot as plt

    ref, target, projected, errors = _calibration_matches(first_raw, H)
    if not len(ref):
        raise ValueError('No calibration matches available for diagnostic plots')
    figures = {}
    if plot_localisations:
        mid = first_raw.shape[1] // 2
        fig, axes = plt.subplots(1, 2, figsize=(12, 7))
        for ax, image, coords, title in (
            (axes[0], first_raw[:, :mid], ref, 'Left: reference localisations'),
            (axes[1], first_raw[:, mid:], target, 'Right: matched localisations'),
        ):
            low, high = np.percentile(image, (1, 99.8))
            ax.imshow(image, cmap='gray', vmin=low, vmax=high)
            ax.scatter(coords[:, 0], coords[:, 1], s=45, facecolors='none', edgecolors='lime')
            for i, (x, y) in enumerate(coords):
                ax.annotate(str(i), (x, y), color='yellow', fontsize=7)
            ax.set_title(title)
        fig.suptitle('Frame 0: matching indices on the raw channels')
        fig.tight_layout()
        figures['localisations'] = fig

    if plot_drift:
        # Among matching pairs consistent with H, choose the highest raw left
        # pixel at the fitted centre. This does not require another spot model.
        consistent = np.flatnonzero(errors <= 3.0)
        if not len(consistent):
            raise ValueError('No calibration matches agree with the homography within 3 pixels')
        mid = first_raw.shape[1] // 2
        left = first_raw[:, :mid]
        coords = np.rint(ref[consistent]).astype(int)
        coords[:, 0] = np.clip(coords[:, 0], 0, mid - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, left.shape[0] - 1)
        index = consistent[np.argmax(left[coords[:, 1], coords[:, 0]])]
        # The original target localisation, mapped to left coordinates, anchors
        # the same fixed right-channel region in the first and last frame.
        cx, cy = projected[index]
        x0 = max(0, int(round(cx)) - box_radius)
        x1 = min(mid, int(round(cx)) + box_radius + 1)
        y0 = max(0, int(round(cy)) - box_radius)
        y1 = min(left.shape[0], int(round(cy)) + box_radius + 1)
        if x0 >= x1 or y0 >= y1:
            raise ValueError('Selected calibration spot lies outside the corrected image')
        crops = [im[y0:y1, x0:x1] for im in (first_right, last_right)]
        low, high = np.percentile(np.concatenate([c.ravel() for c in crops]), (1, 99.8))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, crop, title in zip(axes, crops, ('First frame', 'Last frame')):
            # Brightest pixel in this fixed box is a visual aid, not confirmed
            # identity or a subpixel localisation of the same physical spot.
            yy, xx = np.unravel_index(np.nanargmax(crop), crop.shape)
            x, y = x0 + int(xx), y0 + int(yy)
            ax.imshow(crop, cmap='gray', vmin=low, vmax=high,
                      extent=(x0 - .5, x1 - .5, y1 - .5, y0 - .5))
            ax.scatter([x], [y], marker='+', color='red', s=80)
            ax.set(title=f'{title}: brightest pixel ({x}, {y})', xlabel='x (left pixels)', ylabel='y (left pixels)')
        fig.suptitle('Corrected right channel: same fixed region (not spot tracking)')
        fig.tight_layout()
        figures['drift'] = fig
    return figures


def preprocess_tif(input_path, output_path, *, plot_localisations=False,
                   plot_drift=False, box_radius=12):
    """Write a registered and drift-corrected TIFF using the viewer defaults.

    Output frames retain the input layout: left next to registered right. They
    are float32, preserving subpixel interpolation values, and saved as BigTIFF
    without copying acquisition metadata. Original frame count is preserved;
    split input files are processed individually. Existing output is not replaced.

    When requested, return matplotlib figures in addition to the homography and
    per-frame drift. The drift figure marks the brightest *pixel* in a fixed
    box in each frame; it does not prove that the peaks are the same object.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Input and output must be different files')
    if output_path.exists():
        raise FileExistsError(f'Output already exists: {output_path}')
    if not isinstance(box_radius, int) or box_radius < 1:
        raise ValueError('box_radius must be a positive integer')
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f'Output directory does not exist: {output_path.parent}')

    # OPTIMISATION: time this preparation separately from frame export below.
    # prepare_correction above localises sampled frames in batches and
    # estimates the homography and Picasso RCC drift once.
    correction = prepare_correction(input_path, print) # weird use of print but whatever
    first_raw = first_right = last_right = None
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.preprocess-', suffix='.tif',
                                         dir=output_path.parent, delete=False) as tmp:
            temporary_path = Path(tmp.name)

        max_workers = min(8, os.cpu_count() or 4)
        # Bound in-flight frames so RAM usage stays strictly O(max_workers)
        max_in_flight = max_workers * 2
        
        with Image.open(input_path) as movie, \
             tifffile.TiffWriter(temporary_path, bigtiff=True) as writer, \
             ThreadPoolExecutor(max_workers=max_workers) as executor:
            n_frames = movie.n_frames #type: ignore
            homography = correction['homography']
            drifts = correction['drift']          
            # FIFO queue holds tuples of (frame_index, future)
            pending = deque()
            next_to_submit = 0
            def submit_frame(idx):
                movie.seek(idx)
                raw = np.array(movie)
                if idx == 0 and (plot_localisations or plot_drift):
                    nonlocal first_raw
                    first_raw = raw
                need_right = (
                    (idx == 0 and (plot_localisations or plot_drift)) or
                    (idx == n_frames - 1 and plot_drift)
                )
                fut = executor.submit(
                    _process_single_frame, raw, homography, drifts[idx], need_right
                )
                pending.append((idx, fut))
            # 1. Prime the pipeline with the initial batch of frames
            while next_to_submit < min(max_in_flight, n_frames):
                submit_frame(next_to_submit)
                next_to_submit += 1
            # 2. Drain the oldest completed frame in strict order, then submit 1 new frame
            while pending:
                index, fut = pending.popleft()
                combined, right_copy = fut.result()  # Blocks until frame `index` is ready
                if next_to_submit < n_frames:
                    submit_frame(next_to_submit)
                    next_to_submit += 1
                writer.write(
                    np.clip(np.rint(combined), 0, 65535).astype(np.uint16),
                    photometric='minisblack',
                    contiguous=True,
                    metadata=None
                )
                if index == 0 and right_copy is not None:
                    first_right = right_copy
                if index == n_frames - 1 and right_copy is not None:
                    last_right = right_copy
                if (index + 1) % 100 == 0 or index + 1 == n_frames:
                    print(f'Writing corrected frames: {index + 1}/{n_frames}')

        figures = {}
        if plot_localisations or plot_drift:
            figures = _plot_diagnostics(first_raw, first_right, last_right,
                                        correction['homography'], plot_localisations,
                                        plot_drift, box_radius)
        # Atomic, no-overwrite publication on the output filesystem.
        os.link(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {'output_path': output_path, 'homography': correction['homography'],
            'drift': correction['drift'], 'matches': correction['matches'],
            'inliers': correction['inliers'], 'figures': figures}

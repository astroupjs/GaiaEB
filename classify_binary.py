"""
classify_binary.py
==================
Production inference script for morphological classification of Gaia eclipsing
binary light curves using a fine-tuned ResNet-18 convolutional neural network.

The classifier distinguishes between two binary-star morphological classes:
    - **Overcontact** (EW / W UMa-type): both components fill their Roche lobes,
      yielding continuous, sinusoidal light curves with comparable eclipse depths.
    - **Detached** (EA / EB-type): well-separated components with flat out-of-eclipse
      regions and distinct primary/secondary minima.

Preprocessing pipeline
----------------------
Each raw light curve is converted into a 128×128 RGB image with three channels
that encode complementary geometric representations of the phase-folded photometry:

    Ch 0 – Red   : Polar projection (phase → angle, normalised flux → radius).
    Ch 1 – Green : Cartesian scatter plot (phase vs. normalised flux).
    Ch 2 – Blue  : Second-derivative (curvature) map highlighting eclipse transitions.

Input data
----------
Light curves are stored in an HDF5 archive produced by the Gaia variability pipeline.
Expected structure::

    gaia.hdf5
    ├── parameters/objects/
    │   ├── star_id       (bytes)
    │   ├── G_start_idx   (int)   – index of first point in lc dataset
    │   ├── G_length      (int)   – number of points for this star
    │   ├── T0_fitted     (float) – reference epoch [MJD]
    │   ├── Period        (float) – orbital period [days]
    │   └── Teff          (float) – effective temperature [K]
    └── original_light_curves/G/
        ├── Time          (float) – observation epoch [BJD]
        └── Norm_Flux     (float) – normalised G-band flux

Output
------
Results are written to ``classified_by_type/``:
    all_classifications.csv  – every processed star with probabilities
    detached.csv             – subset predicted as detached
    overcontact.csv          – subset predicted as overcontact

Usage
-----
    python classify_binary.py

Dependencies
------------
    torch, torchvision, h5py, pandas, numpy, scipy, tqdm

Author
------
    Stefan Parimucha
    Pavol Jozef Šafárik University in Košice
"""

import os
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter  # available for optional smoothing
from torchvision import models
from tqdm import tqdm


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

#: Absolute path to the Gaia HDF5 archive.
H5_PATH: str = "/home/parimucha/Analyze/GAIA_Curves/gaia.hdf5"

#: Path to the serialised ResNet-18 state dict produced during training.
MODEL_PATH: str = "best_model_binary.pth"

#: Side length (pixels) of the square image passed to the CNN.
IMAGE_SIZE: int = 128

#: Number of light curves processed in a single GPU/CPU batch.
BATCH_SIZE: int = 256

#: Offset to convert Gaia JD timestamps to MJD (BJD − 2 455 197.5).
JD_TO_MJD_OFFSET: float = 2_455_197.5

#: Inference device – GPU if available, otherwise CPU.
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: Maximum number of stars to classify; set to a very large number for a full run.
N_STARS: int = 100_000_000

#: Directory for output CSV files.
OUT_DIR: str = "classified_by_type"
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================================================================
# 2. PREPROCESSING
# ===========================================================================

def preprocess_lc_production(
    phases: np.ndarray,
    fluxes: np.ndarray,
    size: int = 128,
) -> torch.Tensor | None:
    """Convert a phase-folded light curve into a three-channel image tensor.

    The function applies the **same** normalisation and rendering steps that
    were used during model training, so any modification here must be mirrored
    in the training preprocessing code.

    Parameters
    ----------
    phases:
        Orbital phase values in the range [0, 1), one entry per observation.
    fluxes:
        Corresponding normalised G-band flux values.
    size:
        Output image side length in pixels (default: 128).

    Returns
    -------
    torch.Tensor of shape (3, size, size) and dtype float32, or ``None`` if the
    light curve contains fewer than one valid (finite) data point.

    Channel layout
    --------------
    Index 0 – Red   : polar projection of the phase-folded light curve.
    Index 1 – Green : Cartesian scatter-plot image.
    Index 2 – Blue  : second-derivative (curvature) map.
    """
    # --- Filter non-finite values produced by the Gaia pipeline ---------
    mask = np.isfinite(phases) & np.isfinite(fluxes)
    phases, fluxes = phases[mask], fluxes[mask]

    if len(phases) == 0:
        # No usable data points; signal the caller to skip this star.
        return None

    # Sort by phase so channel images are drawn left-to-right.
    sort_idx = np.argsort(phases)
    phases, fluxes = phases[sort_idx], fluxes[sort_idx]

    # --- Min–max normalisation to [0, 1] ---------------------------------
    f_min, f_max = fluxes.min(), fluxes.max()
    denom = f_max - f_min
    if denom == 0:
        # Flat light curve (constant star or bad data); map everything to zero.
        f_normed = np.zeros_like(fluxes)
    else:
        f_normed = (fluxes - f_min) / (denom + 1e-9)

    f_normed = np.clip(f_normed, 0.0, 1.0)

    # --- Channel 1 (Green): Cartesian scatter plot -----------------------
    # Each observation is rendered as a single lit pixel at
    # (phase × (size−1), flux × (size−1)).
    p_idx = (phases * (size - 1)).astype(int)
    f_idx = (f_normed * (size - 1)).astype(int)
    ch_g = np.zeros((size, size), dtype=np.float32)
    ch_g[f_idx, p_idx] = 1.0

    # --- Channel 0 (Red): Polar projection -------------------------------
    # Phase maps to polar angle; flux maps to radius in [0.2, 1.0] so that
    # the minimum flux sits at r = 0.2 (not at the origin) and the maximum
    # sits at r = 1.0.  The image origin is at the centre of the canvas.
    f_min_n = np.min(f_normed)
    r = 0.2 + 0.8 * ((f_normed - f_min_n) / (1.0 - f_min_n + 1e-9))
    angle = phases * 2 * np.pi - (np.pi / 2)          # start at 12 o'clock
    x_idx = (((r * np.cos(angle)) + 1.1) / 2.2 * (size - 1)).astype(int)
    y_idx = (((r * np.sin(angle)) + 1.1) / 2.2 * (size - 1)).astype(int)
    ch_r = np.zeros((size, size), dtype=np.float32)
    in_bounds = (x_idx >= 0) & (x_idx < size) & (y_idx >= 0) & (y_idx < size)
    ch_r[y_idx[in_bounds], x_idx[in_bounds]] = 1.0

    # --- Channel 2 (Blue): Curvature map ---------------------------------
    # The absolute second derivative highlights rapid flux transitions
    # (eclipse ingress/egress) while suppressing smooth, out-of-eclipse regions.
    ch_b = np.zeros((size, size), dtype=np.float32)
    if len(f_normed) > 4:
        curvature = np.abs(np.gradient(np.gradient(f_normed)))
        curv_idx = (np.clip(curvature * 15, 0, 1) * (size - 1)).astype(int)
        ch_b[curv_idx, p_idx] = 1.0

    # --- Stack channels in training order: [Red, Green, Blue] -----------
    img = np.stack([ch_r, ch_g, ch_b])

    # Per-channel normalisation to [0, 1] to match training augmentation.
    for c in range(3):
        max_val = img[c].max()
        if max_val > 0:
            img[c] /= max_val

    return torch.tensor(img, dtype=torch.float32)


# ===========================================================================
# 3. PRODUCTION INFERENCE LOOP
# ===========================================================================

def run_production() -> None:
    """Classify all eclipsing binary stars in the Gaia HDF5 archive.

    The function:
    1. Loads the trained ResNet-18 classifier from ``MODEL_PATH``.
    2. Streams light curves from the HDF5 file in batches of ``BATCH_SIZE``.
    3. Phase-folds each curve using the fitted T0 and Period.
    4. Converts each curve to a three-channel image and runs inference.
    5. Appends results (star ID, period, Teff, predicted class, probabilities)
       to per-class CSV files and a master CSV.
    6. Prints a summary table with counts, percentages, and throughput.

    Stars with fewer than 15 valid observations are skipped to ensure that
    the image representation has enough points to be informative.

    Output files (written to ``OUT_DIR``)
    --------------------------------------
    all_classifications.csv  – every processed star
    detached.csv             – stars classified as detached (EA/EB)
    overcontact.csv          – stars classified as overcontact (EW)
    """
    start_time = time.time()

    # Integer label → human-readable class name.
    # Label 0 = overcontact (EW), label 1 = detached (EA/EB).
    class_map: dict[int, str] = {1: "detached", 0: "overcontact"}
    stats: dict[str, int] = {name: 0 for name in class_map.values()}
    total_skipped: int = 0

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)   # binary output head

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    # ------------------------------------------------------------------
    # Output file setup – remove stale files from a previous run
    # ------------------------------------------------------------------
    master_file = os.path.join(OUT_DIR, "all_classifications.csv")
    type_files = {name: os.path.join(OUT_DIR, f"{name}.csv") for name in class_map.values()}

    for path in [master_file, *type_files.values()]:
        if os.path.exists(path):
            os.remove(path)

    # ------------------------------------------------------------------
    # Main inference loop over the HDF5 archive
    # ------------------------------------------------------------------
    with h5py.File(H5_PATH, "r") as h5:
        params = h5["parameters/objects"]
        limit = min(N_STARS, len(params["star_id"]))

        # Read all scalar parameter arrays up-front (small memory footprint).
        starts  = params["G_start_idx"][:limit]   # index of first lc point
        lengths = params["G_length"][:limit]       # number of lc points
        t0s     = params["T0_fitted"][:limit]      # reference epoch [MJD]
        periods = params["Period"][:limit]         # orbital period [days]
        teffs   = params["Teff"][:limit]           # effective temperature [K]
        star_ids = params["star_id"][:limit]       # Gaia source ID (bytes)

        lc_dataset = h5["original_light_curves/G"]

        for batch_start in tqdm(range(0, limit, BATCH_SIZE), desc="Classifying Gaia"):
            batch_end = min(batch_start + BATCH_SIZE, limit)

            batch_imgs:    list[torch.Tensor] = []
            batch_meta:    list[str]          = []
            batch_periods: list[float]        = []
            batch_teffs:   list[float]        = []

            # Read the smallest HDF5 chunk that covers the whole batch to
            # minimise the number of I/O requests.
            chunk_start = int(starts[batch_start])
            chunk_end   = int(starts[batch_end - 1]) + int(lengths[batch_end - 1])
            data_chunk  = lc_dataset[chunk_start:chunk_end]

            for i in range(batch_start, batch_end):
                n_pts = int(lengths[i])

                # Skip stars with very sparse coverage; images would be
                # essentially empty and unreliable for classification.
                if n_pts < 15:
                    total_skipped += 1
                    continue

                # Slice this star's light curve from the pre-loaded chunk.
                offset = int(starts[i]) - chunk_start
                curve  = data_chunk[offset : offset + n_pts]

                # Phase-fold: φ = ((t − T0) / P) mod 1
                phases = (
                    (curve["Time"].astype(float) + JD_TO_MJD_OFFSET - t0s[i])
                    / periods[i]
                ) % 1.0
                fluxes = curve["Norm_Flux"].astype(float)

                img_tensor = preprocess_lc_production(phases, fluxes, size=IMAGE_SIZE)

                if img_tensor is not None:
                    batch_imgs.append(img_tensor)
                    batch_meta.append(star_ids[i].decode("utf-8").strip())
                    batch_periods.append(float(periods[i]))
                    batch_teffs.append(float(teffs[i]))
                else:
                    total_skipped += 1

            if not batch_imgs:
                continue   # entire mini-batch was skipped

            # ----------------------------------------------------------
            # Forward pass
            # ----------------------------------------------------------
            inputs = torch.stack(batch_imgs).to(DEVICE)
            with torch.no_grad():
                logits = model(inputs)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                preds  = np.argmax(probs, axis=1)

            # ----------------------------------------------------------
            # Assemble results and append to CSV files
            # ----------------------------------------------------------
            batch_df = pd.DataFrame({
                "star_id":          batch_meta,
                "period":           batch_periods,
                "teff":             batch_teffs,
                "class_name":       [class_map[p] for p in preds],
                # Index 1 = detached, index 0 = overcontact (matches class_map).
                "prob_detached":    probs[:, 1],
                "prob_overcontact": probs[:, 0],
            })

            # Append to master CSV; write header only for the first chunk.
            batch_df.to_csv(
                master_file,
                mode="a",
                index=False,
                header=not os.path.exists(master_file),
            )

            # Write to per-class CSVs.
            for label, name in class_map.items():
                subset = batch_df[batch_df["class_name"] == name]
                if not subset.empty:
                    stats[name] += len(subset)
                    subset.to_csv(
                        type_files[name],
                        mode="a",
                        index=False,
                        header=not os.path.exists(type_files[name]),
                    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_processed = sum(stats.values())
    elapsed         = time.time() - start_time

    print("\n" + "=" * 55)
    print(f"{'BINARY CLASSIFICATION SUMMARY':^55}")
    print("=" * 55)
    print(f"Total Stars Processed : {total_processed:>10}")
    print(f"Total Stars Skipped   : {total_skipped:>10}")
    print(f"Master CSV            : {master_file}")
    print("-" * 55)
    for name, count in stats.items():
        pct = (count / total_processed * 100) if total_processed > 0 else 0.0
        print(f"  {name.upper():<20}: {count:<10} ({pct:>6.2f}%)")
    print("-" * 55)
    throughput = total_processed / elapsed if elapsed > 0 else float("inf")
    print(f"Elapsed : {elapsed:.1f} s  |  Throughput : {throughput:.1f} stars/s")
    print("=" * 55)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    run_production()

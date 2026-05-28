"""
classify_binary.py
==================
Unified production inference script for two classification tasks on eclipsing
binary light curves using fine-tuned convolutional neural networks.

    MODE 1 – ``binary``  : Morphological classification (overcontact vs detached).
    MODE 2 – ``spot``    : Starspot detection (spot vs nospot).

Select the active mode by setting ``RUN_MODE`` in section 1 (CONFIGURATION).

─────────────────────────────────────────────────────────────────────────────
MODE: binary
─────────────────────────────────────────────────────────────────────────────
Architecture  : ResNet-18, 2-output linear head.
Model file    : ``best_model_binary.pth``
Classes       : 0 → overcontact (EW / W UMa-type)
                1 → detached    (EA / EB-type)
Channel encoding (per star, from raw phase + flux):
    Ch 0 – Red   : polar projection (phase → angle, flux → radius)
    Ch 1 – Green : Cartesian scatter plot (phase vs flux)
    Ch 2 – Blue  : second-derivative curvature map
Output columns: star_id, period, teff, class_name,
                prob_detached, prob_overcontact

─────────────────────────────────────────────────────────────────────────────
MODE: spot
─────────────────────────────────────────────────────────────────────────────
Architecture  : MultiScaleSpotNet – ResNet-18 backbone + MLP head.
Model files   : ``best_model_spots_det.pth``  (detached systems)
                ``best_model_spots.pth``       (overcontact systems)
The spot mode requires a prior binary classification result (or a manually
supplied ``binary_type`` column) to choose the correct model per star.
Classes       : 0 → nospot
                1 → spot
Channel encoding (100-point resampled flux array):
    Ch 0 – Gradient map         : first derivative of flux
    Ch 1 – Wavelet detail (db1) : level-1 detail coefficients
    Ch 2 – Adaptive stretch     : 1st–99th percentile contrast normalisation
Output columns: star_id, period, teff, binary_type, class_name,
                prob_spot, prob_nospot

─────────────────────────────────────────────────────────────────────────────
Input data (both modes)
─────────────────────────────────────────────────────────────────────────────
Phase-folded light curves as individual CSV files in ``LC_DIR``.
Each file must contain at least two columns named ``PHASE_COL`` and
``FLUX_COL``.  The filename without extension is used as the star identifier.

Optional metadata CSV (``METADATA_PATH``) – one row per star::

    star_id       – must match the LC filename (without .csv)
    period        – orbital period [days]      (NaN if unknown)
    teff          – effective temperature [K]  (NaN if unknown)
    binary_type   – 'detached' or 'overcontact' (required for spot mode if
                    binary classification results are not yet available;
                    ignored in binary mode)

─────────────────────────────────────────────────────────────────────────────
Output
─────────────────────────────────────────────────────────────────────────────
Results are written to ``OUT_DIR`` (default: ``classified_by_type/``):

  binary mode:
    all_classifications.csv
    detached.csv
    overcontact.csv

  spot mode:
    all_spot_classifications.csv
    spot.csv
    nospot.csv

─────────────────────────────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────────────────────────────
    # Edit RUN_MODE below, then run:
    python classify_binary.py

Dependencies
─────────────────────────────────────────────────────────────────────────────
    torch, torchvision, pandas, numpy, scipy, pywt, tqdm

Author
──────
    Stefan Parimucha
    Pavol Jozef Šafárik University in Košice
"""

import os
import time

import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from torchvision import models
from tqdm import tqdm


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================

# ── Task selection ──────────────────────────────────────────────────────────
#: Set to "binary" for morphological classification,
#: or "spot" for starspot detection.
RUN_MODE: str = "binary"   # "binary" | "spot"

# ── Data paths ──────────────────────────────────────────────────────────────
#: Directory containing one CSV file per star (phase + flux columns).
LC_DIR: str = "/home/light_curves"

#: Column name for orbital phase values in each LC file.
PHASE_COL: str = "phase"

#: Column name for normalised flux values in each LC file.
FLUX_COL: str = "flux"

#: Optional metadata CSV (star_id, period, teff [, binary_type]).
#: Set to None to skip.
METADATA_PATH: str | None = "/home/metadata.csv"

# ── Model paths ─────────────────────────────────────────────────────────────
#: ResNet-18 state dict for binary (morphological) classification.
MODEL_BINARY: str = "best_model_binary.pth"

#: MultiScaleSpotNet state dict for spot detection on *detached* systems.
MODEL_SPOT_DETACHED: str = "best_model_spots_det.pth"

#: MultiScaleSpotNet state dict for spot detection on *overcontact* systems.
MODEL_SPOT_OVERCONTACT: str = "best_model_spots.pth"

# ── Inference settings ───────────────────────────────────────────────────────
#: Side length (pixels) of the square image passed to the CNN.
IMAGE_SIZE: int = 128

#: Number of light curves processed in a single GPU/CPU batch.
BATCH_SIZE: int = 256

#: Inference device – GPU if available, otherwise CPU.
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: Maximum number of stars to classify.
N_STARS: int = 100_000_000

#: Directory for output CSV files.
OUT_DIR: str = "classified_by_type"
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================================================================
# 2. MODEL DEFINITIONS
# ===========================================================================

def build_binary_model() -> nn.Module:
    """ResNet-18 with a 2-class linear head for morphological classification."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


class MultiScaleSpotNet(nn.Module):
    """ResNet-18 backbone + two-layer MLP head for starspot detection.

    The ImageNet-pretrained backbone acts as a feature extractor; its FC
    layer is replaced with ``nn.Identity`` to expose the 512-d GAP vector,
    which then passes to a lightweight MLP with dropout.

    Architecture
    ------------
    Backbone : ResNet-18 (pretrained, FC → Identity) → 512-d feature vector
    Head     : Linear(512 → 256) → ReLU → Dropout(0.1) → Linear(256 → 2)
    """

    def __init__(self) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def build_spot_model() -> MultiScaleSpotNet:
    """Instantiate an untrained MultiScaleSpotNet (weights loaded separately)."""
    return MultiScaleSpotNet()


# ===========================================================================
# 3. PREPROCESSING – BINARY MODE
# ===========================================================================

def preprocess_binary(
    phases: np.ndarray,
    fluxes: np.ndarray,
    size: int = 128,
) -> "torch.Tensor | None":
    """Convert a raw phase-folded light curve into a 3-channel binary-classifier image.

    Parameters
    ----------
    phases : array-like, values in [0, 1)
    fluxes : normalised flux values
    size   : output image side length in pixels

    Returns
    -------
    torch.Tensor of shape (3, size, size) and dtype float32, or None if the
    curve contains no finite points.

    Channels
    --------
    Ch 0 – Red   : polar projection
    Ch 1 – Green : Cartesian scatter plot
    Ch 2 – Blue  : second-derivative curvature map
    """
    mask = np.isfinite(phases) & np.isfinite(fluxes)
    phases, fluxes = phases[mask], fluxes[mask]
    if len(phases) == 0:
        return None

    sort_idx = np.argsort(phases)
    phases, fluxes = phases[sort_idx], fluxes[sort_idx]

    f_min, f_max = fluxes.min(), fluxes.max()
    f_normed = fluxes / f_max
    f_normed = np.clip(f_normed, 0.0, 1.0)
   
    p_idx = (phases * (size - 1)).astype(int)
    f_idx = (f_normed * (size - 1)).astype(int)

    # Ch 1 – Green: Cartesian scatter
    ch_g = np.zeros((size, size), dtype=np.float32)
    ch_g[f_idx, p_idx] = 1.0

    # Ch 0 – Red: Polar projection
    f_min_n = f_normed.min()
    r = 0.2 + 0.8 * ((f_normed - f_min_n) / (1.0 - f_min_n + 1e-9))
    angle = phases * 2 * np.pi - (np.pi / 2)
    x_idx = (((r * np.cos(angle)) + 1.1) / 2.2 * (size - 1)).astype(int)
    y_idx = (((r * np.sin(angle)) + 1.1) / 2.2 * (size - 1)).astype(int)
    ch_r = np.zeros((size, size), dtype=np.float32)
    in_b = (x_idx >= 0) & (x_idx < size) & (y_idx >= 0) & (y_idx < size)
    ch_r[y_idx[in_b], x_idx[in_b]] = 1.0

    # Ch 2 – Blue: Curvature map
    ch_b = np.zeros((size, size), dtype=np.float32)
    if len(f_normed) > 4:
        curv = np.abs(np.gradient(np.gradient(f_normed)))
        cidx = (np.clip(curv * 15, 0, 1) * (size - 1)).astype(int)
        ch_b[cidx, p_idx] = 1.0

    img = np.stack([ch_r, ch_g, ch_b])
    for c in range(3):
        m = img[c].max()
        if m > 0:
            img[c] /= m

    return torch.tensor(img, dtype=torch.float32)


# ===========================================================================
# 4. PREPROCESSING – SPOT MODE
# ===========================================================================

def preprocess_spot(
    phases: np.ndarray,
    fluxes: np.ndarray,
    size: int = 128,
    n_pts: int = 100,
) -> "torch.Tensor | None":
    """Convert a phase-folded light curve into a 3-channel spot-detector image.

    The curve is first resampled onto a uniform 100-point phase grid to match
    the training-time representation used by ``SpotDetectionDataset``.

    Parameters
    ----------
    phases : array-like, values in [0, 1)
    fluxes : normalised flux values
    size   : output image side length in pixels
    n_pts  : number of evenly spaced points to resample to (default: 100)

    Returns
    -------
    torch.Tensor of shape (3, size, size) and dtype float32, or None if the
    curve contains fewer than 8 finite points (minimum for the wavelet channel).

    Channels
    --------
    Ch 0 – Gradient map         : first derivative of flux
    Ch 1 – Wavelet detail (db1) : level-1 Haar detail coefficients
    Ch 2 – Adaptive stretch     : 1st–99th percentile contrast normalisation
    """
    mask = np.isfinite(phases) & np.isfinite(fluxes)
    phases, fluxes = phases[mask], fluxes[mask]
    if len(phases) < 8:
        return None

    # Sort by phase, then resample to a uniform 100-point grid.
    sort_idx = np.argsort(phases)
    phases, fluxes = phases[sort_idx], fluxes[sort_idx]

    base_phase = np.linspace(0.0, 1.0, n_pts, endpoint=True)
    try:
        resampler = interp1d(phases, fluxes, kind="linear",
                             bounds_error=False, fill_value="extrapolate")
        fluxes = resampler(base_phase).astype(np.float32)
    except Exception:
        return None

    phases = base_phase
    p_idx = (phases * (size - 1)).astype(int)

    # Ch 0 – Gradient map
    ch_grad = np.zeros((size, size), dtype=np.float32)
    if len(fluxes) > 2:
        grad = np.gradient(fluxes)
        grad_norm = (grad - grad.min()) / (grad.max() - grad.min() + 1e-9)
        ch_grad[(grad_norm * (size - 1)).astype(int), p_idx] = 1.0

    # Ch 1 – Wavelet detail coefficients (db1, level 1)
    ch_wavelet = np.zeros((size, size), dtype=np.float32)
    if len(fluxes) >= 8:
        coeffs = pywt.wavedec(fluxes, "db1", level=1)
        cD = coeffs[1]
        cD_norm = (cD - cD.min()) / (cD.max() - cD.min() + 1e-9)
        cD_res = np.interp(
            np.linspace(0, 1, len(fluxes)),
            np.linspace(0, 1, len(cD)),
            cD_norm,
        )
        ch_wavelet[(cD_res * (size - 1)).astype(int), p_idx] = 1.0

    # Ch 2 – Adaptive contrast stretch (1st–99th percentile)
    p1, p99 = np.percentile(fluxes, [1, 99])
    f_stretch = np.clip((fluxes - p1) / (p99 - p1 + 1e-9), 0.0, 1.0)
    ch_stretch = np.zeros((size, size), dtype=np.float32)
    ch_stretch[(f_stretch * (size - 1)).astype(int), p_idx] = 1.0

    return torch.tensor(
        np.stack([ch_grad, ch_wavelet, ch_stretch]),
        dtype=torch.float32,
    )


# ===========================================================================
# 5. DATA LOADING HELPERS
# ===========================================================================

def load_lc_csv(filepath: str) -> "tuple[np.ndarray, np.ndarray] | None":
    """Read a single LC CSV and return (phases, fluxes), or None on failure."""
    try:
        df = pd.read_csv(filepath, comment="#")
    except Exception as exc:
        print(f"[WARN] Could not read {filepath}: {exc}")
        return None

    if PHASE_COL not in df.columns or FLUX_COL not in df.columns:
        print(
            f"[WARN] {filepath} missing required columns "
            f"('{PHASE_COL}', '{FLUX_COL}'). Found: {list(df.columns)}"
        )
        return None

    return df[PHASE_COL].to_numpy(float), df[FLUX_COL].to_numpy(float)


def load_metadata(path: "str | None") -> pd.DataFrame:
    """Load the optional metadata CSV, indexed by star_id.

    Returns an empty DataFrame with expected columns on failure.
    """
    empty = pd.DataFrame(
        columns=["star_id", "period", "teff", "binary_type"]
    ).set_index("star_id")

    if path is None or not os.path.exists(path):
        return empty

    try:
        meta = pd.read_csv(path, comment="#")
        if "star_id" not in meta.columns:
            print(f"[WARN] Metadata '{path}' has no 'star_id' column – ignoring.")
            return empty
        return meta.set_index("star_id")
    except Exception as exc:
        print(f"[WARN] Could not read metadata '{path}': {exc}")
        return empty


def _meta_get(meta: pd.DataFrame, star_id: str, col: str, default):
    """Safely retrieve a single value from the metadata DataFrame."""
    if star_id in meta.index and col in meta.columns:
        val = meta.at[star_id, col]
        return default if pd.isna(val) else val
    return default


# ===========================================================================
# 6. INFERENCE RUNNERS
# ===========================================================================

def run_binary(metadata: pd.DataFrame, lc_files: list[str]) -> None:
    """Run morphological (overcontact / detached) classification.

    Loads ``MODEL_BINARY``, iterates over ``lc_files`` in batches, and writes
    results to ``OUT_DIR/all_classifications.csv``, ``detached.csv``, and
    ``overcontact.csv``.
    """
    print(f"\n{'─'*55}")
    print(f"  MODE: binary classification  |  device: {DEVICE}")
    print(f"{'─'*55}\n")

    class_map: dict[int, str] = {0: "overcontact", 1: "detached"}
    stats: dict[str, int]     = {n: 0 for n in class_map.values()}
    total_skipped = 0

    # Load model
    if not os.path.exists(MODEL_BINARY):
        print(f"[ERROR] Model not found: {MODEL_BINARY}")
        return

    model = build_binary_model()
    model.load_state_dict(torch.load(MODEL_BINARY, map_location=DEVICE))
    model.to(DEVICE).eval()

    # Output files
    master_file = os.path.join(OUT_DIR, "all_classifications.csv")
    type_files  = {n: os.path.join(OUT_DIR, f"{n}.csv") for n in class_map.values()}
    for p in [master_file, *type_files.values()]:
        if os.path.exists(p):
            os.remove(p)

    for batch_start in tqdm(range(0, len(lc_files), BATCH_SIZE), desc="Binary classify"):
        batch_files = lc_files[batch_start : batch_start + BATCH_SIZE]

        batch_imgs:    list[torch.Tensor] = []
        batch_ids:     list[str]          = []
        batch_periods: list[float]        = []
        batch_teffs:   list[float]        = []

        for fname in batch_files:
            star_id  = os.path.splitext(fname)[0]
            filepath = os.path.join(LC_DIR, fname)

            result = load_lc_csv(filepath)
            if result is None:
                total_skipped += 1
                continue

            phases, fluxes = result
            valid = np.isfinite(phases) & np.isfinite(fluxes)
            if valid.sum() < 15:
                total_skipped += 1
                continue

            img = preprocess_binary(phases, fluxes, size=IMAGE_SIZE)
            if img is None:
                total_skipped += 1
                continue

            batch_imgs.append(img)
            batch_ids.append(star_id)
            batch_periods.append(_meta_get(metadata, star_id, "period", float("nan")))
            batch_teffs.append(_meta_get(metadata, star_id, "teff",   float("nan")))

        if not batch_imgs:
            continue

        inputs = torch.stack(batch_imgs).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(inputs), dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        batch_df = pd.DataFrame({
            "star_id":          batch_ids,
            "period":           batch_periods,
            "teff":             batch_teffs,
            "class_name":       [class_map[p] for p in preds],
            "prob_detached":    probs[:, 1],
            "prob_overcontact": probs[:, 0],
        })

        batch_df.to_csv(master_file, mode="a", index=False,
                        header=not os.path.exists(master_file))

        for label, name in class_map.items():
            subset = batch_df[batch_df["class_name"] == name]
            if not subset.empty:
                stats[name] += len(subset)
                subset.to_csv(type_files[name], mode="a", index=False,
                               header=not os.path.exists(type_files[name]))

    _print_summary(stats, total_skipped, master_file)


def run_spot(metadata: pd.DataFrame, lc_files: list[str]) -> None:
    """Run starspot detection on pre-classified eclipsing binary light curves.

    The correct model is selected per star based on its ``binary_type``
    (``'detached'`` → ``MODEL_SPOT_DETACHED``, ``'overcontact'`` →
    ``MODEL_SPOT_OVERCONTACT``).  The ``binary_type`` is looked up from:

    1. The ``binary_type`` column in the metadata CSV, OR
    2. ``OUT_DIR/all_classifications.csv`` produced by a prior binary run.

    Stars whose binary type cannot be determined are skipped with a warning.

    Writes results to ``OUT_DIR/all_spot_classifications.csv``, ``spot.csv``,
    and ``nospot.csv``.
    """
    print(f"\n{'─'*55}")
    print(f"  MODE: spot detection  |  device: {DEVICE}")
    print(f"{'─'*55}\n")

    # ── Resolve binary types ─────────────────────────────────────────────────
    # Priority: metadata CSV > previous binary classification CSV
    binary_types: dict[str, str] = {}

    if "binary_type" in metadata.columns:
        for sid, row in metadata.iterrows():
            bt = str(row["binary_type"]).strip().lower()
            if bt in ("detached", "overcontact"):
                binary_types[str(sid)] = bt

    binary_csv = os.path.join(OUT_DIR, "all_classifications.csv")
    if os.path.exists(binary_csv):
        prev = pd.read_csv(binary_csv)
        if "star_id" in prev.columns and "class_name" in prev.columns:
            for _, row in prev.iterrows():
                sid = str(row["star_id"])
                if sid not in binary_types:   # metadata takes priority
                    bt = str(row["class_name"]).strip().lower()
                    if bt in ("detached", "overcontact"):
                        binary_types[sid] = bt
            print(f"  Loaded binary types for {len(binary_types)} stars "
                  f"from {binary_csv}")

    if not binary_types:
        print(
            "[ERROR] No binary_type information found.\n"
            "  Either:\n"
            "    (a) add a 'binary_type' column to the metadata CSV, or\n"
            "    (b) run binary classification first (RUN_MODE='binary')."
        )
        return

    # ── Load both spot models ────────────────────────────────────────────────
    models_loaded: dict[str, nn.Module] = {}
    for bt, path in [("detached", MODEL_SPOT_DETACHED),
                     ("overcontact", MODEL_SPOT_OVERCONTACT)]:
        if not os.path.exists(path):
            print(f"[WARN] Spot model not found for {bt}: {path}")
            continue
        m = build_spot_model()
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.to(DEVICE).eval()
        models_loaded[bt] = m

    if not models_loaded:
        print("[ERROR] No spot models could be loaded.")
        return

    # ── Output files ─────────────────────────────────────────────────────────
    class_map: dict[int, str] = {0: "nospot", 1: "spot"}
    stats: dict[str, int]     = {n: 0 for n in class_map.values()}
    total_skipped = 0

    master_file = os.path.join(OUT_DIR, "all_spot_classifications.csv")
    type_files  = {n: os.path.join(OUT_DIR, f"{n}.csv") for n in class_map.values()}
    for p in [master_file, *type_files.values()]:
        if os.path.exists(p):
            os.remove(p)

    # ── Inference ────────────────────────────────────────────────────────────
    # Group stars by binary_type so each mini-batch is processed by one model.
    det_files  = [f for f in lc_files
                  if binary_types.get(os.path.splitext(f)[0]) == "detached"]
    over_files = [f for f in lc_files
                  if binary_types.get(os.path.splitext(f)[0]) == "overcontact"]
    skipped_no_type = len(lc_files) - len(det_files) - len(over_files)
    if skipped_no_type:
        print(f"[INFO] {skipped_no_type} files skipped – binary_type unknown.")

    for bt, file_subset in [("detached", det_files), ("overcontact", over_files)]:
        if bt not in models_loaded or not file_subset:
            continue

        model = models_loaded[bt]
        desc  = f"Spot [{bt[:3]}]"

        for batch_start in tqdm(range(0, len(file_subset), BATCH_SIZE), desc=desc):
            batch_files = file_subset[batch_start : batch_start + BATCH_SIZE]

            batch_imgs:    list[torch.Tensor] = []
            batch_ids:     list[str]          = []
            batch_periods: list[float]        = []
            batch_teffs:   list[float]        = []
            batch_btypes:  list[str]          = []

            for fname in batch_files:
                star_id  = os.path.splitext(fname)[0]
                filepath = os.path.join(LC_DIR, fname)

                result = load_lc_csv(filepath)
                if result is None:
                    total_skipped += 1
                    continue

                phases, fluxes = result
                valid = np.isfinite(phases) & np.isfinite(fluxes)
                if valid.sum() < 8:
                    total_skipped += 1
                    continue

                img = preprocess_spot(phases, fluxes, size=IMAGE_SIZE)
                if img is None:
                    total_skipped += 1
                    continue

                batch_imgs.append(img)
                batch_ids.append(star_id)
                batch_periods.append(_meta_get(metadata, star_id, "period", float("nan")))
                batch_teffs.append(_meta_get(metadata, star_id, "teff",   float("nan")))
                batch_btypes.append(bt)

            if not batch_imgs:
                continue

            inputs = torch.stack(batch_imgs).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(inputs), dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

            batch_df = pd.DataFrame({
                "star_id":     batch_ids,
                "period":      batch_periods,
                "teff":        batch_teffs,
                "binary_type": batch_btypes,
                "class_name":  [class_map[p] for p in preds],
                "prob_spot":   probs[:, 1],
                "prob_nospot": probs[:, 0],
            })

            batch_df.to_csv(master_file, mode="a", index=False,
                            header=not os.path.exists(master_file))

            for label, name in class_map.items():
                subset = batch_df[batch_df["class_name"] == name]
                if not subset.empty:
                    stats[name] += len(subset)
                    subset.to_csv(type_files[name], mode="a", index=False,
                                  header=not os.path.exists(type_files[name]))

    _print_summary(stats, total_skipped, master_file)


# ===========================================================================
# 7. SHARED UTILITIES
# ===========================================================================

def _print_summary(
    stats: dict[str, int],
    total_skipped: int,
    master_file: str,
    elapsed: float = 0.0,
) -> None:
    """Print a formatted classification summary table."""
    total_processed = sum(stats.values())
    print("\n" + "=" * 55)
    print(f"{'CLASSIFICATION SUMMARY':^55}")
    print("=" * 55)
    print(f"Total Processed : {total_processed:>10}")
    print(f"Total Skipped   : {total_skipped:>10}")
    print(f"Master CSV      : {master_file}")
    print("-" * 55)
    for name, count in stats.items():
        pct = (count / total_processed * 100) if total_processed else 0.0
        print(f"  {name.upper():<20}: {count:<10} ({pct:>6.2f}%)")
    if elapsed > 0:
        tp = total_processed / elapsed
        print(f"{'─'*55}")
        print(f"Elapsed : {elapsed:.1f} s  |  Throughput : {tp:.1f} stars/s")
    print("=" * 55)


# ===========================================================================
# 8. ENTRY POINT
# ===========================================================================

def main() -> None:
    """Validate configuration, discover LC files, and dispatch to the correct runner."""
    start_time = time.time()

    # Validate mode
    if RUN_MODE not in ("binary", "spot"):
        print(f"[ERROR] Unknown RUN_MODE '{RUN_MODE}'. Choose 'binary' or 'spot'.")
        return

    # Discover light-curve files
    if not os.path.isdir(LC_DIR):
        print(f"[ERROR] LC_DIR does not exist: {LC_DIR}")
        return

    lc_files = sorted(
        f for f in os.listdir(LC_DIR) if f.lower().endswith(".csv")
    )[:N_STARS]

    if not lc_files:
        print(f"[ERROR] No CSV files found in {LC_DIR}")
        return

    print(f"Found {len(lc_files)} LC files in {LC_DIR}")
    print(f"Running in mode : {RUN_MODE.upper()}")

    # Load shared metadata
    metadata = load_metadata(METADATA_PATH)

    # Dispatch
    if RUN_MODE == "binary":
        run_binary(metadata, lc_files)
    else:
        run_spot(metadata, lc_files)

    elapsed = time.time() - start_time
    print(f"\nTotal wall time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()

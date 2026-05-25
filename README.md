
## Datasets and model downloads:

Synthetic light curves of detached and overcontact binaries are available at https://doi.org/10.5281/zenodo.20308051

From them, the files selected_data.csv, selected_data_det.csv, and selected_data_over.csv can be very simply created for use in training scripts.

Trained models are available at https://doi.org/10.5281/zenodo.20376568

## How to use it

### Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
PyWavelets>=1.4.0
tqdm>=4.65.0
```

> **GPU support:** If you have a CUDA-capable GPU, install the matching PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`. The script auto-detects the available device.

## Usage

### Training
```bash
# train new binary model
python train_binary.py
```
```bash
# train new spot model on detached binaries
python train_detached.py
```
```bash
# train new spot model on overcontact binaries
python train_overcontact.py
```
Note: These scripts require the files selected_data.csv, selected_data_det.csv, and selected+data_over.csv, which can be created from synthetic light curves. 

## Input data format

### Light curve files

Place one CSV file per star in a directory (configured as `LC_DIR`). Each file must contain at least two columns:

```
phase,flux
0.0012,0.9987
0.0089,0.9981
...
```

The filename without extension is used as the **star identifier** (e.g. `3128456789.csv` → `star_id = 3128456789`). Phase values must be in `[0, 1)`. The column names are configurable via `PHASE_COL` and `FLUX_COL`.

### Metadata file (optional)

A single CSV with one row per star. Missing values are stored as `NaN` in the output.

```
star_id,period,teff,binary_type
3128456789,0.382,5800,overcontact
4291837465,1.947,6200,detached
```

| Column | Description | Required for |
|---|---|---|
| `star_id` | Must match LC filename (without `.csv`) | both modes |
| `period` | Orbital period [days] | output only |
| `teff` | Effective temperature [K] | output only |
| `binary_type` | `detached` or `overcontact` | spot mode (if no prior binary run) |

---



```


---

## Usage

### Step 1 – Configure `classify_binary.py`

Open the script and edit the configuration block at the top:

```python
# ── Task selection ──────────────────────────────────────────────────────────
RUN_MODE: str = "binary"   # "binary" | "spot"

# ── Data paths ──────────────────────────────────────────────────────────────
LC_DIR:        str       = "/path/to/your/light_curves"
PHASE_COL:     str       = "phase"
FLUX_COL:      str       = "flux"
METADATA_PATH: str|None  = "/path/to/metadata.csv"   # or None

# ── Model paths ─────────────────────────────────────────────────────────────
MODEL_BINARY:          str = "best_model_binary.pth"
MODEL_SPOT_DETACHED:   str = "best_model_spots_det.pth"
MODEL_SPOT_OVERCONTACT:str = "best_model_spots.pth"
```

### Step 2 – Run binary classification

```bash
# In classify_binary.py, set: RUN_MODE = "binary"
python classify_binary.py
```

Output written to `classified_by_type/`:

```
classified_by_type/
├── all_classifications.csv   # all stars: star_id, period, teff, class_name, prob_detached, prob_overcontact
├── detached.csv
└── overcontact.csv
```

### Step 3 – Run spot detection

```bash
# In classify_binary.py, set: RUN_MODE = "spot"
python classify_binary.py
```

The script reads `classified_by_type/all_classifications.csv` from the previous step to assign each star to the correct spot model. Output written to `classified_by_type/`:

```
classified_by_type/
├── all_spot_classifications.csv  # all stars: star_id, period, teff, binary_type, class_name, prob_spot, prob_nospot
├── spot.csv
└── nospot.csv
```

> If you want to run spot detection without a prior binary run, add a `binary_type` column (`detached` or `overcontact`) to your metadata CSV.

---

## Channel encoding

### Binary mode — per raw phase/flux curve

| Channel | Name | Description |
|---|---|---|
| Ch 0 (R) | Polar projection | Phase → polar angle, normalised flux → radius |
| Ch 1 (G) | Cartesian scatter | Phase vs. normalised flux |
| Ch 2 (B) | Curvature map | Absolute second derivative, highlights eclipse transitions |

### Spot mode — 100-point resampled flux array

| Channel | Name | Description |
|---|---|---|
| Ch 0 | Gradient map | First derivative of flux; reveals asymmetric slopes from spots |
| Ch 1 | Wavelet detail | db1 level-1 detail coefficients; sensitive to localised flux variations |
| Ch 2 | Adaptive stretch | 1st–99th percentile contrast normalisation; preserves low-amplitude modulation |

---

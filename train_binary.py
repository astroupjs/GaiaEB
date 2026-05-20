"""
train_binary.py
===============
Training script for a ResNet-18 binary classifier that distinguishes between
overcontact (EW / W UMa-type) and detached (EA / EB-type) eclipsing binary
stars from phase-folded, synthetic light curves.

Architecture overview
---------------------
A pretrained ResNet-18 backbone (ImageNet weights) is fine-tuned with a
two-neuron output head.  Each training sample is a 128×128 RGB image whose
three channels encode complementary geometric views of a noisy, sparsely-
sampled light curve (see :class:`AdvancedLCDataset` for details).

Input data format
-----------------
A CSV file ``selected_data.csv`` is expected in the working directory with the
following column layout:

    Columns 0–99  : 100 evenly-spaced, phase-folded flux values in [0, 1].
    Column 100    : string class label – ``'det'`` (detached) or ``'over'``
                    (overcontact).

Training strategy
-----------------
- 80 / 20 random train / validation split.
- Adam optimiser, fixed learning rate 3 × 10⁻⁴, cross-entropy loss.
- Early stopping with patience = 7 epochs on validation loss.
- Best checkpoint saved to ``best_model_binary.pth``.
- Per-epoch metrics appended to ``training_history_binary.csv``.

Data augmentation (applied per sample at runtime)
--------------------------------------------------
- Random circular phase shift  (±5 % of the period).
- Random single-point outlier  (σ = 0.4, applied with probability 0.3).
- Gaussian noise               (σ drawn uniformly from [0.005, 0.02]).
- Random sparse sampling: either ~22 points (sparse, 35 % of samples) or
  ~48 points (moderate, 65 % of samples), clipped to [10, 100].

Outputs
-------
best_model_binary.pth        – state dict of the best validation checkpoint.
training_history_binary.csv  – per-epoch loss and accuracy log.
training_analysis_plot.png   – 4-panel dashboard (loss, accuracy, confusion
                               matrix, sample image comparison).

Usage
-----
    python train_binary.py

Dependencies
------------
    torch, torchvision, pandas, numpy, matplotlib, seaborn, sklearn, scipy, tqdm

Author
------
    Stefan Parimucha
    Pavol Jozef Šafárik University in Košice
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import gaussian_filter  # available for optional smoothing
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm


# ===========================================================================
# 1. DATASET
# ===========================================================================

class AdvancedLCDataset(Dataset):
    """PyTorch Dataset that converts tabular light-curve data into RGB images.

    Each row of the input DataFrame contains 100 phase-folded flux values
    (columns 0–99) and a string class label in column 100.  At training time
    the flux vector is augmented with noise, a random phase shift, and sparse
    sampling before being rendered into a three-channel image tensor.

    Parameters
    ----------
    dataframe:
        DataFrame with 101 columns – 100 flux values followed by the class
        label (``'det'`` or ``'over'``).
    image_size:
        Side length in pixels of the square output image (default: 128).
    outlier_prob:
        Probability of injecting a single random outlier point per sample
        (default: 0.3).
    phase_shift_limit:
        Maximum phase shift expressed as a fraction of the period (default:
        0.05, i.e. ±5 %).  Set to 0 to disable.

    Attributes
    ----------
    data : np.ndarray, shape (N, 100)
        Raw flux arrays, one row per star.
    labels : np.ndarray, shape (N,)
        Integer class labels – 1 for detached, 0 for overcontact.
    base_phase : np.ndarray, shape (100,)
        Evenly-spaced phase grid used for rendering after random sub-sampling.

    Channel encoding
    ----------------
    Index 0 – Red   : polar projection (phase → angle, flux → radius).
    Index 1 – Green : Cartesian scatter plot (phase vs. normalised flux).
    Index 2 – Blue  : second-derivative (curvature) map.

    Notes
    -----
    The channel layout and normalisation are intentionally identical to the
    production preprocessing in ``classify_binary.py``.  Any modification
    here must be mirrored there.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: int = 128,
        outlier_prob: float = 0.3,
        phase_shift_limit: float = 0.05,
    ) -> None:
        # Flux grid: first 100 columns, cast to float32 for PyTorch compatibility.
        self.data = dataframe.iloc[:, :100].values.astype(np.float32)

        # Convert string labels to integers; handle pre-encoded integer columns.
        target_col = dataframe.iloc[:, 100]
        self.labels = (
            target_col.map({"det": 1, "over": 0}).values
            if target_col.dtype == object
            else target_col.values
        )

        self.size              = image_size
        self.outlier_prob      = outlier_prob
        self.phase_shift_limit = phase_shift_limit

        # Fixed phase grid for the full 100-point curve; sub-sampled per call.
        self.base_phase = np.linspace(0.0, 1.0, 100, endpoint=True)

    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------------
    # Image rendering
    # ------------------------------------------------------------------

    def _to_3channel(self, phases: np.ndarray, fluxes: np.ndarray) -> torch.Tensor:
        """Render a sparse phase-folded light curve as a three-channel image.

        Parameters
        ----------
        phases:
            Phase values in [0, 1) for the (sub-sampled) observations.
        fluxes:
            Corresponding normalised flux values.

        Returns
        -------
        torch.Tensor of shape ``(3, size, size)`` and dtype ``float32``.
        """
        # Remove any non-finite values that might have been introduced by augmentation.
        mask = np.isfinite(phases) & np.isfinite(fluxes)
        phases, fluxes = phases[mask], fluxes[mask]

        # Sort by phase so channels are drawn left-to-right.
        sort_idx = np.argsort(phases)
        phases, fluxes = phases[sort_idx], fluxes[sort_idx]

        # Min–max normalisation to [0, 1].
        f_min, f_max = fluxes.min(), fluxes.max()
        f_normed = (fluxes - f_min) / (f_max - f_min + 1e-9)
        f_normed = np.clip(f_normed, 0.0, 1.0)

        # --- Channel 1 (Green): Cartesian scatter plot -------------------
        # Pixel coordinates: x = phase index, y = flux index.
        p_idx = (phases * (self.size - 1)).astype(int)
        f_idx = (f_normed * (self.size - 1)).astype(int)
        ch_g = np.zeros((self.size, self.size), dtype=np.float32)
        ch_g[f_idx, p_idx] = 1.0
        # Optional Gaussian smoothing (disabled by default):
        # ch_g = gaussian_filter(ch_g, sigma=1.0)

        # --- Channel 0 (Red): Polar projection ---------------------------
        # Minimum flux maps to r = 0.2 to keep points away from the origin;
        # maximum flux maps to r = 1.0.
        f_min_n = np.min(f_normed) if len(f_normed) > 0 else 0.0
        r     = 0.2 + 0.8 * ((f_normed - f_min_n) / (1.0 - f_min_n + 1e-9))
        angle = phases * 2 * np.pi - (np.pi / 2)  # phase 0 starts at 12 o'clock
        x_idx = (((r * np.cos(angle)) + 1.1) / 2.2 * (self.size - 1)).astype(int)
        y_idx = (((r * np.sin(angle)) + 1.1) / 2.2 * (self.size - 1)).astype(int)
        ch_r = np.zeros((self.size, self.size), dtype=np.float32)
        in_bounds = (x_idx >= 0) & (x_idx < self.size) & (y_idx >= 0) & (y_idx < self.size)
        ch_r[y_idx[in_bounds], x_idx[in_bounds]] = 1.0
        # Optional Gaussian smoothing (disabled by default):
        # ch_r = gaussian_filter(ch_r, sigma=1.0)

        # --- Channel 2 (Blue): Curvature map -----------------------------
        # The absolute second derivative highlights eclipse ingress/egress
        # transitions while suppressing smooth out-of-eclipse regions.
        ch_b = np.zeros((self.size, self.size), dtype=np.float32)
        if len(f_normed) > 4:
            curvature = np.abs(np.gradient(np.gradient(f_normed)))
            curv_idx  = (np.clip(curvature * 15, 0, 1) * (self.size - 1)).astype(int)
            ch_b[curv_idx, p_idx] = 1.0
            # Optional Gaussian smoothing (disabled by default):
            # ch_b = gaussian_filter(ch_b, sigma=1.2)

        # Stack channels in the order expected by the classifier: [R, G, B].
        img = np.stack([ch_r, ch_g, ch_b])

        # Per-channel normalisation to [0, 1].
        for c in range(3):
            max_val = img[c].max()
            if max_val > 0:
                img[c] /= max_val

        return torch.tensor(img, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Sample retrieval with online augmentation
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an augmented image tensor and its integer class label.

        Augmentation pipeline (applied in order)
        -----------------------------------------
        1. Random circular phase shift of up to ±``phase_shift_limit`` × 100
           grid points via ``np.roll``.
        2. Single random outlier spike with probability ``outlier_prob``.
        3. Gaussian noise with σ ~ Uniform(0.005, 0.02).
        4. Random sparse sub-sampling:
           - 35 % probability: ~22 points  (N ~ Normal(22, 4),  sparse regime)
           - 65 % probability: ~48 points  (N ~ Normal(48, 8), moderate regime)
           Both distributions are clipped to [10, 100].
        """
        fluxes = self.data[idx].copy()

        # --- 1. Random phase shift ---------------------------------------
        if self.phase_shift_limit > 0:
            shift = int(
                np.random.uniform(-self.phase_shift_limit, self.phase_shift_limit) * 100
            )
            fluxes = np.roll(fluxes, shift)

        # --- 2. Random outlier spike -------------------------------------
        if np.random.random() < self.outlier_prob:
            outlier_idx = np.random.randint(0, 100)
            fluxes[outlier_idx] += np.random.normal(0, 0.4)

        # --- 3. Gaussian noise -------------------------------------------
        noise_sigma = np.random.uniform(0.005, 0.02)
        fluxes += np.random.normal(0, noise_sigma, 100)

        # --- 4. Random sparse sub-sampling --------------------------------
        # Two sampling regimes mimic survey cadence variability:
        #   sparse   (~22 pts) → simulates short-baseline or crowded-field data
        #   moderate (~48 pts) → typical Gaia or OGLE sampling density
        if np.random.random() < 0.35:
            n_pts = int(np.clip(np.random.normal(22, 4), 10, 100))
        else:
            n_pts = int(np.clip(np.random.normal(48, 8), 10, 100))

        selected_idx = np.sort(np.random.choice(100, n_pts, replace=False))

        img_tensor   = self._to_3channel(self.base_phase[selected_idx], fluxes[selected_idx])
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return img_tensor, label_tensor


# ===========================================================================
# 2. TRAINING ENGINE & VISUALISATION
# ===========================================================================

def train_model() -> None:
    """Fine-tune ResNet-18 on the eclipsing binary classification task.

    Workflow
    --------
    1. Load and shuffle ``selected_data.csv``; split 80 / 20 into train / val.
    2. Wrap splits in :class:`AdvancedLCDataset` and create DataLoaders.
    3. Initialise ResNet-18 with ImageNet weights; replace the FC head.
    4. Run up to 100 epochs with early stopping (patience = 7 on val loss).
    5. Save the best checkpoint and append per-epoch metrics to a CSV log.
    6. Generate and save a 4-panel diagnostic dashboard.

    Output files
    ------------
    best_model_binary.pth        – best validation-loss checkpoint.
    training_history_binary.csv  – epoch-level loss and accuracy log.
    training_analysis_plot.png   – diagnostic dashboard (loss curves,
                                   accuracy curves, confusion matrix,
                                   sample image comparison).
    """
    LOG_FILE   = "training_history_binary.csv"
    DATA_FILE  = "selected_data.csv"
    MODEL_FILE = "best_model_binary.pth"
    PLOT_FILE  = "training_analysis_plot.png"

    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Training data not found: {DATA_FILE}")
        return

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    df    = pd.read_csv(DATA_FILE).sample(frac=1).reset_index(drop=True)
    split = int(0.8 * len(df))

    train_df = df.iloc[:split].reset_index(drop=True)
    val_df   = df.iloc[split:].reset_index(drop=True)

    train_ds = AdvancedLCDataset(train_df)
    val_ds   = AdvancedLCDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64)

    # ------------------------------------------------------------------
    # Model, optimiser, loss
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start from ImageNet weights and replace the final FC layer for binary output.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    history: list[dict] = []
    best_val_loss = float("inf")
    patience      = 7
    patience_ctr  = 0

    print(f"\nTraining on {device} | {len(train_ds)} train / {len(val_ds)} val samples")
    print(f"{'Epoch':<6} | {'T_Loss':<8} | {'T_Acc':<7} | {'V_Loss':<8} | {'V_Acc':<7}")
    print("-" * 55)

    for epoch in range(100):

        # --- Training pass -------------------------------------------
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Ep {epoch + 1:>3}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            t_loss    += loss.item()
            preds      = logits.argmax(dim=1)
            t_correct += preds.eq(labels).sum().item()
            t_total   += labels.size(0)

        # --- Validation pass -----------------------------------------
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                v_loss    += criterion(logits, labels).item()
                preds      = logits.argmax(dim=1)
                v_correct += preds.eq(labels).sum().item()
                v_total   += labels.size(0)

        # --- Metrics -------------------------------------------------
        avg_t_loss = t_loss / len(train_loader)
        avg_v_loss = v_loss / len(val_loader)
        t_acc      = 100.0 * t_correct / t_total
        v_acc      = 100.0 * v_correct / v_total

        history.append({
            "epoch":  epoch + 1,
            "t_loss": avg_t_loss,
            "t_acc":  t_acc,
            "v_loss": avg_v_loss,
            "v_acc":  v_acc,
        })
        pd.DataFrame(history).to_csv(LOG_FILE, index=False)

        print(
            f"{epoch + 1:<6} | {avg_t_loss:<8.4f} | {t_acc:<6.2f}% | "
            f"{avg_v_loss:<8.4f} | {v_acc:<6.2f}%"
        )

        # --- Checkpoint & early stopping -----------------------------
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            torch.save(model.state_dict(), MODEL_FILE)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                break

    # ------------------------------------------------------------------
    # Diagnostic dashboard
    # ------------------------------------------------------------------
    print("\nTraining complete. Generating diagnostic dashboard...")

    # Reload the best checkpoint for confusion matrix and sample images.
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    h_df = pd.DataFrame(history)
    fig  = plt.figure(figsize=(18, 12))

    # --- Panel 1: Loss curves ----------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(h_df["epoch"], h_df["t_loss"], label="Train Loss",  color="blue")
    ax1.plot(h_df["epoch"], h_df["v_loss"], label="Val Loss",    color="red", linestyle="--")
    ax1.set_title("Training Loss History")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()

    # --- Panel 2: Accuracy curves ------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(h_df["epoch"], h_df["t_acc"], label="Train Acc",  color="green")
    ax2.plot(h_df["epoch"], h_df["v_acc"], label="Val Acc",    color="orange", linestyle="--")
    ax2.set_title("Training Accuracy History")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    # --- Panel 3: Confusion matrix on the validation set -------------
    all_preds: list[int] = []
    all_true:  list[int] = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            preds = model(imgs.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(labels.numpy())

    ax3 = fig.add_subplot(2, 2, 3)
    sns.heatmap(
        confusion_matrix(all_true, all_preds),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Detached", "Overcontact"],
        yticklabels=["Detached", "Overcontact"],
        ax=ax3,
    )
    ax3.set_title("Confusion Matrix (Validation Set)")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("True")

    # --- Panel 4: Sample images (one per class) -----------------------
    # Use original (non-augmented) val_df labels to safely locate one
    # representative sample per class.
    det_samples  = val_df[val_df.iloc[:, 100] == "det"]
    over_samples = val_df[val_df.iloc[:, 100] == "over"]

    ax4 = fig.add_subplot(2, 2, 4)
    if not det_samples.empty and not over_samples.empty:
        img_det,  _ = val_ds[det_samples.index[0]]
        img_over, _ = val_ds[over_samples.index[0]]

        # Side-by-side display: convert CHW → HWC for imshow.
        combined = np.hstack([
            img_det.permute(1, 2, 0).numpy(),
            img_over.permute(1, 2, 0).numpy(),
        ])
        ax4.imshow(combined, origin="lower")
        ax4.set_title("Sample Input: Detached (left) vs Overcontact (right)")
    else:
        ax4.text(0.5, 0.5, "No samples available", ha="center", va="center")
        ax4.set_title("Sample Input Comparison")
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    plt.show()
    print(f"Dashboard saved as '{PLOT_FILE}'.")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    train_model()

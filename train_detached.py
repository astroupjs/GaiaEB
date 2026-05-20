"""
train_detached.py
=================
Training script for a starspot *detection* classifier applied to **detached**
(EA / EB-type) eclipsing binary light curves.

The task is binary: given a phase-folded light curve of a detached system,
predict whether one of the stellar components hosts a starspot that causes a
measurable asymmetry in the out-of-eclipse regions of the observed photometry.

    Label 0 – ``nospot`` : unspotted light curve.
    Label 1 – ``spot``   : light curve distorted by a starspot.

Relationship to ``train_over.py``
----------------------------------
This script is the detached-binary counterpart of ``train_over.py``.  The
dataset class, channel encoding, and model architecture are identical; only
the following differ:

    - Input data  : ``selected_data_det.csv``  (detached systems)
      vs.           ``selected_data_over.csv`` (overcontact systems)
    - ES patience : 7 epochs here vs. 5 epochs in ``train_over.py``.
      Detached light curves have flatter out-of-eclipse regions where spot
      signals are weaker, so additional epochs are allowed before stopping.
    - Output files: ``best_model_spots_det.pth`` and
                    ``training_history_spots_det.csv`` to avoid overwriting
                    the overcontact artefacts.

Architecture overview
---------------------
:class:`MultiScaleSpotNet` wraps a pretrained ResNet-18 backbone (ImageNet
weights) whose global-average-pooling output (512-d) feeds a two-layer MLP
classifier with dropout regularisation.

Input data format
-----------------
A CSV file ``selected_data_det.csv`` in the working directory:

    Columns 0–99  : 100 evenly-spaced, phase-folded flux values.
    Column 100    : string class label – ``'spot'`` or ``'nospot'``.

Channel encoding (see :meth:`SpotDetectionDataset._to_3channel`)
-----------------------------------------------------------------
    Ch 0 – Gradient map    : first derivative of flux, highlights asymmetric
                             slopes introduced by a spot.
    Ch 1 – Wavelet detail  : db1 level-1 detail coefficients, sensitive to
                             high-frequency flux variations near the spot.
    Ch 2 – Adaptive stretch: 1st–99th percentile contrast stretch, preserves
                             low-amplitude out-of-eclipse modulation.

Training strategy
-----------------
- 80 / 20 random train / validation split.
- AdamW optimiser, lr = 1 × 10⁻³, weight decay = 1 × 10⁻⁵.
- Cross-entropy loss.
- Early stopping: patience = 7 epochs, minimum improvement δ = 1 × 10⁻⁴.
- Best checkpoint saved to ``best_model_spots_det.pth``.
- Per-epoch metrics appended to ``training_history_spots_det.csv``.

Data augmentation (training only)
----------------------------------
- Random circular phase shift (±5 % of the period).
- Random single-point outlier (σ = 0.4, probability 0.3).
- Bimodal sparse sub-sampling: ~22 pts (35 %) or ~48 pts (65 %),
  clipped to [10, 100].  Validation uses the full 100-point curve.

Outputs
-------
best_model_spots_det.pth         – state dict of the best validation checkpoint.
training_history_spots_det.csv   – per-epoch loss and accuracy log.
training_stats_dashboard_det.png – 2-panel plot (loss and accuracy curves).

Usage
-----
    python train_detached.py

Dependencies
------------
    torch, torchvision, pandas, numpy, pywt, matplotlib, seaborn, sklearn,
    scipy, tqdm

Author
------
    Stefan Parimucha
    Pavol Jozef Šafárik University in Košice
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm


# ===========================================================================
# 1. DATASET
# ===========================================================================

class SpotDetectionDataset(Dataset):
    """PyTorch Dataset that converts phase-folded light curves into spot-
    sensitive RGB image tensors.

    The channel encoding is designed to amplify the subtle, low-amplitude flux
    asymmetries caused by starspots.  In detached systems the out-of-eclipse
    regions are nearly flat, making spot-induced modulation particularly faint;
    the gradient and wavelet channels are therefore especially important here.

    Parameters
    ----------
    dataframe:
        DataFrame with 101 columns – 100 phase-folded flux values (columns
        0–99) followed by a string class label in column 100 (``'spot'`` or
        ``'nospot'``).
    image_size:
        Side length in pixels of the square output image (default: 128).
    outlier_prob:
        Probability of injecting a single random outlier during training
        (default: 0.3).  Ignored when ``is_training=False``.
    phase_shift_limit:
        Maximum circular phase shift as a fraction of the period (default:
        0.05).  Set to 0 to disable.
    is_training:
        If ``True``, online data augmentation (outlier injection and sparse
        sub-sampling) is applied.  Set to ``False`` for validation / inference
        to use the full 100-point curve without augmentation.

    Attributes
    ----------
    data : np.ndarray, shape (N, 100)
        Raw phase-folded flux arrays, one row per star.
    labels : np.ndarray, shape (N,)
        Integer class labels – 1 for spotted, 0 for unspotted.
    base_phase : np.ndarray, shape (100,)
        Fixed phase grid used after sub-sampling.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: int = 128,
        outlier_prob: float = 0.3,
        phase_shift_limit: float = 0.05,
        is_training: bool = True,
    ) -> None:
        self.data = dataframe.iloc[:, :100].values.astype(np.float32)

        target_col = dataframe.iloc[:, 100]
        self.labels = (
            target_col.map({"spot": 1, "nospot": 0}).values
            if target_col.dtype == object
            else target_col.values
        )

        self.size              = image_size
        self.outlier_prob      = outlier_prob
        self.phase_shift_limit = phase_shift_limit
        self.is_training       = is_training
        self.base_phase        = np.linspace(0.0, 1.0, 100, endpoint=True)

    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------------
    # Image rendering
    # ------------------------------------------------------------------

    def _to_3channel(self, phases: np.ndarray, fluxes: np.ndarray) -> torch.Tensor:
        """Render a sparse phase-folded light curve as a three-channel image.

        Each channel encodes a distinct spot-sensitive representation:

        Channel 0 – Gradient map
            The first derivative of flux is normalised to [0, 1] and rendered
            as a scatter image.  In detached light curves the out-of-eclipse
            regions are nearly flat; a spot introduces a slow, asymmetric
            slope that this channel makes spatially explicit.

        Channel 1 – Wavelet detail coefficients
            A single-level Daubechies db1 (Haar) discrete wavelet transform
            is applied; the detail (high-frequency) coefficients are
            normalised, resampled to the original length, and rendered as a
            scatter image.  This channel is sensitive to rapid, localised flux
            variations caused by spot ingress/egress.

        Channel 2 – Adaptive contrast stretch
            The flux is clipped to the [1st, 99th] percentile range and
            mapped to [0, 1].  This suppresses isolated outliers while
            preserving the low-amplitude, broad modulations that distinguish
            spotted from unspotted out-of-eclipse baselines.

        Parameters
        ----------
        phases:
            Phase values in [0, 1) for the (sub-sampled) observations.
        fluxes:
            Corresponding flux values (not necessarily normalised).

        Returns
        -------
        torch.Tensor of shape ``(3, size, size)`` and dtype ``float32``.
        """
        # Sort by phase so columns map left-to-right.
        sort_idx = np.argsort(phases)
        phases, fluxes = phases[sort_idx], fluxes[sort_idx]
        p_idx = (phases * (self.size - 1)).astype(int)

        # --- Channel 0: Gradient map ---------------------------------
        ch_grad = np.zeros((self.size, self.size), dtype=np.float32)
        if len(fluxes) > 2:
            grad      = np.gradient(fluxes)
            grad_norm = (grad - grad.min()) / (grad.max() - grad.min() + 1e-9)
            grad_idx  = (grad_norm * (self.size - 1)).astype(int)
            ch_grad[grad_idx, p_idx] = 1.0

        # --- Channel 1: Wavelet detail coefficients ------------------
        ch_wavelet = np.zeros((self.size, self.size), dtype=np.float32)
        if len(fluxes) >= 8:
            # Single-level db1 DWT; coeffs[1] contains the detail band.
            coeffs  = pywt.wavedec(fluxes, "db1", level=1)
            cD      = coeffs[1]
            cD_norm = (cD - cD.min()) / (cD.max() - cD.min() + 1e-9)

            # Resample detail coefficients back to the original length so
            # that p_idx can be reused directly.
            cD_resampled = np.interp(
                np.linspace(0, 1, len(fluxes)),
                np.linspace(0, 1, len(cD)),
                cD_norm,
            )
            ch_wavelet[(cD_resampled * (self.size - 1)).astype(int), p_idx] = 1.0

        # --- Channel 2: Adaptive contrast stretch --------------------
        # Percentile clipping suppresses isolated outliers while retaining
        # the broader spot-induced modulation across the out-of-eclipse baseline.
        p1, p99    = np.percentile(fluxes, [1, 99])
        f_norm     = np.clip((fluxes - p1) / (p99 - p1 + 1e-9), 0.0, 1.0)
        ch_stretch = np.zeros((self.size, self.size), dtype=np.float32)
        ch_stretch[(f_norm * (self.size - 1)).astype(int), p_idx] = 1.0

        return torch.tensor(
            np.stack([ch_grad, ch_wavelet, ch_stretch]),
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Sample retrieval with optional online augmentation
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an (optionally augmented) image tensor and its class label.

        Augmentation pipeline (training mode only)
        -------------------------------------------
        1. Circular phase shift of up to ±``phase_shift_limit`` × 100 grid
           points via ``np.roll``.
        2. Single random outlier spike (σ = 0.4) with probability
           ``outlier_prob``.
        3. Bimodal sparse sub-sampling:
           - 35 % probability: ~22 points  (N ~ Normal(22, 4))  – sparse
           - 65 % probability: ~48 points  (N ~ Normal(48, 8))  – moderate
           Both clipped to [10, 100].
        In validation / inference mode (``is_training=False``), the full
        100-point grid is used with no augmentation except the phase shift.
        """
        fluxes = self.data[idx].copy()

        # --- 1. Circular phase shift (always applied) ----------------
        if self.phase_shift_limit > 0:
            shift  = int(
                np.random.uniform(-self.phase_shift_limit, self.phase_shift_limit) * 100
            )
            fluxes = np.roll(fluxes, shift)

        # --- 2. Random outlier injection (training only) -------------
        if self.is_training and np.random.random() < self.outlier_prob:
            fluxes[np.random.randint(0, 100)] += np.random.normal(0, 0.4)

        # --- 3. Sparse sub-sampling (training) or full curve (val) ---
        if self.is_training:
            if np.random.random() < 0.35:
                n_pts = int(np.clip(np.random.normal(22, 4), 10, 100))
            else:
                n_pts = int(np.clip(np.random.normal(48, 8), 10, 100))
        else:
            n_pts = 100  # full-resolution curve for validation / inference

        selected_idx = np.sort(np.random.choice(100, n_pts, replace=False))

        img_tensor   = self._to_3channel(self.base_phase[selected_idx], fluxes[selected_idx])
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return img_tensor, label_tensor


# ===========================================================================
# 2. MODEL ARCHITECTURE
# ===========================================================================

class MultiScaleSpotNet(nn.Module):
    """ResNet-18 backbone with a two-layer MLP classifier for spot detection.

    The ImageNet-pretrained ResNet-18 acts as a general-purpose feature
    extractor.  Its fully-connected head is replaced with ``nn.Identity`` so
    that the 512-d global-average-pooling vector passes directly to a
    lightweight MLP with dropout regularisation.

    Architecture
    ------------
    Backbone : ResNet-18 (pretrained) → 512-d feature vector
    Head     : Linear(512 → 256) → ReLU → Dropout(0.1) → Linear(256 → 2)

    The compact head reduces overfitting on the relatively small spot-training
    dataset while allowing the backbone to retain its broad feature vocabulary.
    """

    def __init__(self) -> None:
        super().__init__()
        backbone    = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()   # remove the original 1000-class head
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: image tensor → class logits.

        Parameters
        ----------
        x:
            Batch of RGB image tensors, shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor of shape ``(B, 2)`` containing raw (pre-softmax) logits.
        """
        features = self.backbone(x)
        return self.classifier(features)


# ===========================================================================
# 3. TRAINING ENGINE & VISUALISATION
# ===========================================================================

def train_model() -> None:
    """Fine-tune MultiScaleSpotNet on starspot detection for detached binaries.

    Workflow
    --------
    1. Load and shuffle ``selected_data_det.csv``; split 80 / 20.
    2. Wrap splits in :class:`SpotDetectionDataset`; create DataLoaders.
    3. Initialise :class:`MultiScaleSpotNet`; configure AdamW + cross-entropy.
    4. Run up to 100 epochs with early stopping
       (patience = 7, min δ = 1 × 10⁻⁴ on val loss).
    5. Save the best checkpoint and log per-epoch metrics to CSV.
    6. Generate and save a 2-panel loss / accuracy dashboard.

    Output files
    ------------
    best_model_spots_det.pth         – best validation-loss checkpoint.
    training_history_spots_det.csv   – epoch-level loss and accuracy log.
    training_stats_dashboard_det.png – loss and accuracy curves.

    Notes
    -----
    Early-stopping patience is set to 7 (vs. 5 for overcontact) because
    detached light curves have flatter out-of-eclipse baselines where spot
    signals are weaker and convergence is slower.
    """
    LOG_FILE   = "training_history_spots_det.csv"
    DATA_PATH  = "selected_data_det.csv"
    MODEL_FILE = "best_model_spots_det.pth"
    PLOT_FILE  = "training_stats_dashboard_det.png"

    # Early-stopping hyperparameters.
    # Patience is larger than for overcontact systems (7 vs. 5) to allow the
    # model more time to learn weaker spot signals in flat out-of-eclipse regions.
    ES_PATIENCE  = 7
    ES_MIN_DELTA = 1e-4

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Training data not found: {DATA_PATH}")
        return

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    df    = pd.read_csv(DATA_PATH).sample(frac=1).reset_index(drop=True)
    split = int(0.8 * len(df))

    train_ds = SpotDetectionDataset(df.iloc[:split], is_training=True)
    val_ds   = SpotDetectionDataset(df.iloc[split:], is_training=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64)

    # ------------------------------------------------------------------
    # Model, optimiser, loss
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MultiScaleSpotNet().to(device)

    # AdamW with mild weight decay to regularise the large backbone.
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    history: list[dict] = []
    best_val_loss    = float("inf")
    es_counter       = 0
    early_stop_epoch = None

    print(f"\nTraining on {device} | {len(train_ds)} train / {len(val_ds)} val samples")

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
            t_correct += logits.argmax(dim=1).eq(labels).sum().item()
            t_total   += labels.size(0)

        # --- Validation pass -----------------------------------------
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits  = model(imgs)
                v_loss    += criterion(logits, labels).item()
                v_correct += logits.argmax(dim=1).eq(labels).sum().item()
                v_total   += labels.size(0)

        # --- Metrics -------------------------------------------------
        avg_t_loss = t_loss / len(train_loader)
        avg_v_loss = v_loss / len(val_loader)
        t_acc      = 100.0 * t_correct / t_total
        v_acc      = 100.0 * v_correct / v_total

        metrics = {
            "epoch":  epoch + 1,
            "t_loss": avg_t_loss,
            "t_acc":  t_acc,
            "v_loss": avg_v_loss,
            "v_acc":  v_acc,
        }
        history.append(metrics)
        pd.DataFrame(history).to_csv(LOG_FILE, index=False)

        print(
            f"Ep {epoch + 1:02} | T_Loss: {avg_t_loss:.4f} | "
            f"V_Loss: {avg_v_loss:.4f} | V_Acc: {v_acc:.2f}%",
            end="",
        )

        # --- Checkpoint & early stopping -----------------------------
        # Require a minimum improvement of ES_MIN_DELTA to reset the counter;
        # this prevents saving trivially small improvements near convergence.
        if avg_v_loss < best_val_loss - ES_MIN_DELTA:
            best_val_loss = avg_v_loss
            es_counter    = 0
            torch.save(model.state_dict(), MODEL_FILE)
            print(" [saved]")
        else:
            es_counter += 1
            print(f" [no improvement {es_counter}/{ES_PATIENCE}]")
            if es_counter >= ES_PATIENCE:
                early_stop_epoch = epoch + 1
                print(
                    f"\nEarly stopping triggered at epoch {early_stop_epoch}. "
                    f"Best val loss: {best_val_loss:.4f}"
                )
                break

    if early_stop_epoch is None:
        print(f"\nTraining completed all epochs. Best val loss: {best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Diagnostic dashboard
    # ------------------------------------------------------------------
    print("\nGenerating diagnostic dashboard...")

    h_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Loss curves
    axes[0].plot(h_df["epoch"], h_df["t_loss"], label="Train")
    axes[0].plot(h_df["epoch"], h_df["v_loss"], label="Val", linestyle="--")
    axes[0].set_title("Loss – Detached Spot Classifier")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    # Panel 2: Accuracy curves
    axes[1].plot(h_df["epoch"], h_df["t_acc"], label="Train")
    axes[1].plot(h_df["epoch"], h_df["v_acc"], label="Val", linestyle="--")
    axes[1].set_title("Accuracy – Detached Spot Classifier")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    plt.show()
    print(f"Dashboard saved as '{PLOT_FILE}'.")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    train_model()

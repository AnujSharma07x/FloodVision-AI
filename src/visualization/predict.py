
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import Sen1Floods11Dataset
from src.models.unet import UNet


# ==========================================
# 1. CONFIGURATION
# ==========================================

DEVICE = torch.device("cpu")

CHECKPOINT_PATH = Path("checkpoints/flood_unet_full_best.pt")
OUTPUT_PATH = Path("outputs/prediction_visualization.png")

THRESHOLD = 0.55


# ==========================================
# 2. MAIN FUNCTION
# ==========================================

def main():
    print("=" * 55)
    print("FloodVision-AI: Prediction Visualization")
    print("=" * 55)
    print("Device:", DEVICE)

    # Check checkpoint
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
        )

    # Load validation dataset for visualization
    # Keep the test set reserved for final evaluation.
    dataset = Sen1Floods11Dataset(split="valid")

    # Find the first sample containing valid pixels
    sample_index = None

    for index in range(len(dataset)):
        candidate = dataset[index]

        if candidate["valid"].any():
            sample_index = index
            break

    if sample_index is None:
        raise RuntimeError("No validation sample has valid pixels.")

    sample = dataset[sample_index]

    image = sample["image"].unsqueeze(0).to(DEVICE)
    mask = sample["mask"].squeeze(0).cpu().numpy()
    valid = sample["valid"].squeeze(0).cpu().numpy().astype(bool)

    # Load the full-dataset checkpoint
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = UNet(in_channels=2, out_channels=1).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint:", CHECKPOINT_PATH.resolve())
    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Checkpoint validation IoU:",
        f'{checkpoint["validation_iou"]:.4f}',
    )
    print("Validation sample index:", sample_index)
    print("Threshold:", THRESHOLD)

    # Generate prediction
    with torch.no_grad():
        logits = model(image)
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

    predictions = probabilities >= THRESHOLD

    # Calculate metrics only over valid pixels
    actual_flood = (mask >= 0.5) & valid
    predicted_flood = predictions & valid

    tp = np.logical_and(predicted_flood, actual_flood).sum()
    fp = np.logical_and(predicted_flood, ~actual_flood & valid).sum()
    fn = np.logical_and(~predicted_flood, actual_flood).sum()

    iou = tp / max(tp + fp + fn, 1)
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)

    valid_pixels = valid.sum()
    actual_pixels = actual_flood.sum()
    predicted_pixels = predicted_flood.sum()

    print("Valid pixels:", int(valid_pixels))
    print("Actual flood pixels:", int(actual_pixels))
    print(
        "Actual flood percentage:",
        f"{100 * actual_pixels / max(valid_pixels, 1):.2f}%",
    )
    print("Predicted flood pixels:", int(predicted_pixels))
    print(
        "Predicted flood percentage:",
        f"{100 * predicted_pixels / max(valid_pixels, 1):.2f}%",
    )
    print("Mean predicted probability:", f"{probabilities[valid].mean():.4f}")
    print("Sample IoU:", f"{iou:.4f}")
    print("Sample Dice:", f"{dice:.4f}")

    # ======================================
    # 3. CREATE VISUALIZATION
    # ======================================

    # Sentinel-1 VV band for a simple grayscale input view
    vv_image = image[0, 0].cpu().numpy()

    # Hide invalid pixels in the displayed image and masks
    vv_display = np.where(valid, vv_image, np.nan)
    actual_display = np.where(valid, mask, np.nan)
    predicted_display = np.where(valid, predictions.astype(float), np.nan)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(vv_display, cmap="gray")
    axes[0].set_title("Sentinel-1 VV Input")

    axes[1].imshow(actual_display, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Actual Flood Mask")

    axes[2].imshow(predicted_display, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title(f"Predicted Flood Mask (Threshold {THRESHOLD})")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"FloodVision-AI | IoU: {iou:.4f} | Dice: {dice:.4f}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("-" * 55)
    print("Visualization saved:", OUTPUT_PATH.resolve())
    print("PREDICTION COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()

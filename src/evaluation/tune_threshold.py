
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import Sen1Floods11Dataset
from src.models.unet import UNet


# ==========================================
# 1. CONFIGURATION
# ==========================================

DEVICE = torch.device("cpu")
BATCH_SIZE = 1

CHECKPOINT_PATH = Path("checkpoints/flood_unet_full_best.pt")

THRESHOLDS = [
    round(i / 100, 2)
    for i in range(10, 91, 5)
]


# ==========================================
# 2. METRICS
# ==========================================

def calculate_metrics(tp, fp, fn):
    iou_denominator = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn

    if iou_denominator == 0:
        iou = 1.0
    else:
        iou = tp / iou_denominator

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (2 * tp) / dice_denominator

    return iou, dice


# ==========================================
# 3. VALIDATION THRESHOLD TUNING
# ==========================================

def main():
    print("=" * 60)
    print("FloodVision-AI: Validation Threshold Tuning")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT_PATH.resolve())

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
        )

    # Load validation dataset only.
    dataset = Sen1Floods11Dataset(split="valid")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Validation samples:", len(dataset))
    print("Thresholds:", THRESHOLDS)

    # Load the trained model.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = UNet(in_channels=2, out_channels=1).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Saved validation IoU:",
        f'{checkpoint["validation_iou"]:.4f}',
    )

    # Store valid-pixel probabilities and ground truth once.
    all_probabilities = []
    all_masks = []
    all_valid = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Running validation"):
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            valid = batch["valid"].to(DEVICE).bool()

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            all_probabilities.append(
                probabilities.cpu()
            )
            all_masks.append(masks.cpu())
            all_valid.append(valid.cpu())

    all_probabilities = torch.cat(all_probabilities, dim=0)
    all_masks = torch.cat(all_masks, dim=0) >= 0.5
    all_valid = torch.cat(all_valid, dim=0)

    # Ground-truth flood pixels only inside valid areas.
    actual = all_masks & all_valid

    total_valid_pixels = all_valid.sum().item()
    total_actual_flood = actual.sum().item()

    if total_valid_pixels == 0:
        raise RuntimeError(
            "Validation dataset contains no valid pixels."
        )

    print("\nValid pixels:", total_valid_pixels)
    print("Actual flood pixels:", total_actual_flood)

    # ======================================
    # Evaluate each threshold
    # ======================================

    results = []

    print("\n" + "-" * 60)
    print(
        f"{'Threshold':>10} "
        f"{'IoU':>12} "
        f"{'Dice':>12} "
        f"{'Precision':>12} "
        f"{'Recall':>12}"
    )
    print("-" * 60)

    for threshold in THRESHOLDS:
        predictions = (
            all_probabilities >= threshold
        ) & all_valid

        tp = (predictions & actual).sum().item()
        fp = (predictions & ~actual & all_valid).sum().item()
        fn = (~predictions & actual).sum().item()

        iou, dice = calculate_metrics(tp, fp, fn)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        result = {
            "threshold": threshold,
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
        }

        results.append(result)

        print(
            f"{threshold:>10.2f} "
            f"{iou:>12.4f} "
            f"{dice:>12.4f} "
            f"{precision:>12.4f} "
            f"{recall:>12.4f}"
        )

    # Prefer the threshold closest to 0.5 if IoU ties.
    best = max(
        results,
        key=lambda item: (
            item["iou"],
            -abs(item["threshold"] - 0.5),
        ),
    )

    print("-" * 60)
    print("\nBEST VALIDATION THRESHOLD")
    print("=" * 40)
    print("Threshold:", best["threshold"])
    print(f"Validation IoU: {best['iou']:.4f}")
    print(f"Validation Dice: {best['dice']:.4f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print("=" * 40)

    print(
        "\nNote: This script only evaluates the validation set."
    )
    print(
        "No model weights were changed or retrained."
    )
    print(
        "Use the validation results to select a threshold; "
        "keep the test set untouched until final evaluation."
    )


if __name__ == "__main__":
    main()

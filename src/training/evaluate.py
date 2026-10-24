
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import Sen1Floods11Dataset
from src.models.unet import UNet


# ==========================================
# 1. CONFIGURATION
# ==========================================

DEVICE = torch.device("cpu")
BATCH_SIZE = 1
THRESHOLD = 0.55

CHECKPOINT_PATH = Path("checkpoints/flood_unet_full_best.pt")


# ==========================================
# 2. EVALUATION
# ==========================================

def main():
    print("=" * 50)
    print("FloodVision-AI: Test Evaluation")
    print("=" * 50)
    print("Device:", DEVICE)
    print("Threshold:", THRESHOLD)

    # Verify checkpoint exists
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
        )

    # Load held-out test dataset
    test_dataset = Sen1Floods11Dataset(split="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Load full-dataset model checkpoint
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = UNet(in_channels=2, out_channels=1).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint:", CHECKPOINT_PATH.resolve())
    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Saved validation IoU:",
        f'{checkpoint["validation_iou"]:.4f}',
    )
    print("Test dataset size:", len(test_dataset))
    print("-" * 50)

    # Aggregate pixel counts over all valid test pixels
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    evaluated_samples = 0

    with torch.no_grad():
        for index, batch in enumerate(test_loader, start=1):
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            valid = batch["valid"].to(DEVICE).bool()

            if not valid.any():
                print(f"Skipping sample {index}: no valid pixels.")
                continue

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            predictions = (probabilities >= THRESHOLD) & valid
            targets = (masks >= 0.55) & valid

            tp = (predictions & targets).sum().item()
            fp = (predictions & ~targets & valid).sum().item()
            fn = (~predictions & targets & valid).sum().item()
            tn = (~predictions & ~targets & valid).sum().item()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
            evaluated_samples += 1

            print(
                f"Test sample {index}/{len(test_loader)} evaluated",
                end="\r",
                flush=True,
            )

    if evaluated_samples == 0:
        raise RuntimeError("No test samples had valid pixels.")

    # Calculate aggregate metrics
    iou = total_tp / max(total_tp + total_fp + total_fn, 1)

    dice = (2 * total_tp) / max(
        2 * total_tp + total_fp + total_fn, 1
    )

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)

    accuracy = (total_tp + total_tn) / max(
        total_tp + total_fp + total_fn + total_tn, 1
    )

    # Print final results
    print("\n\n========== TEST RESULTS ==========")
    print("Evaluated samples:", evaluated_samples)
    print("True positives:", int(total_tp))
    print("False positives:", int(total_fp))
    print("False negatives:", int(total_fn))
    print("True negatives:", int(total_tn))
    print("---------------------------------")
    print(f"Test IoU:       {iou:.4f}")
    print(f"Test Dice:      {dice:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall:    {recall:.4f}")
    print(f"Test Accuracy:  {accuracy:.4f}")
    print("=================================")


if __name__ == "__main__":
    main()

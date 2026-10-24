
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import Sen1Floods11Dataset
from src.models.unet import UNet
from src.training.losses import MaskedBCEDiceLoss


# ==========================================
# 1. CONFIGURATION
# ==========================================

DEVICE = torch.device("cpu")

BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-4

CHECKPOINT_DIR = Path("checkpoints")

BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "flood_unet_full_best.pt"
LATEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "flood_unet_full_latest.pt"


# ==========================================
# 2. METRICS
# ==========================================

def calculate_metrics(tp, fp, fn):
    denominator = tp + fp + fn

    if denominator == 0:
        iou = 1.0
    else:
        iou = tp / denominator

    dice_denominator = 2 * tp + fp + fn

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (2 * tp) / dice_denominator

    return iou, dice


# ==========================================
# 3. TRAINING FUNCTION
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for batch in progress_bar:
        images = batch["image"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)
        valid = batch["valid"].to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks, valid)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += 1

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    if total_samples == 0:
        return 0.0

    return total_loss / total_samples


# ==========================================
# 4. VALIDATION FUNCTION
# ==========================================

def validate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validation", leave=False)

        for batch in progress_bar:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            valid = batch["valid"].to(DEVICE).bool()

            logits = model(images)
            loss = criterion(logits, masks, valid)

            total_loss += loss.item()
            total_samples += 1

            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= 0.5
            actual = masks >= 0.5

            predictions = predictions & valid
            actual = actual & valid

            total_tp += (predictions & actual).sum().item()
            total_fp += (predictions & ~actual & valid).sum().item()
            total_fn += (~predictions & actual & valid).sum().item()

    average_loss = (
        total_loss / total_samples if total_samples else 0.0
    )

    iou, dice = calculate_metrics(total_tp, total_fp, total_fn)

    return average_loss, iou, dice


# ==========================================
# 5. MAIN
# ==========================================

def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("FloodVision-AI: Full Dataset Training")
    print("=" * 55)

    # Load complete datasets
    train_dataset = Sen1Floods11Dataset(split="train")
    valid_dataset = Sen1Floods11Dataset(split="valid")

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Device:             {DEVICE}")
    print(f"Epochs:             {EPOCHS}")
    print(f"Batch size:         {BATCH_SIZE}")
    print("Prediction threshold: 0.5")
    print("=" * 55)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Initialize model
    model = UNet(in_channels=2, out_channels=1).to(DEVICE)

    criterion = MaskedBCEDiceLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_iou = -1.0

    # ======================================
    # TRAINING LOOP
    # ======================================

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 40)

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        valid_loss, valid_iou, valid_dice = validate(
            model,
            valid_loader,
            criterion,
        )

        print(f"Train loss:      {train_loss:.4f}")
        print(f"Validation loss: {valid_loss:.4f}")
        print(f"Validation IoU:  {valid_iou:.4f}")
        print(f"Validation Dice: {valid_dice:.4f}")

        # Save latest checkpoint after every epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_iou": valid_iou,
            "validation_loss": valid_loss,
            "validation_dice": valid_dice,
        }

        torch.save(checkpoint, LATEST_CHECKPOINT_PATH)

        # Save best checkpoint according to validation IoU
        if valid_iou > best_iou:
            best_iou = valid_iou

            torch.save(checkpoint, BEST_CHECKPOINT_PATH)

            print("New best model saved!")

        print(f"Best validation IoU so far: {best_iou:.4f}")

    print("\n" + "=" * 55)
    print("FULL DATASET TRAINING COMPLETED")
    print("=" * 55)
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Best checkpoint:     {BEST_CHECKPOINT_PATH.resolve()}")
    print(f"Latest checkpoint:   {LATEST_CHECKPOINT_PATH.resolve()}")


if __name__ == "__main__":
    main()

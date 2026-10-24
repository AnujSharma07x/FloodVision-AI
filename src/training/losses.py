
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEDiceLoss(nn.Module):
    """BCE + Dice loss that ignores invalid pixels."""

    def __init__(
        self,
        pos_weight=5.0,
        bce_weight=0.5,
        dice_weight=0.5,
        smooth=1.0,
    ):
        super().__init__()

        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32),
        )
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets, valid):
        valid = valid.bool()
        targets = targets.float()
        valid_float = valid.float()

        if not valid.any():
            return logits.sum() * 0.0

        # Binary cross-entropy only on valid pixels.
        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )

        bce_loss = (bce_map * valid_float).sum()
        bce_loss = bce_loss / valid_float.sum().clamp_min(1.0)

        # Dice loss only on valid pixels.
        probs = torch.sigmoid(logits) * valid_float
        safe_targets = targets * valid_float

        dims = (1, 2, 3)
        intersection = (probs * safe_targets).sum(dim=dims)
        denominator = (
            probs.sum(dim=dims) + safe_targets.sum(dim=dims)
        )

        valid_images = valid.sum(dim=dims) > 0

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        dice_loss = (1.0 - dice[valid_images]).mean()

        return (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )

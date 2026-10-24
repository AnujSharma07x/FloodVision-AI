
import torch

from src.training.losses import MaskedBCEDiceLoss


def test_loss_is_finite_and_non_negative():
    criterion = MaskedBCEDiceLoss()

    logits = torch.zeros(1, 1, 4, 4)
    targets = torch.zeros(1, 1, 4, 4)
    valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)

    loss = criterion(logits, targets, valid)

    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_invalid_pixels_do_not_affect_loss():
    criterion = MaskedBCEDiceLoss()

    logits_a = torch.zeros(1, 1, 2, 2)
    logits_b = logits_a.clone()

    targets = torch.zeros(1, 1, 2, 2)
    valid = torch.tensor(
        [[[[True, False], [True, False]]]]
    )

    # Change logits and targets only at invalid pixels.
    logits_b[0, 0, 0, 1] = 100
    logits_b[0, 0, 1, 1] = -100

    targets_b = targets.clone()
    targets_b[0, 0, 0, 1] = 1
    targets_b[0, 0, 1, 1] = 1

    loss_a = criterion(logits_a, targets, valid)
    loss_b = criterion(logits_b, targets_b, valid)

    assert torch.allclose(loss_a, loss_b)


def test_all_invalid_pixels_return_zero_loss():
    criterion = MaskedBCEDiceLoss()

    logits = torch.randn(1, 1, 4, 4, requires_grad=True)
    targets = torch.zeros(1, 1, 4, 4)
    valid = torch.zeros(1, 1, 4, 4, dtype=torch.bool)

    loss = criterion(logits, targets, valid)

    assert loss.item() == 0.0

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()




import torch

from src.models.unet import UNet


def test_unet_output_shape():
    """The model should return one flood logit per pixel."""
    model = UNet()
    model.eval()

    sample = torch.randn(1, 2, 128, 128)

    with torch.no_grad():
        output = model(sample)

    assert output.shape == (1, 1, 128, 128)


def test_unet_output_is_finite():
    """The model should not produce NaN or infinite values."""
    model = UNet()
    model.eval()

    sample = torch.randn(1, 2, 128, 128)

    with torch.no_grad():
        output = model(sample)

    assert torch.isfinite(output).all()

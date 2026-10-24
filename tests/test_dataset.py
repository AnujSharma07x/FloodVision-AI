
import csv

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

import src.data.dataset as dataset_module
from src.data.dataset import Sen1Floods11Dataset


def create_tiff(path, data, nodata=None):
    """Create a small synthetic GeoTIFF for testing."""
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        transform=from_origin(0, 4, 1, 1),
        nodata=nodata,
    ) as dst:
        dst.write(data)


def test_invalid_split_is_rejected():
    with pytest.raises(ValueError):
        Sen1Floods11Dataset(split="unknown")


def test_dataset_loads_tiffs_and_masks_invalid_pixels(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    split_root = tmp_path / "splits"
    image_dir = data_root / "S1Hand"
    label_dir = data_root / "LabelHand"

    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    split_root.mkdir(parents=True)

    image = np.full((2, 4, 4), -10.0, dtype=np.float32)
    image[0, 0, 0] = np.nan

    label = np.zeros((4, 4), dtype=np.float32)
    label[1, 1] = 1

    create_tiff(image_dir / "sample.tif", image)
    create_tiff(label_dir / "label.tif", label)

    with (split_root / "flood_train_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        csv.writer(file).writerow(["sample.tif", "label.tif"])

    monkeypatch.setattr(dataset_module, "DATA_ROOT", data_root)
    monkeypatch.setattr(dataset_module, "SPLIT_ROOT", split_root)

    dataset = Sen1Floods11Dataset(split="train")
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["image"].shape == (2, 4, 4)
    assert sample["mask"].shape == (1, 4, 4)
    assert sample["valid"].shape == (1, 4, 4)

    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32
    assert sample["valid"].dtype == torch.bool

    assert sample["valid"][0, 0, 0].item() is False
    assert sample["valid"][0, 1, 1].item() is True
    assert sample["mask"][0, 1, 1].item() == 1.0

    assert torch.isfinite(sample["image"]).all()
    assert torch.isfinite(sample["mask"]).all()


def test_normalization_statistics_are_finite():
    means = np.array([-10.6284, -17.4934], dtype=np.float32)
    stds = np.array([4.2170, 4.8675], dtype=np.float32)

    image = np.zeros((2, 4, 4), dtype=np.float32)
    normalized = (image - means[:, None, None]) / stds[:, None, None]

    assert normalized.shape == (2, 4, 4)
    assert np.isfinite(normalized).all()
    assert np.all(stds > 0)

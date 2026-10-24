
from pathlib import Path
import csv

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


DATA_ROOT = Path(
    r"C:\Users\KIIT\Desktop\sih hackathon\DRISHTI-repo"
    r"\data\sen1floods11\v1.1\data\flood_events\HandLabeled"
)

SPLIT_ROOT = Path(
    r"C:\Users\KIIT\Desktop\sih hackathon\DRISHTI-repo"
    r"\data\sen1floods11\v1.1\splits\flood_handlabeled"
)


class Sen1Floods11Dataset(Dataset):
    """Load Sentinel-1 images and corresponding binary flood masks."""

    def __init__(self, split="train"):
        if split not in {"train", "valid", "test", "bolivia"}:
            raise ValueError(
                "split must be train, valid, test, or bolivia"
            )

        self.image_dir = DATA_ROOT / "S1Hand"
        self.label_dir = DATA_ROOT / "LabelHand"
        csv_path = SPLIT_ROOT / f"flood_{split}_data.csv"

        if not csv_path.is_file():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        self.samples = []

        with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)

            for row in reader:
                if not row:
                    continue
                if len(row) != 2:
                    raise ValueError(f"Invalid CSV row: {row}")

                image_path = self.image_dir / row[0].strip()
                label_path = self.label_dir / row[1].strip()

                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                if not label_path.is_file():
                    raise FileNotFoundError(label_path)

                self.samples.append((image_path, label_path))

        if not self.samples:
            raise RuntimeError(f"No samples found for split: {split}")

        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
            image_valid = src.dataset_mask() > 0
            image_nodata = src.nodata

        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.float32)
            label_valid = src.dataset_mask() > 0
            label_nodata = src.nodata

        if image.shape[0] != 2:
            raise ValueError(
                f"Expected 2 bands, got {image.shape[0]}: "
                f"{image_path.name}"
            )

        valid = (
            image_valid
            & label_valid
            & np.isfinite(label)
            & np.all(np.isfinite(image), axis=0)
            & np.isin(label, [0, 1])
        )

        if image_nodata is not None and np.isfinite(image_nodata):
            valid &= np.all(image != image_nodata, axis=0)

        if label_nodata is not None and np.isfinite(label_nodata):
            valid &= label != label_nodata

        # Replace invalid input values before normalization.
        image = np.nan_to_num(
            image, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Dataset-level Sentinel-1 statistics, not per-image min-max.
        means = np.array([-10.6284, -17.4934], dtype=np.float32)
        stds = np.array([4.2170, 4.8675], dtype=np.float32)

        image = (image - means[:, None, None]) / stds[:, None, None]

        # Invalid pixels are excluded from loss using the valid mask.
        mask = np.where(valid, label, 0).astype(np.float32)

        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(image)
            ),
            "mask": torch.from_numpy(
                np.ascontiguousarray(mask[None, ...])
            ),
            "valid": torch.from_numpy(
                np.ascontiguousarray(valid[None, ...])
            ),
            "image_path": str(image_path),
        }


if __name__ == "__main__":
    for split in ("train", "valid", "test", "bolivia"):
        dataset = Sen1Floods11Dataset(split=split)
        sample = dataset[0]

        print(f"\n{split}: {len(dataset)} samples")
        print("Image shape:", tuple(sample["image"].shape))
        print("Mask shape:", tuple(sample["mask"].shape))
        print("Valid pixels:", int(sample["valid"].sum()))
        print("Image:", Path(sample["image_path"]).name)

# FloodVision-AI 

**Deep Learning-Based Flood Segmentation Using Sentinel-1 Satellite Imagery**

FloodVision-AI is a deep learning research prototype that detects and segments flood-affected areas from Sentinel-1 Synthetic Aperture Radar (SAR) imagery using a U-Net convolutional neural network.

The project uses the Sen1Floods11 dataset to train and evaluate a binary flood segmentation model. It also includes validation-based threshold selection and prediction visualization.

## Project Status

**Current stage:** Model training, validation, and held-out test evaluation completed.

FloodVision-AI currently focuses on the flood segmentation component. It is not yet a complete operational disaster-management system.

## Features

- Sentinel-1 VV and VH radar input channels.
- Sen1Floods11 HandLabeled dataset support.
- U-Net-based flood segmentation model.
- Masked BCE + Dice loss for handling invalid and NoData pixels.
- Validation-based prediction threshold tuning.
- Evaluation using IoU, Dice, precision, recall, and accuracy.
- Visualization of satellite input, ground-truth mask, and predicted flood mask.
- Separate training, validation, and test data splits.

## Technology Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Model:** U-Net
- **Data Processing:** NumPy
- **Visualization:** Matplotlib
- **Dataset:** Sen1Floods11 v1.1

## Model Overview

The model takes two Sentinel-1 radar channels as input:

- **VV:** Vertical transmit, vertical receive polarization.
- **VH:** Vertical transmit, horizontal receive polarization.

The U-Net model generates a pixel-wise flood probability map. A fixed threshold of `0.55`, selected using validation data, converts the probabilities into a binary flood mask.

The model uses a masked BCE + Dice loss so invalid or NoData pixels do not contribute to the loss calculation.

## Dataset

This project uses the **Sen1Floods11 v1.1 HandLabeled dataset**.

The dataset contains Sentinel-1 imagery and corresponding flood labels. Invalid label pixels are excluded from evaluation using a validity mask.

The dataset is divided into training, validation, and test splits. Two test samples with no valid pixels are skipped during evaluation.

**Note:** The dataset is not included in this repository by default. Obtain it from its official source and configure the dataset paths in `src/data/dataset.py` according to your local setup.

## Project Workflow

1. Load Sentinel-1 VV and VH imagery.
2. Normalize the input channels.
3. Load the ground-truth flood mask and validity mask.
4. Train the U-Net segmentation model.
5. Select the prediction threshold using validation data.
6. Visualize the predicted flood mask.
7. Evaluate the final model on the held-out test split.

## Evaluation Results

The following results were obtained using the best saved checkpoint and a fixed prediction threshold of `0.55`.

| Metric | Test Result |
|---|---:|
| Intersection over Union (IoU) | 0.6053 |
| Dice Score | 0.7541 |
| Precision | 73.47% |
| Recall | 77.47% |
| Accuracy | 93.79% |
| Evaluated test samples | 88 |

Two samples containing no valid pixels were skipped, leaving 88 evaluated samples.

### Threshold Selection

The threshold was selected using aggregate pixel-level IoU on the validation split.

| Validation Metric | Result |
|---|---:|
| IoU at threshold 0.50 | 0.5494 |
| Best validation IoU in the tested threshold range | 0.5719 |
| Selected threshold | 0.55 |
| Dice at selected threshold | 0.7276 |

The threshold was fixed at `0.55` for the reported final test evaluation.

**Metric note:** Accuracy can be misleading in flood segmentation because non-flood pixels may substantially outnumber flood pixels. IoU, Dice, precision, and recall provide additional information about segmentation quality.

## How to Run

Run these commands from the project root directory in PowerShell.

### 1. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Train the model

```powershell
python -m src.training.train
```

The training script saves the best and latest checkpoints under `checkpoints/`.

### 3. Tune the prediction threshold

```powershell
python -m src.evaluation.tune_threshold
```

This evaluates candidate thresholds on the validation split. Do not use the test split to select a threshold.

### 4. Generate a prediction visualization

```powershell
python -m src.visualization.predict
```

The visualization is saved to:

```text
outputs/prediction_visualization.png
```

### 5. Evaluate the model

```powershell
python -m src.training.evaluate
```

The evaluation script reports segmentation metrics on the test split. The reported results use threshold `0.55`.

## Checkpoints and Outputs

- `checkpoints/flood_unet_full_best.pt` - best checkpoint selected using validation IoU.
- `checkpoints/flood_unet_full_latest.pt` — latest saved training checkpoint.
- `outputs/prediction_visualization.png` — sample prediction visualization.

Generated checkpoints and output images may not be included in version control.

## Limitations and Future Work

- Performance is measured on the Sen1Floods11 benchmark and does not guarantee accuracy during real-world flood events.
- Results depend on dataset quality, geographic coverage, preprocessing, and flood-label quality.
- The current prototype does not independently provide a live satellite feed or operational emergency-response recommendations.
- Further work can integrate geospatial flood maps, GIS layers, flood extent estimation, exposed-population analysis, and an interactive disaster-management dashboard.
- Additional geographic and real-event validation is needed before operational use.

## Disclaimer

FloodVision-AI is an academic research prototype for flood segmentation. Its predictions should not be treated as verified emergency information or used as the sole basis for disaster-response decisions.

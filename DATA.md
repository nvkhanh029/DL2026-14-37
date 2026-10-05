# Dataset

## Source

This project uses the UCI Human Activity Recognition Using Smartphones (UCI HAR)
dataset:

- Dataset page: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Download: https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
- UCI dataset ID: 240
- License: CC BY 4.0

Thirty volunteers performed six activities while wearing a waist-mounted
smartphone. The accelerometer and gyroscope signals were sampled at 50 Hz.
This project reads the supplied segmented inertial signals, not the separate
561-feature engineered dataset. The UCI signals have already been filtered and
divided into fixed windows.

## Input versions

The preparation script writes both model input variants under
`data/processed/`:

| Version | File | Channels | Window shape |
| --- | --- | --- | --- |
| Acc | `acc.npz` | `total_acc_x`, `total_acc_y`, `total_acc_z` | `(128, 3)` |
| Acc+Gyro | `acc_gyro.npz` | `total_acc_x`, `total_acc_y`, `total_acc_z`, `body_gyro_x`, `body_gyro_y`, `body_gyro_z` | `(128, 6)` |

`total_acc` includes gravity, which carries useful orientation information
for distinguishing static postures such as sitting, standing, and laying.
`body_gyro` is the three-axis gyroscope signal. The UCI-provided windows contain
128 samples (2.56 seconds) with 50% overlap. We retain these windows as supplied;
there is no additional windowing, filtering, or augmentation.

## Splits and preprocessing

The official training set contains 7,352 windows from 21 subjects, and the
official test set contains 2,947 windows from nine different subjects. The
official training subjects are split into 17 training and four validation
subjects. Validation subjects are chosen deterministically using seed 42 by
default; the selection requires all six activities to occur in both resulting
splits. Set a different seed with `--val-seed` to make another reproducible
subject-level split. The selected subject IDs and split counts are recorded
in `data/processed/summary.json` and each NPZ file.

Labels are converted from UCI's 1-6 encoding to zero-based classes:

| Label | Activity |
| --- | --- |
| 0 | WALKING |
| 1 | WALKING_UPSTAIRS |
| 2 | WALKING_DOWNSTAIRS |
| 3 | SITTING |
| 4 | STANDING |
| 5 | LAYING |

Each channel is normalized independently using its mean and population
standard deviation over all samples and windows in the training split only.
Those training statistics are then applied unchanged to validation and test.
Preparation verifies window shapes, finite values, all six classes in every
split, no subject overlap, and approximately zero mean/unit standard deviation
on normalized training channels.

## Download and prepare

Download the ZIP from the UCI link above and place it at
`data/uci_har.zip`. This repository does not download the dataset at runtime.
The downloaded outer ZIP contains the original `UCI HAR Dataset.zip`; the
preparation script safely extracts both archive layers. From the repository
root, install the runtime dependencies and run:

```powershell
pip install -r requirements.txt
python src/prepare_data.py
```

The raw archive is already present at `data/uci_har.zip` in this workspace.
If the dataset has already been extracted, preparation reuses it. To choose a
different validation subject draw:

```powershell
python src/prepare_data.py --val-seed 7
```

Raw and generated data are kept under `data/` and excluded from version
control. The processed output files are:

- `data/processed/acc.npz`
- `data/processed/acc_gyro.npz`
- `data/processed/summary.json`

Each NPZ contains `X_train`, `X_val`, and `X_test` arrays shaped
`(windows, 128, channels)` with normalized `float32` values; `y_train`,
`y_val`, and `y_test` zero-based `int16` labels; matching `subj_train`,
`subj_val`, and `subj_test` subject IDs; the training `mean` and `std`;
`channels`; `val_seed`; and `val_subjects`.

## Loading batches

The project loader returns PyTorch `DataLoader` objects in train, validation,
test order, followed by the channel count. Batches are CPU tensors so the
training loop can move them to its selected device:

```python
import torch
from src.data import load_processed

train_loader, val_loader, test_loader, channels = load_processed(
    "acc_gyro", batch_size=64
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    # x: (batch, 128, 6), y: (batch,)
```

Use `"acc"` instead of `"acc_gyro"` for three-channel input. Access the
normalized arrays directly with NumPy if a DataLoader is not needed:

```python
import numpy as np

with np.load("data/processed/acc_gyro.npz") as dataset:
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
```

## References

Anguita, D. et al. "A Public Domain Dataset for Human Activity Recognition
Using Smartphones." ESANN 2013. Dataset and documentation are provided by the
UCI Machine Learning Repository at the links above.

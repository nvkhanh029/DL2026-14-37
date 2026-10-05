# Dataset

## Source

This project uses the UCI Human Activity Recognition Using Smartphones (UCI HAR) dataset:

- Dataset page: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Download: https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
- UCI dataset ID: 240
- Dataset version: 1.0
- Download ZIP SHA256: `C00B803081A5C797CD5E4B83700A9810B38D53D9D84E01917E090E1FDBC81031`
- License: CC BY 4.0

Thirty volunteers performed six activities while wearing a waist-mounted smartphone. The accelerometer and gyroscope signals were sampled at 50 Hz. This project reads the supplied segmented inertial signals, not the separate 561-feature engineered dataset. The UCI signals have already been filtered and divided into fixed windows.

## Input versions

The preparation script writes both model input variants under `data/processed/`:

| Version  | File           | Channels                                                                             | Window shape |
| -------- | -------------- | ---------------------------------------------------------------------------------------------------- | ------------ |
| Acc      | `acc.npz`      | `total_acc_x`, `total_acc_y`, `total_acc_z`                                                    | `(128, 3)` |
| Acc+Gyro | `acc_gyro.npz` | `total_acc_x`, `total_acc_y`, `total_acc_z`, `body_gyro_x`, `body_gyro_y`, `body_gyro_z` | `(128, 6)` |

`total_acc` includes gravity, which carries useful orientation information for distinguishing static postures such as sitting, standing, and laying. `body_gyro` is the three-axis gyroscope signal. The UCI-provided windows contain 128 samples (2.56 seconds) with 50% overlap. We retain these windows as supplied; there is no additional windowing, filtering, or augmentation.

## Splits and preprocessing

The official training set contains 7,352 windows from 21 subjects, and the official test set contains 2,947 windows from nine different subjects. The official training subjects are split into 17 training and four validation subjects. Validation subjects are chosen deterministically using seed 42 by default; the selection requires all six activities to occur in both resulting splits. Set a different seed with `--val-seed` to make another reproducible subject-level split. The selected subject IDs and split counts are recorded in `data/processed/summary.json` and each NPZ file.

Labels are converted from UCI's 1-6 encoding to zero-based classes:

| Label | Activity           |
| ----- | ------------------ |
| 0     | WALKING            |
| 1     | WALKING_UPSTAIRS   |
| 2     | WALKING_DOWNSTAIRS |
| 3     | SITTING            |
| 4     | STANDING           |
| 5     | LAYING             |

Each channel is normalized independently using its mean and population standard deviation over all samples and windows in the training split only. Those training statistics are then applied unchanged to validation and test. Preparation verifies window shapes, finite values, all six classes in every split, no subject overlap, and approximately zero mean/unit standard deviation on normalized training channels.

## Download and prepare

From the repository root, install the runtime dependencies and run:

```powershell
pip install -r requirements.txt
python src/prepare_data.py
```

If the dataset is not already extracted, preparation downloads the ZIP from UCI. The outer ZIP contains the original `UCI HAR Dataset.zip`; the preparation script safely extracts both archive layers. To choose a different validation subject draw:

```powershell
python src/prepare_data.py --val-seed 7
```

Raw and generated data are kept under `data/` and excluded from version
control. The processed output files are:

- `data/processed/acc.npz`
- `data/processed/acc_gyro.npz`
- `data/processed/summary.json`

Each NPZ contains `X_train`, `X_val`, and `X_test` arrays shaped `(windows, 128, channels)` with normalized `float32` values; `y_train`, `y_val`, and `y_test` zero-based `int16` labels; matching `subj_train`, `subj_val`, and `subj_test` subject IDs; the training `mean` and `std`; `channels`; `val_seed`; and `val_subjects`.

## Loading the data

Run `python src/prepare_data.py` once before loading.

`load_data(sensors="acc"|"acc_gyro")` returns six NumPy arrays in the order `X_train, y_train, X_val, y_val, X_test, y_test`.

```python
from src.data import load_data

# sensors = "acc" (3 channels) or "acc_gyro" (6 channels)
X_train, y_train, X_val, y_val, X_test, y_test = load_data("acc_gyro")

print(X_train.shape)  # (5800, 128, 6)
print(X_val.shape)    # (1552, 128, 6)
print(X_test.shape)   # (2947, 128, 6)
print(y_train.shape)  # (5800,)
```

| Array | `acc` shape | `acc_gyro` shape |
|---|---|---|
| `X_train` | (5800, 128, 3) | (5800, 128, 6) |
| `X_val` | (1552, 128, 3) | (1552, 128, 6) |
| `X_test` | (2947, 128, 3) | (2947, 128, 6) |

- `X` is float32 and already normalized with the training statistics.
- `y` is an integer array with labels 0-5 (walking, upstairs, downstairs, sitting, standing, laying).
- Any other value for `sensors` raises a `ValueError`.

### PyTorch DataLoaders

`load_processed` returns `DataLoader` objects in train, validation, test order, followed by the channel count. Batches are CPU tensors, so the training loop moves them to its selected device:

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

Use `"acc"` instead of `"acc_gyro"` for three-channel input.

### Reading the processed file directly

If you don't need a DataLoader, read the normalized arrays with NumPy:

```python
import numpy as np

with np.load("data/processed/acc_gyro.npz") as dataset:
    X_train = dataset["X_train"]
    y_train = dataset["y_train"].astype(np.int64)  # stored as int16
```

## References

Anguita, D. et al. "A Public Domain Dataset for Human Activity Recognition
Using Smartphones." ESANN 2013. Dataset and documentation are provided by the
UCI Machine Learning Repository at the links above.

# Human Activity Recognition with Multi-Sensor Deep Learning (DL2026, Group 14, Project 37)

We compare sensor combinations (Acc vs Acc+Gyro) and sequence models (Random Forest, 1D-CNN, LSTM, CNN-LSTM) on the UCI HAR dataset to see how multiple sensors affect activity recognition.

- Dataset: UCI HAR (smartphone IMU, 6 activities, 50 Hz, windows of 128 steps, official train/test split). Source URL, version, split and preprocessing: see [DATA.md](DATA.md).
- Fixed rules: seed 42, metrics = accuracy and macro F1, same split and windowing for all models.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS:        source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare the data

Downloads the UCI HAR zip if it is not already in `data/` and writes `data/processed/` (see DATA.md):

```bash
python src/prepare_data.py
```

## Reproduce the experiments

Run all commands from the repository root. Six runs:

| Run | Model | Sensors |
| --- | --- | --- |
| rf_acc | Random Forest | Acc |
| rf_acc_gyro | Random Forest | Acc+Gyro |
| cnn_acc_gyro | 1D-CNN | Acc+Gyro |
| lstm_acc_gyro | LSTM | Acc+Gyro |
| cnn_lstm_acc | CNN-LSTM | Acc |
| cnn_lstm_acc_gyro | CNN-LSTM | Acc+Gyro |

```bash
python src/train.py --prepare          # prepare data, then run all 6 runs
python src/train.py --only rf_acc      # run selected runs only
python src/train.py --dry-run          # print the commands without running them
```

`src/train.py` only calls the scripts below, so each run can also be started directly:

```bash
python -m src.models.rf --sensors acc            # or acc_gyro
python scripts/train_cnn_lstm.py --models cnn lstm --sensors acc_gyro
python scripts/train_main_model.py --sensors acc acc_gyro
```

Each run writes its metrics to `results/<model>_<sensors>.json`. Predictions for the deep models are in `results/preds/`.

**Note:** a real run overwrites the committed `results/*.json`. Results can differ by up to about 1.5 points across machines and software versions even with seed 42, so differences below 1 point between models are not interpreted as wins.

## Figures

```bash
python src/plot_results.py     # figures/sensor_effect.png and figures/model_effect.png (reads results/*.json)
python src/evaluate.py         # confusion matrices and per-subject plots in results/analysis/ (no training)
```

## Additional analysis (ablation studies)

```bash
python scripts/analysis_window.py     # window size 16/32/64/128, seeds 42 0 1 2
python scripts/analysis_bilstm.py     # LSTM vs bidirectional LSTM
python scripts/analysis_loso.py       # leave-one-subject-out, 30 folds, resumable, about 1 hour on CPU
```

**Warning:** `analysis_window.py` and `analysis_bilstm.py` overwrite their JSON files in `results/analysis/`, even with fewer seeds or epochs.

## Repository layout

- `src/`: data pipeline, models (`src/models/`), entry point `train.py`, `evaluate.py`, `plot_results.py`
- `scripts/`: training and analysis scripts
- `configs/shared.yaml`: shared settings (seed 42)
- `results/`: metrics (JSON), predictions and analysis outputs
- `figures/`: result figures
- `report/`: member sections of the report
- `DATA.md`: dataset source, version, split, preprocessing
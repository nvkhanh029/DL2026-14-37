# AGENTS.md: rules for AI tools in this repo

Project: Human Activity Recognition with Multi-Sensor Deep Learning (DL2026 group project).
Goal: compare sensor combinations (Acc vs Acc+Gyro) and models (RF, CNN, LSTM, CNN-LSTM) on UCI-HAR.
This file is deleted before submission.

## Golden rules
1. Read `configs/shared.yaml` first. Never hard-code the seed, metrics, window size, input shapes or split.
2. Work only on your own branch. Never push to `main`. Open a PR instead.
3. Edit only the files you own (see "File ownership"). Do not touch others' files or the data pipeline.
4. Never use the test set for tuning, early stopping or model selection. Use the validation set.
5. Do not change the split, normalization or seed. If you think a fixed rule is wrong, tell the user, do not fix it yourself.
6. The user must be able to explain every line you write (Q&A can go to any member). Keep code short, simple and commented.

## Fixed rules (from configs/shared.yaml)
- Dataset: UCI-HAR, official train/test split, 6 classes (labels 0-5), 128-step windows at 50 Hz.
- Val split: 4 whole subjects (17, 25, 26, 30) carved from the official train set, seed 42.
- Sizes: train 5800, val 1552, test 2947.
- Normalization: per-channel z-score, statistics from train only (already applied by the data pipeline).
- Inputs: `acc` = (N,128,3) from total_acc; `acc_gyro` = (N,128,6) from total_acc + body_gyro.
- Seed 42: the data code uses it only for the val split. Your training script must seed `random`, `numpy` and `torch` with 42.
- Metrics: accuracy and macro F1, on the test set, for every run.

## Data access
- Preprocessing: `python src/prepare_data.py` creates the processed data. Do not edit it.
- Loader: `src/data.py`. Currently `load_processed(version, batch_size=64, *, data_dir, num_workers)` returns `(train_loader, val_loader, test_loader, n_channels)`. `version` is `"acc"` or `"acc_gyro"`.
- The data person is adding `load_data(sensors="acc" | "acc_gyro")` returning `X_train, y_train, X_val, y_val, X_test, y_test` as arrays (needed by Random Forest). Use it once it exists.
- Do not commit `data/` files.

## Allowed runs (6 total)
- Random Forest: acc, acc_gyro (hand-crafted features per axis: mean, std, min, max, energy)
- CNN (1D): acc_gyro
- LSTM: acc_gyro
- CNN-LSTM (main model): acc, acc_gyro
- Analysis (CNN-LSTM only): window size, bidirectional LSTM, leave-one-subject-out. Outputs go in `results/analysis/`.
- Do not add other models (GRU was dropped) unless the user asks.

## Result format
Each run writes `results/<model>_<sensors>.json`, for example `results/cnn_lstm_acc_gyro.json`:
```json
{"model": "cnn_lstm", "sensors": "acc_gyro", "seed": 42,
 "accuracy": 0.0, "macro_f1": 0.0, "epochs": 0, "train_time_sec": 0.0}
```
- `model` is one of `rf | cnn | lstm | cnn_lstm`. `epochs` is 0 for rf.
- Never write fake or placeholder results. Only write a file after a real run.
- Figures go in `figures/`, named like the run: `figures/cnn_lstm_acc_gyro_confusion.png`.

## File ownership
| Person | Files |
|---|---|
| Data person | `src/data.py`, `src/prepare_data.py`, `DATA.md`, `report/dataset.md` |
| A (Random Forest) | `src/models/rf.py`, `report/baseline_method.md` |
| B (CNN, LSTM) | `src/models/cnn.py`, `src/models/lstm.py`, `report/cnn_lstm_methods.md` |
| C (CNN-LSTM) | `src/models/cnn_lstm.py`, `report/main_method.md` |
| D (Analysis) | `src/evaluate.py`, `scripts/analysis_*.py`, `results/analysis/`, `report/results_discussion.md`, `report/error_analysis.md` |
| Viz/slides | `src/plots.py`, `figures/`, `report/related_work.md`, slides |
| Leader | `README.md`, `src/train.py`, `src/demo.py`, `configs/`, `report/abstract.md`, `report/introduction.md`, `report/conclusion.md`, `report/experimental_setup.md`, `report/references.md`, `report/appendix.md` |

## Branches
One branch per person:
`docs/report-core` (leader), `data/loader`, `viz/figures-slides`, `model/random-forest`, `model/cnn-and-lstm`, `model/cnn-lstm-hybrid`, `analysis/ablation-errors`.
New branches follow `<area>/<topic>`.

## Commits
Format: `<prefix>: <lowercase imperative summary, under 72 characters>`
Prefixes: `feat`, `fix`, `docs`, `exp`, `refactor`, `chore`.
Examples: `feat: add 1d-cnn model`, `docs: write dataset section`, `exp: add cnn-lstm acc_gyro result`.
One purpose per commit. Do not commit data, `.pt`/`.pth` files or notebooks' checkpoints.

## Report sections
- Each person writes their own section as `report/<topic>.md`, using `##` for the section heading and `figures/<name>.png` for figures.
- Write plain, specific English. Report results with interpretation, not just numbers.
- Cite the review paper "Deep Learning in Human Activity Recognition with Wearable Sensors: A Review on Advances" where relevant. Do not invent references or numbers.

## Environment
- The user works on Windows with PowerShell. Give PowerShell commands, or note Git Bash alternatives.
- Dependencies go in `requirements.txt`. Ask before adding a new one.
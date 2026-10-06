"""Single entry point to reproduce all 6 experiments.

Thin wrapper: it calls the existing scripts and does not train anything itself.
Run from the repo root:  python src/train.py --dry-run
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# run name -> command (executed from the repo root)
RUNS = {
    "rf_acc": [PY, "-m", "src.models.rf", "--sensors", "acc"],
    "rf_acc_gyro": [PY, "-m", "src.models.rf", "--sensors", "acc_gyro"],
    "cnn_acc_gyro": [PY, "scripts/train_cnn_lstm.py", "--models", "cnn", "--sensors", "acc_gyro"],
    "lstm_acc_gyro": [PY, "scripts/train_cnn_lstm.py", "--models", "lstm", "--sensors", "acc_gyro"],
    "cnn_lstm_acc": [PY, "scripts/train_main_model.py", "--sensors", "acc"],
    "cnn_lstm_acc_gyro": [PY, "scripts/train_main_model.py", "--sensors", "acc_gyro"],
}


def parse_args():
    p = argparse.ArgumentParser(description="Run the 6 HAR experiments.")
    p.add_argument("--only", nargs="+", choices=list(RUNS), help="run only these runs")
    p.add_argument("--prepare", action="store_true", help="run src/prepare_data.py first")
    p.add_argument("--dry-run", action="store_true", help="print commands without running them")
    return p.parse_args()


def run(cmd, dry_run):
    print(f"$ {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    args = parse_args()
    selected = args.only or list(RUNS)
    if args.prepare:
        run([PY, "src/prepare_data.py"], args.dry_run)
    for name in selected:
        print(f"\n=== {name} ===", flush=True)
        run(RUNS[name], args.dry_run)


if __name__ == "__main__":
    main()
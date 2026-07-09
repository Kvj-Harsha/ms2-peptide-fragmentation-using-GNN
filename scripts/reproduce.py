"""One-command reproduction: baselines -> edge model -> pooled ablation -> eval.

    python scripts/reproduce.py --quick     # tiny subset + few epochs
    python scripts/reproduce.py             # full configured run

Both modes download the Pep2Prob dataset on first run (cached by the `datasets`
library afterwards); --quick then subsamples to a few hundred rows so the whole
pipeline finishes in seconds. For a fully offline smoke test that needs no
download, run `pytest` instead.

Runs everything through the same Config so numbers are consistent and the
edge-vs-pool headline ablation is produced in a single pass.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Tiny subset + few epochs to smoke-test the pipeline.")
    args = ap.parse_args()

    common = []
    if args.quick:
        common = ["--data.train_rows", "500", "--data.val_rows", "100",
                  "--data.test_rows", "100", "--train.epochs", "2"]

    run([PY, "scripts/run_baselines.py", *common])

    run([PY, "scripts/train.py", "--config", "configs/cleavage_gcn.yaml",
         "--model.readout", "edge", "--out_dir", "runs/edge_gcn", *common])
    run([PY, "scripts/evaluate.py", "--ckpt", "runs/edge_gcn/model.pt"])

    run([PY, "scripts/train.py", "--config", "configs/cleavage_gcn.yaml",
         "--model.readout", "pool", "--out_dir", "runs/pool_gcn", *common])
    run([PY, "scripts/evaluate.py", "--ckpt", "runs/pool_gcn/model.pt"])

    print("\nDone. See results/ for the metrics CSVs (edge vs pool vs baselines).")


if __name__ == "__main__":
    main()

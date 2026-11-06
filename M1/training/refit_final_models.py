"""Refit the three selected M1 models from the stored hyperparameters and compare them with ../models.

This is a quick reproducibility check (seconds) that skips the hyperparameter search.

Usage:
    python refit_final_models.py [--out refit]
"""
import argparse
import ast
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_regression
import train_single_peak

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "models"
TASK_MAP = {"single_peak": "single_peak", "Tm": "tm", "dHm": "dh"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "refit"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(MODEL_DIR / "model_summary.csv")
    for _, row in summary.iterrows():
        task = TASK_MAP[row["task"]]
        params = ast.literal_eval(row["best_params"])
        if task == "single_peak":
            X, y = train_single_peak.load_data()
            model = train_single_peak.build_model(row["algorithm"], params).fit(X, y)
            a = model.predict_proba(X)[:, 1]
        else:
            X, y = train_regression.load_data(task)
            model = train_regression.build_model(row["algorithm"], params).fit(X, y)
            a = model.predict(X)
        frozen = joblib.load(MODEL_DIR / row["file"])
        b = frozen.predict_proba(X)[:, 1] if task == "single_peak" else frozen.predict(X)
        with open(out / row["file"], "wb") as f:
            pickle.dump(model, f)
        print(f"{row['task']:12s} {row['algorithm']:8s} n={len(y):3d}  "
              f"max |refit - frozen| on training rows = {np.max(np.abs(a - b)):.3g}")
    print(f"refitted models saved to: {out}")


if __name__ == "__main__":
    main()

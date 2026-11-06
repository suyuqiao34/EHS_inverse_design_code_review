"""Refit the three M0 models on all Round-0 records with the selected hyperparameters.

Selected models (see results/*_model_comparison.csv):
    single peak : XGBoost classifier
    Tm          : GBDT regressor
    dHm         : SVR (RBF kernel)

The refitted models are written to --out and compared with the frozen models in
../models. The SVR refit is deterministic. The XGBoost and GBDT models were
originally fitted without a fixed random seed, so a refit reproduces them only
approximately; the frozen files in ../models are the reference models used in
the paper.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[1] / "data" / "round0"
MODEL_DIR = HERE.parent / "models"
SALT_COLS = ["ID1-salt", "ID2-salt", "ID3-salt"]

FINAL = {
    "single_peak": {
        "file": "round0_single_peak.csv", "target": "single peak",
        "model_file": "single_peak_XGBoost_best_model_from_roc_redraw.pkl",
        "build": lambda: XGBClassifier(
            n_estimators=643, learning_rate=0.07102731973724802, max_depth=7, min_child_weight=8,
            subsample=0.8018497397679697, colsample_bytree=0.8986701562394934, gamma=2.329865598404251,
            reg_lambda=0.11598360514875435, reg_alpha=0.030135320760888637),
    },
    "tm": {
        "file": "round0_tm.csv", "target": "Tm (°C)",
        "model_file": "tm_gbdt_best_model.pkl",
        "build": lambda: GradientBoostingRegressor(
            n_estimators=225, learning_rate=0.1314931673659004, max_depth=6, subsample=0.9803786883261242,
            min_samples_split=6, min_samples_leaf=5),
    },
    "dh": {
        "file": "round0_dh.csv", "target": "ΔHm (J/g)",
        "model_file": "dh_svm_best_model.pkl",
        "build": lambda: SVR(C=64.54842450644585, gamma=4.0235834170741525, epsilon=0.2458273817645959),
    },
}


def load(cfg):
    df = pd.read_csv(DATA_DIR / cfg["file"])
    X = df.drop(columns=SALT_COLS + [cfg["target"]])
    y = pd.to_numeric(df[cfg["target"]], errors="coerce")
    valid = y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "refit"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for task, cfg in FINAL.items():
        X, y = load(cfg)
        model = cfg["build"]()
        if task == "single_peak":
            y = y.astype(int)
        model.fit(X, y)
        joblib.dump(model, out / cfg["model_file"])

        frozen = joblib.load(MODEL_DIR / cfg["model_file"])
        if task == "single_peak":
            a, b = model.predict_proba(X)[:, 1], frozen.predict_proba(X[frozen.feature_names_in_])[:, 1]
        else:
            a, b = model.predict(X), frozen.predict(X[frozen.feature_names_in_])
        print(f"{task:12s} n={len(y):3d}  max |refit - frozen| on training rows = {np.max(np.abs(a - b)):.3g}")
    print(f"refitted models saved to: {out}")


if __name__ == "__main__":
    main()

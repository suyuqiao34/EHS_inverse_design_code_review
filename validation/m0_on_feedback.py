"""Frozen M0 models applied to all 171 feedback formulations (Supplementary Section S9.2).

M0 (MD5-checked Round-0 models, trained before any feedback record existed) and two
mixing-rule baselines are evaluated on the feedback records, overall and by subset
(salt combination seen/unseen in the literature data, binary/ternary, Tm < 0 / >= 0).
The literature records are reported as an in-sample reference.

Usage:
    python m0_on_feedback.py [--out results_rerun]
"""
import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MODEL_DIR = REPO / "M0" / "models"
DATA = REPO / "data" / "round1" / "round1_all_records.csv"

MODELS = {
    "single_peak": ("single_peak_XGBoost_best_model_from_roc_redraw.pkl", "D353C97D2863A72E5F036D9E0A8E8DD3"),
    "Tm": ("tm_gbdt_best_model.pkl", "9D93DDCC6F455D1DFE464DDD68DDD3A9"),
    "dH": ("dh_svm_best_model.pkl", "5BC8D7CE665669D955C028CBE9F5B9B6"),
}
TM_COL, DH_COL, SP_COL, SOURCE_COL = "Tm (°C)", "ΔHm (J/g)", "single peak", "source"
SALT_COLS = ["ID1-salt", "ID2-salt", "ID3-salt"]


def load_models():
    models = {}
    for task, (fname, expected) in MODELS.items():
        path = MODEL_DIR / fname
        md5 = hashlib.md5(path.read_bytes()).hexdigest().upper()
        if md5 != expected:
            raise RuntimeError(f"{fname}: MD5 {md5} does not match expected {expected}")
        models[task] = joblib.load(path)
    return models


def salt_key(row):
    salts = [str(s).strip() for s in row[SALT_COLS] if pd.notna(s) and str(s).strip()]
    return " + ".join(sorted(salts))


def mixing_rule(df, prop):
    props = np.column_stack([df[f"{k}-{prop}"].fillna(0).values for k in (1, 2, 3)])
    return (df[["ID1", "ID2", "ID3"]].fillna(0).values * props).sum(axis=1)


def reg_metrics(y, p):
    return {"n": len(y), "MAE": mean_absolute_error(y, p),
            "RMSE": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p) if len(y) > 1 else np.nan,
            "mean_bias(pred-true)": float(np.mean(p - y))}


def clf_metrics(y, pred, proba):
    out = {"n": len(y), "n_pos": int(y.sum()), "accuracy": accuracy_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0)}
    out["roc_auc"] = roc_auc_score(y, proba) if len(np.unique(y)) == 2 else np.nan
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({"TN": tn, "FP": fp, "FN": fn, "TP": tp})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "results_rerun"))
    out_dir = Path(ap.parse_args().out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = load_models()
    feat_cols = list(models["Tm"].feature_names_in_)
    df = pd.read_csv(DATA)
    X = df.reindex(columns=feat_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    df["salt_combo"] = df.apply(salt_key, axis=1)
    df["n_components"] = df[SALT_COLS].notna().sum(axis=1)
    df["is_new"] = df[SOURCE_COL] == "feedback"
    old_combos = set(df.loc[~df["is_new"], "salt_combo"])
    df["combo_seen_in_old"] = df["salt_combo"].isin(old_combos)

    df["M0_single_peak_pred"] = models["single_peak"].predict(X)
    df["M0_single_peak_proba"] = models["single_peak"].predict_proba(X)[:, 1]
    df["M0_Tm_pred"] = models["Tm"].predict(X)
    df["M0_dH_pred"] = models["dH"].predict(X)
    df["mix_Tm"] = mixing_rule(df, "Tm (°C)")
    df["mix_dH"] = mixing_rule(df, "ΔHm (J/g)")

    # linear calibration y = a * mix + b fitted on literature records only
    for tgt, mix in [(TM_COL, "mix_Tm"), (DH_COL, "mix_dH")]:
        tr = (~df["is_new"]) & df[tgt].notna()
        lr = LinearRegression().fit(df.loc[tr, [mix]], df.loc[tr, tgt])
        df[f"{mix}_calibrated"] = lr.predict(df[[mix]])

    rows = []

    def add(scope, mask):
        sub = df[mask]
        s = sub[sub[SP_COL].notna()]
        if len(s):
            rows.append({"scope": scope, "task": "single_peak", "method": "M0 XGBoost",
                         **clf_metrics(s[SP_COL].astype(int).values,
                                       s["M0_single_peak_pred"].astype(int).values,
                                       s["M0_single_peak_proba"].values)})
        for tgt, pcol, mix, name in [(TM_COL, "M0_Tm_pred", "mix_Tm", "Tm"), (DH_COL, "M0_dH_pred", "mix_dH", "dH")]:
            r = sub[sub[tgt].notna()]
            if not len(r):
                continue
            y = r[tgt].values
            for method, col in [("M0 ML", pcol), ("Mixing rule (direct)", mix),
                                ("Mixing rule (linear-calibrated on old data)", f"{mix}_calibrated")]:
                rows.append({"scope": scope, "task": name, "method": method, **reg_metrics(y, r[col].values)})

    new = df["is_new"]
    add("literature_179_in_sample", ~new)
    add("feedback_all", new)
    add("feedback_unseen_combination", new & ~df["combo_seen_in_old"])
    add("feedback_seen_combination", new & df["combo_seen_in_old"])
    add("feedback_binary", new & (df["n_components"] == 2))
    add("feedback_ternary", new & (df["n_components"] == 3))
    add("feedback_Tm<0", new & (df[TM_COL] < 0))
    add("feedback_Tm>=0", new & (df[TM_COL] >= 0))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "m0_on_feedback_summary.csv", index=False)
    keep = SALT_COLS + ["ID1", "ID2", "ID3", "salt_combo", "n_components", "combo_seen_in_old",
                        SP_COL, "M0_single_peak_pred", "M0_single_peak_proba",
                        TM_COL, "M0_Tm_pred", "mix_Tm", "mix_Tm_calibrated",
                        DH_COL, "M0_dH_pred", "mix_dH", "mix_dH_calibrated"]
    df.loc[new, keep].to_csv(out_dir / "m0_on_feedback_predictions.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(summary.round(3).to_string(index=False))
    print(f"\nfeedback salt combinations: {df.loc[new, 'salt_combo'].nunique()}, "
          f"unseen in literature data: {df.loc[new & ~df['combo_seen_in_old'], 'salt_combo'].nunique()}")
    print(f"results saved to: {out_dir}")


if __name__ == "__main__":
    main()

"""Grouped nested validation of experimental feedback (Table S7a, Fig. 7b,c, Supplementary Fig. S14).

The 171 feedback records are grouped by salt combination (38 combinations) and split
into 5 outer folds, repeated 3 times (seed 42). For every outer fold:
  * M0 (frozen Round-0 models, MD5-checked) predicts the held-out feedback records;
  * a feedback-updated model is tuned with Optuna (TPE, seed 42, 10-fold inner CV;
    single peak 120 trials, Tm 120, dHm 150) on all literature records plus the
    feedback records of the other folds, using the M1 algorithm of each task
    (GBDT classifier, RF regressor, XGBoost regressor), and predicts the same
    held-out records.
No record of a held-out salt combination is used for training or tuning.

Usage:
    python grouped_cv_nested.py [--out results_rerun]
Runtime: several hours on a desktop CPU.
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA_PATH = REPO / "data" / "round1" / "round1_all_records.csv"
M0_DIR = REPO / "M0" / "models"

SEED = 42
OUTER_FOLDS = 5
REPEATS = 3
N_TRIALS = {"single_peak": 120, "Tm": 120, "dH": 150}
TM_COL, DH_COL, SP_COL, SOURCE_COL = "Tm (°C)", "ΔHm (J/g)", "single peak", "source"
SALT_COLS = ["ID1-salt", "ID2-salt", "ID3-salt"]

M0_FILES = {
    "single_peak": ("single_peak_XGBoost_best_model_from_roc_redraw.pkl", "D353C97D2863A72E5F036D9E0A8E8DD3"),
    "Tm": ("tm_gbdt_best_model.pkl", "9D93DDCC6F455D1DFE464DDD68DDD3A9"),
    "dH": ("dh_svm_best_model.pkl", "5BC8D7CE665669D955C028CBE9F5B9B6"),
}


def combo_key(row):
    vals = [str(v).strip() for v in row if pd.notna(v) and str(v).strip()]
    return " + ".join(sorted(vals))


def load_m0_models():
    models = {}
    for task, (fname, expect_md5) in M0_FILES.items():
        p = M0_DIR / fname
        got = hashlib.md5(p.read_bytes()).hexdigest().upper()
        if got != expect_md5:
            raise RuntimeError(f"M0 file hash mismatch for {fname}: got {got}, expected {expect_md5}")
        models[task] = joblib.load(p)
    return models


def suggest_params_tm_rf(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }


def suggest_params_dh_xgb(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 100, log=True),
    }


def suggest_params_sp_gbdt(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
    }


def build_tm_rf(params, seed=SEED):
    return RandomForestRegressor(random_state=seed, n_jobs=1, **params)


def build_dh_xgb(params, seed=SEED):
    return XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=1, **params)


def build_sp_gbdt(params, seed=SEED):
    return GradientBoostingClassifier(random_state=seed, **params)


def tune_task(task, X_train, y_train):
    if task == "Tm":
        cv = KFold(n_splits=10, shuffle=True, random_state=SEED)

        def objective(trial):
            model = build_tm_rf(suggest_params_tm_rf(trial), seed=SEED)
            return float(np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)))
    elif task == "dH":
        cv = KFold(n_splits=10, shuffle=True, random_state=SEED)

        def objective(trial):
            model = build_dh_xgb(suggest_params_dh_xgb(trial), seed=SEED)
            return float(np.mean(cross_val_score(model, X_train, y_train, cv=cv,
                                                 scoring="neg_root_mean_squared_error", n_jobs=-1)))
    else:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

        def objective(trial):
            model = build_sp_gbdt(suggest_params_sp_gbdt(trial), seed=SEED)
            return float(np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)))

    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS[task])
    return study.best_params, study.best_value


def metrics_reg(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "bias": float(np.mean(y_pred - y_true)),
    }


def metrics_clf(y_true, y_pred, y_proba):
    out = {
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_nonSP": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    out["roc_auc"] = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) == 2 else np.nan
    return out


def add_regression_rows(metric_rows, rep, fold, task, model, y_true, y_pred, tm_true):
    metric_rows.append({"repeat": rep, "fold": fold, "task": task, "model": model, "scope": "all_new",
                        **metrics_reg(y_true, y_pred)})
    neg, pos = tm_true < 0, tm_true >= 0
    if neg.sum() > 1:
        metric_rows.append({"repeat": rep, "fold": fold, "task": task, "model": model, "scope": "Tm<0",
                            **metrics_reg(y_true[neg], y_pred[neg])})
    if pos.sum() > 1:
        metric_rows.append({"repeat": rep, "fold": fold, "task": task, "model": model, "scope": "Tm>=0",
                            **metrics_reg(y_true[pos], y_pred[pos])})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "results_rerun"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["combo"] = df[SALT_COLS].apply(combo_key, axis=1)
    is_new = df[SOURCE_COL].eq("feedback")
    is_old = ~is_new

    m0_models = load_m0_models()
    feat_cols = list(m0_models["Tm"].feature_names_in_)
    X = df.reindex(columns=feat_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    new_combos = np.array(sorted(df.loc[is_new, "combo"].unique()))
    rng = np.random.RandomState(SEED)
    pred_rows, metric_rows, tuning_rows = [], [], []
    builders = {"Tm": build_tm_rf, "dH": build_dh_xgb}
    targets = {"Tm": TM_COL, "dH": DH_COL}

    for rep in range(REPEATS):
        perm = rng.permutation(new_combos)
        fold_map = {c: i % OUTER_FOLDS for i, c in enumerate(perm)}
        fold_id = df["combo"].map(fold_map).where(is_new, -1).astype(int).values

        for fold in range(OUTER_FOLDS):
            test_new_mask = is_new.values & (fold_id == fold)
            print(f"[repeat {rep + 1}/{REPEATS}] fold {fold + 1}/{OUTER_FOLDS} ...", flush=True)

            # single peak
            y = pd.to_numeric(df[SP_COL], errors="coerce")
            valid = y.notna().values
            test_mask = test_new_mask & valid
            train_mask = (is_old.values | (is_new.values & (fold_id != fold))) & valid
            if test_mask.sum() > 0:
                y_true = y[test_mask].astype(int).values
                yhat_m0 = m0_models["single_peak"].predict(X.loc[test_mask]).astype(int)
                ypr_m0 = m0_models["single_peak"].predict_proba(X.loc[test_mask])[:, 1]
                metric_rows.append({"repeat": rep, "fold": fold, "task": "single_peak", "model": "M0",
                                    "scope": "all_new", **metrics_clf(y_true, yhat_m0, ypr_m0)})

                Xtr, ytr = X.loc[train_mask], y[train_mask].astype(int)
                best_params, best_cv = tune_task("single_peak", Xtr, ytr)
                tuning_rows.append({"repeat": rep, "fold": fold, "task": "single_peak", "best_cv": best_cv,
                                    "best_params": json.dumps(best_params), "n_train": int(train_mask.sum())})
                m_up = build_sp_gbdt(best_params, seed=SEED).fit(Xtr, ytr)
                yhat_up = m_up.predict(X.loc[test_mask]).astype(int)
                ypr_up = m_up.predict_proba(X.loc[test_mask])[:, 1]
                metric_rows.append({"repeat": rep, "fold": fold, "task": "single_peak", "model": "feedback_updated",
                                    "scope": "all_new", **metrics_clf(y_true, yhat_up, ypr_up)})
                for idx, yt, pm0, pup, sm0, sup in zip(df.index[test_mask], y_true, yhat_m0, yhat_up, ypr_m0, ypr_up):
                    pred_rows.append({"repeat": rep, "fold": fold, "task": "single_peak", "row_index": int(idx),
                                      "combo": df.loc[idx, "combo"], "y_true": int(yt),
                                      "y_pred_M0": int(pm0), "y_pred_updated": int(pup),
                                      "y_proba_M0": float(sm0), "y_proba_updated": float(sup)})

            # Tm and dHm
            for task in ("Tm", "dH"):
                y = pd.to_numeric(df[targets[task]], errors="coerce")
                valid = y.notna().values
                test_mask = test_new_mask & valid
                train_mask = (is_old.values | (is_new.values & (fold_id != fold))) & valid
                if test_mask.sum() <= 1:
                    continue
                y_true = y[test_mask].values
                tm_true = pd.to_numeric(df.loc[test_mask, TM_COL], errors="coerce").values
                yhat_m0 = m0_models[task].predict(X.loc[test_mask])
                add_regression_rows(metric_rows, rep, fold, task, "M0", y_true, yhat_m0, tm_true)

                Xtr, ytr = X.loc[train_mask], y[train_mask]
                best_params, best_cv = tune_task(task, Xtr, ytr)
                tuning_rows.append({"repeat": rep, "fold": fold, "task": task, "best_cv": best_cv,
                                    "best_params": json.dumps(best_params), "n_train": int(train_mask.sum())})
                yhat_up = builders[task](best_params, seed=SEED).fit(Xtr, ytr).predict(X.loc[test_mask])
                add_regression_rows(metric_rows, rep, fold, task, "feedback_updated", y_true, yhat_up, tm_true)
                for idx, yt, pm0, pup in zip(df.index[test_mask], y_true, yhat_m0, yhat_up):
                    pred_rows.append({"repeat": rep, "fold": fold, "task": task, "row_index": int(idx),
                                      "combo": df.loc[idx, "combo"], f"{task}_true": float(yt),
                                      f"{task}_pred_M0": float(pm0), f"{task}_pred_updated": float(pup)})

    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(out / "grouped_cv_per_fold_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "grouped_cv_per_sample_predictions.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(out / "grouped_cv_tuning_log.csv", index=False)

    num_cols = [c for c in ["n", "accuracy", "precision", "recall_pos", "recall_nonSP", "f1", "roc_auc",
                            "MAE", "RMSE", "R2", "bias"] if c in metric_df.columns]
    summary = metric_df.groupby(["task", "scope", "model"], as_index=False)[num_cols].agg(["mean", "std"]).round(4)
    summary.columns = ["_".join([x for x in col if x]).rstrip("_") for col in summary.columns]
    summary.to_csv(out / "grouped_cv_summary.csv", index=False)

    # delta = updated - M0 (negative is better for MAE/RMSE; positive is better for R2/AUC/accuracy)
    piv = metric_df.pivot_table(index=["repeat", "fold", "task", "scope"], columns="model",
                                values=["accuracy", "roc_auc", "recall_nonSP", "MAE", "RMSE", "R2"], aggfunc="mean")
    delta_rows = []
    for idx in piv.index:
        rec = {"repeat": idx[0], "fold": idx[1], "task": idx[2], "scope": idx[3]}
        for metric in ["MAE", "RMSE", "R2", "accuracy", "roc_auc", "recall_nonSP"]:
            key_u, key_m = (metric, "feedback_updated"), (metric, "M0")
            if key_u in piv.columns and key_m in piv.columns:
                rec[f"delta_{metric}(updated-M0)"] = float(piv.loc[idx, key_u] - piv.loc[idx, key_m])
        delta_rows.append(rec)
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out / "grouped_cv_delta_per_fold.csv", index=False)
    delta_summary = delta_df.groupby(["task", "scope"], as_index=False).agg(["mean", "std"]).round(4)
    delta_summary.columns = ["_".join([x for x in col if x]).rstrip("_") for col in delta_summary.columns]
    delta_summary.to_csv(out / "grouped_cv_delta_summary.csv", index=False)

    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))
    print("\n===== DELTA SUMMARY (updated - M0) =====")
    print(delta_summary.to_string(index=False))
    print(f"\nresults saved to: {out}")


if __name__ == "__main__":
    main()

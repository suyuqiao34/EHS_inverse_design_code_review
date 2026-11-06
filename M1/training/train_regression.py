"""Round-1 (M1) Tm and dHm regressors: algorithm comparison and final refit.

Six algorithms are tuned with Optuna (TPE sampler, seed 42, multivariate) by
10-fold CV on an 80/20 split (seed 42); the objective is CV R2 for Tm and CV RMSE
for dHm. Each tuned model is evaluated on the held-out split and on 100 repeated
80/20 splits, and is then refit on all records (sklearn.base.clone(model).fit(X, y)).
The RF (Tm) and XGBoost (dHm) models refit in this way are identical to
../models/RF_best_model.pkl and ../models/XGBoost_best_model.pkl.
A mass-fraction-weighted mixing rule is evaluated on all records as a baseline.

Usage:
    python train_regression.py --task tm [--out results_rerun]
    python train_regression.py --task dh
"""
import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[1] / "data" / "round1"
SALT_COLS = ["ID1-salt", "ID2-salt", "ID3-salt"]
TASKS = {
    # task: (data file, target column, pure-salt property used by the mixing rule, CV objective)
    "tm": ("round1_tm.csv", "Tm (°C)", "Tm (°C)", "r2"),
    "dh": ("round1_dh.csv", "ΔHm (J/g)", "Hm (J/g)", "rmse"),
}

RANDOM_STATE = 42
REPEAT_RUNS = 100
CV = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
N_TRIALS = {"LR": 60, "SVM": 150, "DT": 120, "RF": 120, "GBDT": 120, "XGBoost": 150}
MODEL_NAMES = ["LR", "SVM", "DT", "RF", "GBDT", "XGBoost"]


def load_data(task):
    file, target, _, _ = TASKS[task]
    df = pd.read_csv(DATA_DIR / file)
    X = df.drop(columns=SALT_COLS + [target])
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def mixing_rule_prediction(X, prop_key):
    props = np.column_stack([
        X[next(c for c in X.columns if c.startswith(f"{k}-") and prop_key in c)] for k in (1, 2, 3)])
    return (X[["ID1", "ID2", "ID3"]].values * props).sum(axis=1)


def suggest_params(model_name, trial):
    if model_name == "LR":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True)}
    if model_name == "SVM":
        return {"C": trial.suggest_float("C", 1e-1, 1e4, log=True),
                "gamma": trial.suggest_float("gamma", 1e-6, 10.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-2, 20.0, log=True)}
    if model_name == "DT":
        return {"criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "absolute_error"]),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
                "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-4, 50.0, log=True)}
    if model_name == "RF":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 40),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False])}
    if model_name == "GBDT":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0),
                "loss": trial.suggest_categorical("loss", ["squared_error", "huber"])}
    if model_name == "XGBoost":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 100, log=True)}
    raise ValueError(model_name)


def build_model(model_name, params, seed=RANDOM_STATE):
    if model_name == "LR":
        return Ridge(random_state=seed, **params)
    if model_name == "SVM":
        return SVR(kernel="rbf", **params)
    if model_name == "DT":
        return DecisionTreeRegressor(random_state=seed, **params)
    if model_name == "RF":
        return RandomForestRegressor(random_state=seed, n_jobs=1, **params)
    if model_name == "GBDT":
        return GradientBoostingRegressor(random_state=seed, **params)
    if model_name == "XGBoost":
        return XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=1, **params)
    raise ValueError(model_name)


def tune(task, model_name, X_train, y_train):
    objective_kind = TASKS[task][3]
    scoring = "r2" if objective_kind == "r2" else "neg_root_mean_squared_error"
    sign = 1.0 if objective_kind == "r2" else -1.0

    def objective(trial):
        model = build_model(model_name, suggest_params(model_name, trial))
        return sign * float(np.mean(cross_val_score(model, X_train, y_train, cv=CV, scoring=scoring, n_jobs=-1)))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)
    study = optuna.create_study(direction="maximize" if objective_kind == "r2" else "minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS[model_name])
    return study.best_params, study.best_value


def repeated_runs(model_name, params, X, y):
    def one_run(seed):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
        model = build_model(model_name, params, seed=seed).fit(X_tr, y_tr)
        p_tr, p_te = model.predict(X_tr), model.predict(X_te)
        return {"run": seed,
                "rmse_train": np.sqrt(mean_squared_error(y_tr, p_tr)),
                "rmse_test": np.sqrt(mean_squared_error(y_te, p_te)),
                "r2_train": r2_score(y_tr, p_tr), "r2_test": r2_score(y_te, p_te)}

    return pd.DataFrame(Parallel(n_jobs=-1)(delayed(one_run)(i) for i in range(REPEAT_RUNS)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--out", default=str(HERE / "results_rerun"))
    args = ap.parse_args()
    task, out = args.task, Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv_name = f"cv_{TASKS[task][3]}"

    X, y = load_data(task)
    print(f"{task}: samples: {len(y)} | features: {X.shape[1]}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    rows = []
    for name in MODEL_NAMES:
        print(f"\n[{name}] Optuna, {N_TRIALS[name]} trials (10-fold CV, {TASKS[task][3].upper()}) ...", flush=True)
        best_params, best_cv = tune(task, name, X_train, y_train)
        model = build_model(name, best_params).fit(X_train, y_train)
        p_te = model.predict(X_test)
        with open(out / f"{task}_{name}_best_model.pkl", "wb") as f:
            pickle.dump(clone(model).fit(X, y), f)
        runs = repeated_runs(name, best_params, X, y)
        print(f"[{name}] {cv_name} = {best_cv:.4f} | test R2 = {r2_score(y_test, p_te):.4f} | "
              f"{REPEAT_RUNS}-split test R2 = {runs['r2_test'].mean():.4f} +/- {runs['r2_test'].std(ddof=1):.4f}",
              flush=True)
        rows.append({"model": name, cv_name: best_cv,
                     "rmse": np.sqrt(mean_squared_error(y_test, p_te)),
                     "mae": mean_absolute_error(y_test, p_te),
                     "r2": r2_score(y_test, p_te),
                     "r2_mean": runs["r2_test"].mean(), "r2_std": runs["r2_test"].std(ddof=1),
                     "rmse_mean": runs["rmse_test"].mean(), "rmse_std": runs["rmse_test"].std(ddof=1),
                     "train_r2_mean": runs["r2_train"].mean(), "train_r2_std": runs["r2_train"].std(ddof=1),
                     "train_rmse_mean": runs["rmse_train"].mean(), "train_rmse_std": runs["rmse_train"].std(ddof=1),
                     "best_params": str(best_params)})

    res = pd.DataFrame(rows).sort_values("r2_mean", ascending=False).reset_index(drop=True)
    res.to_csv(out / f"{task}_model_comparison.csv", index=False)

    pred = mixing_rule_prediction(X, TASKS[task][2])
    base = pd.DataFrame([{"model": "Mixing Rule", "n_samples": len(y), "r2": r2_score(y, pred),
                          "rmse": np.sqrt(mean_squared_error(y, pred)), "mae": mean_absolute_error(y, pred),
                          "pred_mean": pred.mean(), "true_mean": y.mean()}])
    base.to_csv(out / f"{task}_mixing_rule_baseline.csv", index=False)

    print("\n" + res.drop(columns="best_params").round(4).to_string())
    print("\nmixing-rule baseline (all records):")
    print(base.round(4).to_string(index=False))
    print(f"\nresults saved to: {out}")


if __name__ == "__main__":
    main()

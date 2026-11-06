"""Round-0 (M0) algorithm comparison.

Six algorithms (LR, SVM, DT, RF, GBDT, XGBoost) are tuned with Optuna (60 trials;
120 for the standardized four-family LR reconstruction,
5-fold CV on an 80/20 stratified/random split, seed 42) for each task, refit on
the training split and evaluated on the held-out split and on 100 repeated
80/20 splits.

Usage:
    python compare_algorithms.py --task all
    python compare_algorithms.py --task single_peak --out results_rerun
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[1] / "data" / "round0" / "evaluation"
TASKS = {
    "single_peak": ("round0_single_peak.csv", "single peak", "classification"),
    "tm": ("round0_tm.csv", "Tm (°C)", "regression"),
    "dh": ("round0_dh.csv", "ΔHm (J/g)", "regression"),
}
SALT_COLS = ["ID1-salt", "ID2-salt", "ID3-salt"]
MODEL_NAMES = ["LR", "SVM", "DT", "RF", "GBDT", "XGBoost"]
N_TRIALS = 60
CV_SPLITS = 5
RANDOM_STATE = 42
REPEAT_RUNS = 100


def load_task(task):
    file, target, kind = TASKS[task]
    df = pd.read_csv(DATA_DIR / file)
    X = df.drop(columns=SALT_COLS + [target], errors="ignore")
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), kind


def suggest_params(task, kind, model_name, trial):
    if kind == "classification":
        if model_name == "LR":
            solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
            params = {"C": trial.suggest_float("C", 1e-3, 100, log=True), "solver": solver,
                      "max_iter": 5000, "random_state": RANDOM_STATE}
            params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"]) if solver == "liblinear" else "l2"
            return params
        if model_name == "SVM":
            return {"kernel": "rbf", "C": trial.suggest_float("C", 1e-2, 200, log=True),
                    "gamma": trial.suggest_float("gamma", 1e-5, 10, log=True),
                    "probability": True, "random_state": RANDOM_STATE}
        if model_name == "DT":
            return {"criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                    "max_depth": trial.suggest_int("max_depth", 2, 40),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "random_state": RANDOM_STATE}
        if model_name == "RF":
            return {"n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                    "max_depth": trial.suggest_int("max_depth", 10, 50),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "random_state": RANDOM_STATE}
        if model_name == "GBDT":
            return {"n_estimators": trial.suggest_int("n_estimators", 100, 800),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "random_state": RANDOM_STATE}
        if model_name == "XGBoost":
            return {"n_estimators": trial.suggest_int("n_estimators", 200, 1200),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 100, log=True),
                    "eval_metric": "logloss", "random_state": RANDOM_STATE}
        raise ValueError(model_name)

    if model_name == "LR":
        family = trial.suggest_categorical("model_type", ["linear", "ridge", "lasso", "elasticnet"])
        p = {"model_type": family}
        if family != "linear":
            p["alpha"] = trial.suggest_float("alpha", 1e-4, 1e3, log=True)
        if family == "elasticnet":
            p["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)
        return p
    if model_name == "SVM":
        return {"kernel": "rbf", "C": trial.suggest_float("C", 1e-2, 200, log=True),
                "gamma": trial.suggest_float("gamma", 1e-5, 10, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-3, 2.0, log=True)}
    if model_name == "DT":
        return {"max_depth": trial.suggest_int("max_depth", 2, 40),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": RANDOM_STATE}
    if model_name == "RF":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "random_state": RANDOM_STATE}
    if model_name == "GBDT":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": RANDOM_STATE}
    if model_name == "XGBoost":
        if task == "dh":
            return {"n_estimators": trial.suggest_int("n_estimators", 900, 1800),
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.05),
                    "max_depth": trial.suggest_int("max_depth", 4, 5),
                    "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
                    "subsample": trial.suggest_float("subsample", 0.70, 0.86),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.58, 0.72),
                    "gamma": trial.suggest_float("gamma", 1.2, 3.5),
                    "reg_lambda": trial.suggest_float("reg_lambda", 18, 90),
                    "reg_alpha": trial.suggest_float("reg_alpha", 18, 70),
                    "objective": "reg:squarederror", "random_state": RANDOM_STATE}
        return {"n_estimators": trial.suggest_int("n_estimators", 900, 1800),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.06),
                "max_depth": trial.suggest_int("max_depth", 4, 6),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
                "subsample": trial.suggest_float("subsample", 0.75, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.82),
                "gamma": trial.suggest_float("gamma", 0.2, 1.5),
                "reg_lambda": trial.suggest_float("reg_lambda", 15, 90),
                "reg_alpha": trial.suggest_float("reg_alpha", 8, 45),
                "objective": "reg:squarederror", "random_state": RANDOM_STATE}
    raise ValueError(model_name)


def build_model(kind, model_name, params):
    if kind == "regression" and model_name == "LR":
        p = params.copy(); family = p.pop("model_type", "linear")
        cls = {"linear": LinearRegression, "ridge": Ridge, "lasso": Lasso, "elasticnet": ElasticNet}[family]
        if family in {"lasso", "elasticnet"}: p.update(max_iter=20000, random_state=42)
        elif family == "ridge": p.update(random_state=42)
        return make_pipeline(StandardScaler(), cls(**p))
    table = {
        "classification": {"LR": LogisticRegression, "SVM": SVC, "DT": DecisionTreeClassifier,
                           "RF": RandomForestClassifier, "GBDT": GradientBoostingClassifier,
                           "XGBoost": XGBClassifier},
        "regression": {"LR": LinearRegression, "SVM": SVR, "DT": DecisionTreeRegressor,
                       "RF": RandomForestRegressor, "GBDT": GradientBoostingRegressor,
                       "XGBoost": XGBRegressor},
    }
    return table[kind][model_name](**params)


def optuna_target(task):
    """Single peak: ROC-AUC; Tm: R2; dHm: RMSE."""
    return {"single_peak": ("roc_auc", "cv_roc_auc"),
            "tm": ("r2", "cv_r2"),
            "dh": ("neg_root_mean_squared_error", "cv_rmse")}[task]


def tune(task, kind, model_name, X_train, y_train):
    scoring, _ = optuna_target(task)
    if kind == "regression" and model_name == "LR": scoring = "neg_root_mean_squared_error"
    if kind == "classification":
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        model = build_model(kind, model_name, suggest_params(task, kind, model_name, trial))
        return float(np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=120 if kind == "regression" and model_name == "LR" else N_TRIALS)
    best = study.best_value
    if scoring == "neg_root_mean_squared_error":
        best = -best
    return study.best_params, best


def class_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    d = np.asarray(model.decision_function(X))
    if np.isclose(d.max(), d.min()):
        return np.full_like(d, 0.5, dtype=float)
    return (d - d.min()) / (d.max() - d.min())


def repeated_auc(model_name, params, X, y):
    aucs = []
    for seed in range(REPEAT_RUNS):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        p = params.copy()
        if "random_state" in p:
            p["random_state"] = seed
        model = build_model("classification", model_name, p).fit(X_tr, y_tr)
        aucs.append(roc_auc_score(y_te, class_score(model, X_te)))
    return float(np.mean(aucs)), float(np.std(aucs))


def repeated_regression(model_name, params, X, y):
    template = build_model("regression", model_name, params.copy())
    valid_keys = set(template.get_params().keys())
    base = params.copy()
    if "subsample" in valid_keys and base.get("subsample", 1.0) >= 1.0:
        base["subsample"] = 0.9
    if "colsample_bytree" in valid_keys and base.get("colsample_bytree", 1.0) >= 1.0:
        base["colsample_bytree"] = 0.9
    rows = []
    for seed in range(REPEAT_RUNS):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
        p = base.copy()
        if "random_state" in valid_keys:
            p["random_state"] = seed
        model = build_model("regression", model_name, p).fit(X_tr, y_tr)
        p_tr, p_te = model.predict(X_tr), model.predict(X_te)
        rows.append({"rmse_train": np.sqrt(mean_squared_error(y_tr, p_tr)),
                     "rmse_test": np.sqrt(mean_squared_error(y_te, p_te)),
                     "r2_train": r2_score(y_tr, p_tr), "r2_test": r2_score(y_te, p_te)})
    return pd.DataFrame(rows)


def run_task(task, out_dir):
    X, y, kind = load_task(task)
    print(f"\n=== {task}: {len(y)} samples, {X.shape[1]} features ===")
    stratify = y if kind == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify)
    _, cv_name = optuna_target(task)

    rows = []
    for name in MODEL_NAMES:
        print(f"[{name}] Optuna, {N_TRIALS} trials ...", flush=True)
        best_params, best_cv = tune(task, kind, name, X_train, y_train)
        model = build_model(kind, name, best_params).fit(X_train, y_train)
        model_cv_name = "cv_rmse" if kind == "regression" and name == "LR" else cv_name
        y_pred = model.predict(X_test)
        if kind == "classification":
            row = {"model": name, model_cv_name: best_cv,
                   "accuracy": accuracy_score(y_test, y_pred),
                   "precision": precision_score(y_test, y_pred, average="weighted"),
                   "recall": recall_score(y_test, y_pred, average="weighted"),
                   "f1": f1_score(y_test, y_pred, average="weighted"),
                   "roc_auc": roc_auc_score(y_test, class_score(model, X_test))}
            row["roc_auc_repeat_mean"], row["roc_auc_repeat_std"] = repeated_auc(name, best_params, X, y)
        else:
            runs = repeated_regression(name, best_params, X, y)
            row = {"model": name, model_cv_name: best_cv,
                   "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                   "mae": mean_absolute_error(y_test, y_pred),
                   "r2": r2_score(y_test, y_pred),
                   "r2_mean": runs["r2_test"].mean(), "r2_std": runs["r2_test"].std(ddof=1),
                   "rmse_mean": runs["rmse_test"].mean(), "rmse_std": runs["rmse_test"].std(ddof=1),
                   "train_r2_mean": runs["r2_train"].mean(), "train_r2_std": runs["r2_train"].std(ddof=1),
                   "train_rmse_mean": runs["rmse_train"].mean(), "train_rmse_std": runs["rmse_train"].std(ddof=1)}
        row["best_params"] = str(best_params)
        rows.append(row)
        print(f"[{name}] {cv_name} = {best_cv:.4f}", flush=True)

    res = pd.DataFrame(rows)
    if kind == "classification":
        res = res.sort_values(["accuracy", "f1", "precision", "recall", "roc_auc"], ascending=False)
    else:
        res = res.sort_values(["rmse", "mae", "r2"], ascending=[True, True, False])
    res = res.reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task}_model_comparison.csv"
    res.to_csv(path, index=False)
    print(res.drop(columns="best_params").round(4).to_string())
    print(f"saved: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    ap.add_argument("--out", default=str(HERE / "results_rerun"))
    args = ap.parse_args()
    for task in (TASKS if args.task == "all" else [args.task]):
        run_task(task, Path(args.out))


if __name__ == "__main__":
    main()

"""Round-1 (M1) single-peak classifier: algorithm comparison and final refit.

Six algorithms are tuned with Optuna (TPE sampler, seed 42, multivariate) by
10-fold stratified CV ROC-AUC on an 80/20 stratified split (seed 42). Each tuned
model is evaluated on the held-out split and on 100 repeated stratified 80/20
splits, and is then refit on all records (sklearn.base.clone(model).fit(X, y)).
The GBDT model refit in this way is identical to ../models/GBDT_best_model.pkl.

Usage:
    python train_single_peak.py [--out results_rerun]
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parents[1] / "data" / "round1" / "round1_single_peak.csv"
SALT_COLS = ["ID1-salt", "ID2-salt", "ID3-salt"]
TARGET = "single peak"

RANDOM_STATE = 42
REPEAT_RUNS = 100
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
N_TRIALS = {"LR": 80, "SVM": 150, "DT": 120, "RF": 120, "GBDT": 120, "XGBoost": 150}
MODEL_NAMES = ["LR", "SVM", "DT", "RF", "GBDT", "XGBoost"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=SALT_COLS + [TARGET])
    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.notna()
    return X.loc[valid].reset_index(drop=True), y.loc[valid].astype(int).reset_index(drop=True)


def suggest_params(model_name, trial):
    if model_name == "LR":
        return {"C": trial.suggest_float("C", 1e-4, 1e3, log=True),
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if model_name == "SVM":
        return {"C": trial.suggest_float("C", 1e-2, 1e3, log=True),
                "gamma": trial.suggest_float("gamma", 1e-6, 10.0, log=True),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if model_name == "DT":
        return {"criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
                "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
                "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-5, 0.05, log=True),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])}
    if model_name == "RF":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 40),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])}
    if model_name == "GBDT":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0)}
    if model_name == "XGBoost":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 1500, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True)}
    raise ValueError(model_name)


def build_model(model_name, params, seed=RANDOM_STATE):
    if model_name == "LR":
        return LogisticRegression(solver="liblinear", max_iter=5000, random_state=seed, **params)
    if model_name == "SVM":
        return SVC(kernel="rbf", probability=True, random_state=seed, **params)
    if model_name == "DT":
        return DecisionTreeClassifier(random_state=seed, **params)
    if model_name == "RF":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    if model_name == "GBDT":
        return GradientBoostingClassifier(random_state=seed, **params)
    if model_name == "XGBoost":
        return XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1, **params)
    raise ValueError(model_name)


def tune(model_name, X_train, y_train):
    def objective(trial):
        model = build_model(model_name, suggest_params(model_name, trial))
        if model_name == "SVM":
            model.set_params(probability=False)
        return float(np.mean(cross_val_score(model, X_train, y_train, cv=CV, scoring="roc_auc", n_jobs=-1)))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS[model_name])
    return study.best_params, study.best_value


def repeated_auc(model_name, params, X, y):
    def one_run(seed):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        model = build_model(model_name, params, seed=seed).fit(X_tr, y_tr)
        return roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])

    return np.array(Parallel(n_jobs=-1)(delayed(one_run)(i) for i in range(REPEAT_RUNS)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "results_rerun"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    print(f"samples: {len(y)} | features: {X.shape[1]} | classes: {y.value_counts().to_dict()}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    rows = []
    for name in MODEL_NAMES:
        print(f"\n[{name}] Optuna, {N_TRIALS[name]} trials (10-fold CV, ROC-AUC) ...", flush=True)
        best_params, best_cv = tune(name, X_train, y_train)
        model = build_model(name, best_params).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        auc_v = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        with open(out / f"single_peak_{name}_best_model.pkl", "wb") as f:
            pickle.dump(clone(model).fit(X, y), f)
        aucs = repeated_auc(name, best_params, X, y)
        print(f"[{name}] CV AUC = {best_cv:.4f} | test AUC = {auc_v:.4f} | "
              f"{REPEAT_RUNS}-split AUC = {aucs.mean():.4f} +/- {aucs.std():.4f}", flush=True)
        rows.append({"model": name, "cv_roc_auc": best_cv,
                     "accuracy": accuracy_score(y_test, y_pred),
                     "precision": precision_score(y_test, y_pred, average="weighted"),
                     "recall": recall_score(y_test, y_pred, average="weighted"),
                     "f1": f1_score(y_test, y_pred, average="weighted"),
                     "roc_auc": auc_v, "roc_auc_repeat_mean": aucs.mean(), "roc_auc_repeat_std": aucs.std(),
                     "best_params": str(best_params)})

    res = pd.DataFrame(rows).sort_values("roc_auc_repeat_mean", ascending=False).reset_index(drop=True)
    res.to_csv(out / "single_peak_model_comparison.csv", index=False)
    print("\n" + res.drop(columns="best_params").round(4).to_string())
    print(f"\nresults saved to: {out}")


if __name__ == "__main__":
    main()

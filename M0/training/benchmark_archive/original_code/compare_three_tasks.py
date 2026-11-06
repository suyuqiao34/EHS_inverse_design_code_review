import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.unicode_minus"] = False


DATASETS = {
    "dh": str(Path(__file__).resolve().parents[1] / "inputs" / "original_m0_dh.csv"),
    "tm": str(Path(__file__).resolve().parents[1] / "inputs" / "original_m0_tm.csv"),
    "single_peak": str(Path(__file__).resolve().parents[1] / "inputs" / "original_m0_single_peak.csv"),
}

N_TRIALS = 60
CV_SPLITS = 5
RANDOM_STATE = 42
ROC_REPEAT_RUNS = 100
DATASET_DISPLAY = {"dh": "ΔH", "tm": "Tm", "single_peak": "SinglePeak"}


def dataset_display_name(dataset_name: str) -> str:
    return DATASET_DISPLAY.get(dataset_name, dataset_name)


def detect_task_type(y: pd.Series) -> str:
    """
    Detect the task type automatically:
    - Classify approximately discrete targets with few unique values as classification.
    - Otherwise, use regression.
    """
    y_numeric = pd.to_numeric(y, errors="coerce").dropna()
    unique_count = y_numeric.nunique()

    if unique_count <= 10:
        return "classification"

    is_integer_like = np.all(np.isclose(y_numeric, np.round(y_numeric)))
    if is_integer_like and unique_count <= max(20, int(len(y_numeric) * 0.05)):
        return "classification"

    return "regression"


def suggest_params(
    dataset_name: str, task_type: str, model_name: str, trial: optuna.Trial
) -> dict:
    if task_type == "classification":
        if model_name == "LR":
            solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
            params = {
                "C": trial.suggest_float("C", 1e-3, 100, log=True),
                "solver": solver,
                "max_iter": 5000,
                "random_state": RANDOM_STATE,
            }
            if solver == "liblinear":
                params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
            else:
                params["penalty"] = "l2"
            return params
        if model_name == "SVM":
            return {
                "kernel": "rbf",
                "C": trial.suggest_float("C", 1e-2, 200, log=True),
                "gamma": trial.suggest_float("gamma", 1e-5, 10, log=True),
                "probability": True,
                "random_state": RANDOM_STATE,
            }
        if model_name == "DT":
            return {
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "max_depth": trial.suggest_int("max_depth", 2, 40),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": RANDOM_STATE,
            }
        if model_name == "RF":
            if dataset_name == "single_peak":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                    "max_depth": trial.suggest_int("max_depth", 10, 50),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                    "max_features": trial.suggest_categorical(
                        "max_features", ["sqrt", "log2", None]
                    ),
                    "random_state": RANDOM_STATE,
                }
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "random_state": RANDOM_STATE,
            }
        if model_name == "GBDT":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": RANDOM_STATE,
            }
        if model_name == "XGBoost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 100, log=True),
                "eval_metric": "logloss",
                "random_state": RANDOM_STATE,
            }

    if model_name == "LR":
        return {}
    if model_name == "SVM":
        return {
            "kernel": "rbf",
            "C": trial.suggest_float("C", 1e-2, 200, log=True),
            "gamma": trial.suggest_float("gamma", 1e-5, 10, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 2.0, log=True),
        }
    if model_name == "DT":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": RANDOM_STATE,
        }
    if model_name == "RF":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": RANDOM_STATE,
        }
    if model_name == "GBDT":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": RANDOM_STATE,
        }
    if model_name == "XGBoost":
        if dataset_name == "dh":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 900, 1800),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.05),
                "max_depth": trial.suggest_int("max_depth", 4, 5),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
                "subsample": trial.suggest_float("subsample", 0.70, 0.86),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.58, 0.72),
                "gamma": trial.suggest_float("gamma", 1.2, 3.5),
                "reg_lambda": trial.suggest_float("reg_lambda", 18, 90),
                "reg_alpha": trial.suggest_float("reg_alpha", 18, 70),
                "objective": "reg:squarederror",
                "random_state": RANDOM_STATE,
            }
        if dataset_name == "tm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 900, 1800),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.06),
                "max_depth": trial.suggest_int("max_depth", 4, 6),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
                "subsample": trial.suggest_float("subsample", 0.75, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.82),
                "gamma": trial.suggest_float("gamma", 0.2, 1.5),
                "reg_lambda": trial.suggest_float("reg_lambda", 15, 90),
                "reg_alpha": trial.suggest_float("reg_alpha", 8, 45),
                "objective": "reg:squarederror",
                "random_state": RANDOM_STATE,
            }
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 100, log=True),
            "objective": "reg:squarederror",
            "random_state": RANDOM_STATE,
        }
    raise ValueError(f"Unsupported model: {model_name}")


def build_model(task_type: str, model_name: str, params: dict):
    if task_type == "classification":
        if model_name == "LR":
            return LogisticRegression(**params)
        if model_name == "SVM":
            return SVC(**params)
        if model_name == "DT":
            return DecisionTreeClassifier(**params)
        if model_name == "RF":
            return RandomForestClassifier(**params)
        if model_name == "GBDT":
            return GradientBoostingClassifier(**params)
        if model_name == "XGBoost":
            return XGBClassifier(**params)
    else:
        if model_name == "LR":
            return LinearRegression(**params)
        if model_name == "SVM":
            return SVR(**params)
        if model_name == "DT":
            return DecisionTreeRegressor(**params)
        if model_name == "RF":
            return RandomForestRegressor(**params)
        if model_name == "GBDT":
            return GradientBoostingRegressor(**params)
        if model_name == "XGBoost":
            return XGBRegressor(**params)
    raise ValueError(f"Unsupported model: {model_name}")


def get_optuna_target(
    dataset_name: str, task_type: str, n_classes: int | None
) -> tuple[str, str, str]:
    """
    Use the scoring objectives from the original notebooks:
    - Single-peak classification: ROC-AUC, or f1_weighted for nonbinary targets.
    - Enthalpy regression: minimize RMSE by maximizing neg_root_mean_squared_error.
    - Melting-temperature regression: R².
    """
    if dataset_name == "single_peak" and task_type == "classification":
        if n_classes == 2:
            return "roc_auc", "maximize", "cv_roc_auc"
        return "f1_weighted", "maximize", "cv_f1_weighted"

    if dataset_name == "dh" and task_type == "regression":
        return "neg_root_mean_squared_error", "maximize", "cv_rmse"

    if dataset_name == "tm" and task_type == "regression":
        return "r2", "maximize", "cv_r2"

    if task_type == "classification":
        return "f1_weighted", "maximize", "cv_f1_weighted"
    return "neg_root_mean_squared_error", "maximize", "cv_rmse"


def tune_one_model(
    dataset_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
    model_name: str,
) -> tuple[dict, float, str]:
    n_classes = y_train.nunique() if task_type == "classification" else None
    scoring, direction, cv_metric_name = get_optuna_target(
        dataset_name, task_type, n_classes
    )

    if task_type == "classification":
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(dataset_name, task_type, model_name, trial)
        model = build_model(task_type, model_name, params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        return float(np.mean(scores))

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=N_TRIALS)

    best_value = study.best_value
    if scoring == "neg_root_mean_squared_error":
        best_value = -best_value
    return study.best_params, best_value, cv_metric_name


def get_score_for_roc(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        decision = np.asarray(decision)
        d_min, d_max = decision.min(), decision.max()
        if np.isclose(d_max, d_min):
            return np.full_like(decision, 0.5, dtype=float)
        return (decision - d_min) / (d_max - d_min)
    raise ValueError("This model has no ROC-score interface (predict_proba / decision_function)")


def plot_mean_roc_for_model(
    dataset_name: str,
    model_name: str,
    best_params: dict,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    save_dir: Path,
) -> tuple[float, float]:
    all_fpr, all_tpr, aucs = [], [], []

    for seed in range(ROC_REPEAT_RUNS):
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all
        )
        params = best_params.copy()
        if "random_state" in params:
            params["random_state"] = seed

        model = build_model("classification", model_name, params)
        model.fit(X_train, y_train)
        y_score = get_score_for_roc(model, X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        aucs.append(roc_auc_score(y_test, y_score))

    mean_fpr = np.linspace(0, 1, 200)
    interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)]
    mean_tpr = np.mean(interp_tprs, axis=0)
    std_tpr = np.std(interp_tprs, axis=0)
    tpr_lower = np.clip(mean_tpr - std_tpr, 0, 1)
    tpr_upper = np.clip(mean_tpr + std_tpr, 0, 1)

    plt.figure(figsize=(7, 6))
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f"Mean ROC ({ROC_REPEAT_RUNS} runs)")
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.2, label="±1 std")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.title(
        f"{dataset_display_name(dataset_name)}-{model_name} Mean ROC Curve ({ROC_REPEAT_RUNS} runs)",
        fontweight="bold",
    )
    plt.legend()
    plt.tight_layout()

    fig_path = save_dir / f"single_peak_{model_name}_mean_roc_curve.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return float(np.mean(aucs)), float(np.std(aucs))


def plot_regression_scatter(
    dataset_name: str,
    model_name: str,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    save_dir: Path,
) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(y_train, y_train_pred, alpha=0.45, label="Train", s=30)
    plt.scatter(y_test, y_test_pred, alpha=0.7, label="Test", marker="^", s=36)

    y_all = np.concatenate([y_train, y_test])
    pred_all = np.concatenate([y_train_pred, y_test_pred])
    mn = float(min(y_all.min(), pred_all.min()))
    mx = float(max(y_all.max(), pred_all.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")

    plt.title(
        f"{dataset_display_name(dataset_name)}-{model_name}: Actual vs Predicted (Train & Test)",
        fontweight="bold",
    )
    plt.xlabel("Actual", fontweight="bold")
    plt.ylabel("Predicted", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    fig_path = save_dir / f"{dataset_name}_{model_name}_train_test_scatter_one_figure.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plot_singlepeak_combined_roc(
    model_roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    save_dir: Path,
) -> None:
    plt.figure(figsize=(7.5, 6.5))

    plot_order = ["SVM", "DT", "GBDT", "XGBoost", "RF", "LR"]
    color_map = {
        "SVM": "#4d4d4d",
        "DT": "#ff4d4d",
        "GBDT": "#1f77ff",
        "XGBoost": "#2ca25f",
        "RF": "#b36ae2",
        "LR": "#d8a307",
    }

    for model_name in plot_order:
        if model_name not in model_roc_data:
            continue
        fpr, tpr, auc_v = model_roc_data[model_name]
        label_name = "Xgboost" if model_name == "XGBoost" else model_name
        plt.plot(
            fpr,
            tpr,
            lw=2.2,
            color=color_map.get(model_name, None),
            label=f"{label_name} (AUC={auc_v:.4f})",
        )

    plt.plot([0, 1], [0, 1], color="black", linestyle=(0, (4, 4)), lw=1.6)
    plt.xlim(-0.03, 1.0)
    plt.ylim(0, 1.08)
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.title("SinglePeak Best Models ROC Curve", fontweight="bold")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()

    fig_path = save_dir / "single_peak_6_models_best_auc_roc_curve.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plot_singlepeak_prf_bar(result_df: pd.DataFrame, save_dir: Path) -> None:
    wanted_order = ["LR", "SVM", "DT", "RF", "GBDT", "XGBoost"]
    plot_df = result_df.set_index("model").reindex(wanted_order).reset_index()

    x = np.arange(len(plot_df))
    width = 0.27

    precision_vals = plot_df["precision"].values
    recall_vals = plot_df["recall"].values
    f1_vals = plot_df["f1"].values

    plt.figure(figsize=(7.2, 6.2))
    plt.bar(
        x - width,
        precision_vals,
        width=width,
        color="#d8b04d",
        edgecolor="#d8b04d",
        label="Precision",
        alpha=0.95,
    )
    plt.bar(
        x,
        recall_vals,
        width=width,
        color="#59d4bf",
        edgecolor="#59d4bf",
        label="Recall",
        alpha=0.95,
    )
    plt.bar(
        x + width,
        f1_vals,
        width=width,
        color="#b34dd9",
        edgecolor="#b34dd9",
        label="F1 Score",
        alpha=0.95,
    )

    plt.xticks(x, wanted_order, rotation=40, fontsize=14, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy", fontsize=24, fontweight="bold")
    plt.legend(loc="upper left", frameon=False, ncol=3, fontsize=12)
    plt.tight_layout()

    fig_path = save_dir / "single_peak_6_models_precision_recall_f1_bar.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()


def plot_regression_r2_bar_for_dataset(
    result_df: pd.DataFrame, dataset_name: str, save_dir: Path
) -> None:
    wanted_order = ["LR", "SVM", "DT", "RF", "GBDT", "XGBoost"]
    plot_df = result_df.set_index("model").reindex(wanted_order).reset_index()

    x = np.arange(len(plot_df))
    width = 0.42

    train_mean = plot_df["train_r2_mean"].values
    train_std = plot_df["train_r2_std"].values
    test_mean = plot_df["r2_mean"].values
    test_std = plot_df["r2_std"].values

    if dataset_name == "dh":
        train_color = "#4d63ff"
        test_color = "#ff6b6b"
    else:
        train_color = "#9647ff"
        test_color = "#f2b67e"

    plt.figure(figsize=(8.2, 6.0))
    plt.bar(
        x - width / 2,
        train_mean,
        width=width,
        yerr=train_std,
        capsize=3,
        color=train_color,
        alpha=0.85,
        label="Training Set",
    )
    plt.bar(
        x + width / 2,
        test_mean,
        width=width,
        yerr=test_std,
        capsize=3,
        color=test_color,
        alpha=0.85,
        label="Testing Set",
    )

    plt.xticks(x, wanted_order, rotation=40, fontsize=14, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    plt.ylim(0, 1.08)
    plt.ylabel("R$^2$", fontsize=26, fontweight="bold")
    plt.xlabel(f"{dataset_display_name(dataset_name)} Models", fontsize=28, fontweight="bold")
    plt.legend(loc="upper right", frameon=False, ncol=2, fontsize=13)
    plt.tight_layout()

    fig_path = save_dir / f"{dataset_name}_6_models_train_test_r2_errorbar.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()


def repeat_regression_runs(
    dataset_name: str,
    model_name: str,
    best_params: dict,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    save_dir: Path,
    n_runs: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_metrics = []

    model_template = build_model("regression", model_name, best_params.copy())
    valid_keys = set(model_template.get_params().keys())

    base_params = best_params.copy()
    # Allow variability between repeated fits by slightly reducing sampling parameters equal to 1.
    if "subsample" in valid_keys and base_params.get("subsample", 1.0) >= 1.0:
        base_params["subsample"] = 0.9
    if (
        "colsample_bytree" in valid_keys
        and base_params.get("colsample_bytree", 1.0) >= 1.0
    ):
        base_params["colsample_bytree"] = 0.9

    for seed in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=seed
        )
        params = base_params.copy()
        if "random_state" in valid_keys:
            params["random_state"] = seed

        model = build_model("regression", model_name, params)
        model.fit(X_train, y_train)

        y_tr = model.predict(X_train)
        y_te = model.predict(X_test)

        rmse_tr = np.sqrt(mean_squared_error(y_train, y_tr))
        rmse_te = np.sqrt(mean_squared_error(y_test, y_te))
        r2_tr = r2_score(y_train, y_tr)
        r2_te = r2_score(y_test, y_te)

        run_metrics.append(
            {
                "run": seed,
                "rmse_train": rmse_tr,
                "rmse_test": rmse_te,
                "r2_train": r2_tr,
                "r2_test": r2_te,
            }
        )

    runs_df = pd.DataFrame(run_metrics)
    runs_path = save_dir / f"{dataset_name}_{model_name}_runs_metrics_{n_runs}.csv"
    runs_df.to_csv(runs_path, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(
        {
            "set": ["Train", "Test"],
            "r2_mean": [runs_df["r2_train"].mean(), runs_df["r2_test"].mean()],
            "r2_std": [runs_df["r2_train"].std(ddof=1), runs_df["r2_test"].std(ddof=1)],
            "rmse_mean": [runs_df["rmse_train"].mean(), runs_df["rmse_test"].mean()],
            "rmse_std": [runs_df["rmse_train"].std(ddof=1), runs_df["rmse_test"].std(ddof=1)],
        }
    )
    summary_path = save_dir / f"{dataset_name}_{model_name}_summary_train_test_r2_rmse.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    return runs_df, summary_df


def evaluate_models(dataset_name: str, csv_path: str) -> pd.DataFrame:
    print(f"\n{'=' * 30}\nDataset: {dataset_name}\nFile: {csv_path}\n{'=' * 30}")

    data_df = pd.read_csv(csv_path)
    X = data_df.iloc[:, 1:-1].copy()
    y_raw = data_df.iloc[:, -1].copy()

    # Match the original code: coerce targets to numeric without feature imputation, scaling, or one-hot preprocessing.
    y = pd.to_numeric(y_raw, errors="coerce")
    valid_mask = y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    task_type = detect_task_type(y)
    print(f"Detected task type: {task_type}")
    print(f"Samples: {len(y)} | Features: {X.shape[1]}")

    if task_type == "classification":
        unique_labels = sorted(y.unique().tolist())
        print(f"Classes: {unique_labels}")
        stratify = y
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model_names = ["LR", "SVM", "DT", "RF", "GBDT", "XGBoost"]
    rows = []
    combined_roc_data = {}

    for model_name in model_names:
        print(f"\nTuning: {model_name} (Optuna, n_trials={N_TRIALS}) ...")
        best_params, best_cv, cv_metric_name = tune_one_model(
            dataset_name, X_train, y_train, task_type, model_name
        )
        best_model = build_model(task_type, model_name, best_params)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        if task_type == "classification":
            row = {
                "model": model_name,
                cv_metric_name: best_cv,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1": f1_score(y_test, y_pred, average="weighted"),
                "roc_auc": np.nan,
                "best_params": str(best_params),
            }

            # Binary AUC requires probabilities or decision_function scores.
            if y.nunique() == 2:
                y_score = get_score_for_roc(best_model, X_test)
                row["roc_auc"] = roc_auc_score(y_test, y_score)
                fpr, tpr, _ = roc_curve(y_test, y_score)
                combined_roc_data[model_name] = (fpr, tpr, row["roc_auc"])

                # For the single-peak dataset, output each model's mean ROC curve across repeated fits.
                if dataset_name == "single_peak":
                    print(
                        f"Generating mean ROC curve: {model_name} "
                        f"({ROC_REPEAT_RUNS} repeated fits) ..."
                    )
                    mean_auc, std_auc = plot_mean_roc_for_model(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        best_params=best_params,
                        X_all=X,
                        y_all=y,
                        save_dir=Path(csv_path).parent,
                    )
                    row["roc_auc_repeat_mean"] = mean_auc
                    row["roc_auc_repeat_std"] = std_auc
            rows.append(row)
        else:
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            y_train_pred = best_model.predict(X_train)
            plot_regression_scatter(
                dataset_name=dataset_name,
                model_name=model_name,
                y_train=np.asarray(y_train),
                y_train_pred=np.asarray(y_train_pred),
                y_test=np.asarray(y_test),
                y_test_pred=np.asarray(y_pred),
                save_dir=Path(csv_path).parent,
            )

            runs_df, summary_df = repeat_regression_runs(
                dataset_name=dataset_name,
                model_name=model_name,
                best_params=best_params,
                X_all=X,
                y_all=y,
                save_dir=Path(csv_path).parent,
                n_runs=100,
            )

            train_row = summary_df[summary_df["set"] == "Train"].iloc[0]
            test_row = summary_df[summary_df["set"] == "Test"].iloc[0]
            rows.append(
                {
                    "model": model_name,
                    cv_metric_name: best_cv,
                    "rmse": rmse,
                    "mae": mean_absolute_error(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred),
                    "r2_mean": test_row["r2_mean"],
                    "r2_std": test_row["r2_std"],
                    "rmse_mean": test_row["rmse_mean"],
                    "rmse_std": test_row["rmse_std"],
                    "train_r2_mean": train_row["r2_mean"],
                    "train_r2_std": train_row["r2_std"],
                    "train_rmse_mean": train_row["rmse_mean"],
                    "train_rmse_std": train_row["rmse_std"],
                    "best_params": str(best_params),
                }
            )

    result_df = pd.DataFrame(rows)

    if task_type == "classification":
        result_df = result_df.sort_values(
            ["accuracy", "f1", "precision", "recall", "roc_auc"], ascending=False
        )
        if dataset_name == "single_peak" and len(combined_roc_data) > 0 and y.nunique() == 2:
            plot_singlepeak_combined_roc(combined_roc_data, Path(csv_path).parent)
            plot_singlepeak_prf_bar(result_df, Path(csv_path).parent)
    else:
        result_df = result_df.sort_values(["rmse", "mae", "r2"], ascending=[True, True, False])
        if dataset_name in {"dh", "tm"}:
            plot_regression_r2_bar_for_dataset(result_df, dataset_name, Path(csv_path).parent)

    result_df = result_df.reset_index(drop=True)
    print("\nModel comparison results:")
    print(result_df.round(4))

    save_name = f"{dataset_name}_model_comparison.csv"
    save_path = Path(csv_path).parent / save_name
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved results to: {save_path}")

    return result_df


def main():
    all_results = {}
    for name, path in DATASETS.items():
        try:
            all_results[name] = evaluate_models(name, path)
        except FileNotFoundError:
            print(f"\n[Skipped] File not found: {path}")
        except Exception as exc:
            print(f"\n[Failed] Dataset {name}: {exc}")

    print("\nAll tasks completed.")
    return all_results


if __name__ == "__main__":
    main()

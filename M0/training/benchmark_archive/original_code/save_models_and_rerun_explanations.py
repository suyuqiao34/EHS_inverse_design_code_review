import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR


ARCHIVE_DIR = Path(__file__).resolve().parents[1]
ROOT = ARCHIVE_DIR / "original_results"
MODEL_DIR = ROOT / "best_models"

TASKS = [
    {
        "name": "tm",
        "model_name": "GBDT",
        "data_path": ARCHIVE_DIR / "inputs" / "original_m0_tm.csv",
        "result_path": ROOT / "tm_model_comparison.csv",
        "model_file": MODEL_DIR / "tm_gbdt_best_model.pkl",
        "shap_png": ROOT / "tm_GBDT_SHAP_summary_beeswarm.png",
        "shap_csv": ROOT / "tm_GBDT_SHAP_mean_abs_importance.csv",
        "imp_png": ROOT / "tm_GBDT_feature_importance_aggregated.png",
        "imp_csv": ROOT / "tm_GBDT_feature_importance_aggregated.csv",
        "imp_color": "#F58518",
        "top_n": 13,
    },
    {
        "name": "dh",
        "model_name": "SVM",
        "data_path": ARCHIVE_DIR / "inputs" / "original_m0_dh.csv",
        "result_path": ROOT / "dh_model_comparison.csv",
        "model_file": MODEL_DIR / "dh_svm_best_model.pkl",
        "shap_png": ROOT / "dh_SVM_SHAP_summary_beeswarm.png",
        "shap_csv": ROOT / "dh_SVM_SHAP_mean_abs_importance.csv",
        "imp_png": ROOT / "dh_SVM_feature_importance_aggregated.png",
        "imp_csv": ROOT / "dh_SVM_feature_importance_aggregated.csv",
        "imp_color": "#D62728",
        "top_n": 13,
    },
]

MAX_DISPLAY = 10
MAX_SAMPLES_FOR_SHAP = 300
BACKGROUND_SAMPLES_FOR_SVM = 120
PERMUTATION_REPEATS = 20
RANDOM_SEED = 42


def apply_plot_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.unicode_minus"] = False


def canonical_feature_name(feature: str) -> str:
    # Match the existing aggregated-importance script: remove the 1-/2-/3- prefix and map ID to proportion.
    name = str(feature).strip()
    name = name.split("-", 1)[1].strip() if "-" in name and name.split("-", 1)[0].isdigit() else name
    if name in {"ID1", "ID2", "ID3"}:
        return "proportion"
    return name


def load_dataset(data_path: Path):
    df = pd.read_csv(data_path)
    X = df.iloc[:, 1:-1].copy()
    y = pd.to_numeric(df.iloc[:, -1], errors="coerce")
    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    return X, y


def load_best_params(result_path: Path, model_name: str) -> dict:
    df = pd.read_csv(result_path, encoding="utf-8-sig")
    row = df[df["model"] == model_name]
    if row.empty:
        raise ValueError(f"Model {model_name} was not found in {result_path.name}")
    params_str = row.iloc[0]["best_params"]
    return ast.literal_eval(params_str) if isinstance(params_str, str) else {}


def build_model(model_name: str, params: dict):
    if model_name == "GBDT":
        return GradientBoostingRegressor(**params)
    if model_name == "SVM":
        return SVR(**params)
    raise ValueError(f"Unsupported model: {model_name}")


def sample_rows(X: pd.DataFrame, n_samples: int, seed: int = RANDOM_SEED):
    if len(X) <= n_samples:
        return X.reset_index(drop=True)
    idx = np.random.default_rng(seed).choice(len(X), size=n_samples, replace=False)
    idx = np.sort(idx)
    return X.iloc[idx].reset_index(drop=True)


def save_shap_outputs(task: dict, model, X: pd.DataFrame):
    model_name = task["model_name"]
    X_eval = sample_rows(X, MAX_SAMPLES_FOR_SHAP, seed=RANDOM_SEED)

    if model_name == "GBDT":
        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(explainer.shap_values(X_eval))
    else:
        bg = sample_rows(X, BACKGROUND_SAMPLES_FOR_SVM, seed=RANDOM_SEED + 1)
        # Explain the SVM with the model-agnostic KernelExplainer.
        explainer = shap.KernelExplainer(model.predict, bg)
        shap_values = np.asarray(explainer.shap_values(X_eval, nsamples=200))

    plt.figure(figsize=(8.5, 10.5))
    shap.summary_plot(
        shap_values,
        X_eval,
        plot_type="dot",
        max_display=MAX_DISPLAY,
        alpha=0.65,
        show=False,
        color_bar=True,
    )
    ax = plt.gca()
    ax.axvline(x=0.0, color="gray", linestyle="-", linewidth=1.0, alpha=0.8)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig(task["shap_png"], dpi=300, bbox_inches="tight")
    plt.close()

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    shap_df = (
        pd.DataFrame({"feature": X_eval.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_df.to_csv(task["shap_csv"], index=False, encoding="utf-8-sig")


def save_feature_importance_outputs(task: dict, model, X: pd.DataFrame, y: pd.Series):
    model_name = task["model_name"]

    if model_name == "GBDT":
        raw_importance = np.asarray(model.feature_importances_, dtype=float)
    else:
        perm = permutation_importance(
            model,
            X,
            y,
            scoring="r2",
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        raw_importance = np.asarray(perm.importances_mean, dtype=float)
        raw_importance = np.clip(raw_importance, 0.0, None)

    df = pd.DataFrame({"feature": X.columns, "importance": raw_importance})
    df["feature"] = df["feature"].map(canonical_feature_name)
    agg_df = (
        df.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    agg_df.to_csv(task["imp_csv"], index=False, encoding="utf-8-sig")

    plot_df = agg_df.head(task["top_n"]).copy()
    y_pos = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    ax.barh(
        y_pos,
        plot_df["importance"].values,
        color=task["imp_color"],
        edgecolor=task["imp_color"],
        alpha=0.82,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"].values, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=18, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(task["imp_png"], dpi=300)
    plt.close()


def run_one_task(task: dict):
    print(f"\n=== {task['name']} - {task['model_name']} ===")
    X, y = load_dataset(task["data_path"])
    params = load_best_params(task["result_path"], task["model_name"])
    model = build_model(task["model_name"], params)
    model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(task["model_file"], "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model: {task['model_file']}")

    save_shap_outputs(task, model, X)
    print(f"Saved SHAP plot: {task['shap_png']}")
    print(f"Saved SHAP data: {task['shap_csv']}")

    save_feature_importance_outputs(task, model, X, y)
    print(f"Saved feature-importance plot: {task['imp_png']}")
    print(f"Saved feature-importance data: {task['imp_csv']}")


def main():
    apply_plot_style()
    for task in TASKS:
        run_one_task(task)
    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()

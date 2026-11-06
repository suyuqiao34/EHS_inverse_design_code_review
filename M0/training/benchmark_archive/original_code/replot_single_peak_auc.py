import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


DATA_PATH = Path(str(Path(__file__).resolve().parents[1] / "inputs" / "original_m0_single_peak.csv"))
RESULT_PATH = Path(Path(__file__).resolve().parents[1] / "original_results" / "single_peak_model_comparison.csv")
OUT_PATH = Path(Path(__file__).resolve().parents[1] / "original_results" / "single_peak_6_models_best_auc_roc_curve.png")
OUT_DATA_PATH = Path(Path(__file__).resolve().parents[1] / "original_results" / "single_peak_6_models_best_auc_roc_curve_data.csv")


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.unicode_minus"] = False


def build_model(model_name: str, params: dict):
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
    raise ValueError(f"Unsupported model: {model_name}")


def get_score_for_roc(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(X))
        d_min, d_max = decision.min(), decision.max()
        if np.isclose(d_max, d_min):
            return np.full_like(decision, 0.5, dtype=float)
        return (decision - d_min) / (d_max - d_min)
    raise ValueError("Model has neither predict_proba nor decision_function.")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Results file not found: {RESULT_PATH}")

    data_df = pd.read_csv(DATA_PATH)
    X = data_df.iloc[:, 1:-1].copy()
    y = pd.to_numeric(data_df.iloc[:, -1], errors="coerce")
    valid_mask = y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].astype(int).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    result_df = pd.read_csv(RESULT_PATH, encoding="utf-8-sig")
    required_models = ["LR", "SVM", "DT", "RF", "GBDT", "XGBoost"]
    result_df = result_df[result_df["model"].isin(required_models)].copy()

    roc_data = {}
    for model_name in required_models:
        row = result_df[result_df["model"] == model_name]
        if row.empty:
            continue
        params = ast.literal_eval(row.iloc[0]["best_params"])
        model = build_model(model_name, params)
        model.fit(X_train, y_train)
        y_score = get_score_for_roc(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_v = roc_auc_score(y_test, y_score)
        roc_data[model_name] = (fpr, tpr, auc_v)

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
        if model_name not in roc_data:
            continue
        fpr, tpr, auc_v = roc_data[model_name]
        legend_name = "Xgboost" if model_name == "XGBoost" else model_name
        plt.plot(
            fpr,
            tpr,
            lw=2.2,
            color=color_map.get(model_name, None),
            label=f"{legend_name} (AUC={auc_v:.4f})",
        )

    plt.plot([0, 1], [0, 1], color="black", linestyle=(0, (4, 4)), lw=1.6)
    plt.xlim(-0.03, 1.0)
    plt.ylim(0, 1.08)
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate", fontweight="bold")
    plt.title("SinglePeak Best Models ROC Curve", fontweight="bold")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    plt.close()

    rows = []
    for model_name in plot_order:
        if model_name not in roc_data:
            continue
        fpr, tpr, auc_v = roc_data[model_name]
        for i, (x, yv) in enumerate(zip(fpr, tpr)):
            rows.append(
                {
                    "model": model_name,
                    "point_idx": i,
                    "fpr": float(x),
                    "tpr": float(yv),
                    "auc": float(auc_v),
                }
            )
    data_df = pd.DataFrame(rows)
    data_df.to_csv(OUT_DATA_PATH, index=False, encoding="utf-8-sig")

    print(f"Replotted: {OUT_PATH}")
    print(f"Exported data: {OUT_DATA_PATH}")


if __name__ == "__main__":
    main()

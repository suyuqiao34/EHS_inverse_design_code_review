from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


DATA_PATH = Path(str(Path(__file__).resolve().parents[1] / "inputs" / "original_m0_single_peak.csv"))
OUTPUT_PATH = Path(Path(__file__).resolve().parents[1] / "original_results" / "single_peak_XGBoost_best_model_retrained.pkl")

# Original best hyperparameters from single_peak_model_comparison.csv
BEST_PARAMS = {
    "n_estimators": 643,
    "learning_rate": 0.07102731973724802,
    "max_depth": 7,
    "min_child_weight": 8,
    "subsample": 0.8018497397679697,
    "colsample_bytree": 0.8986701562394934,
    "gamma": 2.329865598404251,
    "reg_lambda": 0.11598360514875435,
    "reg_alpha": 0.030135320760888637,
}


def load_singlepeak_data(data_path: Path):
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data_df = pd.read_csv(data_path)

    # Keep the same feature/label columns as the original training scripts.
    X = data_df.iloc[:, 1:-1].copy()
    y = pd.to_numeric(data_df.iloc[:, -1], errors="coerce")

    valid_mask = y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].astype(int).reset_index(drop=True)

    return X, y


def build_model() -> XGBClassifier:
    return XGBClassifier(**BEST_PARAMS)


def main():
    X, y = load_singlepeak_data(DATA_PATH)
    print(f"Samples: {len(y)}")
    print(f"Features: {X.shape[1]}")
    print(f"Class counts:\n{y.value_counts().sort_index()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    eval_model = build_model()
    eval_model.fit(X_train, y_train)

    y_pred = eval_model.predict(X_test)
    y_score = eval_model.predict_proba(X_test)[:, 1]

    print("\nHold-out test results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, y_score):.4f}")
    print(classification_report(y_test, y_pred))

    # Save a final model trained on all valid rows in the CSV.
    final_model = build_model()
    final_model.fit(X, y)
    joblib.dump(final_model, OUTPUT_PATH)

    print(f"\nSaved retrained model to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

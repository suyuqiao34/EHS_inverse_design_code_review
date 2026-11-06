import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CSV_PATH = str(Path(__file__).resolve().parents[2] / "inputs" / "original_m0_dh.csv")
N_RUNS = 100
TEST_SIZE = 0.2
N_TRIALS = 120


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.unicode_minus"] = False
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    X_raw = df.iloc[:, 1:-1].copy()
    y_raw = pd.to_numeric(df.iloc[:, -1], errors="coerce")

    X = pd.get_dummies(X_raw, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce")

    valid_mask = y_raw.notna() & (~X.isna().any(axis=1))
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y_raw.loc[valid_mask].reset_index(drop=True)
    return X, y


def build_linear_model(model_type: str, params: dict):
    if model_type == "linear":
        return LinearRegression()
    if model_type == "ridge":
        return Ridge(alpha=params["alpha"], random_state=42)
    if model_type == "lasso":
        return Lasso(alpha=params["alpha"], max_iter=20000, random_state=42)
    if model_type == "elasticnet":
        return ElasticNet(
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            max_iter=20000,
            random_state=42,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def tune_best_model(X_train: pd.DataFrame, y_train: pd.Series):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        model_type = trial.suggest_categorical(
            "model_type", ["linear", "ridge", "lasso", "elasticnet"]
        )
        params = {}
        if model_type in {"ridge", "lasso", "elasticnet"}:
            params["alpha"] = trial.suggest_float("alpha", 1e-4, 1e3, log=True)
        if model_type == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)

        model = build_linear_model(model_type, params)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    best = study.best_params.copy()
    best_model_type = best.pop("model_type")
    best_cv_rmse = -study.best_value
    return best_model_type, best, best_cv_rmse


def plot_scatter(y_train, y_train_pred, y_test, y_test_pred, out_path: Path):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_train, y_train_pred, alpha=0.45, label="Train", s=30)
    plt.scatter(y_test, y_test_pred, alpha=0.7, label="Test", marker="^", s=36)

    y_all = np.concatenate([y_train, y_test])
    pred_all = np.concatenate([y_train_pred, y_test_pred])
    mn = float(min(y_all.min(), pred_all.min()))
    mx = float(max(y_all.max(), pred_all.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")

    plt.xlabel("Actual", fontweight="bold")
    plt.ylabel("Predicted", fontweight="bold")
    plt.title("ΔH-LR: Actual vs Predicted (Train & Test)", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_r2_errorbar(train_mean, train_std, test_mean, test_std, out_path: Path):
    plt.figure(figsize=(6.2, 5.2))
    x = np.array([0, 1])
    means = [train_mean, test_mean]
    stds = [train_std, test_std]
    colors = ["#4d63ff", "#ff6b6b"]

    plt.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.86, width=0.6)
    plt.xticks(x, ["Training Set", "Testing Set"], fontsize=13, fontweight="bold")
    plt.yticks(fontsize=12, fontweight="bold")
    plt.ylabel("R$^2$", fontsize=24, fontweight="bold")
    plt.title("ΔH-Linear Model: R$^2$ Mean ± Std (100 runs)", fontsize=16, fontweight="bold")
    plt.ylim(0, 1.08)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    X, y = load_data(CSV_PATH)
    save_dir = Path(CSV_PATH).parent

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )
    best_model_type, best_params, best_cv_rmse = tune_best_model(X_train, y_train)
    print(f"Best linear family model: {best_model_type}")
    print(f"Best params: {best_params}")
    print(f"Best CV RMSE: {best_cv_rmse:.4f}")

    model = build_linear_model(best_model_type, best_params)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.fit(X_train, y_train)

    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"Train RMSE={rmse_train:.4f}, R2={r2_train:.4f}")
    print(f"Test  RMSE={rmse_test:.4f}, R2={r2_test:.4f}")

    plot_scatter(
        y_train.values,
        y_train_pred,
        y_test.values,
        y_test_pred,
        save_dir / "dh_LR_train_test_scatter_one_figure.png",
    )

    rows = []
    for seed in range(N_RUNS):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed
        )
        model = build_linear_model(best_model_type, best_params)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(X_train, y_train)
        y_tr = pipe.predict(X_train)
        y_te = pipe.predict(X_test)
        rows.append(
            {
                "run": seed,
                "r2_train": r2_score(y_train, y_tr),
                "r2_test": r2_score(y_test, y_te),
                "rmse_train": np.sqrt(mean_squared_error(y_train, y_tr)),
                "rmse_test": np.sqrt(mean_squared_error(y_test, y_te)),
            }
        )

    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(save_dir / "dh_LR_runs_metrics_100.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(
        {
            "set": ["Train", "Test"],
            "r2_mean": [runs_df["r2_train"].mean(), runs_df["r2_test"].mean()],
            "r2_std": [runs_df["r2_train"].std(ddof=1), runs_df["r2_test"].std(ddof=1)],
            "rmse_mean": [runs_df["rmse_train"].mean(), runs_df["rmse_test"].mean()],
            "rmse_std": [runs_df["rmse_train"].std(ddof=1), runs_df["rmse_test"].std(ddof=1)],
        }
    )
    summary_df.to_csv(
        save_dir / "dh_LR_summary_train_test_r2_rmse.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_r2_errorbar(
        train_mean=summary_df.loc[summary_df["set"] == "Train", "r2_mean"].iloc[0],
        train_std=summary_df.loc[summary_df["set"] == "Train", "r2_std"].iloc[0],
        test_mean=summary_df.loc[summary_df["set"] == "Test", "r2_mean"].iloc[0],
        test_std=summary_df.loc[summary_df["set"] == "Test", "r2_std"].iloc[0],
        out_path=save_dir / "dh_LR_train_test_r2_errorbar.png",
    )

    print("Saved: dh_LR_train_test_scatter_one_figure.png")
    print("Saved: dh_LR_runs_metrics_100.csv")
    print("Saved: dh_LR_summary_train_test_r2_rmse.csv")
    print("Saved: dh_LR_train_test_r2_errorbar.png")


if __name__ == "__main__":
    main()

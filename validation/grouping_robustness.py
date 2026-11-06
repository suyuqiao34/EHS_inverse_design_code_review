"""Reproduce Table S7b with fixed M1 hyperparameters and archived task labels.

Regression uses the 238 Tm and 236 enthalpy labels in round1_all_records.csv.
Classification uses the historical 349-label snapshot in round1_single_peak.csv,
which contains 178 literature and 171 feedback records. The collection has 350
records, and the added literature negative label is excluded only from this
historical classification sensitivity analysis. Main nested validation (S7a)
continues to use the complete 350-record table.

All predictions and fold membership are saved. No Optuna or LLM is called.
"""
import argparse
import ast
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
SEED = 42
SALTS = ["ID1-salt", "ID2-salt", "ID3-salt"]
SP, TM, DH, SOURCE = "single peak", "Tm (°C)", "ΔHm (J/g)", "source"
TASK_COLUMNS = {"single_peak": SP, "Tm": TM, "dH": DH}
TRAIN_A = "A literature only"
TRAIN_B = "B literature + remaining feedback"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimator(task, params):
    if task == "single_peak":
        return GradientBoostingClassifier(**params, random_state=SEED)
    if task == "Tm":
        return RandomForestRegressor(**params, random_state=SEED, n_jobs=1)
    return XGBRegressor(**params, random_state=SEED, n_jobs=1)


def metrics(task, rows):
    y, prediction = rows.y_true.values, rows.y_pred.values
    result = {"n_pred": len(y)}
    if task == "single_peak":
        result.update(accuracy=accuracy_score(y, prediction),
                      roc_auc=roc_auc_score(y, rows.y_proba.values) if len(np.unique(y)) == 2 else np.nan,
                      recall_nonSP=float((prediction[y == 0] == 0).mean()) if np.any(y == 0) else np.nan)
    else:
        result.update(MAE=mean_absolute_error(y, prediction),
                      R2=r2_score(y, prediction) if len(y) > 1 else np.nan)
    return result


def make_splits(df, is_feedback, salt_sets):
    definitions, fold_sizes = [], []
    combinations = np.array(sorted(df.loc[is_feedback, "combo"].unique()))
    rng = np.random.RandomState(SEED)
    for repeat in range(20):
        shuffled = rng.permutation(combinations)
        mapping = {combination: i % 5 for i, combination in enumerate(shuffled)}
        folds = df.combo.map(mapping).where(is_feedback, -1).values
        for fold in range(5):
            test = folds == fold
            definitions.append(("S1 random 5-fold by salt combination (20 repeats)", repeat, fold,
                                test, folds != fold))
            fold_sizes.append({"rep": repeat, "fold": fold, "n_rows": int(test.sum()),
                               "n_Tm_label": int((test & df[TM].notna()).sum()),
                               "n_subzero": int((test & (df[TM] < 0)).sum())})
    for fold, combination in enumerate(combinations):
        definitions.append(("S2 leave-one-combination-out", 0, fold,
                            (df.combo == combination).values & is_feedback,
                            (df.combo != combination).values))
    for fold, combination in enumerate(sorted(df.loc[is_feedback, "combo_anhyd"].unique())):
        definitions.append(("S3 leave-one-anhydrous-combination-out", 0, fold,
                            (df.combo_anhyd == combination).values & is_feedback,
                            (df.combo_anhyd != combination).values))
    old_salts = set().union(*salt_sets[~is_feedback])
    new_salts = sorted(set().union(*salt_sets[is_feedback]) - old_salts)
    for fold, salt in enumerate(new_salts):
        contains = salt_sets.apply(lambda values: salt in values).values
        definitions.append(("S4 leave-one-new-salt-out", 0, fold, contains & is_feedback, ~contains))
    return definitions, pd.DataFrame(fold_sizes), new_salts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parent)
    parser.add_argument("--out", type=Path, default=HERE / "results_rerun")
    parser.add_argument("--task", choices=["all", *TASK_COLUMNS], default="all")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data_path = args.repo / "data/round1/round1_all_records.csv"
    classification_path = args.repo / "data/round1/round1_single_peak.csv"
    params_path = args.repo / "M1/models/model_summary.csv"
    df = pd.read_csv(data_path)
    classification = pd.read_csv(classification_path)
    feature_columns = [c for c in classification.columns if c not in SALTS + [SP]]
    X = df[feature_columns].apply(pd.to_numeric, errors="raise")
    if X.isna().any().any():
        raise ValueError("Model features contain missing values")
    label_keys = pd.MultiIndex.from_frame(classification[feature_columns + [SP]])
    classification_valid = pd.MultiIndex.from_frame(df[feature_columns + [SP]]).isin(label_keys)
    if int(classification_valid.sum()) != len(classification):
        raise ValueError("Classification snapshot does not map uniquely to the collection")
    is_feedback = df[SOURCE].eq("feedback").values
    salt_sets = df[SALTS].apply(lambda row: frozenset(str(s) for s in row if pd.notna(s)), axis=1)
    df["combo"] = salt_sets.apply(lambda values: " + ".join(sorted(values)))
    anhydrous = salt_sets.apply(lambda values: frozenset(
        re.sub(r"·\d*H2O", "", salt) for salt in values))
    df["combo_anhyd"] = anhydrous.apply(lambda values: " + ".join(sorted(values)))
    splits, fold_sizes, new_salts = make_splits(df, is_feedback, salt_sets)
    summary = pd.read_csv(params_path).set_index("task")
    parameter_keys = {"single_peak": "single_peak", "Tm": "Tm", "dH": "dHm"}
    parameters = {task: ast.literal_eval(summary.loc[key, "best_params"])
                  for task, key in parameter_keys.items()}
    tasks = list(TASK_COLUMNS) if args.task == "all" else [args.task]
    prediction_rows, fold_rows, timing_rows = [], [], []
    for task in tasks:
        y = pd.to_numeric(df[TASK_COLUMNS[task]], errors="coerce")
        valid = y.notna().values
        if task == "single_peak":
            valid &= classification_valid
        old_train = (~is_feedback) & valid
        target_old = y[old_train].astype(int) if task == "single_peak" else y[old_train]
        started = monotonic()
        model_a = estimator(task, parameters[task]).fit(X.loc[old_train], target_old)
        # A is deterministic and uses the identical literature rows in every fold.
        prediction_a = model_a.predict(X)
        probability_a = model_a.predict_proba(X)[:, 1] if task == "single_peak" else None
        for index, (scheme, repeat, fold, raw_test, raw_train) in enumerate(splits):
            test, train_b = raw_test & valid, raw_train & valid
            if not test.any():
                continue
            if np.any(test & train_b):
                raise AssertionError("Training and test rows overlap")
            if scheme.startswith(("S1 ", "S2 ", "S3 ")):
                overlap = set(df.loc[test, "combo"]) & set(df.loc[train_b & is_feedback, "combo"])
                if overlap:
                    raise AssertionError("Held-out feedback combinations occur in training feedback")
            fit_started = monotonic()
            target_b = y[train_b].astype(int) if task == "single_peak" else y[train_b]
            model_b = estimator(task, parameters[task]).fit(X.loc[train_b], target_b)
            prediction_b = model_b.predict(X.loc[test])
            probability_b = model_b.predict_proba(X.loc[test])[:, 1] if task == "single_peak" else None
            row_indices = np.flatnonzero(test)
            for train_name, train_mask, prediction, probability in (
                    (TRAIN_A, old_train, prediction_a[test], probability_a[test] if probability_a is not None else None),
                    (TRAIN_B, train_b, prediction_b, probability_b)):
                local_rows = []
                for offset, row_index in enumerate(row_indices):
                    record = {"scheme": scheme, "repeat": repeat, "fold": fold, "task": task,
                              "train": train_name, "row_index": int(row_index),
                              "combo": df.loc[row_index, "combo"], "y_true": float(y.iloc[row_index]),
                              "y_pred": float(prediction[offset]),
                              "y_proba": float(probability[offset]) if probability is not None else np.nan,
                              "n_train": int(train_mask.sum()),
                              "n_train_literature": int((train_mask & ~is_feedback).sum()),
                              "n_train_feedback": int((train_mask & is_feedback).sum())}
                    prediction_rows.append(record)
                    local_rows.append(record)
                fold_rows.append({"scheme": scheme, "repeat": repeat, "fold": fold,
                                  "task": task, "train": train_name,
                                  "n_train": int(train_mask.sum()),
                                  **metrics(task, pd.DataFrame(local_rows))})
            timing_rows.append({"task": task, "scheme": scheme, "repeat": repeat, "fold": fold,
                                "fit_seconds": monotonic() - fit_started})
            if index % 10 == 0:
                print(f"{task}: {index + 1}/{len(splits)} folds, {monotonic() - started:.1f}s", flush=True)
        print(f"{task} complete, {monotonic() - started:.1f}s", flush=True)
    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    repeat_rows = []
    for (scheme, task, train_name, repeat), rows in predictions.groupby(["scheme", "task", "train", "repeat"], sort=False):
        repeat_rows.append({"scheme": scheme, "task": task, "train": train_name, "repeat": repeat,
                            **metrics(task, rows)})
    repeats = pd.DataFrame(repeat_rows)
    result = repeats.groupby(["scheme", "task", "train"], sort=False).mean(numeric_only=True).reset_index()
    result = result.drop(columns="repeat")
    suffix = "" if args.task == "all" else "_" + args.task
    predictions.to_csv(args.out / f"grouping_robustness_per_sample_predictions{suffix}.csv", index=False)
    folds.to_csv(args.out / f"grouping_robustness_per_fold_metrics{suffix}.csv", index=False)
    repeats.to_csv(args.out / f"grouping_robustness_per_repeat_metrics{suffix}.csv", index=False)
    result.to_csv(args.out / f"grouping_robustness_results{suffix}.csv", index=False)
    fold_sizes.to_csv(args.out / "grouping_robustness_fold_sizes.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(args.out / f"grouping_robustness_timing{suffix}.csv", index=False)
    manifest = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                "pandas": pd.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                "seed": SEED, "tasks": tasks, "parameters": {task: parameters[task] for task in tasks},
                "files": {str(path.relative_to(args.repo)): sha256(path) for path in
                          [data_path, classification_path, params_path]},
                "script_sha256": sha256(Path(__file__)),
                "n_collection": len(df), "n_classification": int(classification_valid.sum()),
                "n_classification_literature": int((classification_valid & ~is_feedback).sum()),
                "n_classification_feedback": int((classification_valid & is_feedback).sum()),
                "collection_rows_not_in_classification_snapshot": np.flatnonzero(~classification_valid).tolist(),
                "new_salt_count": len(new_salts),
                "note": "Historical S7b classification snapshot; S7a main analysis uses all 350 collection labels."}
    (args.out / f"grouping_robustness_manifest{suffix}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.round(8).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()


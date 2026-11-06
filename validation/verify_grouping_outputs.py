"""Check Table S7b summaries directly from saved predictions, without model fitting."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def auc(y, probability):
    positive, negative = probability[y == 1], probability[y == 0]
    return float(((positive[:, None] > negative).sum() +
                  .5 * (positive[:, None] == negative).sum()) / (len(positive) * len(negative)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    predictions = pd.read_csv(args.results / "grouping_robustness_per_sample_predictions.csv")
    saved = pd.read_csv(args.results / "grouping_robustness_results.csv")
    repeated = []
    for (scheme, task, train, repeat), block in predictions.groupby(["scheme", "task", "train", "repeat"]):
        y, prediction = block.y_true.values, block.y_pred.values
        result = {"scheme": scheme, "task": task, "train": train, "repeat": repeat, "n_pred": len(block)}
        if task == "single_peak":
            result.update(accuracy=float((y == prediction).mean()),
                          roc_auc=auc(y, block.y_proba.values),
                          recall_nonSP=float((prediction[y == 0] == 0).mean()))
        else:
            result.update(MAE=float(abs(prediction - y).mean()),
                          R2=float(1 - ((prediction - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
        repeated.append(result)
    keys = ["scheme", "task", "train"]
    recalculated = pd.DataFrame(repeated).groupby(keys).mean(numeric_only=True).drop(columns="repeat").reset_index()
    joined = saved.merge(recalculated, on=keys, suffixes=("_saved", "_recalculated"), validate="one_to_one")
    if len(joined) != 24:
        raise AssertionError(f"Expected 24 result rows, found {len(joined)}")
    differences = {}
    for metric in ["n_pred", "accuracy", "roc_auc", "recall_nonSP", "MAE", "R2"]:
        a, b = joined[metric + "_saved"].values, joined[metric + "_recalculated"].values
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise AssertionError(f"Missingness differs in {metric}")
        difference = float(np.max(np.abs(a - b), initial=0, where=~np.isnan(a)))
        differences[metric] = difference
        if difference > 1e-10:
            raise AssertionError(f"{metric}: max difference {difference}")
    print(f"Verified 24 Table S7b result rows from {len(predictions)} saved predictions.")
    print(differences)


if __name__ == "__main__":
    main()


"""Robustness of the feedback gain to the grouping scheme (Table S7b).

The M1 algorithms and hyperparameters (../M1/models/model_summary.csv) are kept
fixed. For each scheme, every held-out set of feedback records is predicted by
  A: a model trained on literature records only, and
  B: a model trained on literature records plus all remaining feedback records.

Schemes:
  S1 random 5-fold by salt combination, 20 repeats
  S2 leave-one-combination-out
  S3 leave-one-anhydrous-combination-out (hydrates of the same salt, e.g.
     CaCl2.4H2O and CaCl2.6H2O, are treated as one salt)
  S4 leave-one-new-salt-out: all feedback records containing a salt absent from
     the literature data are held out and every record containing that salt is
     removed from training set B

Usage:
    python grouping_robustness.py [--out results_rerun]
"""
import argparse
import ast
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "data" / "round1" / "round1_all_records.csv"
BEST = REPO / "M1" / "models" / "model_summary.csv"
TM, DH, SP, SOURCE = "Tm (°C)", "ΔHm (J/g)", "single peak", "source"
SALTS = ["ID1-salt", "ID2-salt", "ID3-salt"]
SEED = 42
TRAIN_A, TRAIN_B = "A literature only", "B literature + remaining feedback"

df = pd.read_csv(DATA)
feat = [c for c in df.columns if c not in SALTS + [TM, DH, SP, SOURCE]]
X = df[feat].apply(pd.to_numeric, errors="coerce").fillna(0.0)
is_new = (df[SOURCE] == "feedback").values
salt_sets = df[SALTS].apply(lambda r: frozenset(str(s) for s in r if pd.notna(s)), axis=1)
anhyd = salt_sets.apply(lambda s: frozenset(re.sub(r"·\d*H2O", "", x) for x in s))
df["combo"] = salt_sets.apply(lambda s: " + ".join(sorted(s)))
df["combo_anhyd"] = anhyd.apply(lambda s: " + ".join(sorted(s)))
old_salts = set().union(*salt_sets[~is_new])
new_only_salts = sorted(set().union(*salt_sets[is_new]) - old_salts)

best = pd.read_csv(BEST).set_index("task")
P = {k: ast.literal_eval(best.loc[k, "best_params"]) for k in ["single_peak", "Tm", "dHm"]}
TASKS = {"single_peak": SP, "Tm": TM, "dH": DH}


def make(task):
    if task == "single_peak":
        return GradientBoostingClassifier(**P["single_peak"], random_state=SEED)
    if task == "Tm":
        return RandomForestRegressor(**P["Tm"], random_state=SEED, n_jobs=-1)
    return XGBRegressor(**P["dHm"], random_state=SEED, n_jobs=-1)


def run_scheme(name, splits):
    """splits: list of (test mask, training-set-B mask) over df; training set A is always all literature records."""
    out = []
    leak = 0
    for task, col in TASKS.items():
        y = pd.to_numeric(df[col], errors="coerce")
        ok = y.notna().values
        ys, pa, pb, pa_p, pb_p = [], [], [], [], []
        for te_m, trB_m in splits:
            te = te_m & ok
            if te.sum() == 0:
                continue
            trA = (~is_new) & ok
            trB = trB_m & ok
            assert not (te & trB).any()
            if task == "Tm":
                leak += len(set(df.loc[te, "combo"]) & set(df.loc[trB & is_new, "combo"]))
            for tr, store, store_p in ((trA, pa, pa_p), (trB, pb, pb_p)):
                yt = y[tr].astype(int) if task == "single_peak" else y[tr]
                m = make(task).fit(X[tr], yt)
                store.append(m.predict(X[te]))
                if task == "single_peak":
                    store_p.append(m.predict_proba(X[te])[:, 1])
            ys.append(y[te].values)
        y_all = np.concatenate(ys)
        for tag, p, pp in ((TRAIN_A, pa, pa_p), (TRAIN_B, pb, pb_p)):
            p = np.concatenate(p)
            r = dict(scheme=name, task=task, train=tag, n_pred=len(y_all))
            if task == "single_peak":
                pp = np.concatenate(pp)
                r.update(accuracy=accuracy_score(y_all, p),
                         roc_auc=roc_auc_score(y_all, pp) if len(set(y_all)) == 2 else np.nan,
                         recall_nonSP=(p[y_all == 0] == 0).mean())
            else:
                r.update(MAE=mean_absolute_error(y_all, p), R2=r2_score(y_all, p))
            out.append(r)
    return out, leak


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "results_rerun"))
    out_dir = Path(ap.parse_args().out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, leak_report = [], {}

    rng = np.random.RandomState(SEED)
    combos = np.array(sorted(df.loc[is_new, "combo"].unique()))
    fold_stats, s1 = [], []
    for rep in range(20):
        perm = rng.permutation(combos)
        fold_of = {c: i % 5 for i, c in enumerate(perm)}
        fold = df["combo"].map(fold_of).where(is_new, -1).values
        splits = [((fold == k), (fold != k)) for k in range(5)]
        for k in range(5):
            m = fold == k
            fold_stats.append(dict(rep=rep, fold=k, n_rows=int(m.sum()),
                                   n_Tm_label=int((m & df[TM].notna()).sum()),
                                   n_subzero=int((m & (df[TM] < 0)).sum())))
        r, lk = run_scheme("S1 random 5-fold by salt combination (20 repeats)", splits)
        s1 += r
        leak_report[f"S1_rep{rep}"] = lk
    s1 = pd.DataFrame(s1)
    results.append(s1.groupby(["scheme", "task", "train"]).mean(numeric_only=True).reset_index())
    print("S1 done", flush=True)

    splits = [((df["combo"] == c).values & is_new, (df["combo"] != c).values) for c in combos]
    r, leak_report["S2"] = run_scheme("S2 leave-one-combination-out", splits)
    results.append(pd.DataFrame(r))
    print("S2 done", flush=True)

    acombos = sorted(df.loc[is_new, "combo_anhyd"].unique())
    splits = [((df["combo_anhyd"] == c).values & is_new, (df["combo_anhyd"] != c).values) for c in acombos]
    r, leak_report["S3"] = run_scheme("S3 leave-one-anhydrous-combination-out", splits)
    results.append(pd.DataFrame(r))
    print("S3 done", flush=True)

    splits = []
    for s in new_only_salts:
        has = salt_sets.apply(lambda ss: s in ss).values
        splits.append((has & is_new, ~has))
    r, leak_report["S4"] = run_scheme("S4 leave-one-new-salt-out", splits)
    results.append(pd.DataFrame(r))
    print("S4 done", flush=True)

    res = pd.concat(results, ignore_index=True)
    res.to_csv(out_dir / "grouping_robustness_results.csv", index=False)
    fs = pd.DataFrame(fold_stats)
    fs.to_csv(out_dir / "grouping_robustness_fold_sizes.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("\nleakage check (held-out combinations found in training feedback; must be 0):",
          "all 0" if all(v == 0 for v in leak_report.values()) else leak_report)
    print("\nS1 fold sizes (100 folds):")
    print(fs[["n_rows", "n_Tm_label", "n_subzero"]].describe().round(1).loc[["mean", "std", "min", "max"]].to_string())
    print(f"\nnew salts absent from the literature data ({len(new_only_salts)}): {new_only_salts}")
    print(f"salt combinations: {len(combos)}; anhydrous combinations: {len(acombos)}\n")
    cols = ["scheme", "task", "train", "n_pred", "MAE", "R2", "accuracy", "roc_auc", "recall_nonSP"]
    print(res[cols].round(3).to_string(index=False))
    print(f"\nresults saved to: {out_dir}")


if __name__ == "__main__":
    main()

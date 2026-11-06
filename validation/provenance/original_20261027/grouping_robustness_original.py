"""Test whether the experimental-feedback gains depend on the grouping scheme.

English localization of the historical source. Its historical collection and
parameter inputs are not distributed here. Input basenames and text headers
are shown in English (is_feedback=yes for feedback; task=single_peak/Tm/dH).
For the packaged inputs and runnable entry point, use
validation/grouping_robustness.py.

Check 1: Test whether held-out salt combinations occur in the training folds.
Check 2: Inspect fold sizes and the distribution of subzero-temperature samples.
Check 3: Repeat the A/B comparison using alternative grouping schemes:
   S1 Random five-fold salt-combination groups (original scheme, 20 repeats).
   S2 Leave one salt combination out (one group per split; deterministic).
   S3 Leave one anhydrous salt combination out (CaCl2·4H2O and CaCl2·6H2O share a salt identity).
   S4 Leave one new salt out (hold out all feedback formulations containing a salt absent from the literature; exclude that salt from training).
"""
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
OUT = Path(__file__).resolve().parent
# Historical inputs are not included; use validation/grouping_robustness.py for the packaged workflow.
DATA = OUT / "historical_round1_collection.csv"
BEST = OUT / "historical_best_model_summary.csv"
TM, DH, SP, NEW = "Melting temperature (Tm, ℃)", "Enthalpy value (△Hm, J·g-1)", "single peak", "is_feedback"
SALTS = ["ID1-salt", "ID2-salt", "ID3-salt"]
SEED = 42

df = pd.read_csv(DATA)
feat = [c for c in df.columns if c not in SALTS + [TM, DH, SP, NEW]]
X = df[feat].apply(pd.to_numeric, errors="coerce").fillna(0.0)
is_new = (df[NEW].astype(str).str.strip() == "yes").values
salt_sets = df[SALTS].apply(lambda r: frozenset(str(s) for s in r if pd.notna(s)), axis=1)
anhyd = salt_sets.apply(lambda s: frozenset(re.sub(r"·\d*H2O", "", x) for x in s))
df["combo"] = salt_sets.apply(lambda s: " + ".join(sorted(s)))
df["combo_anhyd"] = anhyd.apply(lambda s: " + ".join(sorted(s)))
old_salts = set().union(*salt_sets[~is_new])
new_only_salts = sorted(set().union(*salt_sets[is_new]) - old_salts)

best = pd.read_csv(BEST, encoding="utf-8-sig").set_index("task")
P = {k: ast.literal_eval(best.loc[k, "best_params"]) for k in ["single_peak", "Tm", "dH"]}
TASKS = {"single_peak": SP, "Tm": TM, "dH": DH}


def make(task):
    if task == "single_peak":
        return GradientBoostingClassifier(**P["single_peak"], random_state=SEED)
    if task == "Tm":
        return RandomForestRegressor(**P["Tm"], random_state=SEED, n_jobs=-1)
    return XGBRegressor(**P["dH"], random_state=SEED, n_jobs=-1)


def run_scheme(name, splits):
    """splits: list of (test_mask_over_df, trainB_mask_over_df). Model A always uses all literature rows."""
    out = []
    leak = 0
    for task, col in TASKS.items():
        y = pd.to_numeric(df[col], errors="coerce")
        ok = y.notna().values
        ys, pa, pb, pa_p, pb_p, tms = [], [], [], [], [], []
        for te_m, trB_m in splits:
            te = te_m & ok
            if te.sum() == 0:
                continue
            trA = (~is_new) & ok
            trB = trB_m & ok
            assert not (te & trB).any()
            if task == "Tm":  # Leakage check: held-out salt combinations must be absent from model B training feedback.
                leak += len(set(df.loc[te, "combo"]) & set(df.loc[trB & is_new, "combo"]))
            for tag, tr, store, store_p in (("A", trA, pa, pa_p), ("B", trB, pb, pb_p)):
                yt = y[tr].astype(int) if task == "single_peak" else y[tr]
                m = make(task).fit(X[tr], yt)
                store.append(m.predict(X[te]))
                if task == "single_peak":
                    store_p.append(m.predict_proba(X[te])[:, 1])
            ys.append(y[te].values)
            tms.append(df.loc[te, TM].values)
        y_all = np.concatenate(ys)
        for tag, p, pp in (("A literature only", pa, pa_p), ("B literature + remaining feedback", pb, pb_p)):
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


results, leak_report = [], {}

# S1 Random five-fold salt-combination groups (original scheme), 20 repeats.
rng = np.random.RandomState(SEED)
combos = np.array(sorted(df.loc[is_new, "combo"].unique()))
fold_stats = []
s1 = []
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

# S2 leave-one-combination-out
splits = [((df["combo"] == c).values & is_new, (df["combo"] != c).values) for c in combos]
r, leak_report["S2"] = run_scheme("S2 leave-one-combination-out", splits)
results.append(pd.DataFrame(r))
print("S2 done", flush=True)

# S3 Leave-one-anhydrous-combination-out
acombos = sorted(df.loc[is_new, "combo_anhyd"].unique())
splits = [((df["combo_anhyd"] == c).values & is_new, (df["combo_anhyd"] != c).values) for c in acombos]
r, leak_report["S3"] = run_scheme("S3 Leave-one-anhydrous-combination-out", splits)
results.append(pd.DataFrame(r))
print("S3 done", flush=True)

# S4 Leave-one-new-salt-out: test feedback containing the salt; exclude every record containing it from model B training.
splits = []
for s in new_only_salts:
    has = salt_sets.apply(lambda ss: s in ss).values
    splits.append((has & is_new, ~has))
r, leak_report["S4"] = run_scheme("S4 leave-one-new-salt-out", splits)
results.append(pd.DataFrame(r))
print("S4 done", flush=True)

res = pd.concat(results, ignore_index=True)
res.to_csv(OUT / "grouping_robustness_original_results.csv", index=False, encoding="utf-8-sig")
fs = pd.DataFrame(fold_stats)
fs.to_csv(OUT / "grouping_robustness_s1_fold_counts.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print("\nLeakage check (held-out combinations found in training feedback; all counts should be zero):",
      "All counts are zero" if all(v == 0 for v in leak_report.values()) else leak_report)
print("\nS1 fold sample distribution (100 folds):")
print(fs[["n_rows", "n_Tm_label", "n_subzero"]].describe().round(1).loc[["mean", "std", "min", "max"]].to_string())
print(f"\nNew salts (absent from literature): {len(new_only_salts)} salts: {new_only_salts}")
print(f"Salt combinations: {len(combos)}; anhydrous combinations: {len(acombos)}\n")
cols = ["scheme", "task", "train", "n_pred", "MAE", "R2", "accuracy", "roc_auc", "recall_nonSP"]
print(res[cols].round(3).to_string(index=False))

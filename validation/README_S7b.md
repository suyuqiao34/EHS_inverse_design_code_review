# Fixed-hyperparameter grouped sensitivity analyses

## Inputs

- `data/round1/round1_all_records.csv`: combined formulation collection.
- `data/round1/round1_single_peak.csv`: archived classification task input.
- `M1/models/model_summary.csv`: algorithms and fixed hyperparameters.

The classification input is matched to collection rows using the saved model
features and labels. The resulting mask is recorded with the outputs. Regression
uses the available temperature and enthalpy labels. Input snapshots, source scripts
and mappings are retained under `validation/provenance`.

## Run

From the repository root:

```bash
python validation/grouping_robustness.py --out validation/results_rerun
python validation/verify_grouping_outputs.py --results validation/results
```

Use `--task single_peak`, `--task Tm` or `--task dH` to run one task.

## Grouping and aggregation

S1 uses five salt-combination folds with 20 repetitions. Metrics are computed
after joining all held-out predictions within a repetition, then averaged over
repetitions. S2 leaves out one salt combination, S3 one anhydrous salt combination,
and S4 one salt absent from the literature input. For these leave-one-group-out
analyses, metrics are computed from the joined held-out predictions.

Model A uses the eligible literature inputs. Model B also uses training-fold
feedback inputs. In S2 and S3, matching literature groups are excluded from model B.
The seed is 42. Fixed-hyperparameter sensitivity analyses and the main nested
validation are separate workflows.

## Outputs

`validation/results/` provides summary tables, per-sample predictions, per-fold
metrics and per-repeat metrics. The verification script independently recalculates
the saved summaries from the predictions. SHA256 values and input mappings
identify the files used for each archived run.

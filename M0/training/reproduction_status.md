# Analysis inputs and replay methods

## Inputs

- `data/round0/round0_*.csv`: task files read by final-fit scripts.
- `data/round0/evaluation/round0_*.csv`: evaluation inputs in their saved row order.
- `data/round0/evaluation/*_row_order_mapping.csv`: mappings between evaluation
  and final-fit rows.
- `data/round0/manuscript_baseline/round0_*.csv`: composition-weighted baseline inputs.
- `data/round1/round1_all_records.csv`: combined collection used for grouped validation.
- `data/round1/round1_*.csv`: task-specific inputs for model training and evaluation.

## Regression summaries

`M0/training/results` contains summaries aggregated from saved regression runs.
Each summary reports the arithmetic mean and sample standard deviation (ddof = 1).
Run-level metrics, original source code and replay settings are preserved in
`benchmark_archive`. Preserve the saved input row order when repeating seeded splits.

## Baselines

`compare_composition_baselines.py` computes direct composition-weighted estimates
and a univariate linear calibration. The calibration is fitted only on the
training subset of each 80/20 partition. The saved analyses use split seeds 0–99.
Outputs include split membership, test predictions, per-split metrics and summary
statistics. Select the input snapshot with `--data-dir` and a rerun destination
with `--out`.

## Linear-family replay

The source reconstruction searches standardized OLS, Ridge, Lasso and ElasticNet.
Numerically recovered replay settings are stored separately from original source
files and run metrics. The temperature replay uses Ridge and the enthalpy replay
uses Lasso. Configuration files and numerical checks are under
`benchmark_archive/replayed`.

## Classification and grouped validation

Classification comparison outputs and a separately labelled stratified recheck
are preserved under `benchmark_archive`. Each output retains its analysis label.

For the fixed-hyperparameter grouped sensitivity analyses, use
`validation/grouping_robustness.py`. Its task-input mapping, per-sample predictions,
per-fold metrics and summaries are included in the archive. The saved predictions
can be independently aggregated with `validation/verify_grouping_outputs.py`.
The main nested-validation workflow has its own entry point,
`validation/grouped_cv_nested.py`.

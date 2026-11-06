# M0 benchmark instructions

The archive contains the original evaluation inputs, saved 100-run regression
metrics, original source code and replay checks. The public regression tables
are aggregated from the saved runs and include the later standardized linear-family
reconstruction.

## Replay commands

Run the following commands from the repository root.

Aggregate the saved regression runs without fitting models:

```bash
python M0/training/benchmark_archive/reproduce_m0_benchmark.py aggregate
```

Replay the recovered GBDT temperature evaluation:

```bash
python M0/training/benchmark_archive/reproduce_m0_benchmark.py replay --task tm --model GBDT
```

Reproduce the submitted baseline snapshot:

```bash
python M0/training/compare_composition_baselines.py
```

A new Optuna search can be run with `M0/training/compare_algorithms.py`.
Portable reruns write to `results_rerun/`. Original-code snapshots are retained
for reference; use the portable entry points above for execution.

## Data and statistical scope

Seeded splitting requires the original evaluation row order. Evaluation files,
final-fit files and explicit row mappings are provided under `data/round0`.
Input files, output definitions and replay configurations are described in
[Analysis inputs and replay methods](reproduction_status.md).

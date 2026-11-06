# Multi-agent inverse design of eutectic hydrated salts

Code, data and trained models accompanying the manuscript

> **Integrating chemical reasoning and property prediction for multi-agent discovery of eutectic hydrated salts**
> Y. Su, Z. Liu, Z. Zhang, H. Gao, Z. Sun, X. Jin, Y. Zhang, Z. Huang, L. Yu, J. Ding, J. Li, X. Wang, G. Wang

The workflow combines LLM-based chemical-compatibility agents (DashScope `qwen-plus`) with a hierarchical
property predictor (single-peak classifier, then melting temperature *T*<sub>m</sub> and melting enthalpy
Δ*H*<sub>m</sub> regressors) to propose binary and ternary eutectic hydrated salts that meet user-defined thermal
targets. It was run in two design–test–learn rounds:

| Round | Predictive module | Training data | Design system in this repository |
|---|---|---|---|
| 0 | **M0**: XGBoost classifier, GBDT (*T*<sub>m</sub>), SVR (Δ*H*<sub>m</sub>) | literature records (`data/round0`) | `M0/design_system` |
| 1 | **M1**: GBDT classifier, random forest (*T*<sub>m</sub>), XGBoost (Δ*H*<sub>m</sub>) | literature records + 171 feedback experiments (`data/round1`) | `M1/design_system` |

## Repository structure

```
data/                          datasets (see data/README.md)
  pure_hydrated_salts_dataset.csv   62-entry pure hydrated-salt library (final version)
  round0/                      Round-0 training data of M0
  round1/                      Round-1 dataset (350 records) and the M1 training files
M0/
  models/                      frozen M0 models + MD5SUMS.txt
  training/                    algorithm comparison and final refit of M0; original results
  design_system/               multi-agent design system used in Round 0 (loads ../models)
M1/
  models/                      frozen M1 models, model_summary.csv (hyperparameters) + MD5SUMS.txt
  training/                    algorithm comparison and final refit of M1; original results
  design_system/               multi-agent design system used in Round 1 (loads ../models)
validation/                    grouped validation of the feedback gain; results/ holds the outputs
figure/                        plotting scripts for Fig. 7a, b, c and g; data/ holds the Fig. 7g summary
```

The two design systems share the same agent architecture. The M1 system differs from the M0 system only in
(i) the predictive models it loads and (ii) parsing of signed and range targets (for example, sub-zero
*T*<sub>m</sub> targets) in `multi_agent_selector.py`. Each system carries the salt library and the structured
salt metadata (`data/*_structured.json`) that it used; the M1 copies contain corrected entries for
Co(NO3)2·6H2O, LiClO3·3H2O and Na2S2O3·5H2O (see `data/README.md`).

## System requirements

* Tested on Windows 10/11 (64-bit) with Python 3.11.7; the code uses only cross-platform packages.
* Dependencies (pinned in `requirements.txt`): scikit-learn 1.5.0, XGBoost 2.0.3, NumPy 1.26.4, pandas 2.1.4,
  joblib 1.5.2, Flask 3.1.2, DashScope 1.25.0, Optuna 4.0.0, SciPy 1.14.1, statsmodels 0.14.0, Matplotlib 3.8.0.
  The pickled models require the pinned scikit-learn and XGBoost versions.
* No special hardware is required. The design systems need internet access and a DashScope API key for the LLM agents.
* Figures use the Times New Roman font; Matplotlib falls back to a default font if it is not installed.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Installation takes a few minutes on a standard desktop computer.

## Running a design system (demo)

The LLM agents call the DashScope API. The key is read from the environment variable `DASHSCOPE_API_KEY`
and is not stored in the repository.

```powershell
$env:DASHSCOPE_API_KEY = "<your DashScope API key>"     # Command Prompt: set DASHSCOPE_API_KEY=<key>
cd M1/design_system                                    # or M0/design_system
python app.py                                          # or double-click start_system.bat on Windows
```

Open <http://127.0.0.1:5000>, upload `pure_hydrated_salts_dataset.csv` from the same folder, enter the target
*T*<sub>m</sub> (°C) and Δ*H*<sub>m</sub> (J g<sup>-1</sup>), for example `14` and `180`, and start the task.
Targets may be single values, ranges (`20-30`) or thresholds (`>150`); the M1 system also accepts negative
targets (`-10`). A run takes from several minutes to tens of minutes, mainly depending on DashScope latency.

Outputs are written to the design-system folder and can be downloaded from the web page:

| File | Content |
|---|---|
| `agent_recommended_results.csv` | ranked candidates: `salt_a/b/c`, `ratio_a/b/c` (mass fractions), `tm_pred`, `dh_pred`, `single_peak_proba`, compliance level and score |
| `results_{perfect,good,acceptable}_compliance.csv` | candidates grouped by compliance with the targets |
| `salt_combinations_formatted.csv` | model-input features of all evaluated combinations |

`example_output/` in each system contains the files produced for the demo target (14 °C, 180 J g<sup>-1</sup>).
Because the agents call an external LLM, the candidate list and its order can change between runs; the
property predictions for a given formulation are deterministic. Without an API key the web service starts,
but the LLM agents are unavailable.

Any salt library with the columns of `pure_hydrated_salts_dataset.csv` (`ID` followed by the 12 descriptors)
can be uploaded. Salts without a matching `data/<ID>_structured.json` file lack the ion information used by
the compatibility rules.

## Reproducing the models

```bash
cd M1/training
python refit_final_models.py               # seconds: refits the three M1 models from model_summary.csv
python train_single_peak.py                # full Optuna comparison of six classifiers
python train_regression.py --task tm       # full comparison for Tm (use --task dh for dHm)

cd ../../M0/training
python fit_final_models.py                 # refits the three M0 models
python compare_algorithms.py --task all    # a new Optuna search on the recovered evaluation row order
python compare_composition_baselines.py    # archived composition-weighted baseline analysis
python benchmark_archive/reproduce_m0_benchmark.py aggregate  # aggregate original regression runs
```

`refit_final_models.py` reproduces the frozen M1 models exactly (maximum prediction difference 0). For M0, the
SVR refit is exact. The M0 XGBoost and GBDT models were originally fitted without a fixed random seed, so a refit
agrees only approximately (maximum differences on the training records of 0.05 in single-peak probability and 0.52 °C in *T*<sub>m</sub>). The frozen files in
`M0/models` are the models used in the paper. The Round-0 algorithm comparison did not fix the Optuna sampler
seed, so a rerun gives similar but not identical hyperparameters. The original comparison tables are provided in
`M0/training/results` and `M1/training/results`. The M0 regression tables aggregate the archived 100-run results, including the
linear-family reconstruction. See [M0 benchmark instructions](M0/training/README_benchmark.md)
for replay commands and [analysis snapshots and reproducibility](M0/training/reproduction_status.md)
for input files, statistical definitions and replay configurations.
A full comparison can take several hours on a desktop CPU. Documented portable
rerun commands write to `results_rerun/` or `refit/`, leaving the supplied results unchanged.

## Grouped validation (Table S7, Supplementary Section S9, Fig. 7b,c)

```bash
cd validation
python m0_on_feedback.py          # seconds: frozen M0 on all 171 feedback formulations (Section S9.2)
python grouping_robustness.py --out results_rerun   # about 10 min: grouping schemes S1-S4 with fixed M1 hyperparameters (Table S7b)
python grouped_cv_nested.py       # several hours: grouped nested validation, 5 folds x 3 repeats (Table S7a)
```

The 38 feedback salt combinations are the outer grouping unit. Held-out feedback
formulations form the test set; literature records remain in the main nested training folds.
The additional sensitivity analysis evaluates the 36 feedback combinations absent from the literature data.
The sensitivity analyses read the archived classification task input; the
main nested validation reads the combined analysis collection. See
`validation/README_S7b.md` for inputs and execution details. The scripts verify the MD5 checksums of the M0 models before use.
The outputs in `validation/results/` include per-sample, per-fold and per-repeat
predictions for the reproduced Table S7b. Verify them without training using
`python validation/verify_grouping_outputs.py --results validation/results` from the repository root.
Source scripts, input mappings and archived runs are retained under
`validation/provenance`.

## Figure 7

```bash
cd figure
python plot_fig7a.py    # Tm-dHm distribution of single-peak literature and feedback records
python plot_fig7b.py    # paired fold-wise Tm MAE (reads validation/results)
python plot_fig7c.py    # absolute Tm error versus measured Tm (reads validation/results)
python plot_fig7g.py --m0-dir <M0 filtered files> --m1-dir <M1 filtered files>
```

Figures are written to `figure/output/`. Fig. 7g requires the compatibility-filtered full-space prediction
files (about 1.4 × 10<sup>8</sup> predicted formulations in total). They are not included because of their size and
are available from the corresponding authors on request. The statistics of the plotted layers are provided in
`figure/data/fig7g_landscape_summary.csv`.

## Model files and checksums

| Model | Task | File | MD5 |
|---|---|---|---|
| M0 | single peak | `M0/models/single_peak_XGBoost_best_model_from_roc_redraw.pkl` | `d353c97d2863a72e5f036d9e0a8e8dd3` |
| M0 | *T*<sub>m</sub> | `M0/models/tm_gbdt_best_model.pkl` | `9d93ddcc6f455d1dfe464ddd68ddd3a9` |
| M0 | Δ*H*<sub>m</sub> | `M0/models/dh_svm_best_model.pkl` | `5bc8d7ce665669d955c028cbe9f5b9b6` |
| M1 | single peak | `M1/models/GBDT_best_model.pkl` | `af70fe8f0b979e8aee93ed3bec3a9931` |
| M1 | *T*<sub>m</sub> | `M1/models/RF_best_model.pkl` | `5d401724367be8dabef21d3dac742689` |
| M1 | Δ*H*<sub>m</sub> | `M1/models/XGBoost_best_model.pkl` | `626600bbde1b913918db6177e1381a16` |

All models take the 39 features listed in `data/README.md`, in the order stored in `model.feature_names_in_`.
Load them with `joblib.load` under the pinned package versions. Only load pickle files from trusted sources.

## License

Code is released under the MIT License (`LICENSE`). Data and trained models are released under the
Creative Commons Attribution 4.0 International License (`LICENSE-DATA`).

## Citation

If you use this code or data, please cite the manuscript above.

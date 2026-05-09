# Eutectic Hydrated Salt Screening System

## Overview

This repository contains the code and example data for a web-based screening system designed to identify candidate eutectic hydrated salt combinations under user-specified thermal targets. The system accepts a CSV file containing salt descriptors, parses target melting point (`Tm`) and enthalpy (`ΔHm`) constraints, performs multi-stage candidate selection and model-based prediction, and exports ranked result files for further analysis.

This public repository has been prepared to support manuscript review and software reproducibility. It retains the executable workflow required for data upload, screening, ranking, and result export.

## Repository Structure

- `app.py`: Flask application entry point and web routes.
- `multi_agent_selector.py`: core multi-agent screening workflow.
- `salt_combination_generator.py`: feature construction, model loading, prediction utilities, and compatibility-related helper functions.
- `strategy_selector.py`: rule-based candidate pre-screening strategy.
- `apply_compatibility_bonus_standalone.py`: post-processing module for compatibility-weighted reranking.
- `templates/index.html`: web interface.
- `data/`: structured metadata for hydrated salts (`*_structured.json`).
- `pure_hydrated_salts_dataset.csv`: default/example input dataset.
- `eutectic_data_single_peak.csv`: eutectic dataset with single-peak labels.
- `eutectic_data_tm.csv`: eutectic dataset with melting-temperature (`Tm`) values.
- `eutectic_data_dh.csv`: eutectic dataset with enthalpy (`ΔHm`) values.
- `agent_recommended_results.csv`: example output file.
- `best model/`: trained prediction models used by the current screening workflow.
- `picture ipynb/Important picture.ipynb`: notebook for visualizing the input data distribution and the screening results for the `14` degree target.
- `picture ipynb/`: CSV files used by the visualization notebook.
- `requirements.txt`: pinned Python dependencies.
- `requirements-visualization.txt`: additional dependencies for running the visualization notebook.
- `.env.example`: environment-variable template for API configuration.
- `start_system.bat`: Windows launcher for convenient local reproduction.

## System Requirements

- Operating system tested by the authors: Windows 10/11
- Python tested by the authors: Python 3.11.7
- Recommended Python version: Python 3.9 or later
- Network access is required for the DashScope/Qwen API calls used by the multi-agent screening workflow.
- No non-standard hardware is required for the provided demo.

## Installation

From the project root directory, create and activate a Python environment, then install the pinned dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Typical installation time on a normal desktop computer is expected to be several minutes, depending on network speed and Python package cache status.

## External API Configuration

The multi-agent workflow calls DashScope (`qwen-plus`). Set the API key as an environment variable before running the web service.

PowerShell:

```powershell
$env:DASHSCOPE_API_KEY="your_dashscope_api_key_here"
```

Command Prompt:

```bat
set DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

Do not put a real API key into the source code. `.env.example` is provided only as a template.

## Included Trained Models

The trained models used by the current screening workflow are included in `best model/`:

- `best model/single_peak_XGBoost_best_model_from_roc_redraw.pkl`: single-peak classification model.
- `best model/tm_gbdt_best_model.pkl`: melting-temperature (`Tm`) regression model.
- `best model/dh_svm_best_model.pkl`: enthalpy (`ΔHm`) regression model.

## Demo Input

Use the included example input file:

- `pure_hydrated_salts_dataset.csv`

This public repository contains a representative processed pure hydrated salt dataset sufficient for running and testing the demo workflow. The full raw pure hydrated salt source dataset and large-scale prediction-space files are not included in this repository and are available from the corresponding author upon reasonable request.

Suggested demo target values:

- Target melting point (`Tm`): `14`
- Target enthalpy (`ΔHm`): `180`

## Running the Demo

### Option 1: Windows Launcher

Run:

```bat
start_system.bat
```

### Option 2: Command Line

Start the local web service:

```bash
python app.py
```

Open the web interface:

- `http://localhost:5000`

In the web interface:

1. Upload `pure_hydrated_salts_dataset.csv`.
2. Enter `14` as the target melting point.
3. Enter `180` as the target enthalpy.
4. Start the screening task.
5. Download the generated result files after the task completes.

Typical demo runtime on a normal desktop computer is expected to be within tens of minutes, depending on network latency for DashScope API calls and the selected screening target.

## Expected Output

The main output files are generated in the project root directory after a screening run:

- `agent_recommended_results.csv`
- `salt_combinations_formatted.csv`
- `results_perfect_compliance.csv`
- `results_good_compliance.csv`
- `results_acceptable_compliance.csv`

This repository includes `agent_recommended_results.csv` as an example output file and schema reference. Important columns include:

- `salt_a`, `salt_b`, `salt_c`
- `ratio_a`, `ratio_b`, `ratio_c`
- `tm_pred`, `dh_pred`
- `single_peak_proba`, `is_single_peak`
- `compliance_level`, `compliance_desc`
- `score`

Because the workflow includes external LLM calls, exact candidate ordering can vary if the upstream model behavior changes. The trained prediction models and rule-based post-processing are included in this package.

## Data and Result Visualizations

The repository includes a Jupyter notebook for visualizing the included data and the screening results generated around the demo target:

- `picture ipynb/Important picture.ipynb`

Demo target values used by the visualization workflow:

- Target melting point (`Tm`): `14`
- Target enthalpy (`ΔHm`): `180`

Install the additional notebook/plotting dependencies before running the visualization workflow:

```bash
pip install -r requirements-visualization.txt
```

The notebook uses the CSV files provided in `picture ipynb/` and exports PNG/PDF figures showing the data distribution, model-result summaries, and the recommended candidate space. Run the notebook from the `picture ipynb/` directory so that the relative input paths resolve correctly.

## Running on Custom Data

Prepare a CSV file with one salt identifier column and the required descriptor columns. The example file `pure_hydrated_salts_dataset.csv` shows the expected column structure:

- `ID`
- `Tm (°C)`
- `ΔHm (J/g)`
- `Molecular Weight`
- `Hydrogen Bond Donor Count`
- `Hydrogen Bond Acceptor Count`
- `Topological Polar Surface Area`
- `Heavy Atom Count`
- `Complexity`
- `Covalently-Bonded Unit Count`
- `Hydration Number`
- `Elem_H`
- `Elem_O`

UTF-8 encoded CSV files are preferred. If UTF-8 decoding fails, the application falls back to GBK.

## Web Routes

- `GET /`: main page.
- `POST /upload_run`: upload data and start a screening task.
- `GET /status`: query task status, progress, and recent logs.
- `GET /download/<which>`: download generated result files.

Supported values for `<which>`:

- `recommend`
- `features`
- `perfect`
- `good`
- `acceptable`

## Notes for Manuscript Review

- The repository includes the source code, example input data, example output file, and trained model files required to run and test the current screening workflow.
- Structured metadata in `data/` are required for compatibility-aware screening and reranking.
- The post-processing module applies an additional compatibility-weighted ranking step after the main screening stage.
- This version has been cleaned to remove deprecated testing code and nonessential legacy entry points.

## License and Availability

This repository is provided to support manuscript review and software reproducibility. Redistribution or reuse outside the peer-review process requires permission from the authors unless a separate public license is provided.

## Disclaimer

This repository is intended to accompany an academic manuscript. Sensitive credentials should be externalized and managed through environment variables or other secure configuration mechanisms.

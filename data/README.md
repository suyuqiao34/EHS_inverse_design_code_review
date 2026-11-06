# Data

All tables are UTF-8 CSV files. Salt formulas use the middle dot for water of crystallisation (for example
`CaCl2·6H2O`). Mass fractions are given as fractions of 1.

## pure_hydrated_salts_dataset.csv

The 62-entry pure hydrated-salt library (final version, used in Round 1). One row per salt:

| Column | Description |
|---|---|
| `ID` | salt formula |
| `Tm (°C)`, `ΔHm (J/g)` | melting temperature and melting enthalpy of the pure salt |
| `Molecular Weight`, `Hydrogen Bond Donor Count`, `Hydrogen Bond Acceptor Count`, `Topological Polar Surface Area`, `Heavy Atom Count`, `Complexity`, `Covalently-Bonded Unit Count` | molecular descriptors from PubChem |
| `Hydration Number` | number of water molecules per formula unit |
| `Elem_H`, `Elem_O` | number of H and O atoms per formula unit |

`M0/design_system/pure_hydrated_salts_dataset.csv` is the version used in Round 0. It differs in two entries:
cobalt nitrate hexahydrate is listed as `Co(NO3)3·6H2O` instead of `Co(NO3)2·6H2O`, and the molecular
descriptors of `LiClO3·3H2O` were later corrected (molecular weight 160.5 to 144.5). The structured metadata of
`Na2S2O3·5H2O` were also updated in Round 1 (`Tm` 50 to 48.5 °C, `ΔHm` 226.6 to 197.5 J/g). The
`data/<salt>_structured.json` files in each design system (one per salt) hold the salt metadata used by the
agents of that system: the PubChem record, thermal properties, molecular descriptors, identifiers (IUPAC name,
SMILES, InChI) and ionic features (cation, cation radius, anion, anion charge, hydration number).

## Formulation records

Every formulation record uses the same 39 model features:

* `k-<descriptor>` for component k = 1, 2, 3: the 12 pure-salt properties listed above (`Tm (°C)`, `ΔHm (J/g)`
  and the ten descriptors) of the k-th constituent salt; all zero for the third component of a binary formulation;
* `ID1`, `ID2`, `ID3`: mass fractions of the three components (`ID3` = 0 for binary formulations).

The columns `ID1-salt`, `ID2-salt` and `ID3-salt` give the constituent salts (empty for an absent third
component); they are not model inputs. Targets:

| Column | Description |
|---|---|
| `single peak` | 1, a single melting endotherm; 0, multi-peak or multistep behaviour |
| `Tm (°C)` | melting temperature of the formulation (for the feedback experiments, the DSC onset temperature) |
| `ΔHm (J/g)` | melting enthalpy of the formulation |

Note that `Tm (°C)` and `ΔHm (J/g)` without a component prefix are formulation targets, whereas
`k-Tm (°C)` and `k-ΔHm (J/g)` are properties of the k-th constituent salt.

### round0/ (M0 training data)

| File | Target |
|---|---|
| `round0_single_peak.csv` | `single peak` |
| `round0_tm.csv` | `Tm (°C)` |
| `round0_dh.csv` | `ΔHm (J/g)` |

The task files are used by the final-fit scripts. Evaluation inputs are preserved
under `round0/evaluation/`; row mappings connect the evaluation and final-fit
files. Baseline inputs are under `round0/manuscript_baseline/`. Use the input
folder specified by each analysis script and preserve its row order for seeded splits.


### round1/ (Round-1 dataset and M1 training data)

`round1_all_records.csv` contains the 350 de-duplicated records of Round 1 (Supplementary Data 1), with
`source` = `literature` (179 records) or `feedback` (171 formulations measured in this work):

| | Literature | Feedback |
|---|---|---|
| single-peak label | 179 (90 single-peak) | 171 (66 single-peak) |
| `Tm (°C)` | 172 | 66 |
| `ΔHm (J/g)` | 170 | 66 |

The task files on which the frozen M1 models were trained (row order preserved):

| File | Target |
|---|---|
| `round1_single_peak.csv` | `single peak` |
| `round1_tm.csv` | `Tm (°C)` |
| `round1_dh.csv` | `ΔHm (J/g)` |

The model-training scripts read the task-specific files. The main grouped
validation reads `round1_all_records.csv`. The fixed-hyperparameter sensitivity
analyses also use the archived classification task file to construct an explicit
label mask. Input mappings and run outputs are provided under `validation/`.

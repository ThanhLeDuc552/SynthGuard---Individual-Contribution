## 1. Synthetic data quality (tier 2 only) — `ml_ids_detect.py`

Dataset roles:
- `--train`: training set for the attack-type classifier (e.g. the multiclass file, real + SMOTE).
- `--test`: traffic flows used **only for testing** (e.g. the original attack file). Never trained on.
- `--synthetic`: generated data (CTGAN / TVAE ...), added on top of `--train`.

```bash
python ml_ids_detect.py --train <TRAIN.csv> --test <TRAFFIC.csv> --synthetic <SYNTHETIC_FILE_OR_FOLDER> [<MORE_FILES_OR_FOLDERS> ...] --out-dir <OUTPUT_FOLDER>
```

Optional flags:

```bash
python ml_ids_detect.py \
    --train <TRAIN.csv> \
    --test <TRAFFIC.csv> \
    --synthetic <SYNTHETIC_FILE_OR_FOLDER> [<MORE_FILES_OR_FOLDERS> ...] \
    --out-dir <OUTPUT_FOLDER> \
    [--attack-col <ATTACK_COLUMN>] \
    [--label-col <LABEL_COLUMN>] \
    [--trees <N_TREES>] \
    [--jobs <N_CPU_CORES>] \
    [--gpu]
```

> PowerShell: put the command on one line, or replace `\` with a backtick `` ` `` at the end of each line.
> On Windows you can also run `.\run_compare.ps1` (defaults below; `-Cpu` for CPU only, `-Trees 100` for a quick check).

| Placeholder | Meaning | Default |
|---|---|---|
| `<TRAIN.csv>` | Training set CSV (BASE) | required |
| `<TRAFFIC.csv>` | Traffic flow CSV used only for testing | required |
| `<SYNTHETIC_FILE_OR_FOLDER>` | One or more synthetic CSV files or folders (every `*.csv` inside is used) | required |
| `<OUTPUT_FOLDER>` | Where results are written | `synthetic_report` |
| `<ATTACK_COLUMN>` | Column with the attack type name | `Attack` |
| `<LABEL_COLUMN>` | Binary label column (excluded from features) | `Label` |
| `<N_TREES>` | XGBoost trees per model (lower = faster, e.g. `100` for a quick check) | `300` |
| `<N_CPU_CORES>` | CPU cores used by XGBoost (`-1` = all) | half of the cores |
| `--gpu` | Train XGBoost on an NVIDIA GPU (needs CUDA-enabled xgboost) | off |

Example (this project):

```bash
python ml_ids_detect.py --train IDS_Data/IDS-multiclass-classification/unsw_nb15_attacks_smote_10k.csv --test "Traffic Flow/unsw_nb15_attacks_original.csv" --synthetic "Synthetic Data (for test)/CTGAN/synthetic_ctgan_unsw_nb15_smote_10k_smote10k_v2.csv" "Synthetic Data (for test)/TVAE/synthetic_tvae_perclass_smote_10k.csv" --out-dir synthetic_report_v3 --gpu
```

Output in `<OUTPUT_FOLDER>`: `summary.csv`, `per_class_f1_TSTR.csv`, `per_class_f1_TRS.csv`, `report.html` (open in a browser).
In the report, the `BASE` row is the model trained on `--train` only; `aug_gain_*` shows whether adding synthetic data helps.

Requirements: `pip install numpy pandas scipy scikit-learn xgboost`

## 2. Two-tier IDS test (tier 1 benign vs attack + tier 2 attack type) — `ids_pipeline.py`

Tier 1 is trained on the binary train fold and is the same for every configuration; only the tier-2 training data
changes per `--config`. The binary test fold (benign + attack) is used only for testing.

```bash
python ids_pipeline.py     --train "<BINARY_DIR>/train/<FILE>_fold_{fold}.csv"     --test  "<BINARY_DIR>/test/<FILE>_fold_{fold}.csv"     --folds <K> [<K> ...]     --config "<NAME>=<FILE.csv>[,<FILE.csv>...]" [--config ...]     [--baseline <NAME>] [--trees <N_TREES>] [--gpu]     --out-dir <OUTPUT_FOLDER>
```

| Placeholder | Meaning | Default |
|---|---|---|
| `{fold}` | Replaced by each number in `--folds` | — |
| `<NAME>=<FILE.csv>,...` | One tier-2 configuration; files are concatenated. Repeat `--config` per configuration | required |
| `<NAME>` in `--baseline` | Configuration the `delta_*` columns compare against | first `--config` |
| `--max-fpr` / `--anomaly-fpr` | False alarm budget of the supervised / anomaly branch | `0.01` / `0.005` |
| `--min-confidence` | Below this tier-2 confidence the flow is labelled `Unknown` | `0.4` |
| `--rare-share` | Attack types below this share of attack flows count as rare | `0.05` |

On Windows: `.un_ids.ps1` runs the four configurations of `FLow test.md` (A original, B multiclass,
C multiclass + CTGAN, D multiclass + TVAE; baseline B). Options: `-Folds 1,2,3,4,5`, `-Trees 100`, `-Cpu`, `-OutDir <dir>`.

Output in `<OUTPUT_FOLDER>`: `report.html` and CSV tables — `summary_configs.csv`, `tier1_detection.csv`,
`tier1_per_attack.csv`, `tier2_f1_per_class.csv`, `end_to_end_recall_per_class.csv`, plus `per_fold_*.csv`.

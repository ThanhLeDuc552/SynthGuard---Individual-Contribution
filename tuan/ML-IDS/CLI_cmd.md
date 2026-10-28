````markdown
## Run

```bash
python ml_ids_detect.py --real <REAL_DATA.csv> --synthetic <SYNTHETIC_FILE_OR_FOLDER> [<MORE_FILES_OR_FOLDERS> ...] --out-dir <OUTPUT_FOLDER>
```

Optional flags:

```bash
python ml_ids_detect.py \
    --real <REAL_DATA.csv> \
    --synthetic <SYNTHETIC_FILE_OR_FOLDER> [<MORE_FILES_OR_FOLDERS> ...] \
    --out-dir <OUTPUT_FOLDER> \
    [--test <REAL_TEST.csv>] \
    [--attack-col <ATTACK_COLUMN>] \
    [--label-col <LABEL_COLUMN>] \
    [--trees <N_TREES>] \
    [--jobs <N_CPU_CORES>] \
    [--gpu]
```

> PowerShell: put the command on one line, or replace `\` with a backtick `` ` `` at the end of each line.

| Placeholder | Meaning | Default |
|---|---|---|
| `<REAL_DATA.csv>` | Real data CSV (the data the generators were trained on) | required |
| `<SYNTHETIC_FILE_OR_FOLDER>` | One or more synthetic CSV files or folders (every `*.csv` inside is used) | required |
| `<OUTPUT_FOLDER>` | Where results are written | `synthetic_report` |
| `<REAL_TEST.csv>` | Separate real test set. If omitted, real data is split 80/20 automatically | auto split |
| `<ATTACK_COLUMN>` | Column with the attack type name | `Attack` |
| `<LABEL_COLUMN>` | Binary label column (excluded from features) | `Label` |
| `<N_TREES>` | XGBoost trees per model (lower = faster, e.g. `100` for a quick check) | `300` |
| `<N_CPU_CORES>` | CPU cores used by XGBoost (`-1` = all) | half of the cores |
| `--gpu` | Train XGBoost on an NVIDIA GPU (needs CUDA-enabled xgboost) | off |

Example:

```bash
python ml_ids_detect.py --real data/real.csv --synthetic synthetic/CTGAN synthetic/TVAE --out-dir synthetic_report --gpu
```

Output in `<OUTPUT_FOLDER>`: `summary.csv`, `per_class_f1_TSTR.csv`, `per_class_f1_TRS.csv`, `report.html` (open in a browser).

Requirements: `pip install numpy pandas scipy scikit-learn xgboost`
````

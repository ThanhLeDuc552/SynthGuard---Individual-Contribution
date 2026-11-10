# Do Minh Thanh – TVAE generator (AI)

Trains one TVAE per attack class and generates synthetic attack records. Data and outputs are not stored here.

## Contents

| Path | What it does |
|---|---|
| `notebooks/tvae_final_perclass_v2.ipynb` | Trains one TVAE per attack class on `final_datasets_v2`, generates the synthetic file and checks its quality. |

## Setup

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

## Running

Unzip `final_datasets_v2.zip` into `do_minh_thanh/datasets/final_datasets_v2/`, then run all cells. Outputs go to `outputs/tvae_final_v2/`:

- `synthetic_tvae_perclass_smote_10k.csv` – synthetic data, same 49 columns as the input
- `models/tvae_<class>.pkl` – 9 trained models
- `manifest_<run_name>.json` – settings, seeds, checksums and quality checks
- `loss_per_class.csv` – training loss per class
- `optuna_tvae.db`, `optuna_trials.csv` – tuning trials (a stopped run resumes)

Settings are in the `CFG` cell. `n_trials=1` uses the settings found by an earlier 15-trial tuning run; set it higher to tune again.

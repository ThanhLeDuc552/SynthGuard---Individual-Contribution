# SAP-DCG Dataset Organization & Version Reference (`data_info.md`)

This document provides a complete inventory of the datasets located in [`SAP-DCG/dataset/`](file:///d:/Swinburne%20Courses/Project%20A/duc_thanh/SAP-DCG/dataset), delineating the **Raw Baseline Sources**, the **New / Current Active Pipeline Datasets**, and the **Old / Legacy Experimental Datasets**.

---

## 1. Summary Hierarchy

```
SAP-DCG/dataset/
├── NF-UNSW-NB15-v3.csv                 [RAW BASELINE] Raw benchmark dataset with shortcut features (~577 MB)
├── NetFlow_v3_Features.csv             [METADATA] NetFlow-v3 schema feature definitions
├── synthetic_deficit.json              [NEW PIPELINE] Pre-computed deficit quotas (n_syn, t_c, n_c)
│
├── dev_splits/                         [NEW PIPELINE] Stratified 80% development set (47 pruned features)
│   ├── NF-UNSW-NB15-v3_train.csv       -> Active training set for SMOTE-NC & CTGAN (690,228 flows)
│   ├── NF-UNSW-NB15-v3_val.csv         -> Validation set for hyperparameter tuning & stopping (98,605 flows)
│   └── NF-UNSW-NB15-v3_test.csv        -> Development test set for benchmark evaluation (197,209 flows)
│
├── holdout_set/                        [NEW PIPELINE] Pristine 20% evaluation partition
│   └── NF-UNSW-NB15-v3.csv             -> Clean holdout set isolated at Step 1 (246,511 flows)
│
├── original/                           [RAW BASELINE BACKUP] Duplicate copy of raw NF-UNSW-NB15-v3.csv
│   ├── NF-UNSW-NB15-v3.csv
│   └── NetFlow_v3_Features.csv
│
├── generators/                         [OLD / LEGACY] Pre-generated static inputs for standalone generators
│   ├── CTGAN/
│   │   ├── unsw_nb15_attacks_smotenc.csv
│   │   └── unsw_nb15_attacks_smote_10k.csv
│   └── TVAE/
│       ├── unsw_nb15_attacks_smotenc.csv
│       └── unsw_nb15_attacks_smote_10k.csv
│
├── IDS-multiclass-classification/      [OLD / LEGACY] Standalone multiclass classifier benchmark files
│   ├── unsw_nb15_attacks_smotenc.csv
│   ├── unsw_nb15_attacks_smotenc_normalized.csv
│   ├── unsw_nb15_attacks_smote_10k.csv
│   └── unsw_nb15_train_smotenc_resampled.csv
│
├── IDS-binary-classification_MIA/      [OLD / LEGACY] 5-Fold cross-validation splits for privacy testing
│   ├── train/ (NF-UNSW-NB15-v3_fold_1..5.csv)
│   └── test/  (NF-UNSW-NB15-v3_fold_1..5.csv)
│
└── oversampled/                        [OLD / LEGACY] Initial heuristic oversampling experiments
    ├── unsw_nb15_attacks_original.csv
    ├── unsw_nb15_attacks_ros_10k.csv
    ├── unsw_nb15_attacks_smote_3k.csv
    └── unsw_nb15_attacks_smote_10k.csv
```

---

## 2. The New / Current Active Pipeline Datasets

These datasets represent the **active SAP-DCG pipeline architecture**. All non-generalizable shortcut learning attributes (`IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT`, and flow timestamps) have been removed, deduplicated across the remaining 47 NetFlow features, and partitioned strictly into an 80% development set and a pristine 20% holdout set.

### 2.1 Development Partitions (`dev_splits/`)
Located at [`SAP-DCG/dataset/dev_splits/`](file:///d:/Swinburne%20Courses/Project%20A/duc_thanh/SAP-DCG/dataset/dev_splits):
- **`NF-UNSW-NB15-v3_train.csv` (690,228 rows)**:
  - The official training dataset for the pipeline.
  - Used in **Round 1** by [`IntraAttackResampler`](file:///d:/Swinburne%20Courses/Project%20A/duc_thanh/SAP-DCG/components/intra_attack_resampler.py) for SMOTE-NC balancing.
  - Used in **Round 2** by [`TabularGenerators`](file:///d:/Swinburne%20Courses/Project%20A/duc_thanh/SAP-DCG/components/tabular_generators.py) to train CTGAN/TVAE.
  - Used in **Round 3** by [`SyntheticSampleFilter`](file:///d:/Swinburne%20Courses/Project%20A/duc_thanh/SAP-DCG/components/synthetic_sample_filter.py) to train the 5-fold Out-Of-Fold (OOF) XGBoost confidence gate strictly on authentic real attacks.
- **`NF-UNSW-NB15-v3_val.csv` (98,605 rows)**:
  - Isolated validation set.
  - Acts as the objective evaluation metric across all sequential rounds (tuning $y_{\max}$, $\theta_{\text{OOF}}$, and early stopping when improvement $\Delta < \epsilon$).
- **`NF-UNSW-NB15-v3_test.csv` (197,209 rows)**:
  - Evaluation test split used to assess downstream Two-Tier IDS performance on unseen traffic.

### 2.2 Pristine Holdout Partition (`holdout_set/`)
Located at [`SAP-DCG/dataset/holdout_set/`](file:///d:/Swinburne%20Courses/Project%20A/duc_thanh/SAP-DCG/dataset/holdout_set):
- **`NF-UNSW-NB15-v3.csv` (246,511 rows)**:
  - The pristine 20% holdout partition extracted at Step 1 before any preprocessing, resampling, or model training.
  - Never exposed during hyperparameter tuning; used solely for final validation against data snooping.

### 2.3 Synthetic Deficit Blueprint (`synthetic_deficit.json`)
Located at [`SAP-DCG/dataset/synthetic_deficit.json`](file:///d:/Swinburne%20Courses/Project%20A/duc_thanh/SAP-DCG/dataset/synthetic_deficit.json):
- Stores original attack class counts ($n_c$), target sample ceiling ($t_c$), and calculated synthetic deficit quotas ($n_{\text{syn}} = \max(0, t_c - n_c)$).

### 2.4 Active Pipeline Generated Outputs (`SAP-DCG/results/`)
While saved in the results directory, these are the active datasets generated by running the pipeline:
- **`unsw_nb15_final_pipeline_train.csv`**: The augmented NetFlow dataset combining real training flows with filtered CTGAN synthetic flows.
- **`synthetic_samples_kept.csv`**: The clean synthetic attack flows that survived physical domain constraints, the 5-fold XGBoost OOF confidence gate, and Edited Nearest Neighbors ($k=3$).

---

## 3. Raw Baseline Inputs (Source Data)

* **`NF-UNSW-NB15-v3.csv`** (Root of `dataset/`) & **`original/NF-UNSW-NB15-v3.csv`**:
  * Original NetFlow-v3 benchmark file (~577 MB) before shortcut stripping and deduplication.
  * Contains socket addresses (`IPV4_SRC_ADDR`, `L4_SRC_PORT`, `IPV4_DST_ADDR`, `L4_DST_PORT`) and flow timestamps.
* **`NetFlow_v3_Features.csv`**:
  * Feature documentation file detailing the NetFlow v3 schema and metric definitions.

---

## 4. Old / Legacy Datasets (Pre-Pipeline Experiments)

These directories contain static datasets and intermediate files created during exploratory phases prior to unifying the multi-round sequential tuning pipeline. **They are retained for experimental provenance and baseline comparison, but are not actively generated or consumed by the current pipeline**:

### 4.1 `oversampled/` (Early Heuristic Oversampling)
- Files: `unsw_nb15_attacks_original.csv`, `unsw_nb15_attacks_ros_10k.csv`, `unsw_nb15_attacks_smote_3k.csv`, `unsw_nb15_attacks_smote_10k.csv`.
- **Purpose**: Initial baseline experiments testing Random Oversampling (ROS) against static SMOTE multipliers (3k and 10k fixed targets) before the dynamic deficit formulation ($n_{\text{syn}}$ governed by $y_{\max}$ and $t_{\max}$) was established.

### 4.2 `generators/` (`CTGAN/` and `TVAE/`)
- Files: `unsw_nb15_attacks_smotenc.csv`, `unsw_nb15_attacks_smote_10k.csv`.
- **Purpose**: Pre-generated static training attack sets dumped to disk for running isolated standalone CTGAN/TVAE training scripts outside of the unified orchestrator. In the current pipeline, these are replaced by dynamic in-memory hand-offs between `IntraAttackResampler` and `TabularGenerators`.

### 4.3 `IDS-multiclass-classification/`
- Files: `unsw_nb15_attacks_smotenc.csv`, `unsw_nb15_attacks_smotenc_normalized.csv`, `unsw_nb15_attacks_smote_10k.csv`, `unsw_nb15_train_smotenc_resampled.csv`.
- **Purpose**: Static attack splits used to benchmark multiclass classifiers (Tier-2 AttackTyper) in standalone scripts without the end-to-end Two-Tier IDS loop.

### 4.4 `IDS-binary-classification_MIA/`
- Directories: `train/` and `test/` containing 5 folds (`NF-UNSW-NB15-v3_fold_1.csv` through `fold_5.csv`).
- **Purpose**: Legacy cross-validation splits structured specifically for Membership Inference Attack (MIA) privacy benchmarking. Replaced in the current pipeline by the integrated 5-Pillar Synthetic Data Quality Evaluation (`evaluate_synthetic_quality`), which computes DCR and closer-to-train ratios directly.

---

## 5. Quick Reference Table

| Category | Path / Filename | Status | Used in Current Pipeline? |
| :--- | :--- | :--- | :--- |
| **Source Data** | `dataset/NF-UNSW-NB15-v3.csv` | Raw Source | Yes (input to Step 1 partitioner) |
| **Source Data** | `dataset/original/` | Source Backup | Reference only |
| **Source Metadata** | `dataset/NetFlow_v3_Features.csv` | Metadata | Reference |
| **Active Splits** | `dataset/dev_splits/NF-UNSW-NB15-v3_train.csv` | **New Active** | **Yes (Training Split)** |
| **Active Splits** | `dataset/dev_splits/NF-UNSW-NB15-v3_val.csv` | **New Active** | **Yes (Validation / Tuning Split)** |
| **Active Splits** | `dataset/dev_splits/NF-UNSW-NB15-v3_test.csv` | **New Active** | **Yes (Test Evaluation Split)** |
| **Active Splits** | `dataset/holdout_set/NF-UNSW-NB15-v3.csv` | **New Active** | **Yes (Pristine 20% Holdout Split)** |
| **Quota Config** | `dataset/synthetic_deficit.json` | **New Active** | **Yes (Target & Deficit Specs)** |
| **Pipeline Output** | `results/unsw_nb15_final_pipeline_train.csv` | **New Active Output**| **Yes (Final Augmented Train Data)** |
| **Pipeline Output** | `results/synthetic_samples_kept.csv` | **New Active Output**| **Yes (Filtered CTGAN Flows)** |
| **Legacy Oversampling**| `dataset/oversampled/` | Old / Legacy | No (historical comparison only) |
| **Legacy Generator Data**| `dataset/generators/` | Old / Legacy | No (replaced by orchestrator) |
| **Legacy Multiclass** | `dataset/IDS-multiclass-classification/` | Old / Legacy | No (replaced by Two-Tier IDS) |
| **Legacy MIA Folds** | `dataset/IDS-binary-classification_MIA/` | Old / Legacy | No (replaced by 5-pillar evaluator) |


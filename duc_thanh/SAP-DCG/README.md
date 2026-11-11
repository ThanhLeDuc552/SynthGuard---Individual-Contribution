# SAP-DCG: Sequential Adaptive Processing & Deep Conditional Generative Framework

A modular, four-stage intrusion detection resampling and evaluation pipeline for network flow datasets (specifically **NF-UNSW-NB15-v3**).

The framework integrates CTGAN deep conditional generation derived from Minh's pipeline (`minh/`) and two-tier intrusion detection & quality evaluation derived from Tuan's pipeline (`tuan/`):

1. **Intra-Attack SMOTE-NC Resampling** with a class-specific Synthetic Deficit calculation:
   $$t_c = \max\left(n_c, \min(t_{\max}, y_{\max} \cdot n_c)\right), \quad n_{\text{syn}} = \max(0, t_c - n_c)$$
2. **Deep Tabular Generative Modeling** (CTGAN / TVAE) trained under a fixed gradient update budget (42,000 steps), incorporating Minh's NetFlow preprocessing, PAC batch alignment, and conditional rejection sampling (`sample_by_class`) / pool sampling (`sample_pool_by_class`).
3. **Protocol-Aware Synthetic Sample Filtering** using inverse log/scaling transformations, NetFlow physical invariants, a 5-fold cross-validated XGBoost Out-Of-Fold (OOF) confidence gate, and Edited Nearest Neighbors (ENN, $n=3$) boundary checking.
4. **Two-Tier Intrusion Detection System (IDS)** derived from Tuan's ML-IDS:
   - **Tier 1 (Benign vs Attack Detector)**: Supervised XGBoost + IsolationForest & signed-log robust-Z anomaly detector (calibrated to `max_fpr`).
   - **Tier 2 (Multiclass AttackTyper)**: Multi-class XGBoost with balanced sample weights ($w = \text{balanced}^{0.5}$) and confidence gating (low confidence $\to$ `"Unknown"`).
5. **Unified Multi-Round Sequential Tuner**: Tunes hyperparameters across all four components sequentially with Optuna and $\epsilon$ early stopping criteria.
6. **5-Pillar Synthetic Data Quality Evaluation**: Measures Coverage, Validity, Fidelity (KS & correlation), Privacy (DCR & closer-to-train rate), and Utility (TRTR, TSTR, TR+S, ML Efficacy Ratio).

---

## Directory Structure

```
SAP-DCG/
├── README.md                     # Framework documentation & usage guide
├── run_pipeline.py               # Standalone CLI execution script
├── __init__.py                   # Package exports for SAP-DCG
│
├── dataset/                      # Self-contained dataset storage
│   ├── original/                 # Pristine raw NetFlow v3 benchmark dataset
│   │   ├── NF-UNSW-NB15-v3.csv
│   │   └── NetFlow_v3_Features.csv
│   ├── dev_splits/               # 80% Dev partition splits (train, val, test)
│   ├── holdout_set/              # 20% Pristine Holdout test set
│   ├── IDS-multiclass-classification/ # Resampled augmented datasets
│   └── generators/               # Generative checkpoints (CTGAN / TVAE weights)
│
├── components/                   # Modular pipeline components
│   ├── __init__.py
│   ├── intra_attack_resampler.py # Component 1: Steps 1-6 SMOTE-NC & deficit logic
│   ├── tabular_generators.py     # Component 2: CTGAN & TVAE models (Minh's logic)
│   ├── synthetic_sample_filter.py# Component 3: Domain rules, 5-Fold OOF gate & ENN (n=3)
│   └── final_ids_model.py        # Component 4: Two-Tier IDS model & 5-pillar metrics (Tuan's logic)
│
├── orchestration/                # Multi-round coordination & optimization
│   ├── __init__.py
│   ├── pipeline_tuner.py         # Unified Tuner consolidating Rounds 1–4 optimization
│   └── pipeline_orchestrator.py  # Master 4-round sequential tuning pipeline
│
├── notebooks/                    # Interactive Jupyter notebooks
│   └── nf_unsw_resampling_v2.ipynb # Steps 1–3 data preparation & Component 1 execution
│
└── docs/                         # Specifications & research documents
    ├── components.txt            # Component requirements & multi-round pipeline specs
    ├── dataset.txt               # NF-UNSW-NB15-v3 preprocessing & mathematical formulas
    ├── evaluation.txt            # Baseline comparison methods & target metrics
    ├── problems.txt              # Data science & AI challenges addressed
    └── problems to read.txt      # Shortcut learning references
```

---

## The 4 Pipeline Components

| Component       | Module                                  | Key Responsibilities & Derived Logic                                                                                                                                                                                                                                                                   |
| :-------------- | :-------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Component 1** | `components/intra_attack_resampler.py`  | Calculates Synthetic Deficit $n_{\text{syn}}$ per minority attack class, normalizes continuous features, and resamples attack flows via SMOTE-NC without altering Benign distributions.                                                                                                                |
| **Component 2** | `components/tabular_generators.py`      | **Minh's Logic**: NetFlow preprocessor (`NetFlowPreprocessor`), PAC-aligned batch sizing, conditional rejection sampling (`sample_by_class`) to enforce class consistency, TVAE pool sampling (`sample_pool_by_class`), and post-generation validity clipping/rounding under 42,000 gradient updates. |
| **Component 3** | `components/synthetic_sample_filter.py` | Enforces physical NetFlow domain invariants (TCP flag consistency, packet/byte constraints); gates samples via 5-Fold OOF XGBoost confidence; prunes boundary outliers via ENN ($n=3$).                                                                                                               |
| **Component 4** | `components/final_ids_model.py`         | **Tuan's Logic**: Two-Tier IDS architecture (Tier 1 Binary + Anomaly gating, Tier 2 AttackTyper multiclass). Computes Macro F1, Balanced Accuracy, Detection Rate (DR), False Alarm Rate (FAR), MCC, PR-AUC, and 5-pillar synthetic quality metrics (`evaluate_synthetic_quality`).               |

---

## Multi-Round Sequential Tuning Pipeline

The [`PipelineTuner`](orchestration/pipeline_tuner.py) coordinates the 4 sequential rounds:

1. **Round 1 (Component 1)**: Tunes SMOTE-NC multiplier factor $y_{\text{max}} \in [5, 10, 15, 20, 25, 30, 40]$ based on validation Macro F1 of the IDS model. Advances when $\Delta < \epsilon$.
2. **Round 2 (Component 2)**: Tunes generator hyperparameters with Optuna using Minh's search space (`generator_dim`, `discriminator_dim`, `generator_lr`, `discriminator_lr`, `discriminator_steps`, `log_frequency`, `embedding_dim`, `pac`, `batch_size`) against validation Macro F1. Advances when $\Delta < \epsilon$.
3. **Round 3 (Component 3)**: Tunes the OOF confidence gate threshold $\in [0.4, 0.5, 0.6, 0.7, 0.8]$ based on validation Macro F1. Advances when $\Delta < \epsilon$.
4. **Final Round (Component 4)**: Optimizes final Two-Tier IDS model hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `weight_power`, `min_confidence`, `max_fpr`) using Tuan's search space. Evaluates performance across Validation, Test, and pristine Holdout sets, computes normalized Wasserstein & centroid distances, and reports 5-pillar synthetic data quality.

---

## Standalone Usage

```bash
# Display CLI options
python run_pipeline.py --help

# Execute full pipeline with CTGAN generator, convergence epsilon = 0.002, and custom destination output folder
python run_pipeline.py --generator ctgan --epsilon 0.002 --t-max 10000 --output-dir outputs/experiment_1

# Execute full pipeline with TVAE generator
python run_pipeline.py --generator tvae --epsilon 0.002 --t-max 10000 --output-dir outputs/tvae_experiment
```

### Destination Output Artifacts

When executing `run_pipeline.py` with `--output-dir <path>` (or `--out-dir` / `--dest-dir`), all pipeline artifacts and checkpoints are automatically exported to that destination directory:

| Artifact | File Format | Description |
| :--- | :--- | :--- |
| `evaluation_report.md` | Markdown (`.md`) | Comprehensive human-readable report: overall metrics table (Validation, Test, Holdout), per-class specific precision/recall/F1 metrics, confusion matrix, 5-pillar synthetic quality metrics, and distribution distances. |
| `best_params.json` | JSON | All optimal hyperparameters discovered across Rounds 1–4 (optimal $y_{\max}$, deficit quotas, generator budget & checkpoint, OOF confidence threshold, ENN neighbors, and optimal Two-Tier IDS parameters). |
| `pipeline_tuning_summary.json` | JSON | Complete numerical evaluation payload across all partitions and tuning rounds for programmatic consumption. |
| `round2_best_weights.pt` | PyTorch Weights | Saved weights and preprocessor state for the trained CTGAN / TVAE generator. |
| `final_ids_model.joblib` | Model Checkpoint | Serialized state of the trained Two-Tier Intrusion Detection System (Tier-1 detector, Anomaly detector, Tier-2 AttackTyper, calibrated thresholds). |
| `unsw_nb15_final_pipeline_train.csv` | CSV Dataset | The finalized, augmented NetFlow training dataset with balanced minority attack classes. |
| `synthetic_samples_kept.csv` | CSV Dataset | The quality-filtered synthetic attack flows generated by CTGAN/TVAE and retained by the sample filter. |


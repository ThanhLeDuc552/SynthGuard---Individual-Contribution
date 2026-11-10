# SAP-DCG: Sequential Adaptive Processing & Deep Conditional Generative Framework

A modular, four-stage intrusion detection resampling and evaluation pipeline for network flow datasets (specifically **NF-UNSW-NB15-v3**).

The framework tackles extreme class imbalance and minority memorization by combining:

1. **Intra-Attack SMOTE-NC Resampling** with a class-specific Synthetic Deficit calculation.
2. **Deep Tabular Generative Modeling** (CTGAN / TVAE) trained with a fixed gradient update budget (42k steps).
3. **Protocol-Aware Synthetic Sample Filtering** using inverse log/scaling transformations, NetFlow physical invariants, a 5-fold cross-validated XGBoost Out-Of-Fold (OOF) confidence gate, and Edited Nearest Neighbors (ENN, $n=3$) boundary checking.
4. **Multiclass Intrusion Detection System (IDS)** benchmarking with XGBoost evaluated across Validation, Test, and pristine Holdout partitions.
5. **Unified Multi-Round Sequential Tuner**: Tunes hyperparameters across all four components sequentially, advancing to subsequent rounds when improvement drops below $\epsilon$.

---

## Directory Structure

```
SAP-DCG/
├── README.md                     # Framework documentation & usage guide
├── run_pipeline.py               # Root standalone CLI execution script
├── __init__.py                   # Package exports for SAP-DCG
│
├── components/                   # Modular pipeline components (pure, no tuning code)
│   ├── __init__.py
│   ├── intra_attack_resampler.py # Component 1: Steps 1-6 SMOTE-NC & deficit logic
│   ├── tabular_generators.py     # Component 2: CTGAN & TVAE models (42k steps)
│   ├── synthetic_sample_filter.py# Component 3: Rules, 5-Fold OOF gate & ENN (n=3)
│   └── final_ids_model.py        # Component 4: Multiclass XGBoost IDS model & metrics
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

| Component       | Module                                  | Key Responsibilities                                                                                                                                                                                     |
| :-------------- | :-------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Component 1** | `components/intra_attack_resampler.py`  | Calculates Synthetic Deficit $n_{\text{syn}} = \max(0, t_c - n_c)$ where $t_c = \max(n_c, \min(t_{\max}, y_{\max} \cdot n_c))$, normalizes continuous features, and resamples attack flows via SMOTE-NC. |
| **Component 2** | `components/tabular_generators.py`      | Manages CTGAN and TVAE architectures under a strict budget of 42,000 gradient updates; persists checkpoints and samples $10 \times n_{\text{syn}}$ candidate pools.                                      |
| **Component 3** | `components/synthetic_sample_filter.py` | Enforces physical NetFlow domain invariants (TCP flag consistency, packet/byte constraints); gates samples via 5-Fold OOF XGBoost confidence; prunes boundary outliers via ENN ($n=3$).                  |
| **Component 4** | `components/final_ids_model.py`         | Fits production XGBoost multiclass threat classifier; calculates unweighted Macro F1, Balanced Accuracy, Weighted F1, Binary F1, per-class breakdown, and confusion matrix.                              |

---

## Multi-Round Sequential Tuning Pipeline

The [`PipelineTuner`](orchestration/pipeline_tuner.py) coordinates the 4 sequential rounds:

1. **Round 1 (Component 1)**: Tunes SMOTE-NC multiplier factor $y_{\text{max}} \in [5, 10, 15, 20, 25, 30, 40]$ based on validation Macro F1 of the IDS model. Advances when $\Delta < \epsilon$.
2. **Round 2 (Component 2)**: Tunes generator hyperparameters (learning rates, embedding dim, batch size) using Optuna against validation Macro F1. Advances when $\Delta < \epsilon$.
3. **Round 3 (Component 3)**: Tunes the OOF confidence gate threshold $\in [0.4, 0.5, 0.6, 0.7, 0.8]$ based on validation Macro F1. Advances when $\Delta < \epsilon$.
4. **Final Round (Component 4)**: Optimizes final IDS model hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`) on the best resampled training data. Evaluates final performance on Test and Holdout sets, and computes normalized Wasserstein and standardized centroid distribution distances to the validation distribution.

---

## Standalone Usage

```bash
# Display CLI options
python run_pipeline.py --help

# Execute full pipeline with CTGAN generator and convergence epsilon = 0.002
python run_pipeline.py --generator ctgan --epsilon 0.002 --t-max 10000

# Execute full pipeline with TVAE generator
python run_pipeline.py --generator tvae --epsilon 0.002 --t-max 10000
```

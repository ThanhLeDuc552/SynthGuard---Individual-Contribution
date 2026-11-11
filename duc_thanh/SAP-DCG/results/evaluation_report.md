# SAP-DCG Pipeline Comprehensive Evaluation Report

> **Generated:** 2026-11-11 07:51:02  
> **Destination Folder:** `/content/drive/MyDrive/SAP-DCG`  
> **Generator Architecture:** `CTGAN` (42,000 gradient updates)  
> **Target Class Ceiling ($t_{\max}$):** `10,000` | **Convergence $\epsilon$:** `0.002` | **Optimization Metric:** `macro_f1`  

---

## 1. Overall IDS Performance Benchmark

Comparison of downstream Two-Tier IDS evaluation metrics across the validation, test, and holdout partitions:

| Metric | Validation Set | Test Set | Holdout Set | Target Direction |
| :--- | :---: | :---: | :---: | :---: |
| **Macro F1-Score** | `0.5735` | `0.5396` | `0.5591` | Higher is better (Max 1.0) |
| **Detection Rate (DR / Attack Recall)** | `1.0000` | `1.0000` | `1.0000` | Higher is better (Max 1.0) |
| **False Alarm Rate (FAR / Benign FPR)** | `0.0052` | `0.0049` | `0.0048` | Lower is better (Min 0.0) |
| **Matthews Correlation (MCC)** | `0.9692` | `0.9705` | `0.9715` | Higher is better (Max 1.0) |
| **Balanced Accuracy** | `0.7078` | `0.6779` | `0.6895` | Higher is better (Max 1.0) |
| **Overall Accuracy** | `0.9703` | `0.9693` | `0.9699` | Higher is better (Max 1.0) |
| **Weighted F1-Score** | `0.9735` | `0.9724` | `0.9728` | Higher is better (Max 1.0) |
| **Binary F1-Score (Attack vs Benign)** | `0.9713` | `0.9726` | `0.9735` | Higher is better (Max 1.0) |
| **Isolated Tier-2 Attack Macro F1** | `0.5893` | `0.5463` | `0.5691` | Higher is better (Max 1.0) |
| **Rare Attack F1-Score** | `0.4951` | `0.4182` | `0.4674` | Higher is better (Max 1.0) |
| **End-to-End Rare Attack Recall** | `0.6909` | `0.6362` | `0.6637` | Higher is better (Max 1.0) |
| **Unknown Attack Alert Rate** | `0.0802` | `0.0792` | `0.0746` | Informational |

---

## 2. Per-Class Specific Performance Breakdown

Detailed classification metrics (Precision, Recall, F1-Score, and Support) per network flow category:

### 2.1 Validation Split

| Network Flow Class | Category Type | Support | Precision | Recall (DR) | F1-Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Analysis** | Rare Attack (Minority) | 98 | 29.56% | 95.92% | **45.19%** | ⚠️ Rare Threat |
| **Backdoor** | Rare Attack (Minority) | 311 | 29.10% | 17.68% | **22.00%** | ⚠️ Rare Threat |
| **Benign** | Benign Flow | 90,659 | 100.00% | 99.48% | **99.74%** | Normal |
| **DoS** | Frequent Attack | 430 | 45.25% | 46.51% | **45.87%** | Standard Threat |
| **Exploits** | Frequent Attack | 3,239 | 90.37% | 65.48% | **75.94%** | Standard Threat |
| **Fuzzers** | Frequent Attack | 2,159 | 70.23% | 88.51% | **78.32%** | Standard Threat |
| **Generic** | Frequent Attack | 609 | 67.56% | 74.55% | **70.88%** | Standard Threat |
| **Reconnaissance** | Frequent Attack | 953 | 68.78% | 56.87% | **62.26%** | Standard Threat |
| **Shellcode** | Rare Attack (Minority) | 135 | 39.11% | 71.85% | **50.65%** | ⚠️ Rare Threat |
| **Worms** | Rare Attack (Minority) | 11 | 71.43% | 90.91% | **80.00%** | ⚠️ Rare Threat |

### 2.2 Test Split

| Network Flow Class | Category Type | Support | Precision | Recall (DR) | F1-Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Analysis** | Rare Attack (Minority) | 196 | 27.49% | 94.39% | **42.58%** | ⚠️ Rare Threat |
| **Backdoor** | Rare Attack (Minority) | 621 | 28.00% | 16.91% | **21.08%** | ⚠️ Rare Threat |
| **Benign** | Benign Flow | 181,319 | 100.00% | 99.51% | **99.75%** | Normal |
| **DoS** | Frequent Attack | 860 | 40.38% | 40.00% | **40.19%** | Standard Threat |
| **Exploits** | Frequent Attack | 6,478 | 90.13% | 63.32% | **74.39%** | Standard Threat |
| **Fuzzers** | Frequent Attack | 4,318 | 68.51% | 87.33% | **76.79%** | Standard Threat |
| **Generic** | Frequent Attack | 1,217 | 66.06% | 74.03% | **69.82%** | Standard Threat |
| **Reconnaissance** | Frequent Attack | 1,907 | 68.34% | 59.20% | **63.44%** | Standard Threat |
| **Shellcode** | Rare Attack (Minority) | 270 | 40.00% | 65.93% | **49.79%** | ⚠️ Rare Threat |
| **Worms** | Rare Attack (Minority) | 22 | 43.59% | 77.27% | **55.74%** | ⚠️ Rare Threat |

### 2.3 Holdout Split

| Network Flow Class | Category Type | Support | Precision | Recall (DR) | F1-Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Analysis** | Rare Attack (Minority) | 245 | 28.54% | 98.78% | **44.28%** | ⚠️ Rare Threat |
| **Backdoor** | Rare Attack (Minority) | 777 | 26.81% | 15.70% | **19.81%** | ⚠️ Rare Threat |
| **Benign** | Benign Flow | 226,650 | 100.00% | 99.52% | **99.76%** | Normal |
| **DoS** | Frequent Attack | 1,075 | 41.04% | 39.81% | **40.42%** | Standard Threat |
| **Exploits** | Frequent Attack | 8,097 | 89.78% | 64.42% | **75.01%** | Standard Threat |
| **Fuzzers** | Frequent Attack | 5,397 | 69.94% | 87.99% | **77.94%** | Standard Threat |
| **Generic** | Frequent Attack | 1,522 | 67.36% | 74.31% | **70.67%** | Standard Threat |
| **Reconnaissance** | Frequent Attack | 2,383 | 66.22% | 57.99% | **61.83%** | Standard Threat |
| **Shellcode** | Rare Attack (Minority) | 337 | 34.70% | 65.28% | **45.31%** | ⚠️ Rare Threat |
| **Worms** | Rare Attack (Minority) | 28 | 75.00% | 85.71% | **80.00%** | ⚠️ Rare Threat |

---

## 3. Confusion Matrix (Test Partition)

| True \ Predicted | Analysis | Backdoor | Benign | DoS | Exploits | Fuzzers | Generic | Reconnaissance | Shellcode | Worms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Analysis** | 185 | 1 | 0 | 5 | 3 | 0 | 0 | 0 | 0 | 0 |
| **Backdoor** | 4 | 105 | 0 | 29 | 147 | 197 | 64 | 43 | 12 | 2 |
| **Benign** | 0 | 10 | 180,423 | 12 | 9 | 0 | 1 | 1 | 0 | 0 |
| **DoS** | 44 | 12 | 0 | 344 | 167 | 147 | 32 | 39 | 24 | 1 |
| **Exploits** | 286 | 89 | 0 | 319 | 4,102 | 835 | 220 | 366 | 95 | 16 |
| **Fuzzers** | 67 | 112 | 0 | 81 | 20 | 3,771 | 68 | 37 | 13 | 1 |
| **Generic** | 21 | 10 | 0 | 29 | 35 | 122 | 901 | 28 | 39 | 0 |
| **Reconnaissance** | 66 | 30 | 0 | 25 | 60 | 392 | 66 | 1,129 | 84 | 2 |
| **Shellcode** | 0 | 6 | 0 | 8 | 3 | 40 | 12 | 9 | 178 | 0 |
| **Worms** | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 17 |

---

## 4. 5-Pillar Synthetic Data Quality Benchmark

Evaluated on candidate synthetic flows prior to final augmentation:

### Pillar 1: Coverage
- **Classes Covered:** `5/10`
- **Missing Classes:** `['Analysis', 'Benign', 'Exploits', 'Fuzzers', 'Worms']`

### Pillar 2: Validity
- **Invalid Row Rate:** `0.00%` (Rows failing integrity rules)
- **Negative Value Cell Rate:** `0.000%`
- **Non-Integer Cell Rate in Count Columns:** `0.000%`
- **Over Maximum Bound Cell Rate:** `0.000%`

### Pillar 3: Fidelity
- **Mean Kolmogorov-Smirnov (KS) Statistic:** `0.1974` (Lower = Closer continuous distributions)
- **Mean Spearman Correlation Difference:** `0.191`
- **Tail Divergence (q99 Log-Ratio):** `0.4423`

### Pillar 4: Privacy
- **Exact Copy Rate:** `22.020%`
- **Distance to Closest Record (DCR) Train 5th Percentile:** `0.0`
- **Distance to Closest Record (DCR) Holdout 5th Percentile:** `0.6835`
- **Closer to Train Rate:** `85.36%` (~50% indicates no training data memorization)

### Pillar 5: Utility
- **Baseline Real Train (TRTR) Macro F1:** `0.6224`
- **Synthetic Only (TSTR) Macro F1:** `0.0714`
- **Machine Learning Efficacy Ratio (TSTR / TRTR):** `0.1147`
- **Combined Real + Synthetic (TR+S) Macro F1:** `0.6205`
- **Augmentation Gain (TR+S - TRTR):** `-0.0018`
- **Rare Attack Augmentation Gain:** `-0.0021`

---

## 5. Dataset Distribution Distances to Validation Distribution

- **Mean Normalized Wasserstein Distance:** `0.00119` (Lower is closer)
- **Standardized Centroid Euclidean Distance:** `0.0967`
- **Continuous Features Analyzed:** `36`

Top Divergent Continuous Features:
| Feature | Normalized Wasserstein Distance |
| :--- | :---: |
| `MAX_IP_PKT_LEN` | 0.00906 |
| `LONGEST_FLOW_PKT` | 0.00903 |
| `MAX_TTL` | 0.00264 |
| `TCP_WIN_MAX_OUT` | 0.00256 |
| `MIN_TTL` | 0.00215 |

---

## 6. Optimal Hyperparameters Summary

### Round 1: Intra-Attack Resampler (SMOTE-NC)
- **Optimal $y_{\max}$:** `5`
- **Downstream Score:** `0.568`
- **Synthetic Deficit Quota:** `{"Exploits": 0, "Fuzzers": 0, "Reconnaissance": 3327, "Generic": 5739, "DoS": 6990, "Backdoor": 7826, "Shellcode": 3780, "Analysis": 2748, "Worms": 316}`

### Round 2: Tabular Generators (CTGAN)
- **Model Architecture:** `CTGAN`
- **Gradient Updates Budget:** `42,000`
- **Weights Checkpoint:** `/content/drive/MyDrive/SAP-DCG/round2_best_weights.pt`
- **Downstream Score:** `0.568`

### Round 3: Synthetic Sample Filter
- **Optimal OOF Confidence Threshold:** `0.4`
- **ENN Boundary Cleaner Neighbors:** `3`
- **Retained High-Fidelity Synthetic Flows:** `813`
- **Downstream Score:** `0.5712`

### Final Round: Final IDS Model (Two-Tier Classifier)
```json
{
  "n_estimators": 100,
  "max_depth": 10,
  "learning_rate": 0.052220677761467035,
  "subsample": 1.0,
  "colsample_bytree": 0.8,
  "random_state": 42,
  "n_jobs": 4,
  "tree_method": "hist",
  "weight_power": 0.75,
  "min_confidence": 0.3,
  "max_fpr": 0.01
}
```

---

## 7. Exported Artifacts Manifest

All pipeline outputs and checkpoints have been exported to `/content/drive/MyDrive/SAP-DCG`:

| Artifact Filename | Type | Description |
| :--- | :--- | :--- |
| `evaluation_report.md` | Markdown Report | Comprehensive evaluation, per-class metrics, quality benchmarks |
| `best_params.json` | JSON Config | All optimal hyperparameters across components 1-4 |
| `pipeline_tuning_summary.json` | JSON Metrics | Complete structured numerical metrics across all splits and rounds |
| `round2_best_weights.pt` | PyTorch Weights | Trained CTGAN generator model checkpoint |
| `final_ids_model.joblib` | Model State | Trained Two-Tier IDS detector (Tier-1 + Anomaly + Tier-2) |
| `unsw_nb15_final_pipeline_train.csv` | Dataset CSV | Final augmented NetFlow training set with balanced minority flows |
| `synthetic_samples_kept.csv` | Dataset CSV | Quality-filtered synthetic flows generated by the pipeline |

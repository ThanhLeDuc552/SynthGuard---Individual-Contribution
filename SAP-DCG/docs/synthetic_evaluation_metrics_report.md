# Synthetic Dataset Evaluation Metrics & Gap Analysis Report

**Scope & Sources Analyzed:**
- **Requirements:** `SAP-DCG/docs/statistics evaluation.txt`
- **Latest Model Results:**
  - **CTGAN:** `SAP-DCG/results/ctgan/result_2/` (`pipeline_tuning_summary.json`, `evaluation_report.md`, `synthetic_samples_kept.csv`)
  - **TVAE:** `SAP-DCG/results/tvae/result_1/` (`pipeline_tuning_summary.json`, `evaluation_report.md`, `detailed_tvae_report.md`, `synthetic_samples_kept.csv`)
  - **Cross-Model Comparison:** `SAP-DCG/results/ctgan_vs_tvae_comparison_report.md`
- **Evaluation Implementation:**
  - `SAP-DCG/components/final_ids_model.py` (`evaluate_synthetic_quality`, `FinalIDSModel.evaluate`)
  - `SAP-DCG/orchestration/pipeline_tuner.py` (`_compute_distribution_distances`)

---

## 1. Metrics Currently Used to Evaluate the Synthetic Dataset

In the SAP-DCG pipeline, the created synthetic dataset is evaluated across **three layers**: (A) a **5-Pillar Synthetic Data Quality Benchmark** on filtered synthetic flows, (B) **Continuous Feature Distribution Distances** on the resampled attack training set against validation flows, and (C) **Downstream Two-Tier IDS Performance** used for tuning and final validation.

### A. 5-Pillar Synthetic Data Quality Benchmark (`evaluate_synthetic_quality`)
Implemented in `components/final_ids_model.py` (lines 711–875) and called in `pipeline_tuner.py` (lines 979–984) on `best_clean_syn` against `real_train` and `holdout_df` (falling back to `test_df` if no holdout is provided):

| Pillar | Metric Key | Definition / Formula | Target Direction |
| :--- | :--- | :--- | :--- |
| **1. Coverage** | `classes_covered` | Number of unique `Attack` classes present in `syn_clean` out of total classes in `real_train` (`got / total`) | Higher (`10/10` ideal) |
| | `missing_classes` | Sorted list of `real_train` classes absent from `syn_clean` | Empty `[]` ideal |
| **2. Validity** | `invalid_row_rate` | Fraction of synthetic rows failing any of the 3 column-bound checks below | `0.0` (Lower is better) |
| | `neg_cell_rate` | Fraction of cells `< 0` in columns that are non-negative in `real_train` | `0.0` |
| | `nonint_cell_rate` | Fraction of cells with `x % 1 != 0` in columns that are strictly integer in `real_train` | `0.0` |
| | `over_max_cell_rate` | Fraction of cells exceeding the per-column maximum observed in `real_train` | `0.0` |
| **3. Fidelity** | `ks_mean` | Mean 2-sample Kolmogorov–Smirnov statistic (`ks_2samp(...).statistic`) across numeric columns, averaged over covered classes with $\ge 2$ real & synthetic samples | Lower is better (`0.0` ideal) |
| | `corr_diff_mean` | Mean absolute difference between `real_train` and `syn_clean` Spearman correlation matrices per covered class | Lower is better |
| | `tail_q99_log_ratio` | Mean $|\log_{10}((q_{99,\text{syn}} + 1) / (q_{99,\text{real}} + 1))|$ across numeric columns and covered classes | Lower is better |
| **4. Privacy** | `exact_copy_rate` | Fraction of sampled synthetic rows (up to `cap=15,000`) with Euclidean distance $< 10^{-9}$ to nearest `real_train` row in signed-log standardized space | Lower is better (`0.0`) |
| | `dcr_train_p5` | 5th-percentile Distance to Closest Record (1-NN Euclidean) from synthetic rows to `real_train` | Higher / non-zero |
| | `dcr_holdout_p5` | 5th-percentile DCR from synthetic rows to `real_test` (holdout) | Comparable to `dcr_train_p5` |
| | `closer_to_train_rate` | Fraction of synthetic rows closer to `real_train` than to `real_test` ($d_{\text{train}} < d_{\text{holdout}}$) | `~0.50` (no memorization) |
| **5. Utility** | `trtr_macro_f1` | Train-Real-Test-Real baseline XGBoost Macro F1 on `real_test` | Baseline reference |
| | `tstr_macro_f1` | Train-Synthetic-Test-Real XGBoost Macro F1 on `real_test` (0.0 if $<2$ classes in `syn_clean`) | Closer to `trtr_macro_f1` |
| | `ml_efficacy_ratio` | `tstr_macro_f1 / trtr_macro_f1` | Closer to `1.0` |
| | `trs_macro_f1` | Train-(Real + Synthetic)-Test-Real XGBoost Macro F1 | Higher than `trtr_macro_f1` |
| | `trs_rare_f1` | Mean F1 on rare attack classes (training share `< RARE_SHARE_THRESHOLD`) under TR+S | Higher than `trtr_rare_f1` |
| | `aug_gain_macro_f1` | `trs_macro_f1 - trtr_macro_f1` | Positive (`> 0`) |
| | `aug_gain_rare_f1` | `trs_rare_f1 - trtr_rare_f1` | Positive (`> 0`) |

### B. Continuous Feature Distribution Distances (`_compute_distribution_distances`)
Implemented in `orchestration/pipeline_tuner.py` (lines 1002–1050), comparing attack flows (`Label == 1`) in `best_train_resampled` against `val_df` across **36 continuous features**:
- **`mean_normalized_wasserstein_distance`**: Mean 1-D Wasserstein distance per continuous feature, normalized by the validation feature range ($\text{ptp}(v_{\text{val}})$).
- **`centroid_euclidean_distance`**: Euclidean distance between the mean vectors of attack flows in `best_train_resampled` and `val_df`, standardized by validation feature standard deviation.
- **`top_divergent_features`**: Top 5 continuous features ranked by normalized Wasserstein distance.

### C. Downstream Two-Tier IDS Metrics (`FinalIDSModel.evaluate`)
Used as the optimization objective (`macro_f1` in Rounds 1 & 3) and final benchmark across Validation, Test, and Holdout splits:
- **Balance & Overall:** `macro_f1`, `balanced_accuracy`, `accuracy`, `weighted_f1`, plus per-class `precision`, `recall`, `f1`, and `support`.
- **Security / IDS Operational:** `detection_rate` (Attack Recall), `false_alarm_rate` (Benign FPR), `binary_f1`, `mcc` (Matthews Correlation Coefficient), `unknown_rate` (Unknown Attack Alert Rate).
- **Attack & Minority Focus:** `t2_macro_f1` (Isolated Tier-2 Attack Macro F1), `t2_rare_f1` (Tier-2 Rare Attack F1), `e2e_rare_recall` (End-to-End Rare Attack Recall), `e2e_attack_recall`.

---

## 2. Latest Observed Results (CTGAN `result_2` vs. TVAE `result_1`)

### 2.1 Synthetic Quality & Distribution Distances

| Metric Category | Metric | CTGAN (`result_2`) | TVAE (`result_1`) | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Augmentation Volume** | Round 1 Best $y_{\max}$ | `25` | `1` | TVAE `y_max=1` set all `syn_deficit=0` |
| | Kept Synthetic Flows (`best_clean_syn`) | `429` | `0` *(in pipeline)* | `tvae/result_1/synthetic_samples_kept.csv` separately has `429` rows |
| **Pillar 1: Coverage** | `classes_covered` | `5/10` | *Not executed (`{}`)* | Missing: `Analysis`, `Benign`, `Exploits`, `Fuzzers`, `Worms` |
| **Pillar 2: Validity** | `invalid_row_rate` / cell rates | `0.00%` (all `0.0`) | *Not executed (`{}`)* | All 429 CTGAN rows passed column bounds |
| **Pillar 3: Fidelity** | `ks_mean` | `0.2905` | *Not executed (`{}`)* | — |
| | `corr_diff_mean` | `0.2916` | *Not executed (`{}`)* | — |
| | `tail_q99_log_ratio` | `0.6743` | *Not executed (`{}`)* | — |
| **Pillar 4: Privacy** | `exact_copy_rate` | `0.1655` (`16.55%`) | *Not executed (`{}`)* | High exact/near-exact match rate |
| | `dcr_train_p5` / `dcr_holdout_p5` | `0.0` / `0.6449` | *Not executed (`{}`)* | — |
| | `closer_to_train_rate` | `0.9231` (`92.31%`) | *Not executed (`{}`)* | Strong proximity to training set vs. holdout |
| **Pillar 5: Utility** | `trtr_macro_f1` / `tstr_macro_f1` | `0.6189` / `0.0462` | *Not executed (`{}`)* | `ml_efficacy_ratio = 0.0746` |
| | `trs_macro_f1` (`aug_gain_macro_f1`) | `0.6236` (`+0.0047`) | *Not executed (`{}`)* | `trs_rare_f1 = 0.5818` (`aug_gain_rare_f1 = +0.0052`) |
| **Distribution Dist.** | `mean_normalized_wasserstein_distance` | `0.00118` | `0.00115` | Computed on `best_train_resampled` (dominated by real rows) |
| *(36 continuous)* | `centroid_euclidean_distance` | `0.092` | `0.1` | Top divergent: `MAX_IP_PKT_LEN`, `LONGEST_FLOW_PKT`, `TCP_WIN_MAX_OUT`, `MAX_TTL`, `MIN_TTL` |

### 2.2 Downstream IDS Benchmark (Test & Holdout Summary)

| Metric | CTGAN (`result_2`) Test / Holdout | TVAE (`result_1`) Test / Holdout |
| :--- | :---: | :---: |
| **Macro F1** | `0.5153` / `0.5520` | `0.5105` / `0.5404` |
| **Balanced Accuracy** | `0.5425` / `0.5783` | `0.5183` / `0.5468` |
| **Detection Rate (DR)** | `1.0000` / `1.0000` | `1.0000` / `1.0000` |
| **False Alarm Rate (FAR)** | `0.0048` / `0.0046` | `0.0048` / `0.0046` |
| **MCC** | `0.9714` / `0.9723` | `0.9713` / `0.9722` |
| **Tier-2 Rare Attack F1** | `0.3437` / `0.4483` | `0.3640` / `0.4501` |
| **E2E Rare Attack Recall** | `0.3146` / `0.4044` | `0.2728` / `0.3453` |
| **Unknown Alert Rate** | `0.0539` / `0.0527` | `0.1203` / `0.1196` |

---

## 3. Do Current Metrics Satisfy `docs/statistics evaluation.txt`?

`SAP-DCG/docs/statistics evaluation.txt` specifies:
> - `tap trung hon vao reliability` *(focus more on reliability)*
> - `syn: secu, balance, phg phap thong ke: reliable, numerical` *(synthetic: security, balance, statistical methods: reliable, numerical)*

### Overall Verdict: **Partially Satisfied (Needs Reliability, Security-Validity, and Balance Upgrades)**

| Requirement | Status | What Is Satisfied | Gaps & Limitations Observed |
| :--- | :---: | :--- | :--- |
| **Numerical Statistical Methods** (`phg phap thong ke: reliable, numerical`) | **Partially Satisfied** | All synthetic quality and distance metrics are quantitative: KS statistic, Spearman correlation diff, 99th-percentile log-ratio, Wasserstein distance, centroid Euclidean distance, 1-NN DCR percentiles, and F1/MCC. | 1. **No uncertainty / significance testing:** All metrics are single-seed (`seed=42`) point estimates; no confidence intervals, bootstrap CIs, multi-seed variance, or KS $p$-values (`.pvalue` is discarded).<br>2. **Small-sample unreliability:** In CTGAN `result_2`, `Backdoor` has only `2` kept synthetic rows (`DoS`: `20`, `Shellcode`: `25`). Computing KS, Spearman correlation matrices, and $q_{99}$ ratios on $n=2$ or $n=20$ is statistically noisy.<br>3. **TVAE `synthetic_quality` missing (`{}`):** Because `y_max=1` yielded `0` kept samples in Round 3, TVAE's 5-pillar benchmark was skipped entirely.<br>4. **Diluted distribution distance:** `_compute_distribution_distances` compares `best_train_resampled` (690,228 real + 429 synthetic for CTGAN; 100% real for TVAE) against `val_df`, rather than isolating pure synthetic flows (`synthetic_samples_kept.csv`) vs. real flows. |
| **Reliability Focus** (`tap trung hon vao reliability`) | **Partially Satisfied** | Privacy checks (`exact_copy_rate`, `dcr_train_p5`, `closer_to_train_rate`) and TR+S augmentation gain (`+0.0047` Macro F1, `+0.0052` Rare F1) test whether synthetic rows help without pure memorization. | 1. **No acceptance gates / pass-fail thresholds:** CTGAN `result_2` exhibits `exact_copy_rate = 16.55%`, `dcr_train_p5 = 0.0`, and `closer_to_train_rate = 92.31%` (strong memorization signal), yet no reliability gate flags or rejects the synthetic set.<br>2. **Unweighted class aggregation:** `ks_mean`, `corr_diff_mean`, and `tail_q99_log_ratio` average equally across covered classes regardless of sample count, while silently skipping classes with $<2$ synthetic samples. |
| **Security** (`secu`) | **Partially Satisfied** | Downstream IDS metrics directly measure security efficacy (`detection_rate`, `false_alarm_rate`, `unknown_rate`, `t2_rare_f1`, `e2e_rare_recall`, `mcc`), and Privacy Pillar checks data leakage/memorization. | **Missing NetFlow/protocol semantic validity:** `validity` only checks single-column non-negativity, integrality, and max bounds. Cross-feature network invariants (e.g., `MIN_TTL <= MAX_TTL`, `SHORTEST_FLOW_PKT <= LONGEST_FLOW_PKT <= MAX_IP_PKT_LEN`, `IN_BYTES >= IN_PKTS`, valid TCP flag / protocol consistency) are not verified—even though `final_ids_model.py` docstring mentions `L1-L3 network metrics` that are not implemented. |
| **Balance** (`balance`) | **Partially Satisfied** | Downstream evaluation tracks `macro_f1`, `balanced_accuracy`, `t2_rare_f1`, `e2e_rare_recall`, and per-class F1; synthetic quality tracks `coverage` (`classes_covered`, `missing_classes`) and `trs_rare_f1`. | 1. **No synthetic class-balance metric:** There is no imbalance ratio, normalized entropy, or deficit-fulfillment rate comparing `synthetic_samples_kept` counts against `syn_deficit` (`429` kept vs. `44,146` total deficit in CTGAN `result_2`, with `0` rows for `Analysis` and `Worms`).<br>2. **No per-class synthetic fidelity/utility breakdown:** `fidelity` and `tstr_macro_f1` are only reported as global averages (`tstr_macro_f1 = 0.0462`), hiding which rare classes succeeded or collapsed. |

---

## 4. Concrete Recommendations to Fully Satisfy `statistics evaluation.txt`

1. **Isolate Pure Synthetic vs. Real Distribution Distances:** Compute Wasserstein distance, KS, and centroid distance directly on `synthetic_samples_kept.csv` vs. `real_train`/`val_df` (per class and overall), in addition to `best_train_resampled`, so synthetic quality is not masked by the 690,228 real rows.
2. **Add Statistical Reliability & Sample-Size Guards:** Report bootstrap 95% confidence intervals (or multi-seed mean $\pm$ std), retain KS $p$-values (with multiple-testing correction), and enforce a minimum sample threshold (e.g., $n \ge 30$) or sample-weighted reporting before computing correlation/tail metrics.
3. **Implement Cross-Feature Network Security Validity (`L1–L3`):** Add domain-specific NetFlow invariant checks (e.g., packet length ordering, TTL ordering, byte-to-packet consistency, and valid protocol/TCP-flag combinations) to Pillar 2 (`validity`).
4. **Add Explicit Balance & Per-Class Synthetic Metrics:** Report per-class synthetic counts vs. `syn_deficit` (deficit fulfillment rate), class-balance entropy / imbalance ratio of `synthetic_samples_kept.csv`, and per-class fidelity/TSTR scores.
5. **Ensure Complete Coverage Across Models:** Run `evaluate_synthetic_quality` on `results/tvae/result_1/synthetic_samples_kept.csv` (429 rows) as well, or document why `y_max=1` bypassed synthetic augmentation, so CTGAN and TVAE can be compared on equal footing.

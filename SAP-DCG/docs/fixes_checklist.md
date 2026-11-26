# SAP-DCG: Distribution Shift & Pipeline Fixes Checklist

Tracks all fixes from [`docs/distribution_shift_fixes.md`](distribution_shift_fixes.md) and related pipeline improvements.

---

## 1. Data & Feature-Representation Fixes
- [x] Move `ICMP_TYPE` and `ICMP_IPV4_TYPE` to categorical/discrete code columns (`TVAE_CODE_COLS` in [`components/tvae_generator.py`](../components/tvae_generator.py); already in `CAT_CANDIDATES` in [`components/ctgan_generator.py`](../components/ctgan_generator.py)).
- [x] Encode spiky OS window constants (`TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT`) as top-30 discrete codes with rare-value restoration in `TVAEPreprocessor` and `NetFlowPreprocessor`.
- [x] Replace plain `log1p` on continuous features with `QuantileTransformer(output_distribution="normal")` (with `log1p` fallback for `< 2` rows) in `TVAEPreprocessor` and `NetFlowPreprocessor`.
- [x] Freeze constant / near-constant (`> 99.9%` identical, e.g., `DST_TO_SRC_IAT_MIN`) columns during generation and restore on inverse in `TVAEPreprocessor` (already present in `NetFlowPreprocessor`).
- [x] Add two-part hurdle representation (`<col>__is_zero` binary categorical flag + positive-only `QuantileTransformer`) for zero-inflated continuous columns (`5%–99.9%` zeros) in `TVAEPreprocessor` and `NetFlowPreprocessor`.
- [x] Reparameterize coupled bounds (`MIN_TTL <= MAX_TTL`, `SHORTEST_FLOW_PKT <= LONGEST_FLOW_PKT <= MAX_IP_PKT_LEN`) into non-negative deltas before generation and reconstruct on inverse in `TVAEPreprocessor` and `NetFlowPreprocessor`.
- [x] Exclude `DERIVED_FEATURES` (`TCP_FLAGS`, `SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_SECOND_BYTES`, `SRC_TO_DST_AVG_THROUGHPUT`, `DST_TO_SRC_AVG_THROUGHPUT`) from SMOTE-NC ([`components/intra_attack_resampler.py`](../components/intra_attack_resampler.py)) and deep generators (`CTGANGenerator`, `TVAEGenerator`), and recompute them deterministically via `recompute_derived_features()` before `SyntheticSampleFilter` and `FinalIDSModel`.
- [ ] Replace `30x` SMOTE-NC pre-balancing for generator training with mild `<= 3x` SMOTE-NC restricted to ultra-rare classes (`n_c < 500`) via `build_generator_train_attacks()` ([`orchestration/pipeline_tuner.py`](../orchestration/pipeline_tuner.py), [`orchestration/pipeline_orchestrator.py`](../orchestration/pipeline_orchestrator.py)).
- [ ] *(Optional Alternative)* Replace mild `<= 3x` SMOTE-NC with shared-representation transfer learning (pretrain one conditional generator on all real attacks, then fine-tune per rare class with validation early stopping).

---

## 2. Filter Fixes (`components/synthetic_sample_filter.py`)
- [x] Deduplicate intra-synthetic flows and remove exact copies of `real_train_df` in `SyntheticSampleFilter.filter_samples()`.
- [x] Add missing NetFlow rules (`ICMP_TYPE = 0` / `ICMP_IPV4_TYPE = 0` when `PROTOCOL in [6, 17]`, `LONGEST_FLOW_PKT <= MAX_IP_PKT_LEN`, `IN_BYTES >= IN_PKTS`, `OUT_BYTES >= OUT_PKTS`) and **reject** severely invalid rows instead of silently clamping them.
- [x] Add a local density-adaptive $k$-NN manifold-support distance gate in `signed_log1p + StandardScaler` space (`dist(x_syn, nn_real_same_class) <= alpha * r_k(nn_real_same_class)`).
- [ ] Add C2ST / density-ratio rejection sampling (lightweight real-vs-synthetic discriminator $D(x)$) to drive C2ST ROC-AUC toward `0.50`.

---

## 3. Model & Tuning Fixes
- [ ] Scale generator capacity, KL/L2 regularization, and internal validation early stopping to class size $n_c$ (prevent memorization/mode-collapse on tiny classes like `Worms`).
- [ ] Condition generation on `(Attack, PROTOCOL)` (or route TCP / UDP / ICMP through separate conditional branches).
- [ ] Benchmark tree/diffusion tabular generators (**ARF**, **ForestDiffusion**, **TabDDPM / TabSyn**) against CTGAN/TVAE in [`notebooks/synthetic_metrics_evaluation.ipynb`](../notebooks/synthetic_metrics_evaluation.ipynb).
- [ ] Update Round 1–3 tuning objective to a composite score (`Val Macro F1 - lambda_1 * Mean KS - lambda_2 * (1 - PFIR)`).

---

## 4. Verification & Re-Evaluation
- [ ] Run syntax/unit checks on modified components and re-run `CTGAN` and `TVAE` pipelines with the updated preprocessors.
- [ ] Re-run [`notebooks/synthetic_metrics_evaluation.ipynb`](../notebooks/synthetic_metrics_evaluation.ipynb) to verify improvements in C2ST ROC-AUC, PFIR, Generative Precision/Recall, and downstream Holdout IDS metrics.

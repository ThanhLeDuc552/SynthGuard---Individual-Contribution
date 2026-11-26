# SAP-DCG: Key Concepts, Metrics & Implementation Trade-Offs

This document explains five core concepts and evaluation metrics used in **SAP-DCG**, including how each works in [`notebooks/synthetic_metrics_evaluation.ipynb`](../notebooks/synthetic_metrics_evaluation.ipynb) / [`test/filter_test.ipynb`](../test/filter_test.ipynb) and the **benefits and cons** of implementing them actively in the pipeline.

---

## 1. C2ST / C2ST ROC-AUC (Classifier Two-Sample Test)

### Explanation
A **Classifier Two-Sample Test (C2ST)** trains a binary classifier (`Real = 0` vs. `Synthetic = 1`, evaluated via 5-fold cross-validated ROC-AUC using `HistGradientBoostingClassifier`) to measure how easily synthetic flows can be separated from real attack flows.
- **Interpretation:**
  - `ROC-AUC = 0.50`: Ideal indistinguishability (differentiating real vs. synthetic is pure coin-flip luck).
  - `ROC-AUC -> 1.00`: Strong distribution shift or structural artifacts give synthetic rows away.
- **Observed Baseline (`result_2`):** CTGAN = `1.0000`, TVAE = `0.9999`, Filtered TVAE (`test/filter_test.ipynb`) = `0.9989`.
- **As an Implementation Feature (Fix 2.4 — C2ST Rejection / Density-Ratio Weighting):** Train a real-vs-synthetic discriminator $D(x)$ inside [`SyntheticSampleFilter`](../components/synthetic_sample_filter.py) and reject samples with high $P(\text{syn} \mid x)$ or resample proportionally to $\frac{1 - D(x)}{D(x)}$.

### Benefits if Implemented
- Directly targets the exact multivariate artifacts (smeared zeros, `TCP_WIN_MAX` spikes, micro-correlations) that tree-based IDS classifiers exploit.
- Provides a single, hard-to-fool multivariate realism score alongside 1D marginal metrics.

### Cons / Risks
- **Class-balance pitfall:** If evaluated across all classes pooled without matching class proportions, C2ST can score `> 0.50` purely from oversampled class ratios ($P_{\text{syn}}(Y) \neq P_{\text{real}}(Y)$) rather than bad feature distributions ($P(X \mid Y)$).
- **Can reward memorization:** Exact copies of real rows score `AUC = 0.50`, so C2ST must always be paired with `NewRowSynthesis`, `DCR Closer-to-Train`, and `Generative Recall`.
- **High rejection rate when used as a filter:** If the generator hasn't fixed underlying schema issues first, C2ST rejection will discard most candidates and starve rare classes.

---

## 2. Generator Capacity

### Explanation
**Generator capacity** is the expressive size of the neural generator — determined by layer widths/depth, latent/embedding dimensionality, and optimization budget:
- **TVAE ([`components/tvae_generator.py`](../components/tvae_generator.py)):** `compress_dims` / `decompress_dims` (`128x128`, `256x256`, `256x256x256`), `embedding_dim` (`64`, `128`, `256`), and `epochs` (`300`).
- **CTGAN ([`components/ctgan_generator.py`](../components/ctgan_generator.py)):** `generator_dim` / `discriminator_dim` (`256x256`, `256x256x256`, `512x512`) and `embedding_dim`.
- **As an Implementation Feature (Fix 3.1):** Scale capacity, `batch_size`, `l2scale` / KL regularization, and validation early stopping to each class's real sample size $n_c$ (smaller capacity & stronger regularization for tiny classes like `Worms`, full capacity for larger classes).

### Benefits if Implemented
- Prevents tiny classes (`Worms`: `79` real rows; `Analysis`: `687` real rows) from memorizing training points (`DCR Closer-to-Train -> 100%`) or collapsing onto a single mode when SMOTE-NC is reduced or removed.
- Matches batch size to real class counts (`32–64` instead of `200`, avoiding 1-batch-per-epoch instability on `Worms`).
- Speeds up training on small classes via lower width and validation early stopping.

### Cons / Risks
- **Underfitting if scaled too low:** Too small a latent `embedding_dim` or network width on complex multi-modal classes can blur correlations (`Spearman MACE`) and reduce `Generative Recall`.
- **More configuration complexity:** Introduces size-tiered rules or per-tier hyperparameters to maintain and validate.
- **Incompatible with weight transfer if widths vary:** If you pretrain one base model and fine-tune per class, layer widths (`dims`, `embedding_dim`) must stay identical across classes — so scaling has to happen via learning rate, `l2scale`, early stopping, and encoder freezing instead.

---

## 3. Conditional Generation (`(Attack, PROTOCOL)`)

### Explanation
Instead of generating all protocols (`TCP = 6`, `UDP = 17`, `ICMP = 1`, etc.) from one unconditioned latent distribution per attack class, **conditional generation** explicitly conditions on `(Attack, PROTOCOL)`:
- **Unified CTGAN:** Condition on a composite discrete column `Attack_Proto = f"{Attack}__{PROTOCOL}"` and sample sub-quotas matching the real per-class protocol distribution $P_{\text{real}}(\text{PROTOCOL} \mid \text{Attack}=c)$.
- **Per-class TVAE:** Split each attack class into protocol branches (`TCP`, `UDP`, `ICMP/Other`) with protocol-appropriate feature masking (e.g., excluding `ICMP_*` on TCP flows and fixing them to `0`), sampling proportionally to $P_{\text{real}}(\text{ProtoBranch} \mid \text{Attack}=c)$.

### Benefits if Implemented
- Eliminates protocol cross-contamination (e.g., TCP flows emitting non-zero `ICMP_TYPE` or UDP flows emitting non-zero `TCP_FLAGS`/`TCP_WIN_MAX`) **by construction**.
- Preserves empirical protocol ratios per attack class, improving both **Categorical TVD** and **PFIR** before filtering even runs.

### Cons / Risks
- **Data fragmentation on rare classes:** Splitting a small class like `Worms` (`79` rows) or `Analysis` (`687` rows) into multiple protocol branches leaves minor protocols (e.g. a few UDP/ICMP rows) with too few samples to train a deep generator reliably (requiring a minimum-count fallback).
- **More models & runtime (TVAE):** Branching 7 attack classes $\times$ 2–3 protocols increases the number of sub-models to manage, save, and sample.

---

## 4. Mean KS (Mean 2-Sample Kolmogorov–Smirnov Statistic)

### Explanation
For each continuous feature $j$, the 2-sample KS statistic measures the maximum vertical gap between the real and synthetic empirical cumulative distribution functions (ECDFs):
$$\text{KS}_j = \sup_x \left| F_{\text{real}, j}(x) - F_{\text{syn}, j}(x) \right| \in [0, 1]$$
**Mean KS** averages $\text{KS}_j$ across all continuous features (`0.0` = identical 1D marginals).
- **Observed Baseline (`result_2`):** CTGAN = `0.2653`, TVAE = `0.1533`.
- **As an Implementation Feature (Fix 3.4 — Composite Tuning Penalty):** Include `Mean KS` in the Round 1–3 tuning objective (`Val Macro F1 - lambda_1 * Mean KS - lambda_2 * (1 - PFIR)`).

### Benefits if Implemented
- Fast, hyperparameter-free, non-parametric check of whether continuous marginals (durations, IATs, packet sizes) match real distributions.
- When added to the tuning objective, stops the tuner from selecting `y_max` or threshold settings that boost `macro_f1` slightly while severely distorting continuous distributions.

### Cons / Risks
- **Blind to joint correlations & protocol rules:** Two columns can each have `KS = 0.0` individually while violating cross-feature relationships (e.g., `SHORTEST_FLOW_PKT > LONGEST_FLOW_PKT`).
- **Sensitive to rare-class reweighting if not class-matched:** Comparing a rare-class-heavy synthetic pool against an `Exploits`/`Fuzzers`-heavy real attack pool shifts Mean KS even for good generators unless evaluated within covered/matched classes.
- **Requires tuning $\lambda_1$:** Too large a penalty in Round 1–3 can over-prioritize marginal fit at the expense of downstream IDS rare-class recall.

---

## 5. PFIR (Protocol / Cross-Feature Invariant Pass Rate)

### Explanation
**PFIR** is the fraction of synthetic rows that simultaneously satisfy **all** NetFlow physical, protocol, and real-range invariants (`MIN_TTL <= MAX_TTL`, `SHORTEST_FLOW_PKT <= LONGEST_FLOW_PKT <= MAX_IP_PKT_LEN`, `MIN_IP_PKT_LEN <= MAX_IP_PKT_LEN`, `IN_BYTES >= IN_PKTS`, `OUT_BYTES >= OUT_PKTS`, `TCP_FLAGS == 0` when UDP, `ICMP_TYPE == 0` when TCP/UDP, non-negative durations/counts, and real `[min, max]` bounds).
- **Observed Baseline (`result_2`):** CTGAN = `8.39%`, TVAE = `9.11%`.
- **Implemented in Filter (Fix 2.2):** [`SyntheticSampleFilter.validate_netflow_invariants()`](../components/synthetic_sample_filter.py#L209-L249) strictly rejects rows failing invariants (lifting filtered TVAE PFIR to **`100.00%`** in [`test/filter_test.ipynb`](../test/filter_test.ipynb)).

### Benefits if Implemented
- Guarantees 100% physically and protocol-valid synthetic NetFlow records enter the IDS training set, removing impossible combinations that C2ST and tree classifiers exploit.
- Provides an exact, interpretable diagnostic of generator structural quality.

### Cons / Risks
- **Severe yield collapse if used without generator fixes:** As seen in [`test/filter_test.ipynb`](../test/filter_test.ipynb), enforcing strict PFIR on an *unfixed* TVAE run rejected `93.95%` (`20,263 / 21,567`) of candidates, wiping out `Shellcode`, `Worms`, `Reconnaissance`, and `Analysis` completely and leaving `95%` of survivors in `Generic`.
- **Selection bias:** Rejecting 90%+ of candidates via PFIR without fixing the generator's feature representation (coupled deltas + categorical `ICMP_TYPE`) biases kept samples toward only whichever protocol/sub-cluster happened to satisfy the rules by chance.

"""Synthetic data quality for IDS: train on real data, then compare synthetic datasets (CTGAN / TVAE / ROS / SMOTE ...).

Flow:
  1. Split real data 80/20 by group (identical rows stay on one side). The 20% is the real test set and privacy holdout.
  2. Train the baseline: XGBoost attack-type classifier on real train only (TRTR).
  3. For each synthetic CSV:
       - keep only newly generated rows (drop rows already in the real data, and exact copies of test rows)
       - Coverage / Validity: classes generated, negative / fractional / above-real-max values
       - Fidelity: KS, Spearman correlation diff, q99 tail, L1-L3 (packet size, direction, TCP flags), per attack type
       - Privacy: DCR synthetic->train vs synthetic->holdout, exact copy rate
       - Utility: TSTR + ML Efficacy Ratio, TR+S (real + synthetic), SHAP rank stability
  4. Write summary.csv, per_class_f1_TSTR.csv, per_class_f1_TRS.csv, report.html.

Data here contains attack rows only (no benign), so utility is measured as attack-type classification, not detection.
Metric names follow "Document/Metric for IDS train.xlsx".

Usage:
  python ml_ids_detect.py --real REAL.csv [--test TEST.csv] --synthetic FILE_OR_FOLDER ... [--out-dir synthetic_report]
                          [--trees 300] [--jobs N] [--gpu]
"""
import argparse, glob, os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import DMatrix, XGBClassifier

SEED = 42
DEVICE = "cpu"       # "cuda" with --gpu (XGBoost only; k-NN and KS stay on CPU)
RARE_SHARE = 0.02    # classes below 2% of real train count as rare (rare_f1)


# ---------------- preprocessing ----------------

class Preprocessor:
    """Numeric: inf/NaN -> 0, optional clip to real train range. Strings: ordinal encoding (unseen -> -1)."""

    def fit(self, df, exclude):
        cols = [c for c in df.columns if c not in exclude]
        self.cat = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        self.num = [c for c in cols if c not in self.cat]
        self.vocab = {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))} for c in self.cat}
        X = self._num(df)
        self.lo, self.hi = X.min(), X.max()
        return self

    @property
    def features(self):
        return self.num + self.cat

    def _num(self, df):
        X = df[self.num].apply(pd.to_numeric, errors="coerce")
        return X.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")

    def transform(self, df, clip=True):
        """clip=False for distances: out-of-range synthetic values must not be hidden."""
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        X = self._num(df)
        if clip:
            X = X.clip(self.lo, self.hi, axis=1)
        for c in self.cat:
            X[c] = df[c].astype(str).map(self.vocab[c]).fillna(-1).astype("float32")
        return X


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def group_split(X, test_size, seed=SEED):
    """Split so rows with identical features stay on the same side (no leakage)."""
    g = pd.util.hash_pandas_object(X, index=False).values
    return next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(X, groups=g))


def make_xgb(n_classes, **kw):
    p = dict(n_estimators=300, learning_rate=0.08, max_depth=8, subsample=0.8, colsample_bytree=0.8,
             tree_method="hist", n_jobs=-1, random_state=SEED, device=DEVICE)
    p.update(kw)
    if n_classes == 2:
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **p)
    return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **p)


# ---------------- metrics ----------------

# column -> (good direction, description) for the report
METRICS = {
    "Coverage": {
        "n_rows": ("", "synthetic rows evaluated (after removing the two columns below)"),
        "real_rows_removed": ("", "rows identical to real data already in the file (e.g. oversampled files include the originals)"),
        "holdout_copies_removed": ("↓", "rows identical to a test row, removed to avoid test leakage into training"),
        "classes_covered": ("↑", "attack types generated / attack types in real data"),
        "missing_classes": ("", "attack types not generated (mode collapse)"),
    },
    "Validity": {
        "invalid_row_rate": ("↓", "share of rows with at least one impossible value (negative / fractional count / above real max)"),
        "neg_cell_rate": ("↓", "share of negative cells in columns that are never negative in real data"),
        "nonint_cell_rate": ("↓", "share of fractional cells in count columns (bytes, pkts, flags...)"),
        "over_max_cell_rate": ("↓", "share of cells above the real maximum"),
    },
    "Fidelity": {
        "ks_mean": ("↓", "mean KS statistic (per attack type, per feature)"),
        "ks_worst_feature": ("", "feature with the largest KS"),
        "corr_diff_mean": ("↓", "mean |Spearman_real - Spearman_syn| (per attack type)"),
        "tail_q99_log_ratio": ("↓", "mean |log10(q99_syn / q99_real)|: tail mismatch"),
        "L1_pktsize_entropy_diff": ("↓", "|packet size distribution entropy (NUM_PKTS_*) real - syn|"),
        "L2_direction_jsd": ("↓", "JSD of IN_BYTES/(IN+OUT) ratio, real vs syn"),
        "L3_flag_entropy_diff": ("↓", "|TCP_FLAGS entropy real - syn|"),
        "L3_flag_valid_rate": ("↑", "share of (rounded) TCP_FLAGS values that exist in real data"),
    },
    "Privacy": {
        "exact_copy_rate": ("↓", "share of synthetic rows identical to a real train row"),
        "dcr_train_p5": ("", "5th percentile distance to train (subsampled to holdout size)"),
        "dcr_holdout_p5": ("", "5th percentile distance to holdout"),
        "closer_to_train_rate": ("→0.5", "share of rows closer to train than to holdout; >> 0.5 = memorizing/copying train"),
    },
    "Utility": {
        "tstr_macro_f1": ("↑", "Train on Synthetic, Test on Real: macro-F1 over all attack types"),
        "tstr_mcc": ("↑", "MCC of TSTR"),
        "ml_efficacy_ratio": ("→1", "TSTR macro-F1 / TRTR macro-F1"),
        "trs_macro_f1": ("↑", "Train on Real + Synthetic, Test on Real: macro-F1"),
        "trs_rare_f1": ("↑", "TR+S: mean F1 of rare attack types (< 2% of train)"),
        "aug_gain_macro_f1": ("↑", "trs_macro_f1 - TRTR: does synthetic data help?"),
        "aug_gain_rare_f1": ("↑", "trs_rare_f1 - TRTR rare_f1"),
        "shap_rank_stability": ("→1", "Spearman of feature importance (SHAP) TSTR vs TRTR"),
    },
}


def _sample(n, cap, seed=SEED):
    return np.sort(np.random.default_rng(seed).choice(n, min(n, cap), replace=False))


def train_and_score(Xtr, ytr, Xte, yte, labels, rare, trees, jobs=-1):
    """Train an XGBoost attack-type classifier, test on real data.
    Returns (summary, per-class F1, mean |SHAP| per feature)."""
    cls = np.unique(ytr)
    if len(cls) < 2:
        return None, None, None
    ci = np.searchsorted(cls, ytr)
    m = make_xgb(len(cls), n_estimators=trees, n_jobs=jobs).fit(
        Xtr, ci, sample_weight=compute_sample_weight("balanced", ci) ** 0.5)
    pred = cls[m.predict_proba(Xte).argmax(1)]
    f1 = f1_score(yte, pred, labels=labels, average=None, zero_division=0)
    sh = m.get_booster().predict(DMatrix(Xte), pred_contribs=True)
    sh = np.abs(sh).mean(axis=tuple(range(sh.ndim - 1)))[:-1]   # drop bias column
    return ({"macro_f1": f1.mean(), "mcc": matthews_corrcoef(yte, pred), "rare_f1": f1[np.isin(labels, rare)].mean()},
            pd.Series(f1, labels), sh)


def _entropy(values):
    p = pd.Series(values).value_counts(normalize=True).values
    return float(-(p * np.log2(p)).sum())


def network_metrics(R, S):
    """L1-L3 on the NetFlow schema; skipped when columns are missing."""
    r = {}
    pk = [c for c in R if c.startswith("NUM_PKTS_")]
    if pk:
        def h(D):
            P = D[pk].clip(lower=0).values
            P = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
            return float(-(P * np.log2(np.where(P > 0, P, 1))).sum(1).mean())
        r["L1_pktsize_entropy_diff"] = abs(h(R) - h(S))
    if {"IN_BYTES", "OUT_BYTES"} <= set(R.columns):
        def asym(D):
            i, o = D.IN_BYTES.clip(lower=0), D.OUT_BYTES.clip(lower=0)
            return np.histogram(i / np.maximum(i + o, 1e-9), bins=20, range=(0, 1))[0] + 1e-9
        r["L2_direction_jsd"] = float(jensenshannon(asym(R), asym(S), base=2) ** 2)
    if "TCP_FLAGS" in R:
        fr, fs = R.TCP_FLAGS.round(), S.TCP_FLAGS.round()
        r["L3_flag_entropy_diff"] = abs(_entropy(fr) - _entropy(fs))
        r["L3_flag_valid_rate"] = float(fs.isin(set(fr)).mean())
    return r


def _row_hash(d, cols, attack_col):
    d = d[cols].copy()
    for c in cols:
        d[c] = d[c].astype(str).str.strip() if c == attack_col else pd.to_numeric(d[c], errors="coerce").astype("float64")
    return pd.util.hash_pandas_object(d, index=False).values


def new_rows(syn, real, test, attack_col):
    """Keep generated rows only: subtract real rows (by multiplicity), then drop exact copies of test rows.
    Columns with NaN/inf in real data are ignored when matching (oversampling tools fill those cells)."""
    both = pd.concat([real, test])
    finite = lambda c: c == attack_col or np.isfinite(pd.to_numeric(both[c], errors="coerce")).all()
    cols = [c for c in real.columns if c in syn.columns and finite(c)]
    h = _row_hash(syn, cols, attack_col)
    seen = pd.Series(_row_hash(both, cols, attack_col)).value_counts()
    rank = pd.Series(h).groupby(h).cumcount().values
    is_real = rank < pd.Series(h).map(seen).fillna(0).values
    in_test = np.isin(h, _row_hash(test, cols, attack_col)) & ~is_real
    return syn[~(is_real | in_test)].reset_index(drop=True), int(is_real.sum()), int(in_test.sum())


# ---------------- train + compare ----------------

def compare(real, test, files, out_dir, attack_col="Attack", label_col="Label", trees=300, cap=20000, jobs=-1):
    pre = Preprocessor().fit(real, {attack_col, label_col})
    names = lambda d: d[attack_col].astype(str).str.strip().values
    yr, yt = names(real), names(test)
    labels = np.unique(yr)
    share = pd.Series(yr).value_counts(normalize=True)
    rare = share[share < RARE_SHARE].index.values
    Xr, Xt = pre.transform(real), pre.transform(test)
    print(f"real train={len(real)} test/holdout={len(test)} classes={len(labels)} rare={list(rare)}")

    # 1. train baseline (TRTR)
    base, f1_base, sh_base = train_and_score(Xr, yr, Xt, yt, labels, rare, trees, jobs)
    print(f"TRTR macro-F1={base['macro_f1']:.4f} rare-F1={base['rare_f1']:.4f}")
    rows = [{"generator": "REAL", "variant": "TRTR (baseline)", "n_rows": len(real),
             "classes_covered": f"{len(labels)}/{len(labels)}", "missing_classes": "",
             "tstr_macro_f1": base["macro_f1"], "tstr_mcc": base["mcc"], "ml_efficacy_ratio": 1.0,
             "trs_macro_f1": base["macro_f1"], "trs_rare_f1": base["rare_f1"]}]
    f1_tstr, f1_trs = {"TRTR": f1_base}, {"TRTR": f1_base}

    # real-data statistics shared by every synthetic file
    R = pre._num(real)
    int_cols = R.columns[(R % 1 == 0).all()]
    nonneg = R.columns[R.min() >= 0]
    hi = R.max()
    L = lambda d: signed_log(pre.transform(d, clip=False)).values
    Lr = L(real)
    mu, sd = Lr.mean(0), Lr.std(0)
    sd[sd == 0] = 1
    Er, Eh = (Lr - mu) / sd, (L(test) - mu) / sd
    nn_full = NearestNeighbors(n_neighbors=1).fit(Er)
    nn_tr = NearestNeighbors(n_neighbors=1).fit(Er[_sample(len(Er), len(Eh))])   # same size as holdout
    nn_ho = NearestNeighbors(n_neighbors=1).fit(Eh)

    # 2. compare each synthetic file
    for f in files:
        gen = os.path.basename(os.path.dirname(os.path.abspath(f)))
        variant = os.path.splitext(os.path.basename(f))[0].replace("synthetic_", "") \
            .replace(gen.lower() + "_", "").replace("unsw_nb15_", "").replace("attacks_", "")
        if "oversampl" in gen.lower():
            gen = "OVERSAMPLE"
        tag = f"{gen}/{variant}"
        print(f"\n--- {tag} ---")
        syn, n_real, n_ho = new_rows(pd.read_csv(f), real, test, attack_col)
        ys = names(syn)
        keep = np.isin(ys, labels)                       # drop attack types absent from real data
        syn, ys = syn[keep].reset_index(drop=True), ys[keep]
        print(f"  removed {n_real} existing real rows, {n_ho} test copies -> {len(syn)} rows left")
        if len(syn) < 2:
            print("  no generated rows left -> skipped")
            continue
        row = {"generator": gen, "variant": variant, "n_rows": len(syn),
               "real_rows_removed": n_real, "holdout_copies_removed": n_ho}
        got = np.unique(ys)
        row["classes_covered"] = f"{len(got)}/{len(labels)}"
        row["missing_classes"] = ", ".join(sorted(set(labels) - set(got)))

        # validity
        S = pre._num(syn)
        neg, nonint, over = S[nonneg] < 0, S[int_cols] % 1 != 0, S > hi
        bad = neg.any(axis=1) | nonint.any(axis=1) | over.any(axis=1)
        row.update(invalid_row_rate=bad.mean(), neg_cell_rate=neg.values.mean(),
                   nonint_cell_rate=nonint.values.mean(), over_max_cell_rate=over.values.mean())

        # fidelity: compare each attack type with the same type in real data (class mix differs by design)
        ks_all, cor, tail = [], [], []
        for c in got:
            a, b = R[yr == c], S[ys == c]
            if len(b) < 2:
                continue
            k = pd.Series({x: ks_2samp(a[x], b[x]).statistic for x in R.columns})
            ks_all.append(k)
            cor.append(np.abs(a.corr("spearman").fillna(0) - b.corr("spearman").fillna(0)).values.mean())
            tail.append(np.abs(np.log10((b.quantile(.99).clip(lower=0) + 1) / (a.quantile(.99).clip(lower=0) + 1))).mean())
        if ks_all:
            ks = pd.concat(ks_all, axis=1).mean(1)
            row.update(ks_mean=ks.mean(), ks_worst_feature=ks.idxmax(), corr_diff_mean=np.mean(cor),
                       tail_q99_log_ratio=np.mean(tail))
        row.update(network_metrics(R[np.isin(yr, got)], S))

        # privacy: distances in standardized signed-log space, values NOT clipped
        i = _sample(len(syn), cap)
        Es = (L(syn.iloc[i]) - mu) / sd
        d_full = nn_full.kneighbors(Es)[0][:, 0]
        d_tr, d_ho = nn_tr.kneighbors(Es)[0][:, 0], nn_ho.kneighbors(Es)[0][:, 0]
        row.update(exact_copy_rate=(d_full < 1e-9).mean(), dcr_train_p5=np.percentile(d_tr, 5),
                   dcr_holdout_p5=np.percentile(d_ho, 5), closer_to_train_rate=(d_tr < d_ho).mean())

        # utility
        Xs = pre.transform(syn)
        tstr, f1_s, sh_s = train_and_score(Xs, ys, Xt, yt, labels, rare, trees, jobs)
        if tstr:
            row.update(tstr_macro_f1=tstr["macro_f1"], tstr_mcc=tstr["mcc"],
                       ml_efficacy_ratio=tstr["macro_f1"] / max(base["macro_f1"], 1e-9),
                       shap_rank_stability=spearmanr(sh_s, sh_base).statistic)
            f1_tstr[tag] = f1_s
        trs, f1_m, _ = train_and_score(pd.concat([Xr, Xs], ignore_index=True), np.r_[yr, ys],
                                       Xt, yt, labels, rare, trees, jobs)
        row.update(trs_macro_f1=trs["macro_f1"], trs_rare_f1=trs["rare_f1"],
                   aug_gain_macro_f1=trs["macro_f1"] - base["macro_f1"],
                   aug_gain_rare_f1=trs["rare_f1"] - base["rare_f1"])
        f1_trs[tag] = f1_m
        print(f"  TSTR={row.get('tstr_macro_f1', float('nan')):.4f}  TR+S={row['trs_macro_f1']:.4f} "
              f"(gain {row['aug_gain_macro_f1']:+.4f})  invalid_rows={row['invalid_row_rate']:.3f}  "
              f"closer_to_train={row['closer_to_train_rate']:.3f}")
        rows.append(row)

    # 3. write outputs
    summary = pd.DataFrame(rows).set_index(["generator", "variant"])
    per_class = {"TSTR": pd.DataFrame(f1_tstr), "TR+S": pd.DataFrame(f1_trs)}
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"))
    for k, v in per_class.items():
        v.to_csv(os.path.join(out_dir, f"per_class_f1_{k.replace('+', '')}.csv"))
    write_html(summary, per_class, rare, len(real), len(test), os.path.join(out_dir, "report.html"))
    print("\n=== SUMMARY ===")
    cols = ["classes_covered", "invalid_row_rate", "ks_mean", "closer_to_train_rate",
            "tstr_macro_f1", "ml_efficacy_ratio", "trs_macro_f1", "aug_gain_macro_f1", "aug_gain_rare_f1"]
    print(summary[[c for c in cols if c in summary]].round(4).to_string())
    print(f"\nwritten -> {out_dir}/summary.csv, per_class_f1_*.csv, report.html")
    return summary, per_class


def write_html(summary, per_class, rare, n_train, n_test, path):
    fmt = lambda v: "" if pd.isna(v) else f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    parts = []
    for group, cols in METRICS.items():
        cols = {c: v for c, v in cols.items() if c in summary}
        t = summary[list(cols)].copy()
        t.columns = [f"{c} {d}".strip() for c, (d, _) in cols.items()]
        legend = "".join(f"<li><b>{c}</b> {d} — {e}</li>" for c, (d, e) in cols.items())
        parts.append(f"<h2>{group}</h2>{t.to_html(na_rep='', formatters={c: fmt for c in t.columns})}"
                     f"<ul class='legend'>{legend}</ul>")
    for k, v in per_class.items():
        parts.append(f"<h2>F1 per attack type — {k}</h2>"
                     f"<p>Rows = training setup, columns = attack type in the real test set. "
                     f"Rare types (&lt; 2% of train): {', '.join(rare)}.</p>{v.T.to_html(float_format='{:.3f}'.format)}")
    notes = f"""
<h2>Reading notes</h2><ul>
<li>Real and synthetic data contain attack rows only (no Normal/Benign), so Utility is measured as
 <b>attack-type classification</b> ({len(per_class['TSTR'].index)} classes); detection (DR/FAR) is not measured.</li>
<li>The test set is always real data ({n_test} rows), split by groups of identical rows from train ({n_train} rows);
 synthetic data never enters the test set.</li>
<li>Privacy (DCR vs holdout) is only meaningful if the generators were <b>not</b> trained on the holdout rows. If CTGAN/TVAE
 were trained on the full original file, closer_to_train_rate is biased — confirm how data was split before generation.</li>
<li>Variants ros_10k / smote_* = generator trained on oversampled data, so they inherit ROS copies / SMOTE interpolations.</li>
<li>OVERSAMPLE = the original ROS/SMOTE files. They include the real data, so only generated rows are kept (real rows
 subtracted by multiplicity). Rows identical to a test row are removed for every generator (see holdout_copies_removed).
 ROS/SMOTE were built on the full original data (including test), so indirect leakage remains: SMOTE interpolates from
 test rows, ROS copied test rows (removed). OVERSAMPLE only adds rare classes, so its TSTR covers a few classes only —
 compare mainly on TR+S.</li>
<li>Each number is a single run (seed {SEED}); differences below ~0.01 macro-F1 are not conclusive.</li>
<li>Not computed: FID/MMD, MIA, Differential Privacy (not applicable to standard CTGAN/TVAE).</li></ul>"""
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Synthetic Data Quality</title>
<style>
body{{font-family:system-ui,Segoe UI,sans-serif;margin:24px;color:#1f2328;background:#fff;max-width:1400px}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #d0d7de;padding-bottom:4px}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0}} th,td{{border:1px solid #d0d7de;padding:4px 8px;text-align:right}}
th{{background:#f6f8fa}} tbody th{{text-align:left}} tr:nth-child(even) td{{background:#fafbfc}}
.legend{{font-size:12px;color:#57606a}} .wrap{{overflow-x:auto}}
</style></head><body>
<h1>Synthetic data quality (CTGAN / TVAE / oversampling) for IDS — UNSW-NB15 NetFlow</h1>
<p>Metrics follow "Metric for IDS train.xlsx". ↑ higher is better, ↓ lower is better, →x closer to x is better.
The REAL / TRTR row is the baseline trained on real data only.</p>
<div class="wrap">{''.join(parts)}</div>{notes}</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def csv_files(paths):
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(p, "**", "*.csv"), recursive=True)) if os.path.isdir(p) else [p]
    return out


# ---------------- CLI ----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="real data CSV")
    ap.add_argument("--test", help="real test CSV; if omitted, --real is split 80/20 by group (20%% = test + privacy holdout)")
    ap.add_argument("--synthetic", nargs="+", required=True, help="CSV files or folders (all *.csv inside)")
    ap.add_argument("--attack-col", default="Attack")
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--trees", type=int, default=300, help="XGBoost trees per model")
    ap.add_argument("--cap", type=int, default=20000, help="max synthetic rows used for DCR")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="CPU cores for XGBoost (default: half of the cores, -1 = all)")
    ap.add_argument("--gpu", action="store_true", help="train XGBoost on an NVIDIA GPU (CUDA)")
    ap.add_argument("--out-dir", default="synthetic_report")
    a = ap.parse_args()

    if a.gpu:
        import xgboost
        DEVICE = "cuda"
        xgboost.set_config(verbosity=0)   # silence "mismatched devices" warning when predicting from CPU data

    data = pd.read_csv(a.real)
    if a.test:
        real, test = data, pd.read_csv(a.test)
    else:
        tr, te = group_split(data.drop(columns=[c for c in (a.attack_col, a.label_col) if c in data]), 0.2)
        real, test = data.iloc[tr].reset_index(drop=True), data.iloc[te].reset_index(drop=True)
        print("[no --test] split 80/20 by group (identical rows stay on one side)")
    compare(real, test, csv_files(a.synthetic), a.out_dir, a.attack_col, a.label_col, a.trees, a.cap, a.jobs)

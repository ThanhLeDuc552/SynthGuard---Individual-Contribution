"""Two-tier IDS test: does the IDS detect attacks (benign vs attack) and tell attack types apart?

Tier 1 (same for every configuration): benign vs attack, trained on the binary train fold.
    alert = supervised XGBoost  OR  anomaly detector (IsolationForest + robust z, learns benign only)
Tier 2 (one classifier per configuration): which attack type, trained on the files of that configuration.
    flows alerted by tier 1 go to tier 2; low confidence, or caught only by the anomaly branch -> "Unknown".

Flow test (see "FLow test.md"): tier 1 is fixed, tier 2 is trained on different data per configuration, e.g.
    A_original         = original attack file
    B_multiclass       = multiclass file (real + SMOTE)
    C_multiclass+CTGAN = multiclass + CTGAN
    D_multiclass+TVAE  = multiclass + TVAE
so any difference between configurations comes from the tier-2 training data only. The test fold is never trained on.

Tables written to --out-dir (CSV + report.html):
    1. Configuration comparison: tier-2 alone (true attacks), end-to-end (both tiers), delta vs --baseline
    2. Tier 1: detection rate, false alarm rate, precision, F1, MCC, PR-AUC ... (benign vs attack)
    3. Tier 1 per attack type: how often each type is alerted
    4. Tier 2 F1 per attack type, per configuration
    5. End-to-end recall per class (alerted AND correctly typed; Benign = not alerted), per configuration

Nothing is split, filtered or removed in your data. Overlap between a configuration's training data and the test
attacks is measured and reported (test_attacks_in_train).

Usage:
  python ids_pipeline.py --train "…/train/NF-UNSW-NB15-v3_fold_{fold}.csv" --test "…/test/NF-UNSW-NB15-v3_fold_{fold}.csv" \
      --folds 1 2 3 4 5 --config "NAME=file1.csv[,file2.csv…]" [--config …] [--baseline NAME] --out-dir ids_report [--gpu]
"""
import argparse, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, precision_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SEED = 42
UNKNOWN = "Unknown"
DEVICE = "cpu"      # "cuda" with --gpu (XGBoost only; IsolationForest stays on CPU)


# ---------------- preprocessing ----------------

class Preprocessor:
    """Numeric: inf/NaN -> median of the fitting data, clip to its range. Strings: ordinal (unseen -> -1).
    Fitted separately per tier, so each tier fills missing cells the way its own training data was filled."""

    def fit(self, df, exclude):
        cols = [c for c in df.columns if c not in exclude]
        self.cat = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        self.num = [c for c in cols if c not in self.cat]
        self.vocab = {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))} for c in self.cat}
        X = self._clean(df)
        self.fill = X.median().fillna(0)
        X = X.fillna(self.fill)
        self.lo, self.hi = X.min(), X.max()
        return self

    def _clean(self, df):
        return df[self.num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    @property
    def features(self):
        return self.num + self.cat

    def transform(self, df, clip=True):
        """clip=False for the anomaly branch: out-of-range values are exactly what it should notice."""
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        X = self._clean(df).fillna(self.fill).astype("float32")
        if clip:
            X = X.clip(self.lo, self.hi, axis=1)
        for c in self.cat:
            X[c] = df[c].astype(str).map(self.vocab[c]).fillna(-1).astype("float32")
        return X


class AnomalyDetector:
    """Learns benign only. Two signals, each normalised by its own threshold (> 1 means anomalous):
    IsolationForest (rare feature combinations) and robust z on a log scale (values far beyond anything benign)."""

    def fit(self, X_ben, X_ref, fpr):
        self.iforest = IsolationForest(n_estimators=300, max_samples=min(4096, len(X_ben)),
                                       random_state=SEED, n_jobs=-1).fit(X_ben)
        L = self._log(X_ben)
        self.med = L.median()
        self.iqr = (L.quantile(0.75) - L.quantile(0.25)).clip(lower=1.0)
        self.thr_if = np.quantile(self._if(X_ref), 1 - fpr / 2)   # each branch gets half of the false alarm budget
        self.thr_z = np.quantile(self._z(X_ref), 1 - fpr / 2)
        return self

    @staticmethod
    def _log(X):
        return np.sign(X) * np.log1p(np.abs(X))

    def _if(self, X):
        return -self.iforest.score_samples(X)

    def _z(self, X):
        return ((self._log(X) - self.med) / self.iqr).abs().max(axis=1).values

    def score(self, X):
        return np.maximum(self._if(X) / self.thr_if, self._z(X) / max(self.thr_z, 1e-9))


def group_split(X, test_size, seed=SEED):
    """Rows with identical features stay on the same side (no leakage between fit and validation)."""
    g = pd.util.hash_pandas_object(X, index=False).values
    return next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(X, groups=g))


def make_xgb(n_classes, **kw):
    p = dict(n_estimators=300, learning_rate=0.08, max_depth=8, subsample=0.8, colsample_bytree=0.8,
             tree_method="hist", n_jobs=-1, random_state=SEED, device=DEVICE)
    p.update(kw)
    if n_classes == 2:
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **p)
    return XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **p)


# ---------------- the two tiers ----------------

class Detector:
    """Tier 1: benign vs attack. Alert = supervised score > threshold OR anomaly score > 1.
    Thresholds are calibrated on a validation part of the training fold, so the false alarm rate is set, not hoped for."""

    def __init__(self, max_fpr, anomaly_fpr, trees, jobs, val_size=0.15, weight_power=0.5):
        self.max_fpr, self.anomaly_fpr, self.trees, self.jobs = max_fpr, anomaly_fpr, trees, jobs
        self.val_size, self.weight_power = val_size, weight_power

    def fit(self, X, y):
        tr, va = group_split(X, self.val_size)
        ben_tr, ben_va = tr[y[tr] == 0], va[y[va] == 0]
        if not len(ben_va):
            raise SystemExit("tier 1 needs benign rows in the validation part of --train")
        self.anomaly = AnomalyDetector().fit(X.iloc[ben_tr], X.iloc[ben_va], self.anomaly_fpr)
        w = compute_sample_weight("balanced", y[tr]) ** self.weight_power
        self.model = make_xgb(2, n_estimators=self.trees, n_jobs=self.jobs).fit(X.iloc[tr], y[tr], sample_weight=w)
        s_va = self.model.predict_proba(X.iloc[va])[:, 1]
        self.thr = float(np.clip(np.quantile(s_va[y[va] == 0], 1 - self.max_fpr), 0.05, 0.95))
        return self

    def predict(self, pre, df):
        """Strict '>' on purpose: many benign flows share the exact same score; '>=' would alert all of them
        when the calibrated threshold lands on that tied value."""
        sup = self.model.predict_proba(pre.transform(df))[:, 1]
        ano = self.anomaly.score(pre.transform(df, clip=False))
        return sup, ano, sup > self.thr, ano > 1


class AttackTyper:
    """Tier 2: attack type, trained on attack rows only (benign rows, if present, are ignored)."""

    def __init__(self, attack_col, label_col, benign, trees, jobs, weight_power=0.5):
        self.attack_col, self.label_col, self.benign = attack_col, label_col, benign
        self.trees, self.jobs, self.weight_power = trees, jobs, weight_power

    def fit(self, df):
        name = df[self.attack_col].astype(str).str.strip().values
        df, name = df[name != self.benign], name[name != self.benign]
        self.pre = Preprocessor().fit(df, {self.attack_col, self.label_col})
        self.classes_ = np.unique(name)
        ci = np.searchsorted(self.classes_, name)
        w = compute_sample_weight("balanced", ci) ** self.weight_power
        self.model = make_xgb(len(self.classes_), n_estimators=self.trees, n_jobs=self.jobs).fit(
            self.pre.transform(df), ci, sample_weight=w)
        return self

    def predict_proba(self, df):
        return self.model.predict_proba(self.pre.transform(df))


# ---------------- evaluation ----------------

def targets(df, attack_col, label_col, benign):
    name = df[attack_col].astype(str).str.strip().values
    y = df[label_col].astype(int).values if label_col in df else (name != benign).astype(int)
    return y, np.where(y == 1, name, benign)


def row_hashes(df, cols):
    return pd.util.hash_pandas_object(df[cols].apply(pd.to_numeric, errors="coerce").astype("float64"),
                                      index=False).values


def tier1_metrics(y, alert, hit_s, hit_a, score):
    ben = y == 0
    tn, fp, fn, tp = confusion_matrix(y, alert.astype(int), labels=[0, 1]).ravel()
    return {"detection_rate": tp / (tp + fn), "false_alarm_rate": fp / (fp + tn),
            "precision": precision_score(y, alert.astype(int), zero_division=0),
            "f1": f1_score(y, alert.astype(int), zero_division=0),
            "mcc": matthews_corrcoef(y, alert.astype(int)),
            "pr_auc": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score),
            "dr_supervised": hit_s[~ben].mean(), "far_supervised": hit_s[ben].mean(),
            "dr_anomaly": hit_a[~ben].mean(), "far_anomaly": hit_a[ben].mean(),
            "dr_caught_only_by_anomaly": (hit_a & ~hit_s)[~ben].mean(),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def run_fold(k, train, test, cfgs, a):
    t0 = time.time()
    exclude = {a.attack_col, a.label_col}
    y_tr, _ = targets(train, a.attack_col, a.label_col, a.benign)
    if len(np.unique(y_tr)) < 2:
        raise SystemExit("tier 1 needs both benign and attack rows in --train")
    y_te, n_te = targets(test, a.attack_col, a.label_col, a.benign)
    att = y_te == 1
    labels_att = sorted(np.unique(n_te[att]))
    labels_all = [a.benign] + labels_att
    share = pd.Series(n_te[att]).value_counts(normalize=True)
    rare = sorted(share[share < a.rare_share].index)
    rm = np.isin(labels_att, rare)
    print(f"\n===== fold {k}: train={len(train)} test={len(test)} (attacks {att.sum()}, benign {(~att).sum()}) "
          f"rare={rare} =====")

    # ---- tier 1
    pre1 = Preprocessor().fit(train, exclude)
    det = Detector(a.max_fpr, a.anomaly_fpr, a.trees, a.jobs).fit(pre1.transform(train), y_tr)
    sup, ano, hit_s, hit_a = det.predict(pre1, test)
    alert = hit_s | hit_a
    score = np.maximum(sup / det.thr, ano)
    t1 = {"fold": k, "threshold": det.thr, **tier1_metrics(y_te, alert, hit_s, hit_a, score)}
    t1_class = [{"fold": k, "class": c, "n": int((n_te == c).sum()), "alert_rate": alert[n_te == c].mean(),
                 "by_supervised": hit_s[n_te == c].mean(), "by_anomaly": hit_a[n_te == c].mean(),
                 "only_anomaly": (hit_a & ~hit_s)[n_te == c].mean()} for c in labels_all]
    print(f"tier 1: DR={t1['detection_rate']:.4f} FAR={t1['false_alarm_rate']:.4f} MCC={t1['mcc']:.4f} "
          f"PR-AUC={t1['pr_auc']:.4f}  ({time.time() - t0:.0f}s)")

    # ---- tier 2, one classifier per configuration
    feat = [c for c in test.columns if c not in exclude]
    fin = [c for c in feat if np.isfinite(pd.to_numeric(test[c], errors="coerce")).all()]   # skip NaN/inf columns
    h_te = row_hashes(test[att], fin)
    overall, per_class = [], []
    for name, cfg in cfgs.items():
        typer = AttackTyper(a.attack_col, a.label_col, a.benign, a.trees, a.jobs).fit(cfg)
        P = typer.predict_proba(test)
        conf, kind = P.max(1), typer.classes_[P.argmax(1)]

        pred_iso = kind[att]       # tier 2 alone on the true attack flows (no gating by tier 1)
        f1_iso = f1_score(n_te[att], pred_iso, labels=labels_att, average=None, zero_division=0)

        kind2 = np.where(conf < a.min_confidence, UNKNOWN, kind)
        kind2 = np.where(hit_a & ~hit_s, UNKNOWN, kind2)        # only the anomaly branch saw it -> unknown attack
        final = np.where(alert, kind2, a.benign)                # what the real IDS would output per flow
        p, r, f, sup_n = precision_recall_fscore_support(n_te, final, labels=labels_all, zero_division=0)
        rm_all = np.r_[False, rm]

        overall.append({"fold": k, "config": name, "train_rows": len(cfg),
                        "t2_macro_f1": f1_iso.mean(), "t2_mcc": matthews_corrcoef(n_te[att], pred_iso),
                        "t2_rare_f1": f1_iso[rm].mean() if rm.any() else np.nan,
                        "e2e_macro_f1": f.mean(), "e2e_rare_recall": r[rm_all].mean() if rm.any() else np.nan,
                        "e2e_attack_recall": float((final[att] == n_te[att]).mean()),
                        "unknown_rate": float((final[alert] == UNKNOWN).mean()) if alert.any() else np.nan,
                        "test_attacks_in_train": float(np.isin(h_te, row_hashes(cfg, fin)).mean())})
        for i, c in enumerate(labels_all):
            per_class.append({"fold": k, "config": name, "class": c, "support": int(sup_n[i]),
                              "e2e_precision": p[i], "e2e_recall": r[i], "e2e_f1": f[i],
                              "t2_f1": f1_iso[i - 1] if i else np.nan})
        o = overall[-1]
        print(f"  {name:28s} tier2 macro-F1={o['t2_macro_f1']:.4f}  end-to-end macro-F1={o['e2e_macro_f1']:.4f}  "
              f"rare recall={o['e2e_rare_recall']:.4f}  ({time.time() - t0:.0f}s)")
        del typer, P
    return t1, t1_class, overall, per_class, rare


# ---------------- tables ----------------

OVERALL_COLS = {
    "t2_macro_f1": ("↑", "Tier 2 alone, true attack flows: macro-F1 over attack types"),
    "t2_rare_f1": ("↑", "Tier 2 alone: mean F1 of the rare attack types"),
    "e2e_macro_f1": ("↑", "End-to-end (both tiers): macro-F1 over Benign + attack types ('Unknown' counts as wrong)"),
    "e2e_rare_recall": ("↑", "End-to-end: mean recall of the rare types = alerted AND typed correctly"),
    "e2e_attack_recall": ("↑", "End-to-end: share of ALL attack flows alerted and typed correctly"),
    "unknown_rate": ("", "Share of alerted flows labelled Unknown"),
    "test_attacks_in_train": ("↓", "Share of test attack flows that also appear in this configuration's training data"),
    "delta_t2_macro_f1": ("↑", "t2_macro_f1 minus the baseline configuration (same fold)"),
    "delta_e2e_macro_f1": ("↑", "e2e_macro_f1 minus the baseline configuration"),
    "delta_e2e_rare_recall": ("↑", "e2e_rare_recall minus the baseline configuration"),
}
TIER1_COLS = {
    "detection_rate": "Share of attack flows alerted (DR / TPR)",
    "false_alarm_rate": "Share of benign flows alerted (FAR / FPR)",
    "precision": "Share of alerts that are real attacks",
    "f1": "F1 of the alert decision",
    "mcc": "Matthews correlation of the alert decision",
    "pr_auc": "PR-AUC on the combined score (alert <=> score > 1)",
    "roc_auc": "ROC-AUC on the combined score",
    "dr_supervised": "DR of the supervised branch alone",
    "far_supervised": "FAR of the supervised branch alone",
    "dr_anomaly": "DR of the anomaly branch alone",
    "far_anomaly": "FAR of the anomaly branch alone",
    "dr_caught_only_by_anomaly": "Attacks that only the anomaly branch caught",
    "threshold": "Supervised threshold calibrated on the validation part of --train",
    "TN": "True negatives", "FP": "False positives", "FN": "False negatives", "TP": "True positives",
}


COUNT_COLS = {"n", "TN", "FP", "FN", "TP"}


def ms(m, s, folds, nd=3, pm="±"):
    if pd.isna(m):
        return ""
    return f"{m:.{nd}f}" + (f" {pm} {s:.{nd}f}" if folds > 1 and not pd.isna(s) else "")


def fmt_frame(m, s, folds, nd=3, pm="±"):
    out = pd.DataFrame(index=m.index, columns=m.columns, dtype=object)
    for c in m.columns:
        out[c] = [ms(a, b, folds, 0 if c in COUNT_COLS else nd, pm) for a, b in zip(m[c], s[c])]
    return out


def build_tables(t1, t1_class, overall, per_class, baseline, rare):
    t1, t1_class = pd.DataFrame(t1), pd.DataFrame(t1_class)
    overall, per_class = pd.DataFrame(overall), pd.DataFrame(per_class)
    for m in ["t2_macro_f1", "e2e_macro_f1", "e2e_rare_recall"]:
        b = overall[overall.config == baseline].set_index("fold")[m]
        overall["delta_" + m] = overall[m] - overall["fold"].map(b)

    out = {}
    cols = list(OVERALL_COLS)
    g = overall.groupby("config", sort=False)[cols]
    out["configs"] = (g.mean(), g.std(ddof=0))

    tc = [c for c in TIER1_COLS if c in t1]
    out["tier1"] = (t1[tc].mean(), t1[tc].std(ddof=0))

    gc = t1_class.groupby("class", sort=False)[["n", "alert_rate", "by_supervised", "by_anomaly", "only_anomaly"]]
    m, s = gc.mean(), gc.std(ddof=0)
    order = ["Benign"] + list(m.drop(index="Benign").sort_values("alert_rate").index) if "Benign" in m.index else list(m.index)
    out["tier1_class"] = (m.loc[order], s.loc[order])

    star = lambda c: c + "*" if c in rare else c
    for key, col in [("t2_f1", "t2_f1"), ("e2e_recall", "e2e_recall")]:
        g = per_class.groupby(["config", "class"], sort=False)[col]
        mm, ss = g.mean().unstack(), g.std(ddof=0).unstack().fillna(0)
        cls = [c for c in mm.columns if not (col == "t2_f1" and c == "Benign")]
        mm, ss = mm[cls].reindex(list(overall.config.unique())), ss[cls].reindex(list(overall.config.unique()))
        mm.columns = ss.columns = [star(c) for c in cls]
        out[key] = (mm, ss)
    out["raw"] = (overall, per_class, t1, t1_class)
    return out


def save_csvs(tabs, out_dir, folds):
    overall, per_class, t1, _ = tabs["raw"]
    def both(m, s):
        return m if folds == 1 else m.join(s, rsuffix="_std") if isinstance(m, pd.DataFrame) else m
    both(*tabs["configs"]).to_csv(os.path.join(out_dir, "summary_configs.csv"))
    pd.DataFrame({"mean": tabs["tier1"][0], "std": tabs["tier1"][1]}).to_csv(os.path.join(out_dir, "tier1_detection.csv"))
    both(*tabs["tier1_class"]).to_csv(os.path.join(out_dir, "tier1_per_attack.csv"))
    both(*tabs["t2_f1"]).to_csv(os.path.join(out_dir, "tier2_f1_per_class.csv"))
    both(*tabs["e2e_recall"]).to_csv(os.path.join(out_dir, "end_to_end_recall_per_class.csv"))
    overall.to_csv(os.path.join(out_dir, "per_fold_configs.csv"), index=False)
    per_class.to_csv(os.path.join(out_dir, "per_fold_per_class.csv"), index=False)
    t1.to_csv(os.path.join(out_dir, "per_fold_tier1.csv"), index=False)


def write_html(tabs, a, rare, folds, path):
    h = lambda df, **kw: df.to_html(border=0, na_rep="", **kw)
    cm, cs = tabs["configs"]
    t_cfg = fmt_frame(cm, cs, folds)
    t_cfg.columns = [f"{c} {OVERALL_COLS[c][0]}".strip() for c in t_cfg.columns]
    legend = "".join(f"<li><b>{c}</b> {d} — {e}</li>" for c, (d, e) in OVERALL_COLS.items())
    tm, ts = tabs["tier1"]
    t_t1 = pd.DataFrame({"value": [ms(tm[c], ts[c], folds, 0 if c in COUNT_COLS else 4) for c in tm.index],
                         "meaning": [TIER1_COLS[c] for c in tm.index]}, index=tm.index)
    km, ks = tabs["tier1_class"]
    t_cls = fmt_frame(km, ks, folds)
    t_cls.index = [c + ("*" if c in rare else "") for c in t_cls.index]
    t_cls.columns = ["flows", "alerted (DR; Benign row = FAR)", "by supervised", "by anomaly", "only anomaly"]
    f1m, f1s = tabs["t2_f1"]
    rm, rs = tabs["e2e_recall"]
    notes = f"""<h2>Reading notes</h2><ul>
<li>Folds: {folds}{' (mean ± std over folds)' if folds > 1 else ''}. Test flows are never trained on. Single seed ({SEED});
 differences below ~0.01 are not conclusive. Types marked * are rare (&lt; {a.rare_share:.0%} of attack flows): {', '.join(rare)}.</li>
<li>Tier 1 is identical for every configuration (trained on the binary train fold); only the tier-2 training data differs.</li>
<li>The baseline for the delta columns is <b>{a.baseline_name}</b>.</li>
<li><b>test_attacks_in_train</b> shows how much of the test attack traffic a configuration has already seen in training.
 Where it is high, absolute scores are optimistic and configurations with a higher share are favoured.</li>
<li>Missing cells (NaN/inf) are filled with the median of each tier's own training data; no rows are removed or split.</li>
<li>Flows alerted only by the anomaly branch, or typed with confidence &lt; {a.min_confidence}, are reported as Unknown and count as wrong.</li>
<li>The holdout set is not used here.</li></ul>"""
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>IDS two-tier test</title>
<style>
body{{font-family:system-ui,Segoe UI,sans-serif;margin:24px;color:#1f2328;background:#fff;max-width:1500px}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #d0d7de;padding-bottom:4px}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0}} th,td{{border:1px solid #d0d7de;padding:4px 8px;text-align:right}}
th{{background:#f6f8fa}} tbody th{{text-align:left}} tr:nth-child(even) td{{background:#fafbfc}}
.legend{{font-size:12px;color:#57606a}} .wrap{{overflow-x:auto}} td:last-child{{text-align:right}}
</style></head><body>
<h1>IDS two-tier test: benign vs attack detection and attack-type classification</h1>
<p>Tier 1 (benign / attack) → alerted flows → tier 2 (attack type). ↑ higher is better, ↓ lower is better.</p>
<div class="wrap">
<h2>1. Configuration comparison (tier-2 training data)</h2>{h(t_cfg)}<ul class="legend">{legend}</ul>
<h2>2. Tier 1: benign vs attack detection</h2>{h(t_t1)}
<h2>3. Tier 1 per class (alert rate)</h2>{h(t_cls)}
<h2>4. Tier 2 alone: F1 per attack type</h2>{h(fmt_frame(f1m, f1s, folds))}
<h2>5. End-to-end recall per class (alerted AND correctly typed; Benign = not alerted)</h2>{h(fmt_frame(rm, rs, folds))}
</div>{notes}</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def print_tables(tabs, folds):
    pm = "+/-"   # Windows consoles often can't print "±"
    cm, cs = tabs["configs"]
    print("\n=== 1. CONFIGURATION COMPARISON ===")
    print(fmt_frame(cm, cs, folds, pm=pm).to_string())
    tm, ts = tabs["tier1"]
    print("\n=== 2. TIER 1: BENIGN vs ATTACK ===")
    print(pd.Series({c: ms(tm[c], ts[c], folds, 0 if c in COUNT_COLS else 4, pm) for c in tm.index}).to_string())
    km, ks = tabs["tier1_class"]
    print("\n=== 3. TIER 1 PER CLASS (alert rate; Benign row = FAR) ===")
    print(fmt_frame(km, ks, folds, pm=pm).to_string())
    print("\n=== 4. TIER 2 ALONE: F1 PER ATTACK TYPE ===")
    print(fmt_frame(*tabs["t2_f1"], folds, pm=pm).to_string())
    print("\n=== 5. END-TO-END RECALL PER CLASS ===")
    print(fmt_frame(*tabs["e2e_recall"], folds, pm=pm).to_string())


# ---------------- CLI ----------------

def parse_config(s):
    name, _, files = s.partition("=")
    if not name.strip() or not files.strip():
        raise SystemExit(f'--config expects "NAME=file1.csv[,file2.csv...]", got: {s}')
    return name.strip(), [f.strip() for f in files.split(",") if f.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="binary train CSV for tier 1; use {fold} for several folds")
    ap.add_argument("--test", required=True, help="binary test CSV (benign + attack); use {fold} for several folds")
    ap.add_argument("--folds", type=int, nargs="+", default=[1], help="fold numbers replacing {fold}")
    ap.add_argument("--config", action="append", required=True, metavar="NAME=FILE[,FILE...]",
                    help="tier-2 training data of one configuration (files are concatenated); repeat per configuration")
    ap.add_argument("--baseline", help="configuration the delta columns compare against (default: the first)")
    ap.add_argument("--attack-col", default="Attack")
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--benign", default="Benign")
    ap.add_argument("--max-fpr", type=float, default=0.01, help="false alarm budget of the supervised branch")
    ap.add_argument("--anomaly-fpr", type=float, default=0.005, help="false alarm budget of the anomaly branch")
    ap.add_argument("--min-confidence", type=float, default=0.4, help="below this tier-2 confidence -> Unknown")
    ap.add_argument("--rare-share", type=float, default=0.05, help="attack types below this share of attack flows are rare")
    ap.add_argument("--trees", type=int, default=300, help="XGBoost trees per model")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="CPU cores for XGBoost (default: half of the cores, -1 = all)")
    ap.add_argument("--gpu", action="store_true", help="train XGBoost on an NVIDIA GPU (CUDA)")
    ap.add_argument("--out-dir", default="ids_report")
    a = ap.parse_args()

    if a.gpu:
        import xgboost
        DEVICE = "cuda"
        xgboost.set_config(verbosity=0)   # silence "mismatched devices" warning when predicting from CPU data

    cfgs = {}
    for c in a.config:
        name, files = parse_config(c)
        cfgs[name] = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        if a.attack_col not in cfgs[name]:
            raise SystemExit(f"config {name}: column '{a.attack_col}' not found")
        print(f"config {name}: {len(cfgs[name])} rows from {len(files)} file(s)")
    a.baseline_name = a.baseline or next(iter(cfgs))
    if a.baseline_name not in cfgs:
        raise SystemExit(f"--baseline {a.baseline_name} is not one of {list(cfgs)}")

    several = "{fold}" in a.train or "{fold}" in a.test
    folds = a.folds if several else a.folds[:1]
    T1, T1C, OV, PC, rare = [], [], [], [], []
    for k in folds:
        train = pd.read_csv(a.train.format(fold=k))
        test = pd.read_csv(a.test.format(fold=k))
        t1, t1c, ov, pc, rare = run_fold(k, train, test, cfgs, a)
        T1.append(t1); T1C += t1c; OV += ov; PC += pc
        del train, test

    tabs = build_tables(T1, T1C, OV, PC, a.baseline_name, rare)
    os.makedirs(a.out_dir, exist_ok=True)
    save_csvs(tabs, a.out_dir, len(folds))
    write_html(tabs, a, rare, len(folds), os.path.join(a.out_dir, "report.html"))
    print_tables(tabs, len(folds))
    print(f"\nwritten -> {a.out_dir}/report.html and CSV tables")

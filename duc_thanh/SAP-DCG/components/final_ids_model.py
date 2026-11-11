#!/usr/bin/env python3
"""
Final Intrusion Detection System (IDS) Model (Component 4)

Derived from Tuan's IDS pipeline (`tuan/ML-IDS/ids_pipeline.py` and `tuan/ML-IDS/ml_ids_detect.py`):
- Two-Tier IDS Architecture:
  - Tier 1: Benign vs Attack Detection (Supervised XGBoost + IsolationForest/Robust-Z Anomaly Detector).
    - Anomaly Detector: Trained strictly on Benign flows. Evaluates IsolationForest + signed-log robust-z score.
    - Supervised Detector: Binary XGBoost with balanced sample weights (weight_power=0.5).
    - Threshold Calibration: Calibrated on a held-out validation split against target false-alarm rate (max_fpr).
  - Tier 2: Multiclass Attack Classifier (AttackTyper).
    - Multi-class XGBoost with balanced sample weights, trained on attack flows.
    - Gated decision logic: Low confidence (< min_confidence) or anomaly-only alerts labeled as "Unknown".
- Single-tier Fallback Mode:
  - Multiclass classifier directly trained when only attack flows or flat labels are present.
- Comprehensive Metrics Suite:
  - End-to-end two-tier metrics: DR (detection rate), FAR (false alarm rate), MCC, PR-AUC, ROC-AUC,
    t2_macro_f1 (isolated tier 2), e2e_macro_f1 (end-to-end), e2e_rare_recall, unknown_rate.
  - Standard multiclass metrics: macro_f1, weighted_f1, balanced_accuracy, accuracy, binary_f1,
    per_class breakdown, and confusion matrix.
  - Synthetic data quality comparison suite (Coverage, Validity, Fidelity, Privacy, Utility).
"""

import os
import sys
import time
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.append(str(curr_dir))

from intra_attack_resampler import resolve_paths

UNKNOWN = "Unknown"
DEFAULT_BENIGN_LABEL = "Benign"
RARE_SHARE_THRESHOLD = 0.05


# ============================================================================
# 1. Preprocessor (Tuan's ML-IDS Preprocessing Logic)
# ============================================================================
class Preprocessor:
    """
    Handles numerical sanitization (inf/NaN -> median, bounds clipping)
    and categorical ordinal encoding (unseen categories -> -1).
    clip=False is preserved for anomaly detection and out-of-distribution sensing.
    """

    def __init__(self, exclude: Optional[List[str]] = None):
        self.exclude = set(exclude or ["Label", "Attack", "OOF_Confidence"])
        self.cat: List[str] = []
        self.num: List[str] = []
        self.vocab: Dict[str, Dict[str, int]] = {}
        self.fill: pd.Series = pd.Series(dtype=float)
        self.lo: pd.Series = pd.Series(dtype=float)
        self.hi: pd.Series = pd.Series(dtype=float)

    def fit(self, df: pd.DataFrame, exclude: Optional[set] = None) -> "Preprocessor":
        if exclude is not None:
            self.exclude = set(exclude)
        cols = [c for c in df.columns if c not in self.exclude]
        self.cat = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        self.num = [c for c in cols if c not in self.cat]
        self.vocab = {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))} for c in self.cat}

        X = self._clean(df)
        self.fill = X.median().fillna(0.0)
        X = X.fillna(self.fill)
        self.lo, self.hi = X.min(), X.max()
        return self

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        clean_cols = [c for c in self.num if c in df.columns]
        return df[clean_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    @property
    def features(self) -> List[str]:
        return self.num + self.cat

    def transform(self, df: pd.DataFrame, clip: bool = True) -> pd.DataFrame:
        f32_min, f32_max = np.finfo(np.float32).min, np.finfo(np.float32).max
        clean_df = self._clean(df).fillna(self.fill).clip(lower=f32_min, upper=f32_max).astype("float32")
        if clip:
            clean_df = clean_df.clip(self.lo, self.hi, axis=1)

        for c in self.cat:
            if c in df.columns:
                clean_df[c] = df[c].astype(str).map(self.vocab[c]).fillna(-1).astype("float32")
            else:
                clean_df[c] = np.float32(-1)

        return clean_df


# ============================================================================
# 2. Anomaly Detector (Tuan's Benign Anomaly Engine)
# ============================================================================
class AnomalyDetector:
    """
    Learns benign flows only.
    Two signals, each normalized by its own threshold (> 1 means anomalous):
      1. IsolationForest: rare feature combinations.
      2. Robust Z on a signed-log scale: values far beyond anything benign.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.iforest: Optional[IsolationForest] = None
        self.med: pd.Series = pd.Series(dtype=float)
        self.iqr: pd.Series = pd.Series(dtype=float)
        self.thr_if: float = 1.0
        self.thr_z: float = 1.0

    def fit(self, X_ben: pd.DataFrame, X_ref: pd.DataFrame, fpr: float = 0.005) -> "AnomalyDetector":
        self.iforest = IsolationForest(
            n_estimators=300,
            max_samples=min(4096, len(X_ben)),
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(X_ben)

        L = self._signed_log(X_ben)
        self.med = L.median()
        self.iqr = (L.quantile(0.75) - L.quantile(0.25)).clip(lower=1.0)

        # False alarm budget split across branches
        ref_if = self._if(X_ref)
        ref_z = self._z(X_ref)
        self.thr_if = float(np.quantile(ref_if, 1 - fpr / 2))
        self.thr_z = float(np.quantile(ref_z, 1 - fpr / 2))
        return self

    @staticmethod
    def _signed_log(X: pd.DataFrame) -> pd.DataFrame:
        return np.sign(X) * np.log1p(np.abs(X))

    def _if(self, X: pd.DataFrame) -> np.ndarray:
        if self.iforest is None:
            return np.zeros(len(X))
        return -self.iforest.score_samples(X)

    def _z(self, X: pd.DataFrame) -> np.ndarray:
        log_x = self._signed_log(X)
        z = ((log_x - self.med) / self.iqr).abs().max(axis=1).values
        return z

    def score(self, X: pd.DataFrame) -> np.ndarray:
        s_if = self._if(X) / max(self.thr_if, 1e-9)
        s_z = self._z(X) / max(self.thr_z, 1e-9)
        return np.maximum(s_if, s_z)


# ============================================================================
# 3. Helper Functions
# ============================================================================
def group_split(X: pd.DataFrame, test_size: float = 0.15, seed: int = 42):
    """Rows with identical features stay on the same side (no leakage)."""
    g = pd.util.hash_pandas_object(X, index=False).values
    return next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(X, groups=g))


def make_xgb(n_classes: int, seed: int = 42, device: str = "cpu", **kw) -> xgb.XGBClassifier:
    p = dict(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )
    if device == "cuda":
        p["device"] = "cuda"
    p.update(kw)
    if n_classes == 2:
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **p)
    return xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **p)


# ============================================================================
# 4. Final IDS Model Class (Two-Tier + Standalone Support)
# ============================================================================
class FinalIDSModel:
    """
    Component 4 Final Intrusion Detection System Classifier.
    Derived from Tuan's Two-Tier IDS Architecture (Tier 1 Binary + Anomaly, Tier 2 AttackTyper).

    Features:
      - Two-tier flow detection & attack typing when Benign flows are present.
      - Supervised + Anomaly detector gating (IsolationForest + robust Z-score).
      - Calibration on validation fold to control False Alarm Rate (FAR / max_fpr).
      - Multiclass classification with balanced sample weights.
      - Comprehensive evaluation metrics matching Tuan's ML-IDS specification.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.08,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        weight_power: float = 0.5,
        max_fpr: float = 0.01,
        anomaly_fpr: float = 0.005,
        min_confidence: float = 0.4,
        rare_share: float = RARE_SHARE_THRESHOLD,
        random_state: int = 42,
        n_jobs: int = -1,
        tree_method: str = "hist",
        device: str = "cpu",
        use_two_tier: bool = True,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "tree_method": tree_method,
        }
        self.device = device
        self.weight_power = weight_power
        self.max_fpr = max_fpr
        self.anomaly_fpr = anomaly_fpr
        self.min_confidence = min_confidence
        self.rare_share = rare_share
        self.random_state = random_state
        self.use_two_tier = use_two_tier

        # Sub-modules
        self.pre1 = Preprocessor(exclude=["Label", "Attack", "OOF_Confidence"])
        self.pre2 = Preprocessor(exclude=["Label", "Attack", "OOF_Confidence"])
        self.tier1_model: Optional[xgb.XGBClassifier] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.tier1_thr: float = 0.5

        self.tier2_model: Optional[xgb.XGBClassifier] = None
        self.attack_classes_: np.ndarray = np.array([])
        self.label_encoder = LabelEncoder()

        # Standalone model fallback
        self.standalone_model: Optional[xgb.XGBClassifier] = None
        self.feature_cols: List[str] = []
        self.is_trained: bool = False
        self.last_val_metrics: Dict[str, Any] = {}
        self.last_test_metrics: Dict[str, Any] = {}

    def _extract_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts binary y (0=benign, 1=attack) and attack names."""
        name = df["Attack"].astype(str).str.strip().values if "Attack" in df.columns else np.array(["Unknown"] * len(df))
        if "Label" in df.columns:
            y = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int).values
        else:
            y = (name != DEFAULT_BENIGN_LABEL).astype(int)
        attack_types = np.where(y == 1, name, DEFAULT_BENIGN_LABEL)
        return y, attack_types

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
    ):
        """
        Trains the IDS classifier.
        If Benign flows are present and use_two_tier=True, fits Tier 1 (Detector + Anomaly)
        and Tier 2 (AttackTyper). Otherwise, trains a direct multiclass XGBoost.
        """
        t0 = time.time()
        y_tr, attack_names_tr = self._extract_labels(train_df)
        has_benign = (y_tr == 0).any() and (y_tr == 1).any()

        # Determine feature columns
        self.feature_cols = [c for c in train_df.columns if c not in ["Label", "Attack", "OOF_Confidence"]]

        if self.use_two_tier and has_benign:
            if verbose:
                print(f"    [IDS Train] Training Two-Tier IDS Architecture (Trees={self.params['n_estimators']}, lr={self.params['learning_rate']})...")

            # Tier 1 Preprocessor
            self.pre1.fit(train_df)
            X_tr1 = self.pre1.transform(train_df)

            # Split validation part for calibration if not provided externally
            if val_df is not None and (val_df["Label"] == 0).any():
                X_va_clean = self.pre1.transform(val_df)
                y_va, _ = self._extract_labels(val_df)
                ben_tr_idx = np.where(y_tr == 0)[0]
                ben_va_idx = np.where(y_va == 0)[0]
                X_ben_tr = X_tr1.iloc[ben_tr_idx]
                X_ben_va = X_va_clean.iloc[ben_va_idx]
            else:
                tr_idx, va_idx = group_split(X_tr1, test_size=0.15, seed=self.random_state)
                y_va = y_tr[va_idx]
                X_va_clean = X_tr1.iloc[va_idx]
                ben_tr = tr_idx[y_tr[tr_idx] == 0]
                ben_va = va_idx[y_va == 0]
                if len(ben_va) == 0:
                    ben_va = ben_tr[:max(1, len(ben_tr) // 5)]
                X_ben_tr = X_tr1.iloc[ben_tr]
                X_ben_va = X_tr1.iloc[ben_va]

            # Fit Anomaly Detector
            self.anomaly_detector = AnomalyDetector(random_state=self.random_state)
            self.anomaly_detector.fit(X_ben_tr, X_ben_va, fpr=self.anomaly_fpr)

            # Fit Supervised Tier 1 XGBoost
            w_t1 = compute_sample_weight("balanced", y_tr) ** self.weight_power
            self.tier1_model = make_xgb(
                2,
                seed=self.random_state,
                device=self.device,
                **self.params,
            ).fit(X_tr1, y_tr, sample_weight=w_t1)

            # Calibrate Tier 1 threshold on validation benign flows
            s_va = self.tier1_model.predict_proba(X_va_clean)[:, 1]
            ben_scores = s_va[y_va == 0]
            if len(ben_scores) > 0:
                self.tier1_thr = float(np.clip(np.quantile(ben_scores, 1 - self.max_fpr), 0.05, 0.95))
            else:
                self.tier1_thr = 0.5

            # Tier 2: AttackTyper (Trained strictly on Attack flows)
            attack_mask = (y_tr == 1)
            df_att = train_df[attack_mask].copy()
            att_names = attack_names_tr[attack_mask]

            self.pre2.fit(df_att)
            self.attack_classes_ = np.unique(att_names)
            ci = np.searchsorted(self.attack_classes_, att_names)
            w_t2 = compute_sample_weight("balanced", ci) ** self.weight_power

            X_tr2 = self.pre2.transform(df_att)
            self.tier2_model = make_xgb(
                len(self.attack_classes_),
                seed=self.random_state,
                device=self.device,
                **self.params,
            ).fit(X_tr2, ci, sample_weight=w_t2)

            self.label_encoder.classes_ = np.array([DEFAULT_BENIGN_LABEL] + list(self.attack_classes_))
        else:
            # Standalone multiclass XGBoost fallback
            if verbose:
                print(f"    [IDS Train] Training Standalone Multiclass IDS (Trees={self.params['n_estimators']}, lr={self.params['learning_rate']})...")

            self.pre1.fit(train_df)
            X_clean = self.pre1.transform(train_df)
            self.label_encoder.fit(attack_names_tr)
            y_enc = self.label_encoder.transform(attack_names_tr)
            n_classes = len(self.label_encoder.classes_)
            w = compute_sample_weight("balanced", y_enc) ** self.weight_power

            self.standalone_model = make_xgb(
                n_classes,
                seed=self.random_state,
                device=self.device,
                **self.params,
            ).fit(X_clean, y_enc, sample_weight=w)

        self.is_trained = True
        elapsed = time.time() - t0
        if verbose:
            print(f"    [IDS Train] Complete in {elapsed:.2f}s ({len(train_df):,} flows).")

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts flow alert status (0/1) and specific attack category.
        Returns (binary_alert, attack_prediction).
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")

        if self.use_two_tier and self.tier1_model is not None and self.tier2_model is not None:
            # Tier 1 predictions
            X1_sup = self.pre1.transform(df, clip=True)
            X1_ano = self.pre1.transform(df, clip=False)
            sup_score = self.tier1_model.predict_proba(X1_sup)[:, 1]
            ano_score = self.anomaly_detector.score(X1_ano) if self.anomaly_detector else np.zeros(len(df))

            hit_s = sup_score > self.tier1_thr
            hit_a = ano_score > 1.0
            alert = hit_s | hit_a

            # Tier 2 predictions
            X2 = self.pre2.transform(df, clip=True)
            P = self.tier2_model.predict_proba(X2)
            conf = P.max(axis=1)
            kind = self.attack_classes_[P.argmax(axis=1)]

            # Gate decisions
            kind2 = np.where(conf < self.min_confidence, UNKNOWN, kind)
            kind2 = np.where(hit_a & ~hit_s, UNKNOWN, kind2)
            final_predictions = np.where(alert, kind2, DEFAULT_BENIGN_LABEL)
            return alert.astype(int), final_predictions
        else:
            X = self.pre1.transform(df, clip=True)
            y_pred_idx = self.standalone_model.predict(X)
            preds = self.label_encoder.inverse_transform(y_pred_idx)
            alert = (preds != DEFAULT_BENIGN_LABEL).astype(int)
            return alert, preds

    def evaluate(
        self,
        eval_df: pd.DataFrame,
        split_name: str = "val",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation computing:
          - Macro F1, Weighted F1, Balanced Accuracy, Accuracy, Binary F1
          - Tier 1 Detection Rate (DR), False Alarm Rate (FAR), MCC, PR-AUC, ROC-AUC
          - Tier 2 isolated F1, end-to-end Macro F1, rare threat recall
          - Per-class breakdown and confusion matrix
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Call train() first.")

        t0 = time.time()
        y_true_bin, true_names = self._extract_labels(eval_df)
        pred_alert, pred_names = self.predict(eval_df)

        att_mask = (y_true_bin == 1)
        classes_in_true = sorted(np.unique(true_names))
        labels_att = [c for c in classes_in_true if c != DEFAULT_BENIGN_LABEL]

        # Rare class identification
        if att_mask.any():
            share = pd.Series(true_names[att_mask]).value_counts(normalize=True)
            rare_classes = sorted(share[share < self.rare_share].index)
        else:
            rare_classes = []

        # Standard Multiclass Metrics
        acc = accuracy_score(true_names, pred_names)
        bal_acc = balanced_accuracy_score(true_names, pred_names)
        macro_f1 = f1_score(true_names, pred_names, average="macro", zero_division=0)
        weighted_f1 = f1_score(true_names, pred_names, average="weighted", zero_division=0)

        # Binary Detection Metrics (Tier 1)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, pred_alert, labels=[0, 1]).ravel()
        dr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        bin_f1 = f1_score(y_true_bin, pred_alert, zero_division=0)
        mcc = float(matthews_corrcoef(y_true_bin, pred_alert))

        # Two-Tier Specific Metrics
        t2_macro_f1 = float(np.nan)
        t2_rare_f1 = float(np.nan)
        e2e_rare_recall = float(np.nan)

        if self.use_two_tier and self.tier2_model is not None and att_mask.any():
            # Isolated Tier 2 on true attacks
            df_att = eval_df[att_mask]
            X2 = self.pre2.transform(df_att, clip=True)
            P = self.tier2_model.predict_proba(X2)
            pred_iso = self.attack_classes_[P.argmax(axis=1)]

            f1_iso = f1_score(true_names[att_mask], pred_iso, labels=labels_att, average=None, zero_division=0)
            t2_macro_f1 = float(f1_iso.mean()) if len(f1_iso) > 0 else 0.0

            if rare_classes:
                rare_mask = np.isin(labels_att, rare_classes)
                if rare_mask.any():
                    t2_rare_f1 = float(f1_iso[rare_mask].mean())

        # End-to-end Rare Recall
        if rare_classes:
            r_rare = [
                float((pred_names[true_names == rc] == rc).mean())
                for rc in rare_classes if (true_names == rc).sum() > 0
            ]
            if r_rare:
                e2e_rare_recall = float(np.mean(r_rare))

        e2e_attack_recall = float((pred_names[att_mask] == true_names[att_mask]).mean()) if att_mask.any() else 0.0
        unknown_rate = float((pred_names[pred_alert == 1] == UNKNOWN).mean()) if (pred_alert == 1).any() else 0.0

        # Per-Class Precision / Recall / F1
        p_c, r_c, f_c, sup_c = precision_recall_fscore_support(
            true_names, pred_names, labels=classes_in_true, zero_division=0
        )
        per_class_metrics = {}
        for idx, cls in enumerate(classes_in_true):
            per_class_metrics[cls] = {
                "precision": round(float(p_c[idx]), 4),
                "recall": round(float(r_c[idx]), 4),
                "f1": round(float(f_c[idx]), 4),
                "support": int(sup_c[idx]),
                "is_rare": cls in rare_classes,
            }

        cm = confusion_matrix(true_names, pred_names, labels=classes_in_true)
        df_cm = pd.DataFrame(cm, index=classes_in_true, columns=classes_in_true)

        results = {
            "split": split_name,
            "macro_f1": round(float(macro_f1), 4),
            "e2e_macro_f1": round(float(macro_f1), 4),
            "t2_macro_f1": round(float(t2_macro_f1), 4) if not np.isnan(t2_macro_f1) else round(float(macro_f1), 4),
            "t2_rare_f1": round(float(t2_rare_f1), 4) if not np.isnan(t2_rare_f1) else 0.0,
            "e2e_rare_recall": round(float(e2e_rare_recall), 4) if not np.isnan(e2e_rare_recall) else 0.0,
            "e2e_attack_recall": round(float(e2e_attack_recall), 4),
            "detection_rate": round(float(dr), 4),
            "false_alarm_rate": round(float(far), 4),
            "binary_f1": round(float(bin_f1), 4),
            "mcc": round(float(mcc), 4),
            "unknown_rate": round(float(unknown_rate), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "accuracy": round(float(acc), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "rare_classes": rare_classes,
            "per_class": per_class_metrics,
            "confusion_matrix": df_cm,
            "elapsed_sec": round(time.time() - t0, 2),
        }

        if split_name == "val":
            self.last_val_metrics = results
        elif split_name in ["test", "holdout"]:
            self.last_test_metrics = results

        if verbose:
            print(f"\n" + "=" * 80)
            print(f"=== IDS MODEL EVALUATION [{split_name.upper()}] ===")
            print("=" * 80)
            print(f"  Macro F1:          {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
            print(f"  Balanced Accuracy: {bal_acc:.4f} ({bal_acc * 100:.2f}%)")
            print(f"  Overall Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
            print(f"  Detection Rate:    {dr:.4f} ({dr * 100:.2f}%) [Recall on Attacks]")
            print(f"  False Alarm Rate:  {far:.4f} ({far * 100:.2f}%) [FPR on Benign]")
            print(f"  Matthews Corr (MCC):{mcc:.4f}")
            print(f"  Binary F1:         {bin_f1:.4f}")
            if not np.isnan(t2_macro_f1):
                print(f"  Isolated Tier-2 F1:{t2_macro_f1:.4f}")
            if not np.isnan(e2e_rare_recall):
                print(f"  Rare Threat Recall:{e2e_rare_recall:.4f} ({e2e_rare_recall * 100:.2f}%) [Rare Types: {rare_classes}]")
            if not np.isnan(t2_rare_f1):
                print(f"  Rare Threat F1:    {t2_rare_f1:.4f}")

            print(f"\n  --- Per-Class Metrics Breakdown [{split_name.upper()}] ---")
            header_str = f"  {'Class':<18} | {'Category':<16} | {'Support':>8} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9}"
            print(header_str)
            print("  " + "-" * (len(header_str) - 2))
            for cls, c_met in per_class_metrics.items():
                ctype = "Benign" if cls == DEFAULT_BENIGN_LABEL else ("Rare Attack" if c_met["is_rare"] else "Frequent Attack")
                print(
                    f"  {cls:<18} | {ctype:<16} | {c_met['support']:>8,d} | "
                    f"{c_met['precision'] * 100:>8.2f}% | {c_met['recall'] * 100:>8.2f}% | {c_met['f1'] * 100:>8.2f}%"
                )
            print("=" * 80 + "\n")

        return results

    def save_model(self, filepath: Union[str, Path]) -> Path:
        """Saves trained Two-Tier and fallback IDS models, preprocessors, and thresholds."""
        filepath = Path(filepath).resolve()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "tier1_model": getattr(self, "tier1_model", None),
            "anomaly_detector": getattr(self, "anomaly_detector", None),
            "tier2_model": getattr(self, "tier2_model", None),
            "standalone_model": getattr(self, "standalone_model", None),
            "pre1": getattr(self, "pre1", None),
            "pre2": getattr(self, "pre2", None),
            "label_encoder": getattr(self, "label_encoder", None),
            "attack_classes_": getattr(self, "attack_classes_", None),
            "tier1_thr": getattr(self, "tier1_thr", 0.5),
            "rare_share": getattr(self, "rare_share", RARE_SHARE_THRESHOLD),
            "use_two_tier": getattr(self, "use_two_tier", True),
            "params": getattr(self, "params", {}),
            "weight_power": getattr(self, "weight_power", 0.5),
            "min_confidence": getattr(self, "min_confidence", 0.4),
            "max_fpr": getattr(self, "max_fpr", 0.01),
            "anomaly_fpr": getattr(self, "anomaly_fpr", 0.005),
            "is_trained": getattr(self, "is_trained", False),
            "feature_cols": getattr(self, "feature_cols", []),
            "last_val_metrics": getattr(self, "last_val_metrics", None),
            "last_test_metrics": getattr(self, "last_test_metrics", None),
        }
        try:
            import joblib
            joblib.dump(state, filepath)
        except Exception:
            import pickle
            with open(filepath, "wb") as f:
                pickle.dump(state, f)
        print(f"    [IDS Model Checkpoint] Saved trained model to: {filepath.name}")
        return filepath

    def load_model(self, filepath: Union[str, Path]) -> bool:
        """Loads trained IDS model from file."""
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            print(f"    [Warning] Checkpoint not found at {filepath}")
            return False
        try:
            import joblib
            state = joblib.load(filepath)
        except Exception:
            import pickle
            with open(filepath, "rb") as f:
                state = pickle.load(f)
        for k, v in state.items():
            setattr(self, k, v)
        print(f"    [IDS Model Checkpoint] Loaded model from: {filepath.name}")
        return True


# ============================================================================
# 5. Full Synthetic Data Quality Suite (Tuan's 5-Pillar Evaluation)
# ============================================================================
def evaluate_synthetic_quality(
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    attack_col: str = "Attack",
    label_col: str = "Label",
    cap: int = 15000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates synthetic data across 5 pillars:
      1. Coverage (classes covered, missing classes)
      2. Validity (invalid row rate, negative cells, fractional counts, over max)
      3. Fidelity (mean KS statistic, correlation diff, tail q99 ratio, L1-L3 network metrics)
      4. Privacy (exact copy rate, DCR train p5, DCR holdout p5, closer to train rate)
      5. Utility (TRTR baseline, TSTR, TR+S, ML Efficacy Ratio, Augmentation Gain)
    """
    pre = Preprocessor(exclude=[attack_col, label_col])
    pre.fit(real_train)

    names = lambda d: d[attack_col].astype(str).str.strip().values
    yr, yt = names(real_train), names(real_test)
    labels = sorted(np.unique(yr))
    share = pd.Series(yr).value_counts(normalize=True)
    rare = list(share[share < RARE_SHARE_THRESHOLD].index)

    Xr, Xt = pre.transform(real_train), pre.transform(real_test)

    # 1. Baseline TRTR
    base_m = make_xgb(len(labels), seed=seed).fit(
        Xr, np.searchsorted(labels, yr),
        sample_weight=compute_sample_weight("balanced", np.searchsorted(labels, yr)) ** 0.5
    )
    pred_base = [labels[i] for i in base_m.predict(Xt)]
    f1_base = f1_score(yt, pred_base, labels=labels, average=None, zero_division=0)
    trtr_macro = float(f1_base.mean())
    trtr_rare = float(f1_base[np.isin(labels, rare)].mean()) if rare else trtr_macro

    # Synthetic processing
    ys = names(synthetic_df)
    keep_mask = np.isin(ys, labels)
    syn_clean = synthetic_df[keep_mask].reset_index(drop=True)
    ys = ys[keep_mask]

    # Coverage
    got = sorted(np.unique(ys))
    coverage = {
        "classes_covered": f"{len(got)}/{len(labels)}",
        "missing_classes": sorted(set(labels) - set(got)),
    }

    # Validity
    R_num = pre._clean(real_train).fillna(pre.fill)
    S_num = pre._clean(syn_clean).fillna(pre.fill)
    int_cols = [c for c in pre.num if c in R_num.columns and (R_num[c] % 1 == 0).all()]
    nonneg = [c for c in pre.num if c in R_num.columns and R_num[c].min() >= 0]
    hi = R_num.max()

    neg = S_num[nonneg] < 0 if nonneg else pd.DataFrame()
    nonint = (S_num[int_cols] % 1 != 0) if int_cols else pd.DataFrame()
    over = (S_num > hi) if not S_num.empty else pd.DataFrame()

    bad_rows = (neg.any(axis=1) if not neg.empty else False) | \
               (nonint.any(axis=1) if not nonint.empty else False) | \
               (over.any(axis=1) if not over.empty else False)

    validity = {
        "invalid_row_rate": round(float(bad_rows.mean()), 4) if len(bad_rows) > 0 else 0.0,
        "neg_cell_rate": round(float(neg.values.mean()), 4) if not neg.empty else 0.0,
        "nonint_cell_rate": round(float(nonint.values.mean()), 4) if not nonint.empty else 0.0,
        "over_max_cell_rate": round(float(over.values.mean()), 4) if not over.empty else 0.0,
    }

    # Fidelity
    ks_list, cor_list, tail_list = [], [], []
    for c in got:
        a = R_num[yr == c]
        b = S_num[ys == c]
        if len(b) < 2 or len(a) < 2:
            continue
        k_stat = [ks_2samp(a[col], b[col]).statistic for col in R_num.columns if col in b.columns]
        ks_list.append(np.mean(k_stat))
        cor_diff = np.abs(a.corr("spearman").fillna(0) - b.corr("spearman").fillna(0)).values.mean()
        cor_list.append(cor_diff)
        tail_diff = np.abs(np.log10((b.quantile(0.99).clip(lower=0) + 1) / (a.quantile(0.99).clip(lower=0) + 1))).mean()
        tail_list.append(tail_diff)

    fidelity = {
        "ks_mean": round(float(np.mean(ks_list)), 4) if ks_list else 1.0,
        "corr_diff_mean": round(float(np.mean(cor_list)), 4) if cor_list else 1.0,
        "tail_q99_log_ratio": round(float(np.mean(tail_list)), 4) if tail_list else 1.0,
    }

    # Privacy (DCR on signed-log standardized space)
    def signed_log(X):
        return np.sign(X) * np.log1p(np.abs(X))

    Lr = signed_log(pre.transform(real_train, clip=False)).values
    mu, sd = Lr.mean(0), Lr.std(0)
    sd[sd == 0] = 1.0
    Er = (Lr - mu) / sd
    Eh = (signed_log(pre.transform(real_test, clip=False)).values - mu) / sd

    rng = np.random.default_rng(seed)
    sub_n = min(len(syn_clean), cap)
    idx_syn = rng.choice(len(syn_clean), sub_n, replace=False) if len(syn_clean) > sub_n else np.arange(len(syn_clean))
    Es = (signed_log(pre.transform(syn_clean.iloc[idx_syn], clip=False)).values - mu) / sd

    nn_tr = NearestNeighbors(n_neighbors=1).fit(Er[rng.choice(len(Er), min(len(Er), len(Eh)), replace=False)])
    nn_ho = NearestNeighbors(n_neighbors=1).fit(Eh)

    d_tr = nn_tr.kneighbors(Es)[0][:, 0]
    d_ho = nn_ho.kneighbors(Es)[0][:, 0]

    privacy = {
        "exact_copy_rate": round(float((d_tr < 1e-9).mean()), 4),
        "dcr_train_p5": round(float(np.percentile(d_tr, 5)), 4),
        "dcr_holdout_p5": round(float(np.percentile(d_ho, 5)), 4),
        "closer_to_train_rate": round(float((d_tr < d_ho).mean()), 4),
    }

    # Utility (TSTR & TR+S)
    Xs = pre.transform(syn_clean)
    cls_s = np.unique(ys)
    if len(cls_s) >= 2:
        m_tstr = make_xgb(len(cls_s), seed=seed).fit(
            Xs, np.searchsorted(cls_s, ys),
            sample_weight=compute_sample_weight("balanced", np.searchsorted(cls_s, ys)) ** 0.5
        )
        pred_tstr = [cls_s[i] for i in m_tstr.predict(Xt)]
        f1_tstr = f1_score(yt, pred_tstr, labels=labels, average=None, zero_division=0)
        tstr_macro = float(f1_tstr.mean())
    else:
        tstr_macro = 0.0

    # TR+S
    X_trs = pd.concat([Xr, Xs], ignore_index=True)
    y_trs = np.r_[yr, ys]
    cls_trs = np.unique(y_trs)
    m_trs = make_xgb(len(cls_trs), seed=seed).fit(
        X_trs, np.searchsorted(cls_trs, y_trs),
        sample_weight=compute_sample_weight("balanced", np.searchsorted(cls_trs, y_trs)) ** 0.5
    )
    pred_trs = [cls_trs[i] for i in m_trs.predict(Xt)]
    f1_trs = f1_score(yt, pred_trs, labels=labels, average=None, zero_division=0)
    trs_macro = float(f1_trs.mean())
    trs_rare = float(f1_trs[np.isin(labels, rare)].mean()) if rare else trs_macro

    utility = {
        "trtr_macro_f1": round(trtr_macro, 4),
        "tstr_macro_f1": round(tstr_macro, 4),
        "ml_efficacy_ratio": round(float(tstr_macro / max(trtr_macro, 1e-9)), 4),
        "trs_macro_f1": round(trs_macro, 4),
        "trs_rare_f1": round(trs_rare, 4),
        "aug_gain_macro_f1": round(float(trs_macro - trtr_macro), 4),
        "aug_gain_rare_f1": round(float(trs_rare - trtr_rare), 4),
    }

    return {
        "coverage": coverage,
        "validity": validity,
        "fidelity": fidelity,
        "privacy": privacy,
        "utility": utility,
    }


__all__ = [
    "FinalIDSModel",
    "Preprocessor",
    "AnomalyDetector",
    "evaluate_synthetic_quality",
    "UNKNOWN",
    "DEFAULT_BENIGN_LABEL",
]


if __name__ == "__main__":
    print("Component 4 (final_ids_model.py) loaded successfully.")

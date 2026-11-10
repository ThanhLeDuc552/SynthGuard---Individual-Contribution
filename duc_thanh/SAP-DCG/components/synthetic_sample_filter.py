#!/usr/bin/env python3
"""
Synthetic Sample Filtering & Post-Processing Pipeline (Component 3)

Implements Component 3 according to components.txt and dataset.txt:
- Reverse transformations:
    - Reverse log-transform / inverse scaling
    - Integer rounding on discrete counts
    - Physical NetFlow protocol invariant rules
- OOF (Out-Of-Fold) Confidence Gate:
    - XGBoost classifier trained on 5-fold cross-validation STRICTLY on real training dataset.
    - Filters synthetic samples: keeps only samples with confidence >= threshold for their intended class.
- ENN (Edited Nearest Neighbors):
    - Nearest neighbors (n=3) against real training flows.
    - If majority of 3 nearest real neighbors belong to a different class (close to wrong class), remove.

Hyperparameter tuning (OOF threshold) has been consolidated into `PipelineTuner`.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.append(str(curr_dir))

from intra_attack_resampler import CATEGORICAL_FEATURES


class SyntheticSampleFilter:
    """
    Component 3 Post-Processing & Filtering Engine.

    Guarantees synthetic data fidelity by applying:
      1. Domain-specific physical protocol invariant rules and inverse scaling.
      2. 5-Fold Cross-Validated Out-Of-Fold (OOF) XGBoost confidence gating.
      3. Edited Nearest Neighbors (ENN, n=3) boundary pruning against real training flows.
    """

    def __init__(
        self,
        random_state: int = 42,
        default_oof_threshold: float = 0.6,
        enn_neighbors: int = 3,
    ):
        self.random_state = random_state
        self.default_oof_threshold = default_oof_threshold
        self.enn_neighbors = enn_neighbors

        self.oof_models: List[xgb.XGBClassifier] = []
        self.label_encoder: Optional[LabelEncoder] = None
        self.real_feature_cols: List[str] = []
        self.enn_engine: Optional[NearestNeighbors] = None
        self.enn_scaler: Optional[StandardScaler] = None
        self.real_labels_enn: Optional[np.ndarray] = None
        self.valid_protocols: Optional[set] = None

    # ------------------------------------------------------------------------
    # 1. Reverse Transformation & Physical NetFlow Protocol Rules
    # ------------------------------------------------------------------------
    def apply_reverse_transform_and_rules(
        self,
        df: pd.DataFrame,
        scaler: Optional[Any] = None,
        was_log_transformed: bool = False,
        continuous_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Applies inverse scaling, log1p inversion, integer rounding,
        and NetFlow physical protocol invariant validation rules.
        """
        df_out = df.copy()
        feature_cols = [c for c in df_out.columns if c not in ["Label", "Attack"]]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
        cont_cols = [c for c in feature_cols if c not in cat_cols] if continuous_cols is None else continuous_cols

        # 1. Inverse feature scaling if fitted scaler provided
        if scaler is not None and hasattr(scaler, "inverse_transform"):
            try:
                df_out[cont_cols] = scaler.inverse_transform(df_out[cont_cols])
            except Exception:
                pass

        # 2. Reverse log-transform if log1p was applied
        if was_log_transformed:
            df_out[cont_cols] = np.expm1(df_out[cont_cols].clip(lower=0))

        # 3. Non-negativity constraint across all flow attributes
        num_cols = df_out.select_dtypes(include=[np.number]).columns
        df_out[num_cols] = df_out[num_cols].clip(lower=0.0)

        # 4. Integer rounding on discrete and count metrics
        round_cols = [
            "IN_PKTS", "OUT_PKTS", "IN_BYTES", "OUT_BYTES",
            "MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
            "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
            "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
            "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
            "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
            "NUM_PKTS_1024_TO_1514_BYTES", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
            "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
        ]
        for col in round_cols + cat_cols:
            if col in df_out.columns and col != "L7_PROTO":
                df_out[col] = df_out[col].round().astype(int)

        # 5. NetFlow physical invariant protocol rules
        # Rule A: Zero packets cannot transmit positive bytes
        zero_out = (df_out["OUT_PKTS"] == 0) & (df_out["OUT_BYTES"] > 0)
        df_out.loc[zero_out, "OUT_BYTES"] = 0

        zero_in = (df_out["IN_PKTS"] == 0) & (df_out["IN_BYTES"] > 0)
        df_out.loc[zero_in, "IN_BYTES"] = 0

        # Rule B: Non-TCP protocol flows cannot have non-zero TCP flags
        if "PROTOCOL" in df_out.columns:
            non_tcp = df_out["PROTOCOL"] != 6
            for flag_col in ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"]:
                if flag_col in df_out.columns:
                    df_out.loc[non_tcp, flag_col] = 0

            # Rule C: Protocol valid numbers constraint
            if self.valid_protocols:
                invalid_proto = ~df_out["PROTOCOL"].isin(self.valid_protocols)
                if invalid_proto.any():
                    mode_proto = 6
                    df_out.loc[invalid_proto, "PROTOCOL"] = mode_proto

        # Rule D: Packet length and TTL sanity invariants
        if "MIN_IP_PKT_LEN" in df_out.columns and "MAX_IP_PKT_LEN" in df_out.columns:
            inv_ip = df_out["MIN_IP_PKT_LEN"] > df_out["MAX_IP_PKT_LEN"]
            df_out.loc[inv_ip, "MAX_IP_PKT_LEN"] = df_out.loc[inv_ip, "MIN_IP_PKT_LEN"]

        if "SHORTEST_FLOW_PKT" in df_out.columns and "LONGEST_FLOW_PKT" in df_out.columns:
            inv_pkt = df_out["SHORTEST_FLOW_PKT"] > df_out["LONGEST_FLOW_PKT"]
            df_out.loc[inv_pkt, "LONGEST_FLOW_PKT"] = df_out.loc[inv_pkt, "SHORTEST_FLOW_PKT"]

        if "MIN_TTL" in df_out.columns and "MAX_TTL" in df_out.columns:
            inv_ttl = df_out["MIN_TTL"] > df_out["MAX_TTL"]
            df_out.loc[inv_ttl, "MAX_TTL"] = df_out.loc[inv_ttl, "MIN_TTL"]

        return df_out

    # ------------------------------------------------------------------------
    # 2. OOF Confidence Gate: 5-Fold XGBoost Predictor on Real Dataset
    # ------------------------------------------------------------------------
    def fit_oof_predictor(self, real_train_df: pd.DataFrame, n_splits: int = 5):
        """
        Trains a 5-fold cross-validated XGBoost ensemble STRICTLY on the real training dataset.
        """
        print(f">>> [SampleFilter OOF] Training {n_splits}-Fold XGBoost OOF Predictor on Real Dataset ({len(real_train_df):,} flows)...")
        t0 = time.time()

        feature_cols = [c for c in real_train_df.columns if c not in ["Label", "Attack"]]
        self.real_feature_cols = feature_cols
        self.valid_protocols = set(real_train_df["PROTOCOL"].unique()) if "PROTOCOL" in real_train_df.columns else None

        # Encode attack targets
        self.label_encoder = LabelEncoder()
        y_real = self.label_encoder.fit_transform(real_train_df["Attack"])
        X_real = real_train_df[feature_cols].copy()

        # Sanitize real features
        for col in feature_cols:
            X_real[col] = pd.to_numeric(X_real[col], errors="coerce").fillna(0.0)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        self.oof_models = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_real, y_real), 1):
            X_tr, y_tr = X_real.iloc[train_idx], y_real[train_idx]

            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                tree_method="hist",
                random_state=self.random_state + fold,
                verbosity=0,
                n_jobs=4,
            )
            model.fit(X_tr, y_tr)
            self.oof_models.append(model)

        print(f"    Trained {len(self.oof_models)} OOF fold models across {len(self.label_encoder.classes_)} classes in {time.time() - t0:.2f}s.\n")

    def predict_oof_confidence(self, synthetic_df: pd.DataFrame) -> np.ndarray:
        """
        Computes the ensemble mean predicted probability that each synthetic sample
        belongs to its intended attack category.
        """
        if not self.oof_models or self.label_encoder is None:
            raise RuntimeError("OOF predictor not fitted. Call fit_oof_predictor first.")

        X_syn = synthetic_df[self.real_feature_cols].copy()
        for col in self.real_feature_cols:
            X_syn[col] = pd.to_numeric(X_syn[col], errors="coerce").fillna(0.0)

        # Average probabilities across 5 folds
        all_probs = np.zeros((len(synthetic_df), len(self.label_encoder.classes_)))
        for model in self.oof_models:
            all_probs += model.predict_proba(X_syn)
        all_probs /= len(self.oof_models)

        # Map intended class names to integer indices
        class_to_idx = {name: idx for idx, name in enumerate(self.label_encoder.classes_)}
        intended_indices = np.array([class_to_idx.get(atk, -1) for atk in synthetic_df["Attack"]])

        # Extract confidence for intended class
        confidences = np.zeros(len(synthetic_df))
        valid_mask = intended_indices >= 0
        if valid_mask.any():
            row_indices = np.where(valid_mask)[0]
            col_indices = intended_indices[valid_mask]
            confidences[row_indices] = all_probs[row_indices, col_indices]

        return confidences

    # ------------------------------------------------------------------------
    # 3. ENN (Edited Nearest Neighbors, n=3) Boundary Check
    # ------------------------------------------------------------------------
    def fit_enn(self, real_train_df: pd.DataFrame):
        """
        Fits NearestNeighbors index (n_neighbors=3) on real training flows.
        """
        print(f">>> [SampleFilter ENN] Fitting Edited Nearest Neighbors (k={self.enn_neighbors}) on Real Dataset...")
        feature_cols = [c for c in real_train_df.columns if c not in ["Label", "Attack"]]
        X_real = real_train_df[feature_cols].copy()
        for col in feature_cols:
            X_real[col] = pd.to_numeric(X_real[col], errors="coerce").fillna(0.0)

        self.enn_scaler = StandardScaler()
        X_scaled = self.enn_scaler.fit_transform(X_real)

        self.enn_engine = NearestNeighbors(n_neighbors=self.enn_neighbors, algorithm="auto", n_jobs=4)
        self.enn_engine.fit(X_scaled)
        self.real_labels_enn = real_train_df["Attack"].values
        print(f"    ENN index ready on {len(X_scaled):,} real reference flows.\n")

    def enn_filter(self, synthetic_df: pd.DataFrame) -> np.ndarray:
        """
        Checks whether synthetic sample is close to the wrong class.
        Returns boolean mask where True = keep (neighbor majority agrees), False = remove.
        """
        if self.enn_engine is None or self.enn_scaler is None:
            raise RuntimeError("ENN engine not fitted. Call fit_enn first.")

        X_syn = synthetic_df[self.real_feature_cols].copy()
        for col in self.real_feature_cols:
            X_syn[col] = pd.to_numeric(X_syn[col], errors="coerce").fillna(0.0)

        X_syn_scaled = self.enn_scaler.transform(X_syn)
        _, indices = self.enn_engine.kneighbors(X_syn_scaled)

        keep_mask = np.ones(len(synthetic_df), dtype=bool)
        synthetic_classes = synthetic_df["Attack"].values

        for i in range(len(synthetic_df)):
            neighbor_labels = self.real_labels_enn[indices[i]]
            intended = synthetic_classes[i]
            # Majority agreement among 3 neighbors
            agreement_count = np.sum(neighbor_labels == intended)
            # If majority disagrees (agreement < 2 out of 3), sample is encroaching wrong class
            if agreement_count < 2:
                keep_mask[i] = False

        return keep_mask

    # ------------------------------------------------------------------------
    # 4. Master Filtering Execution (Reverse Rules + OOF + ENN)
    # ------------------------------------------------------------------------
    def filter_samples(
        self,
        synthetic_df: pd.DataFrame,
        real_train_df: Optional[pd.DataFrame] = None,
        syn_deficit: Optional[Dict[str, int]] = None,
        oof_threshold: Optional[float] = None,
        run_enn: bool = True,
        scaler: Optional[Any] = None,
        was_log_transformed: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes full Component 3 filtering pipeline:
          1. Reverse transformation & physical rules.
          2. OOF confidence gating.
          3. ENN neighborhood agreement gating.
          4. Cap each class at target deficit n_syn (selecting top confidence samples).
        """
        t0 = time.time()
        thresh = oof_threshold if oof_threshold is not None else self.default_oof_threshold

        if synthetic_df.empty:
            return {"kept_df": pd.DataFrame(), "rejected_df": pd.DataFrame(), "diagnostics": {}}

        print("=" * 80)
        print(">>> [SampleFilter] Executing Synthetic Sample Filtering Pipeline")
        print("=" * 80)
        print(f"Input Candidate Pool:  {len(synthetic_df):,} flows")
        print(f"OOF Confidence Thresh: {thresh}")
        print(f"ENN Boundary Check:    {'Enabled (k=3)' if run_enn else 'Disabled'}")

        # 1. Reverse transform & physical rules
        df_rules = self.apply_reverse_transform_and_rules(
            synthetic_df, scaler=scaler, was_log_transformed=was_log_transformed
        )

        # 2. Ensure OOF predictor and ENN are fitted if real_train_df provided
        if not self.oof_models and real_train_df is not None:
            self.fit_oof_predictor(real_train_df)
        if run_enn and self.enn_engine is None and real_train_df is not None:
            self.fit_enn(real_train_df)

        # 3. OOF confidence evaluation
        confidences = self.predict_oof_confidence(df_rules)
        df_rules["OOF_Confidence"] = confidences
        oof_mask = confidences >= thresh

        # 4. ENN neighborhood evaluation
        if run_enn and self.enn_engine is not None:
            enn_mask = self.enn_filter(df_rules)
        else:
            enn_mask = np.ones(len(df_rules), dtype=bool)

        combined_keep_mask = oof_mask & enn_mask

        df_kept_raw = df_rules[combined_keep_mask].copy()
        df_rejected = df_rules[~combined_keep_mask].copy()

        # 5. Cap samples at target n_syn (ranking by highest OOF confidence)
        final_kept_records = []
        if syn_deficit is not None:
            for cls_name, n_syn in syn_deficit.items():
                if n_syn <= 0:
                    continue
                df_cls_kept = df_kept_raw[df_kept_raw["Attack"] == cls_name]
                # Sort descending by OOF confidence and take top n_syn
                df_cls_budgeted = df_cls_kept.sort_values(by="OOF_Confidence", ascending=False).head(n_syn)
                final_kept_records.append(df_cls_budgeted)

            if final_kept_records:
                df_final_kept = pd.concat(final_kept_records, ignore_index=True)
            else:
                df_final_kept = df_kept_raw
        else:
            df_final_kept = df_kept_raw

        # Drop internal confidence score from final dataset
        if "OOF_Confidence" in df_final_kept.columns:
            df_final_kept = df_final_kept.drop(columns=["OOF_Confidence"])

        elapsed = time.time() - t0
        print(f"\n    Filtering Results:")
        print(f"      - Input Candidate Pool:     {len(synthetic_df):,}")
        print(f"      - Passed OOF Gate:          {oof_mask.sum():,} ({oof_mask.sum() / len(synthetic_df) * 100:.2f}%)")
        print(f"      - Passed ENN Check:         {enn_mask.sum():,} ({enn_mask.sum() / len(synthetic_df) * 100:.2f}%)")
        print(f"      - Passed Both Gates:        {combined_keep_mask.sum():,} ({combined_keep_mask.sum() / len(synthetic_df) * 100:.2f}%)")
        print(f"      - Final Budgeted Synthetic: {len(df_final_kept):,} flows ({elapsed:.2f}s)")
        print("=" * 80 + "\n")

        return {
            "kept_df": df_final_kept,
            "rejected_df": df_rejected,
            "diagnostics": {
                "input_count": len(synthetic_df),
                "oof_pass_count": int(oof_mask.sum()),
                "enn_pass_count": int(enn_mask.sum()),
                "both_pass_count": int(combined_keep_mask.sum()),
                "final_kept_count": len(df_final_kept),
                "threshold_used": thresh,
                "elapsed_sec": round(elapsed, 2),
            },
        }


__all__ = [
    "SyntheticSampleFilter",
]


if __name__ == "__main__":
    print("Component 3 (synthetic_sample_filter.py) loaded successfully.")


#!/usr/bin/env python3
"""
Final Intrusion Detection System (IDS) Model (Component 4)

Implements Component 4 according to components.txt and dataset.txt:
- ids model: Production-grade XGBoost multiclass threat classifier.
- train function: Fits on augmented / resampled training sets with optional validation monitoring.
- evaluate function: Calculates Macro F1, Weighted F1, Balanced Accuracy, Binary F1,
  per-class metrics, and confusion matrix.
  Evaluates on validation set (val_df) throughout tuning rounds, and test set when complete.

Hyperparameter tuning (Optuna) has been consolidated into `PipelineTuner`.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import xgboost as xgb

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.append(str(curr_dir))

from intra_attack_resampler import resolve_paths


class FinalIDSModel:
    """
    Component 4 Final Intrusion Detection System Classifier.

    Acts as the primary performance benchmark across all pipeline rounds:
      - Multi-class threat classification with evaluation on Validation, Test, and Holdout sets.
      - Calculates Macro F1 (unweighted, critical for minority threats), Balanced Accuracy,
        Weighted F1, Binary F1, and per-class metrics.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = 4,
        tree_method: str = "hist",
        device: str = "cpu",
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
            "eval_metric": "mlogloss",
        }
        if device == "cuda":
            self.params["device"] = "cuda"

        self.model: Optional[xgb.XGBClassifier] = None
        self.label_encoder = LabelEncoder()
        self.feature_cols: List[str] = []
        self.is_trained: bool = False
        self.last_val_metrics: Dict[str, Any] = {}
        self.last_test_metrics: Dict[str, Any] = {}

    def _prepare_xy(self, df: pd.DataFrame, fit_encoder: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
        """Sanitizes feature matrix X and encodes target y."""
        if not self.feature_cols:
            self.feature_cols = [c for c in df.columns if c not in ["Label", "Attack", "OOF_Confidence"]]

        X = df[self.feature_cols].copy()
        for col in self.feature_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

        target_series = df["Attack"].astype(str)
        if fit_encoder:
            y = self.label_encoder.fit_transform(target_series)
        else:
            classes = list(self.label_encoder.classes_)
            encoded = []
            for item in target_series:
                if item in classes:
                    encoded.append(classes.index(item))
                else:
                    encoded.append(0)
            y = np.array(encoded)

        return X, y

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Trains the final XGBoost IDS model on the provided training set.
        """
        t0 = time.time()
        X_train, y_train = self._prepare_xy(train_df, fit_encoder=True)

        eval_set = None
        if val_df is not None:
            X_val, y_val = self._prepare_xy(val_df, fit_encoder=False)
            eval_set = [(X_val, y_val)]

        self.model = xgb.XGBClassifier(**self.params)

        if eval_set and early_stopping_rounds:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=verbose,
            )
        else:
            self.model.fit(X_train, y_train, verbose=verbose)

        self.is_trained = True
        elapsed = time.time() - t0
        if verbose:
            print(f"    [IDS Train] Trained {self.params['n_estimators']} trees on {len(X_train):,} flows in {elapsed:.2f}s")

    def evaluate(
        self,
        eval_df: pd.DataFrame,
        split_name: str = "val",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates the model on eval_df and computes comprehensive IDS metrics:
          - Macro F1 (unweighted, critical for minority threats)
          - Weighted F1
          - Balanced Accuracy
          - Accuracy
          - Binary F1 (Benign vs All Attacks)
          - Per-class breakdown
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("IDS Model has not been trained yet. Call train() first.")

        t0 = time.time()
        X_eval, y_true = self._prepare_xy(eval_df, fit_encoder=False)
        y_pred = self.model.predict(X_eval)

        # Standard Multiclass Metrics
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Binary F1: Benign (Label == 0) vs Attack (Label > 0)
        benign_indices = np.where(self.label_encoder.classes_ == "Benign")[0]
        benign_idx = benign_indices[0] if len(benign_indices) > 0 else 0

        y_true_bin = (y_true != benign_idx).astype(int)
        y_pred_bin = (y_pred != benign_idx).astype(int)
        binary_f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        # Per-Class Breakdown
        class_names = list(self.label_encoder.classes_)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(class_names)), zero_division=0
        )

        per_class_metrics = {}
        for idx, cls in enumerate(class_names):
            per_class_metrics[cls] = {
                "precision": round(float(precision[idx]), 4),
                "recall": round(float(recall[idx]), 4),
                "f1": round(float(f1[idx]), 4),
                "support": int(support[idx]),
            }

        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

        results = {
            "split": split_name,
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "accuracy": round(float(acc), 4),
            "binary_f1": round(float(binary_f1), 4),
            "per_class": per_class_metrics,
            "confusion_matrix": df_cm,
            "elapsed_sec": round(time.time() - t0, 2),
        }

        if split_name == "val":
            self.last_val_metrics = results
        elif split_name in ["test", "holdout"]:
            self.last_test_metrics = results

        if verbose:
            print(f"=== IDS Model Evaluation [{split_name.upper()}] ===")
            print(f"  Macro F1:          {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
            print(f"  Balanced Accuracy: {bal_acc:.4f} ({bal_acc * 100:.2f}%)")
            print(f"  Weighted F1:       {weighted_f1:.4f}")
            print(f"  Binary F1:         {binary_f1:.4f}")
            print(f"  Overall Accuracy:  {acc:.4f}")

        return results


__all__ = [
    "FinalIDSModel",
]


if __name__ == "__main__":
    print("Component 4 (final_ids_model.py) loaded successfully.")


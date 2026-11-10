#!/usr/bin/env python3
"""
Tabular Deep Generative Models: CTGAN & TVAE (Component 2)

Implements Component 2 according to components.txt and dataset.txt:
- Model management for CTGAN and TVAE tabular generators.
- Weight persistence & checkpoint storage (saving/loading best weights).
- Training loop adhering to fixed gradient update budget (default: 42,000 steps).
- Drawing up to 10 * n_syn candidate synthetic records per minority class.

Hyperparameter tuning (Optuna) has been consolidated into `PipelineTuner`.
"""

import os
import sys
import time
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

# Prevent OpenMP runtime library conflict on Linux/Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from ctgan import CTGAN, TVAE
except ImportError:
    raise ImportError("ctgan package is required. Install via `pip install ctgan`.")

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.append(str(curr_dir))

from intra_attack_resampler import CATEGORICAL_FEATURES, resolve_paths


class TabularGenerators:
    """
    Manager for CTGAN and TVAE Generative Models for Network Intrusion Data.

    Features:
      - Model selection: 'ctgan' or 'tvae'.
      - Fixed gradient update budget (default: 42,000 steps).
      - Checkpoint persistence for model weights.
      - Scaled generation drawing up to 10 * n_syn per class for post-processing.
    """

    def __init__(
        self,
        model_type: str = "ctgan",
        fixed_gradient_updates: int = 42000,
        random_state: int = 42,
        device: Optional[str] = None,
        weights_dir: Optional[Union[str, Path]] = None,
    ):
        self.model_type = model_type.lower()
        if self.model_type not in ["ctgan", "tvae"]:
            raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'ctgan' or 'tvae'.")

        self.fixed_gradient_updates = fixed_gradient_updates
        self.random_state = random_state

        # Resolve compute device
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Checkpoint directory
        if weights_dir is None:
            base_dir, _, final_dir = resolve_paths()
            self.weights_dir = final_dir / f"generators/{self.model_type.upper()}/weights"
        else:
            self.weights_dir = Path(weights_dir).resolve()
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Any] = {}
        self.best_weights_path: Optional[Path] = None
        self.training_diagnostics: Dict[str, Any] = {}

    def calculate_epochs(self, n_samples: int, batch_size: int, max_epochs_cap: int = 5000) -> int:
        """
        Calculates epochs needed to achieve exactly fixed_gradient_updates:
          steps_per_epoch = max(1, n_samples // batch_size)
          epochs = ceil(fixed_gradient_updates / steps_per_epoch)
        """
        steps_per_epoch = max(1, n_samples // batch_size)
        epochs = int(np.ceil(self.fixed_gradient_updates / steps_per_epoch))
        if max_epochs_cap is not None:
            epochs = min(epochs, max_epochs_cap)
        return max(1, epochs)

    def _resolve_batch_size_and_pac(self, n_samples: int, requested_batch_size: int) -> Tuple[int, int]:
        """
        Ensures batch_size is compatible with CTGAN's pac constraint:
          batch_size % pac == 0 and batch_size % 2 == 0
        """
        if n_samples >= 100:
            pac = 10
        elif n_samples >= 20:
            pac = 2
        else:
            pac = 1

        step = pac if pac % 2 == 0 else pac * 2
        batch_size = (requested_batch_size // step) * step
        batch_size = max(step, min(batch_size, (n_samples // step) * step))
        if batch_size == 0:
            batch_size = step
        return batch_size, pac

    def train(
        self,
        train_attacks: pd.DataFrame,
        syn_deficit: Dict[str, int],
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        batch_size: int = 200,
        epochs: Optional[int] = None,
        verbose: bool = False,
        **model_kwargs,
    ) -> Dict[str, Any]:
        """
        Trains a generator model for each minority attack class requiring synthetic generation.

        Parameters:
          train_attacks: Real attack flows from Step 4
          syn_deficit: Dictionary of n_syn deficit values per class (from Step 5)
          feature_cols: List of flow feature column names
          categorical_cols: List of categorical feature names for CTGAN/TVAE handling
          batch_size: Desired training batch size
          epochs: Optional epoch override. If None, derived from fixed_gradient_updates (42k)
          model_kwargs: Additional hyperparameters (e.g., generator_lr, discriminator_lr)

        Returns:
          Diagnostics dictionary containing training times and epoch counts.
        """
        t0 = time.time()
        if feature_cols is None:
            feature_cols = [c for c in train_attacks.columns if c not in ["Label", "Attack"]]
        if categorical_cols is None:
            categorical_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

        target_classes = [cls for cls, n_syn in syn_deficit.items() if n_syn > 0]
        print(f"\n>>> [Generators Train] Training {self.model_type.upper()} for {len(target_classes)} minority attack classes...")
        print(f"    Target Classes:          {target_classes}")
        print(f"    Fixed Gradient Updates:  {self.fixed_gradient_updates:,} steps")
        print(f"    Compute Device:          {self.device.upper()}")

        self.models = {}
        diagnostics = {}

        for cls_name in target_classes:
            df_cls = train_attacks[train_attacks["Attack"] == cls_name][feature_cols].copy()
            n_samples = len(df_cls)
            if n_samples == 0:
                continue

            cls_bs, pac = self._resolve_batch_size_and_pac(n_samples, batch_size)
            cls_epochs = epochs if epochs is not None else self.calculate_epochs(n_samples, cls_bs)

            print(f"    - Training '{cls_name}' ({n_samples} real flows): {cls_epochs} epochs (bs={cls_bs}, pac={pac})...")
            t_cls = time.time()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.model_type == "ctgan":
                    model = CTGAN(
                        epochs=cls_epochs,
                        batch_size=cls_bs,
                        pac=pac,
                        verbose=verbose,
                        cuda=(self.device != "cpu"),
                        **model_kwargs,
                    )
                else:  # tvae
                    model = TVAE(
                        epochs=cls_epochs,
                        batch_size=cls_bs,
                        cuda=(self.device != "cpu"),
                        **model_kwargs,
                    )

                # Fit model on class slice with specified categorical columns
                model.fit(df_cls, discrete_columns=categorical_cols)

            elapsed_cls = time.time() - t_cls
            self.models[cls_name] = model
            diagnostics[cls_name] = {
                "samples": n_samples,
                "epochs": cls_epochs,
                "batch_size": cls_bs,
                "elapsed_sec": round(elapsed_cls, 2),
            }

        total_elapsed = time.time() - t0
        self.training_diagnostics = diagnostics
        print(f"    All {len(self.models)} generator models trained in {total_elapsed:.2f}s.\n")
        return diagnostics

    def sample(
        self,
        syn_deficit: Dict[str, int],
        multiplier: int = 10,
    ) -> pd.DataFrame:
        """
        Instructs the generators to draw up to multiplier * n_syn records
        (Step 8: draw up to 10 * n_syn for subsequent post-processing in Component 3).
        """
        print(f">>> [Generators Sample] Drawing up to {multiplier}x n_syn synthetic flows per minority class...")
        synthetic_records = []

        for cls_name, model in self.models.items():
            n_syn = syn_deficit.get(cls_name, 0)
            if n_syn <= 0:
                continue

            draw_count = int(multiplier * n_syn)
            print(f"    - Sampling {draw_count:,} candidate flows for '{cls_name}' ({n_syn:,} needed)...")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_syn = model.sample(draw_count)

            df_syn["Attack"] = cls_name
            df_syn["Label"] = 1
            synthetic_records.append(df_syn)

        if synthetic_records:
            df_synthetic = pd.concat(synthetic_records, ignore_index=True)
        else:
            df_synthetic = pd.DataFrame()

        print(f"    Generated Raw Candidate Pool: {len(df_synthetic):,} synthetic flows across {len(synthetic_records)} classes.\n")
        return df_synthetic

    def save_weights(self, filename: Optional[str] = None) -> Path:
        """Saves current generator weights/checkpoints for all class models."""
        if filename is None:
            filename = f"{self.model_type}_best_weights.pt"
        save_path = self.weights_dir / filename

        with open(save_path, "wb") as f:
            pickle.dump(self.models, f)

        self.best_weights_path = save_path
        print(f"    [Generators Checkpoint] Saved weights to: {save_path.name}")
        return save_path

    def load_weights(self, filepath: Optional[Union[str, Path]] = None) -> bool:
        """Loads saved generator checkpoints."""
        load_path = Path(filepath) if filepath else self.best_weights_path
        if load_path is None or not load_path.exists():
            print(f"    [Warning] Checkpoint not found at {load_path}")
            return False

        with open(load_path, "rb") as f:
            self.models = pickle.load(f)
        print(f"    [Generators Checkpoint] Loaded weights from: {load_path.name}")
        return True


__all__ = [
    "TabularGenerators",
]


if __name__ == "__main__":
    print("Component 2 (tabular_generators.py) loaded successfully.")


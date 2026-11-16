#!/usr/bin/env python3
"""
TVAE Tabular Deep Generative Model (Component 2 - TVAE)

Derived from Do Minh Thanh's TVAE configuration (`do_minh_thanh/notebooks/tvae_final_perclass_v2.ipynb`):
- TVAE-specific NetFlow preprocessing & validity snapper (`TVAEPreprocessor`):
  - Numeric cleaning with median imputation.
  - Discrete code column snapping to observed valid codes (`min_code_count = 3`).
  - Log1p transform on skewed numeric features exceeding `log_threshold = 1000.0`, expm1 on inverse.
  - Post-generation clipping to [col_min, col_max] and integer rounding.
- Per-minority-class `sdv.single_table.TVAESynthesizer` training.
- Per-class sampling and checkpoint/model export (`tvae_per_class_models/`).
"""

import os
import sys
import time
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd

# Prevent OpenMP runtime library conflict on Linux/Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.append(str(curr_dir))

from intra_attack_resampler import resolve_paths

# Do Minh Thanh's TVAE configuration (`do_minh_thanh/notebooks/tvae_final_perclass_v2.ipynb`)
TVAE_CODE_COLS = [
    "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE",
]
TVAE_DIMS = {
    "128x128": (128, 128),
    "256x256": (256, 256),
    "256x256x256": (256, 256, 256),
}
TVAE_DEFAULT_PARAMS = {
    "epochs": 300,
    "batch_size": 1000,
    "embedding_dim": 64,
    "dims": "128x128",
    "l2scale": 0.0007235103476873236,
    "loss_factor": 1,
}


class TVAEPreprocessor:
    """
    NetFlow preprocessor & validity snapper for SDV TVAESynthesizer.
    Ported from `do_minh_thanh/notebooks/tvae_final_perclass_v2.ipynb`.
    """

    def __init__(self, label_col: str = "Attack", min_code_count: int = 3, log_threshold: float = 1000.0):
        self.label_col = label_col
        self.min_code_count = min_code_count
        self.log_threshold = log_threshold
        self.feats: List[str] = []
        self.code_cols: List[str] = []
        self.int_cols: List[str] = []
        self.log_cols: List[str] = []
        self.med: Optional[pd.Series] = None
        self.valid_codes: Dict[str, np.ndarray] = {}
        self.col_min: Optional[pd.Series] = None
        self.col_max: Optional[pd.Series] = None
        self.order: List[str] = []

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if self.med is None:
            self.med = X.median()
        return X.fillna(self.med).fillna(0)

    def _snap(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.code_cols:
            v = self.valid_codes[c]
            x = X[c].values.astype(float)
            if len(v) == 0:
                continue
            if len(v) == 1:
                X[c] = v[0]
                continue
            i = np.clip(np.searchsorted(v, x), 1, len(v) - 1)
            X[c] = np.where(np.abs(x - v[i - 1]) <= np.abs(v[i] - x), v[i - 1], v[i])
        return X

    def fit(self, df: pd.DataFrame) -> "TVAEPreprocessor":
        self.order = list(df.columns)
        self.feats = [c for c in df.columns if c not in (self.label_col, "Label", "OOF_Confidence")]
        self.code_cols = [c for c in TVAE_CODE_COLS if c in self.feats]
        self.med = None
        X = self._clean(df)

        self.valid_codes = {}
        for c in self.code_cols:
            vc = X[c].value_counts()
            valid = np.sort(vc[vc >= self.min_code_count].index.values.astype(float))
            if len(valid) == 0:
                valid = np.sort(vc.index.values.astype(float))
            self.valid_codes[c] = valid

        X = self._snap(X)
        self.int_cols = [c for c in self.feats if np.all(np.mod(X[c], 1) == 0)]
        self.log_cols = [
            c for c in self.feats
            if c not in self.code_cols and X[c].min() >= 0 and X[c].max() > self.log_threshold
        ]
        self.col_min = X.min()
        self.col_max = X.max()
        return self

    def to_model(self, df: pd.DataFrame) -> pd.DataFrame:
        Z = self._snap(self._clean(df))
        for c in self.log_cols:
            Z[c] = np.log1p(Z[c].clip(lower=0))
        for c in self.code_cols:
            Z[c] = Z[c].map(lambda v: f"{v:g}")
        return Z

    def from_model(self, Z: pd.DataFrame) -> pd.DataFrame:
        X = Z.copy()
        for c in self.code_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(self.med[c] if self.med is not None else 0.0)
        for c in self.log_cols:
            X[c] = np.expm1(X[c])
        X = X[self.feats].clip(lower=self.col_min, upper=self.col_max, axis=1)
        X[self.int_cols] = X[self.int_cols].round()
        return X


class TVAEGenerator:
    """
    Manager for SDV TVAE Generative Model for Network Intrusion Data.
    Trains per-minority-class `sdv.single_table.TVAESynthesizer` models
    with `TVAEPreprocessor`.
    """

    def __init__(
        self,
        fixed_gradient_updates: int = 42000,
        random_state: int = 42,
        device: Optional[str] = None,
        weights_dir: Optional[Union[str, Path]] = None,
    ):
        self.model_type = "tvae"
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
            _, _, final_dir = resolve_paths()
            self.weights_dir = final_dir / "generators/TVAE/weights"
        else:
            self.weights_dir = Path(weights_dir).resolve()
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Any] = {}
        self.unified_model: Optional[Any] = None
        self.tvae_preprocessor: Optional[TVAEPreprocessor] = None
        self.best_weights_path: Optional[Path] = None
        self.per_class_model_paths: Dict[str, str] = {}
        self.training_diagnostics: Dict[str, Any] = {}

    def train(
        self,
        train_attacks: pd.DataFrame,
        syn_deficit: Dict[str, int],
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        unified_generator: bool = False,
        verbose: bool = False,
        **model_kwargs,
    ) -> Dict[str, Any]:
        """
        Per-class SDV TVAESynthesizer training ported from
        `do_minh_thanh/notebooks/tvae_final_perclass_v2.ipynb`.
        """
        try:
            from sdv.metadata import Metadata
            from sdv.single_table import TVAESynthesizer
        except ImportError as exc:
            raise ImportError(
                "sdv package is required for TVAE. Install via `pip install sdv>=1.0.0`."
            ) from exc

        import torch
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        t0 = time.time()
        target_classes = [cls for cls, n_syn in syn_deficit.items() if n_syn > 0]

        resolved_epochs = int(epochs if epochs is not None else model_kwargs.get("epochs", TVAE_DEFAULT_PARAMS["epochs"]))
        resolved_batch_size = int(batch_size if batch_size is not None else model_kwargs.get("batch_size", TVAE_DEFAULT_PARAMS["batch_size"]))
        resolved_emb = int(embedding_dim if embedding_dim is not None else model_kwargs.get("embedding_dim", TVAE_DEFAULT_PARAMS["embedding_dim"]))
        dims_spec = model_kwargs.get("dims", TVAE_DEFAULT_PARAMS["dims"])
        if isinstance(dims_spec, str):
            dims_tuple = TVAE_DIMS.get(dims_spec) or tuple(int(d) for d in dims_spec.split("x"))
        else:
            dims_tuple = tuple(dims_spec)
        l2scale = float(model_kwargs.get("l2scale", TVAE_DEFAULT_PARAMS["l2scale"]))
        loss_factor = int(model_kwargs.get("loss_factor", TVAE_DEFAULT_PARAMS["loss_factor"]))
        use_cuda = (self.device != "cpu")

        print("\n>>> [Generators Train] Training SDV TVAE (Per-Class)...")
        print(f"    Target Classes: {target_classes}")
        print(f"    Params: epochs={resolved_epochs}, bs={resolved_batch_size}, emb={resolved_emb}, dims={dims_tuple}, l2={l2scale:.2e}, lf={loss_factor}")
        print(f"    Compute Device: {self.device.upper()}")

        self.tvae_preprocessor = TVAEPreprocessor(label_col="Attack")
        self.tvae_preprocessor.fit(train_attacks)

        self.unified_model = None
        self.models = {}
        diagnostics: Dict[str, Any] = {}

        for cls_name in target_classes:
            df_cls = train_attacks[train_attacks["Attack"] == cls_name].reset_index(drop=True)
            if df_cls.empty:
                continue

            Z = self.tvae_preprocessor.to_model(df_cls)
            md_ = Metadata.detect_from_dataframes({"t": Z}, infer_keys=None)
            for col in Z.columns:
                md_.update_column(
                    col,
                    table_name="t",
                    sdtype="categorical" if col in self.tvae_preprocessor.code_cols else "numerical",
                )

            cls_batch_size = max(2, min(resolved_batch_size, (len(Z) // 2) * 2)) if len(Z) >= 2 else len(Z)

            t_cls = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = TVAESynthesizer(
                    md_,
                    epochs=resolved_epochs,
                    batch_size=cls_batch_size,
                    embedding_dim=resolved_emb,
                    compress_dims=dims_tuple,
                    decompress_dims=dims_tuple,
                    l2scale=l2scale,
                    loss_factor=loss_factor,
                    enforce_min_max_values=True,
                    enforce_rounding=False,
                    enable_gpu=use_cuda,
                    verbose=verbose,
                )
                model.fit({"t": Z})

            self.models[cls_name] = model
            cls_diag: Dict[str, Any] = {
                "samples": len(Z),
                "epochs": resolved_epochs,
                "batch_size": cls_batch_size,
                "embedding_dim": resolved_emb,
                "dims": list(dims_tuple),
                "l2scale": l2scale,
                "loss_factor": loss_factor,
                "elapsed_sec": round(time.time() - t_cls, 2),
            }
            try:
                loss_df = model.get_loss_values()
                if loss_df is not None and not loss_df.empty:
                    loss_col = "Loss" if "Loss" in loss_df.columns else loss_df.columns[-1]
                    cls_diag["initial_loss"] = round(float(loss_df[loss_col].iloc[0]), 6)
                    cls_diag["final_loss"] = round(float(loss_df[loss_col].iloc[-1]), 6)
                    cls_diag["loss_epochs_recorded"] = int(len(loss_df))
            except Exception:
                pass
            diagnostics[cls_name] = cls_diag

        self.training_diagnostics = diagnostics
        print(f"    SDV TVAE training completed in {time.time() - t0:.2f}s.\n")
        return diagnostics

    def get_generator_details(self) -> Dict[str, Any]:
        """Returns detailed TVAE generator metadata, hyperparameters, and per-class diagnostics."""
        prep_info: Dict[str, Any] = {}
        if self.tvae_preprocessor is not None:
            prep_info = {
                "num_features": len(self.tvae_preprocessor.feats),
                "code_cols": list(self.tvae_preprocessor.code_cols),
                "log_cols": list(self.tvae_preprocessor.log_cols),
                "int_cols_count": len(self.tvae_preprocessor.int_cols),
                "min_code_count": self.tvae_preprocessor.min_code_count,
                "log_threshold": self.tvae_preprocessor.log_threshold,
            }

        return {
            "library": "sdv.single_table.TVAESynthesizer",
            "mode": "per_class",
            "num_per_class_models": len(self.models),
            "trained_classes": list(self.models.keys()),
            "per_class_models": self.training_diagnostics,
            "per_class_model_files": dict(self.per_class_model_paths),
            "preprocessor": prep_info,
        }

    def sample(
        self,
        syn_deficit: Dict[str, int],
        multiplier: int = 10,
        max_rounds: int = 10,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Samples synthetic flows according to syn_deficit using per-class SDV TVAESynthesizer
        (or pool sampling if a unified model checkpoint is loaded).
        Applies inverse transformation with validity clipping and integer rounding.
        """
        print(f">>> [Generators Sample] Drawing candidate synthetic pool ({multiplier}x deficit quota)...")
        quotas = {cls: int(multiplier * n) for cls, n in syn_deficit.items() if n > 0}
        if not quotas:
            return pd.DataFrame()

        t0 = time.time()
        raw_parts = []

        if self.unified_model is not None:
            # TVAE pool sampling fallback (Minh's sample_pool_by_class)
            need = dict(quotas)
            kept = {c: [] for c in quotas}
            for _ in range(max_rounds * 2):
                if all(v <= 0 for v in need.values()):
                    break
                batch_draw = min(50_000, max(5_000, sum(need.values()) * 2))
                pool = self.unified_model.sample(batch_draw)
                for cls in quotas:
                    if need[cls] <= 0:
                        continue
                    if "Attack" in pool.columns:
                        m = pool[pool["Attack"] == cls]
                    else:
                        m = pool.head(need[cls]).copy()
                        m["Attack"] = cls

                    take = m.head(need[cls])
                    kept[cls].append(take)
                    need[cls] -= len(take)

            for v in kept.values():
                if v:
                    raw_parts.append(pd.concat(v, ignore_index=True))
        else:
            # Per-class generator sampling
            for cls_name, model in self.models.items():
                if cls_name not in quotas:
                    continue
                draw_count = quotas[cls_name]
                batch = model.sample(num_rows=int(draw_count))
                batch = batch["t"] if isinstance(batch, dict) else batch
                if self.tvae_preprocessor is not None:
                    batch = self.tvae_preprocessor.from_model(batch)
                batch["Attack"] = cls_name
                raw_parts.append(batch)

        if not raw_parts:
            return pd.DataFrame()

        df_syn = pd.concat(raw_parts, ignore_index=True)

        # Ensure Label = 1 for all synthetic attack flows
        df_syn["Label"] = 1

        elapsed = time.time() - t0
        print(f"    Generated Raw Candidate Pool: {len(df_syn):,} synthetic flows across {len(quotas)} classes in {elapsed:.2f}s.\n")
        return df_syn

    def save_weights(self, filename: Optional[Union[str, Path]] = None) -> Path:
        """Saves TVAE generator weights, TVAEPreprocessor state, and per-class SDV model files."""
        if filename is None:
            save_path = self.weights_dir / "tvae_best_weights.pt"
        else:
            p = Path(filename)
            if p.is_absolute() or str(filename).find("/") != -1 or str(filename).find("\\") != -1:
                save_path = p.resolve()
            else:
                save_path = self.weights_dir / p

        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "models": self.models,
            "unified_model": self.unified_model,
            "tvae_preprocessor": self.tvae_preprocessor,
            "diagnostics": self.training_diagnostics,
        }
        with open(save_path, "wb") as f:
            pickle.dump(state, f)

        self.best_weights_path = save_path
        print(f"    [Generators Checkpoint] Saved weights to: {save_path.name}")

        # Export individual per-class SDV TVAE models alongside the unified checkpoint
        if self.models:
            per_class_dir = save_path.parent / "tvae_per_class_models"
            per_class_dir.mkdir(parents=True, exist_ok=True)
            self.per_class_model_paths = {}
            for cls_name, model in self.models.items():
                cls_path = per_class_dir / f"tvae_{cls_name}.pkl"
                if hasattr(model, "save"):
                    model.save(str(cls_path))
                else:
                    with open(cls_path, "wb") as f:
                        pickle.dump(model, f)
                self.per_class_model_paths[cls_name] = str(cls_path)
            print(f"    [Generators Checkpoint] Exported {len(self.per_class_model_paths)} per-class TVAE models to: {per_class_dir.name}/")

        return save_path

    def load_weights(self, filepath: Optional[Union[str, Path]] = None) -> bool:
        """Loads saved TVAE generator checkpoints."""
        load_path = Path(filepath) if filepath else self.best_weights_path
        if load_path is None or not load_path.exists():
            print(f"    [Warning] Checkpoint not found at {load_path}")
            return False

        with open(load_path, "rb") as f:
            state = pickle.load(f)

        if isinstance(state, dict) and "unified_model" in state:
            self.models = state.get("models", {})
            self.unified_model = state.get("unified_model", None)
            self.tvae_preprocessor = state.get("tvae_preprocessor", None)
            self.training_diagnostics = state.get("diagnostics", {})
        else:
            self.models = state

        print(f"    [Generators Checkpoint] Loaded weights from: {load_path.name}")
        return True


__all__ = [
    "TVAEGenerator",
    "TVAEPreprocessor",
    "TVAE_CODE_COLS",
    "TVAE_DIMS",
    "TVAE_DEFAULT_PARAMS",
]


if __name__ == "__main__":
    print("Component 2 (tvae_generator.py) loaded successfully.")

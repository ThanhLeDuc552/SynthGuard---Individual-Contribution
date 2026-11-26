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

from intra_attack_resampler import resolve_paths, DERIVED_FEATURES, recompute_derived_features

from sklearn.preprocessing import QuantileTransformer

# Do Minh Thanh's TVAE configuration (`do_minh_thanh/notebooks/tvae_final_perclass_v2.ipynb`)
TVAE_CODE_COLS = [
    "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS", "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE",
]
TVAE_SPIKY_CODE_COLS = ["TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"]
TVAE_TOP_K_SPIKY = 30
TVAE_CONST_THRESHOLD = 0.999
TVAE_ZERO_INFLATED_MIN_RATE = 0.05
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

    def __init__(self, label_col: str = "Attack", min_code_count: int = 3, log_threshold: float = 1000.0, random_state: int = 42):
        self.label_col = label_col
        self.min_code_count = min_code_count
        self.log_threshold = log_threshold
        self.random_state = random_state
        self.feats: List[str] = []
        self.active_feats: List[str] = []
        self.const_cols: Dict[str, float] = {}
        self.orig_code_cols: List[str] = []
        self.spiky_cols: List[str] = []
        self.spiky_top: Dict[str, set] = {}
        self.spiky_rare_vals: Dict[str, np.ndarray] = {}
        self.zero_inflated_cols: List[str] = []
        self.zero_flag_cols: List[str] = []
        self.pos_med: Dict[str, float] = {}
        self.code_cols: List[str] = []
        self.cont_cols: List[str] = []
        self.int_cols: List[str] = []
        self.log_cols: List[str] = []
        self.qt: Optional[QuantileTransformer] = None
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

    @staticmethod
    def _apply_coupled_reparam(X: pd.DataFrame) -> pd.DataFrame:
        """Reparameterize coupled packet-length and TTL bounds into non-negative deltas."""
        X = X.copy()
        if "MIN_TTL" in X.columns and "MAX_TTL" in X.columns:
            X["MAX_TTL"] = (X["MAX_TTL"] - X["MIN_TTL"]).clip(lower=0.0)
        if "SHORTEST_FLOW_PKT" in X.columns and "LONGEST_FLOW_PKT" in X.columns:
            long_orig = X["LONGEST_FLOW_PKT"].copy()
            X["LONGEST_FLOW_PKT"] = (long_orig - X["SHORTEST_FLOW_PKT"]).clip(lower=0.0)
            if "MAX_IP_PKT_LEN" in X.columns:
                X["MAX_IP_PKT_LEN"] = (X["MAX_IP_PKT_LEN"] - long_orig).clip(lower=0.0)
        return X

    @staticmethod
    def _invert_coupled_reparam(X: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct coupled bounds and derived-zero consistency from generated deltas."""
        X = X.copy()
        if "MIN_TTL" in X.columns and "MAX_TTL" in X.columns:
            X["MAX_TTL"] = X["MIN_TTL"] + X["MAX_TTL"].clip(lower=0.0)
        if "SHORTEST_FLOW_PKT" in X.columns and "LONGEST_FLOW_PKT" in X.columns:
            X["LONGEST_FLOW_PKT"] = X["SHORTEST_FLOW_PKT"] + X["LONGEST_FLOW_PKT"].clip(lower=0.0)
            if "MAX_IP_PKT_LEN" in X.columns:
                X["MAX_IP_PKT_LEN"] = X["LONGEST_FLOW_PKT"] + X["MAX_IP_PKT_LEN"].clip(lower=0.0)
        if "SRC_TO_DST_AVG_THROUGHPUT" in X.columns and "IN_BYTES" in X.columns:
            X.loc[X["IN_BYTES"] <= 0, "SRC_TO_DST_AVG_THROUGHPUT"] = 0.0
        if "DST_TO_SRC_AVG_THROUGHPUT" in X.columns and "OUT_BYTES" in X.columns:
            X.loc[X["OUT_BYTES"] <= 0, "DST_TO_SRC_AVG_THROUGHPUT"] = 0.0
        return X

    def _snap(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.orig_code_cols:
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
        self.feats = [c for c in df.columns if c not in (self.label_col, "Label", "OOF_Confidence") and c not in DERIVED_FEATURES]
        self.med = None
        X = self._clean(df)

        # 1. Freeze near-constant / constant columns (>99.9% identical)
        self.const_cols = {}
        for c in self.feats:
            if X[c].nunique(dropna=True) <= 1 or X[c].value_counts(normalize=True, dropna=True).iloc[0] > TVAE_CONST_THRESHOLD:
                mode_val = X[c].mode(dropna=True)
                self.const_cols[c] = float(mode_val.iloc[0]) if len(mode_val) > 0 else 0.0
        self.active_feats = [c for c in self.feats if c not in self.const_cols]

        # 2. Discrete code columns (including ICMP_TYPE/ICMP_IPV4_TYPE) & spiky OS window codes
        self.orig_code_cols = [c for c in TVAE_CODE_COLS if c in self.active_feats]
        self.spiky_cols = [c for c in TVAE_SPIKY_CODE_COLS if c in self.active_feats]

        self.valid_codes = {}
        for c in self.orig_code_cols:
            vc = X[c].value_counts()
            valid = np.sort(vc[vc >= self.min_code_count].index.values.astype(float))
            if len(valid) == 0:
                valid = np.sort(vc.index.values.astype(float))
            self.valid_codes[c] = valid

        X = self._snap(X)
        self.spiky_top = {}
        self.spiky_rare_vals = {}
        for c in self.spiky_cols:
            top_vals = set(X[c].value_counts().index[:TVAE_TOP_K_SPIKY].astype(float))
            self.spiky_top[c] = top_vals
            self.spiky_rare_vals[c] = X.loc[~X[c].astype(float).isin(top_vals), c].dropna().to_numpy(dtype=float)

        # 3. Coupled delta reparameterization before bounds & continuous transform fitting
        X_rep = self._apply_coupled_reparam(X)
        self.int_cols = [c for c in self.feats if np.all(np.mod(X[c], 1) == 0)]
        self.col_min = X_rep[self.feats].min()
        self.col_max = X_rep[self.feats].max()

        self.cont_cols = [c for c in self.active_feats if c not in self.orig_code_cols and c not in self.spiky_cols]

        # 4. Zero-inflated continuous columns -> hurdle binary flag + positive-only QuantileTransformer
        self.zero_inflated_cols = []
        self.zero_flag_cols = []
        self.pos_med = {}
        X_cont = X_rep[self.cont_cols].copy()
        for c in self.cont_cols:
            if X_cont[c].min() >= 0:
                zero_rate = float((X_cont[c] == 0).mean())
                if TVAE_ZERO_INFLATED_MIN_RATE <= zero_rate <= TVAE_CONST_THRESHOLD:
                    self.zero_inflated_cols.append(c)
                    self.zero_flag_cols.append(f"{c}__is_zero")
                    pos_vals = X_cont.loc[X_cont[c] > 0, c]
                    med_pos = float(pos_vals.median()) if len(pos_vals) > 0 else 0.0
                    self.pos_med[c] = med_pos
                    X_cont.loc[X_cont[c] == 0, c] = med_pos

        self.code_cols = self.orig_code_cols + self.spiky_cols + self.zero_flag_cols
        self.log_cols = [
            c for c in self.cont_cols
            if X_rep[c].min() >= 0 and X_rep[c].max() > self.log_threshold
        ]

        if len(self.cont_cols) > 0 and len(X_cont) >= 2:
            n_q = min(1000, len(X_cont))
            self.qt = QuantileTransformer(
                n_quantiles=n_q,
                output_distribution="normal",
                subsample=min(100_000, len(X_cont)),
                random_state=self.random_state,
            )
            self.qt.fit(X_cont)
        else:
            self.qt = None
        return self

    def to_model(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._snap(self._clean(df))
        X_rep = self._apply_coupled_reparam(X)
        Z = X_rep[self.active_feats].copy()

        for c in self.zero_inflated_cols:
            is_zero = (Z[c] == 0)
            Z[f"{c}__is_zero"] = np.where(is_zero, "1", "0")
            Z.loc[is_zero, c] = self.pos_med[c]

        if self.qt is not None and len(self.cont_cols) > 0:
            Z[self.cont_cols] = self.qt.transform(Z[self.cont_cols])
        else:
            for c in self.log_cols:
                if c in Z.columns:
                    Z[c] = np.log1p(Z[c].clip(lower=0))

        for c in self.orig_code_cols:
            Z[c] = Z[c].map(lambda v: f"{v:g}")
        for c in self.spiky_cols:
            top_vals = self.spiky_top[c]
            Z[c] = Z[c].map(lambda v: f"{float(v):g}" if float(v) in top_vals else "__OTHER__")
        return Z

    def from_model(self, Z: pd.DataFrame) -> pd.DataFrame:
        X = Z.copy()
        rng = np.random.default_rng(self.random_state)

        for c in self.orig_code_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(self.med[c] if self.med is not None else 0.0)
        for c in self.spiky_cols:
            m_other = X[c].astype(str) == "__OTHER__"
            if m_other.any() and len(self.spiky_rare_vals.get(c, [])) > 0:
                X.loc[m_other, c] = rng.choice(self.spiky_rare_vals[c], size=int(m_other.sum()))
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(self.med[c] if self.med is not None else 0.0)

        if self.qt is not None and len(self.cont_cols) > 0:
            X[self.cont_cols] = self.qt.inverse_transform(X[self.cont_cols])
        else:
            for c in self.log_cols:
                if c in X.columns:
                    X[c] = np.expm1(X[c])

        for c in self.zero_inflated_cols:
            flag_col = f"{c}__is_zero"
            if flag_col in X.columns:
                X.loc[X[flag_col].astype(str) == "1", c] = 0.0
                X = X.drop(columns=[flag_col])

        for c, v in self.const_cols.items():
            X[c] = v

        X = X[self.feats].clip(lower=self.col_min, upper=self.col_max, axis=1)
        X = self._invert_coupled_reparam(X)
        X[self.int_cols] = X[self.int_cols].round()
        X = recompute_derived_features(X)
        target_cols = [c for c in self.order if c in X.columns]
        extra_cols = [c for c in X.columns if c not in target_cols]
        return X[target_cols + extra_cols]


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
        self.best_weights_size_bytes: int = 0
        self.per_class_model_paths: Dict[str, str] = {}
        self.per_class_model_sizes: Dict[str, int] = {}
        self.training_diagnostics: Dict[str, Any] = {}
        self.last_run_config: Dict[str, Any] = {}
        self.last_total_train_sec: float = 0.0
        self.last_sampling_diagnostics: Dict[str, Any] = {}

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

        self.tvae_preprocessor = TVAEPreprocessor(label_col="Attack", random_state=self.random_state)
        self.tvae_preprocessor.fit(train_attacks)

        self.unified_model = None
        self.models = {}
        self.per_class_model_paths = {}
        self.per_class_model_sizes = {}
        self.last_run_config = {
            "epochs": resolved_epochs,
            "requested_batch_size": resolved_batch_size,
            "embedding_dim": resolved_emb,
            "dims": list(dims_tuple),
            "l2scale": l2scale,
            "loss_factor": loss_factor,
            "device": self.device,
            "random_state": self.random_state,
            "fixed_gradient_updates": self.fixed_gradient_updates,
            "target_classes": target_classes,
        }
        diagnostics: Dict[str, Any] = {}

        for idx, cls_name in enumerate(target_classes, 1):
            df_cls = train_attacks[train_attacks["Attack"] == cls_name].reset_index(drop=True)
            if df_cls.empty:
                print(f"    [{idx}/{len(target_classes)}] Skipping '{cls_name}': 0 training rows.")
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
            steps_per_epoch = max(1, len(Z) // max(1, cls_batch_size))
            print(
                f"    [{idx}/{len(target_classes)}] Training TVAE '{cls_name}': "
                f"samples={len(Z):,}, deficit={syn_deficit.get(cls_name, 0):,}, "
                f"bs={cls_batch_size}, epochs={resolved_epochs}, steps/epoch={steps_per_epoch}..."
            )

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
            cls_elapsed = round(time.time() - t_cls, 2)
            cls_diag: Dict[str, Any] = {
                "samples": len(Z),
                "syn_deficit": int(syn_deficit.get(cls_name, 0)),
                "epochs": resolved_epochs,
                "batch_size": cls_batch_size,
                "steps_per_epoch": steps_per_epoch,
                "total_gradient_updates": int(steps_per_epoch * resolved_epochs),
                "embedding_dim": resolved_emb,
                "dims": list(dims_tuple),
                "l2scale": l2scale,
                "loss_factor": loss_factor,
                "elapsed_sec": cls_elapsed,
            }
            loss_summary = ""
            try:
                loss_df = model.get_loss_values()
                if loss_df is not None and not loss_df.empty:
                    loss_col = "Loss" if "Loss" in loss_df.columns else loss_df.columns[-1]
                    losses = loss_df[loss_col].astype(float)
                    cls_diag["initial_loss"] = round(float(losses.iloc[0]), 6)
                    cls_diag["final_loss"] = round(float(losses.iloc[-1]), 6)
                    cls_diag["min_loss"] = round(float(losses.min()), 6)
                    cls_diag["loss_epochs_recorded"] = int(len(loss_df))
                    loss_summary = f" | loss: {cls_diag['initial_loss']:.4f} -> {cls_diag['final_loss']:.4f} (min={cls_diag['min_loss']:.4f})"
            except Exception:
                pass
            print(f"      -> Done '{cls_name}' in {cls_elapsed:.2f}s{loss_summary}")
            diagnostics[cls_name] = cls_diag

        self.training_diagnostics = diagnostics
        self.last_total_train_sec = round(time.time() - t0, 2)
        print(f"    SDV TVAE training completed across {len(self.models)}/{len(target_classes)} per-class models in {self.last_total_train_sec:.2f}s.\n")
        return diagnostics

    def get_generator_details(self) -> Dict[str, Any]:
        """Returns detailed TVAE generator metadata, main checkpoint info, and per-class diagnostics."""
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

        per_class_files_detailed: Dict[str, Any] = {}
        for cls_name, cls_path in self.per_class_model_paths.items():
            size_b = self.per_class_model_sizes.get(cls_name, 0)
            per_class_files_detailed[cls_name] = {
                "path": cls_path,
                "size_bytes": size_b,
                "size_mb": round(size_b / (1024 * 1024), 3),
            }

        return {
            "library": "sdv.single_table.TVAESynthesizer",
            "mode": "per_class",
            "main_checkpoint": {
                "path": str(self.best_weights_path) if self.best_weights_path else None,
                "size_bytes": self.best_weights_size_bytes,
                "size_mb": round(self.best_weights_size_bytes / (1024 * 1024), 3),
                "total_training_elapsed_sec": self.last_total_train_sec,
                "hyperparameters": dict(self.last_run_config),
            },
            "num_per_class_models": len(self.models),
            "trained_classes": list(self.models.keys()),
            "per_class_models": self.training_diagnostics,
            "per_class_model_files": per_class_files_detailed,
            "last_sampling_summary": self.last_sampling_diagnostics,
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
            samp_diag: Dict[str, Any] = {}
            for cls_name, model in self.models.items():
                if cls_name not in quotas:
                    continue
                t_s = time.time()
                draw_count = quotas[cls_name]
                batch = model.sample(num_rows=int(draw_count))
                batch = batch["t"] if isinstance(batch, dict) else batch
                if self.tvae_preprocessor is not None:
                    batch = self.tvae_preprocessor.from_model(batch)
                batch["Attack"] = cls_name
                raw_parts.append(batch)
                s_elapsed = round(time.time() - t_s, 2)
                samp_diag[cls_name] = {"quota": int(draw_count), "generated": int(len(batch)), "elapsed_sec": s_elapsed}
                print(f"    - Sampled '{cls_name}': {len(batch):,}/{draw_count:,} flows in {s_elapsed:.2f}s")
            self.last_sampling_diagnostics = samp_diag

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
        self.best_weights_size_bytes = save_path.stat().st_size if save_path.exists() else 0
        print(f"    [Generators Checkpoint] Saved main checkpoint: {save_path.name} ({self.best_weights_size_bytes / (1024 * 1024):.2f} MB)")

        # Export individual per-class SDV TVAE models alongside the unified checkpoint
        if self.models:
            per_class_dir = save_path.parent / "tvae_per_class_models"
            per_class_dir.mkdir(parents=True, exist_ok=True)
            self.per_class_model_paths = {}
            self.per_class_model_sizes = {}
            for cls_name, model in self.models.items():
                cls_path = per_class_dir / f"tvae_{cls_name}.pkl"
                if hasattr(model, "save"):
                    model.save(str(cls_path))
                else:
                    with open(cls_path, "wb") as f:
                        pickle.dump(model, f)
                cls_size = cls_path.stat().st_size if cls_path.exists() else 0
                self.per_class_model_paths[cls_name] = str(cls_path)
                self.per_class_model_sizes[cls_name] = cls_size
                print(f"      - Exported '{cls_name}': {cls_path.name} ({cls_size / (1024 * 1024):.2f} MB)")
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

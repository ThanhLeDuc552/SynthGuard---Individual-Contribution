#!/usr/bin/env python3
"""
Tabular Deep Generative Models: CTGAN & TVAE (Component 2)

Derived from Minh's CTGAN/TVAE training and tuning pipeline (`minh/src/ctgan_pipeline.py`,
`minh/src/fix_pipeline.py`, and `minh/notebooks/07_tune_ctgan.ipynb`):
- NetFlow-aware preprocessing (fit_prep, transform, inverse):
  - Categorical handling with TOP_K common codes and rare value reconstruction.
  - Dropping constant and quasi-constant columns, restored on inverse.
  - Log1p transform on skewed numeric features (skew > 5), expm1 on inverse.
  - Post-generation validity constraints: clipping to real max, non-negative bounds,
    and rounding integer count columns (packets, bytes).
- Batch size and pac alignment (step = pac if pac % 2 == 0 else pac * 2).
- Conditional generation with rejection sampling (sample_by_class / sample_matched) for CTGAN:
  Enforces exact class match, filters misconditioned flows, reports shortfalls.
- Pool sampling by class quota for TVAE (sample_pool_by_class).
- Checkpoint persistence and gradient update budget alignment.
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

# Candidate categorical / discrete code columns in NetFlow schemas
CAT_CANDIDATES = [
    "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE",
    "MIN_TTL", "MAX_TTL"
]
ID_COLS = ["DNS_QUERY_ID"]
TOP_K_CATEGORICAL = 30
SKEW_LIMIT = 5.0

CTGAN_DIMS = {
    "256x256": (256, 256),
    "256x256x256": (256, 256, 256),
    "512x512": (512, 512),
}

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


class NetFlowPreprocessor:
    """
    Learns feature transformations, bounds, and categorical encodings strictly from real training data.
    Derived from `minh/src/fix_pipeline.py`.
    """

    def __init__(self, label_col: str = "Attack", top_k: int = TOP_K_CATEGORICAL, skew_limit: float = SKEW_LIMIT):
        self.label_col = label_col
        self.top_k = top_k
        self.skew_limit = skew_limit
        self.meta: Dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "NetFlowPreprocessor":
        p: Dict[str, Any] = {}
        clean_df = df.replace([np.inf, -np.inf], np.nan)

        # Categoricals: candidate columns + any object columns present
        obj_cols = [c for c in clean_df.columns if clean_df[c].dtype == "object" and c != self.label_col]
        p["cat"] = [c for c in CAT_CANDIDATES if c in clean_df.columns and c != self.label_col]
        for c in obj_cols:
            if c not in p["cat"]:
                p["cat"].append(c)

        # Drop constant, ID, or quasi-constant (>99.9% identical) columns
        drop_dict = {}
        for c in clean_df.columns:
            if c == self.label_col or c == "Label":
                continue
            if c in ID_COLS or clean_df[c].nunique(dropna=True) <= 1:
                mode_val = clean_df[c].mode(dropna=True)
                drop_dict[c] = mode_val.iloc[0] if len(mode_val) > 0 else 0
            elif clean_df[c].value_counts(normalize=True, dropna=True).iloc[0] > 0.999:
                drop_dict[c] = clean_df[c].mode(dropna=True).iloc[0]

        p["drop"] = drop_dict
        p["cat"] = [c for c in p["cat"] if c not in p["drop"]]
        p["num"] = [c for c in clean_df.columns if c not in p["cat"] + list(p["drop"]) + [self.label_col, "Label"]]

        # Top K categories and rare values for reconstruction
        p["top"] = {c: set(clean_df[c].value_counts().index[:self.top_k]) for c in p["cat"]}
        p["rare_vals"] = {c: clean_df.loc[~clean_df[c].isin(p["top"][c]), c].dropna().to_numpy() for c in p["cat"]}

        # Skewed continuous features -> log1p
        p["log"] = [c for c in p["num"] if clean_df[c].min() >= 0 and clean_df[c].skew() > self.skew_limit]
        p["median"] = clean_df[p["num"]].median().fillna(0.0)
        p["max"] = clean_df[p["num"]].max().fillna(1.0)
        p["min"] = clean_df[p["num"]].min().fillna(0.0)
        p["is_int"] = {c: bool(np.all(np.isclose(clean_df[c].dropna() % 1, 0))) for c in p["num"]}
        p["order"] = list(df.columns)

        self.meta = p
        return self

    @staticmethod
    def _as_str(series: pd.Series) -> np.ndarray:
        if series.dtype.kind in "iuf":
            return series.map(
                lambda v: "nan" if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v))
            ).to_numpy()
        return series.astype(str).to_numpy()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.meta
        clean_df = df.replace([np.inf, -np.inf], np.nan)
        drop_cols = [c for c in p["drop"] if c in clean_df.columns]
        out = clean_df.drop(columns=drop_cols).copy()

        for c in p["num"]:
            if c in out.columns:
                out[c] = out[c].fillna(p["median"][c])
                if c in p["log"]:
                    out[c] = np.log1p(out[c].clip(lower=0))

        for c in p["cat"]:
            if c in out.columns:
                out[c] = np.where(out[c].isin(p["top"][c]), self._as_str(out[c]), "other")

        return out

    def inverse(self, syn: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
        p = self.meta
        rng = np.random.default_rng(seed)
        out = syn.copy()

        for c in p["num"]:
            if c in out.columns:
                if c in p["log"]:
                    out[c] = np.expm1(out[c])
                # Validity clipping: lower bound at 0 if real was non-negative, upper bound at real max
                lower_bound = 0.0 if p["min"][c] >= 0 else p["min"][c]
                out[c] = out[c].clip(lower=lower_bound, upper=p["max"][c])
                if p["is_int"][c]:
                    out[c] = out[c].round()

        for c in p["cat"]:
            if c in out.columns:
                m = out[c] == "other"
                if m.any() and len(p["rare_vals"][c]) > 0:
                    out.loc[m, c] = rng.choice(p["rare_vals"][c], size=int(m.sum()))
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

        for c, v in p["drop"].items():
            out[c] = v

        # Align columns to original order
        target_cols = [c for c in p["order"] if c in out.columns]
        extra_cols = [c for c in out.columns if c not in target_cols]
        return out[target_cols + extra_cols]


class TabularGenerators:
    """
    Manager for CTGAN and TVAE Generative Models for Network Intrusion Data.
    Derived from Minh's training and tuning architecture.

    Features:
      - NetFlow preprocessing & post-generation validity enforcement.
      - Fixed gradient update budget calculation (default: 42,000 steps).
      - Conditional generation with rejection sampling (sample_by_class) for CTGAN.
      - Pool sampling for TVAE (sample_pool_by_class).
      - Weight persistence & checkpoint storage.
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
        self.unified_model: Optional[Any] = None
        self.preprocessor: Optional[NetFlowPreprocessor] = None
        self.tvae_preprocessor: Optional[TVAEPreprocessor] = None
        self.best_weights_path: Optional[Path] = None
        self.per_class_model_paths: Dict[str, str] = {}
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

    def _resolve_batch_size_and_pac(self, n_samples: int, requested_batch_size: int, requested_pac: int = 10) -> Tuple[int, int]:
        """
        Ensures batch_size is compatible with CTGAN's pac constraint:
          batch_size % pac == 0 and batch_size % 2 == 0
          bs not larger than the data
        """
        if n_samples >= 100:
            pac = requested_pac
        elif n_samples >= 20:
            pac = min(requested_pac, 2)
        else:
            pac = 1

        step = pac if pac % 2 == 0 else pac * 2
        batch_size = (requested_batch_size // step) * step
        batch_size = max(step * 2, min(batch_size, (n_samples // step) * step))
        if batch_size == 0:
            batch_size = max(step * 2, step)
        return int(batch_size), int(pac)

    def train(
        self,
        train_attacks: pd.DataFrame,
        syn_deficit: Dict[str, int],
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        batch_size: int = 500,
        epochs: Optional[int] = None,
        pac: int = 10,
        embedding_dim: int = 128,
        unified_generator: bool = True,
        verbose: bool = False,
        **model_kwargs,
    ) -> Dict[str, Any]:
        """
        Trains CTGAN/TVAE on attack flows.
        Supports both:
          - Unified generator (Minh's design): One generator conditioned on 'Attack' column.
          - Per-minority-class generators: Slices model per minority class.
        """
        import torch
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        t0 = time.time()
        target_classes = [cls for cls, n_syn in syn_deficit.items() if n_syn > 0]
        print(f"\n>>> [Generators Train] Training {self.model_type.upper()} ({'Unified' if unified_generator else 'Per-Class'})...")
        print(f"    Target Classes:          {target_classes}")
        print(f"    Fixed Gradient Updates:  {self.fixed_gradient_updates:,} steps")
        print(f"    Compute Device:          {self.device.upper()}")

        if self.model_type == "tvae":
            return self._train_tvae(
                train_attacks=train_attacks,
                syn_deficit=syn_deficit,
                batch_size=batch_size,
                epochs=epochs,
                embedding_dim=embedding_dim,
                verbose=verbose,
                **model_kwargs,
            )

        # Initialize and fit NetFlow preprocessor (CTGAN path)
        self.tvae_preprocessor = None
        self.preprocessor = NetFlowPreprocessor(label_col="Attack")
        self.preprocessor.fit(train_attacks)
        transformed_attacks = self.preprocessor.transform(train_attacks)

        diagnostics = {}
        use_cuda = (self.device != "cpu")

        if unified_generator:
            # Train a unified generator across all attack flows
            n_samples = len(transformed_attacks)
            bs, resolved_pac = self._resolve_batch_size_and_pac(n_samples, batch_size, pac)
            calc_epochs = epochs if epochs is not None else self.calculate_epochs(n_samples, bs)

            discrete_cols = [c for c in self.preprocessor.meta["cat"] if c in transformed_attacks.columns]
            if "Attack" not in discrete_cols and "Attack" in transformed_attacks.columns:
                discrete_cols.append("Attack")

            print(f"    - Fitting Unified {self.model_type.upper()} ({n_samples:,} flows, {calc_epochs} epochs, bs={bs}, pac={resolved_pac})...")

            clean_kwargs = {k: v for k, v in model_kwargs.items() if k not in ["epochs", "batch_size", "pac", "cuda", "verbose"]}
            if "generator_dim" in clean_kwargs and isinstance(clean_kwargs["generator_dim"], str):
                clean_kwargs["generator_dim"] = CTGAN_DIMS.get(clean_kwargs["generator_dim"], (256, 256))
            if "discriminator_dim" in clean_kwargs and isinstance(clean_kwargs["discriminator_dim"], str):
                clean_kwargs["discriminator_dim"] = CTGAN_DIMS.get(clean_kwargs["discriminator_dim"], (256, 256))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.model_type == "ctgan":
                    model = CTGAN(
                        epochs=calc_epochs,
                        batch_size=bs,
                        pac=resolved_pac,
                        embedding_dim=embedding_dim,
                        cuda=use_cuda,
                        verbose=verbose,
                        **clean_kwargs,
                    )
                else:
                    model = TVAE(
                        epochs=calc_epochs,
                        batch_size=bs,
                        embedding_dim=embedding_dim,
                        cuda=use_cuda,
                        verbose=verbose,
                        **clean_kwargs,
                    )

                model.fit(transformed_attacks, discrete_columns=discrete_cols)

            self.unified_model = model
            self.models = {"unified": model}
            diagnostics["unified"] = {
                "samples": n_samples,
                "epochs": calc_epochs,
                "batch_size": bs,
                "elapsed_sec": round(time.time() - t0, 2),
            }
        else:
            # Per-minority-class models
            self.models = {}
            for cls_name in target_classes:
                df_cls = transformed_attacks[transformed_attacks["Attack"] == cls_name].copy()
                n_samples = len(df_cls)
                if n_samples == 0:
                    continue

                bs, resolved_pac = self._resolve_batch_size_and_pac(n_samples, batch_size, pac)
                cls_epochs = epochs if epochs is not None else self.calculate_epochs(n_samples, bs)

                discrete_cols = [c for c in self.preprocessor.meta["cat"] if c in df_cls.columns]
                t_cls = time.time()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if self.model_type == "ctgan":
                        model = CTGAN(
                            epochs=cls_epochs,
                            batch_size=bs,
                            pac=resolved_pac,
                            embedding_dim=embedding_dim,
                            cuda=use_cuda,
                            verbose=verbose,
                            **model_kwargs,
                        )
                    else:
                        model = TVAE(
                            epochs=cls_epochs,
                            batch_size=bs,
                            embedding_dim=embedding_dim,
                            cuda=use_cuda,
                            verbose=verbose,
                            **model_kwargs,
                        )
                    model.fit(df_cls, discrete_columns=discrete_cols)

                self.models[cls_name] = model
                diagnostics[cls_name] = {
                    "samples": n_samples,
                    "epochs": cls_epochs,
                    "batch_size": bs,
                    "elapsed_sec": round(time.time() - t_cls, 2),
                }

        total_elapsed = time.time() - t0
        self.training_diagnostics = diagnostics
        print(f"    Generators training completed in {total_elapsed:.2f}s.\n")
        return diagnostics

    def _train_tvae(
        self,
        train_attacks: pd.DataFrame,
        syn_deficit: Dict[str, int],
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        embedding_dim: Optional[int] = None,
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

        print(f"\n>>> [Generators Train] Training SDV TVAE (Per-Class)...")
        print(f"    Target Classes: {target_classes}")
        print(f"    Params: epochs={resolved_epochs}, bs={resolved_batch_size}, emb={resolved_emb}, dims={dims_tuple}, l2={l2scale:.2e}, lf={loss_factor}")
        print(f"    Compute Device: {self.device.upper()}")

        self.preprocessor = None
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
        """Returns detailed generator metadata, hyperparameters, and per-class diagnostics."""
        if self.model_type != "tvae":
            return {
                "library": "ctgan.CTGAN",
                "mode": "unified" if self.unified_model is not None else "per_class",
                "diagnostics": self.training_diagnostics,
            }

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
        Samples synthetic flows according to syn_deficit.
        For CTGAN: Uses conditional rejection sampling (`sample_by_class`).
        For TVAE: Uses pool sampling (`sample_pool_by_class`).
        Applies inverse transformation with validity clipping and integer rounding.
        """
        print(f">>> [Generators Sample] Drawing candidate synthetic pool ({multiplier}x deficit quota)...")
        quotas = {cls: int(multiplier * n) for cls, n in syn_deficit.items() if n > 0}
        if not quotas:
            return pd.DataFrame()

        t0 = time.time()
        raw_parts = []

        if self.unified_model is not None:
            if self.model_type == "ctgan":
                # Conditional rejection sampling (Minh's sample_by_class)
                report = []
                for cls, quota_n in quotas.items():
                    kept, drawn, rounds, hits = [], 0, 0, 0
                    need = quota_n
                    while need > 0 and rounds < max_rounds:
                        rate = hits / drawn if drawn else 0.5
                        size = int(min(50_000, np.ceil(need * max(2.0, 1.5 / max(rate, 0.005))) + 10))
                        try:
                            batch = self.unified_model.sample(size, condition_column="Attack", condition_value=cls)
                        except Exception:
                            # Fallback if condition not supported
                            batch = self.unified_model.sample(size)

                        drawn += len(batch)
                        if "Attack" in batch.columns:
                            match = batch[batch["Attack"] == cls]
                        else:
                            match = batch.copy()
                            match["Attack"] = cls

                        hits += len(match)
                        kept.append(match.head(need))
                        need = quota_n - sum(len(k) for k in kept)
                        rounds += 1

                    got = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
                    report.append((cls, quota_n, len(got), drawn, len(got) / drawn if drawn else 0.0))
                    raw_parts.append(got)

                if verbose:
                    rep_df = pd.DataFrame(report, columns=["class", "quota", "got", "drawn", "match_rate"])
                    print(rep_df.to_string(index=False))
            else:
                # TVAE pool sampling (Minh's sample_pool_by_class)
                need = dict(quotas)
                kept = {c: [] for c in quotas}
                drawn = 0
                hits = {c: 0 for c in quotas}
                for _ in range(max_rounds * 2):
                    if all(v <= 0 for v in need.values()):
                        break
                    batch_draw = min(50_000, max(5_000, sum(need.values()) * 2))
                    pool = self.unified_model.sample(batch_draw)
                    drawn += len(pool)
                    for cls in quotas:
                        if need[cls] <= 0:
                            continue
                        if "Attack" in pool.columns:
                            m = pool[pool["Attack"] == cls]
                        else:
                            m = pool.head(need[cls]).copy()
                            m["Attack"] = cls

                        hits[cls] += len(m)
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
                if self.model_type == "tvae" and self.tvae_preprocessor is not None:
                    batch = model.sample(num_rows=int(draw_count))
                    batch = batch["t"] if isinstance(batch, dict) else batch
                    batch = self.tvae_preprocessor.from_model(batch)
                else:
                    batch = model.sample(draw_count)
                batch["Attack"] = cls_name
                raw_parts.append(batch)

        if not raw_parts:
            return pd.DataFrame()

        df_raw = pd.concat(raw_parts, ignore_index=True)

        # Apply inverse transformation for validity (bounds clipping, expm1, integer rounding)
        if self.model_type == "tvae" and self.tvae_preprocessor is not None:
            df_syn = df_raw
        elif self.preprocessor is not None:
            df_syn = self.preprocessor.inverse(df_raw, seed=self.random_state)
        else:
            df_syn = df_raw

        # Ensure Label = 1 for all synthetic attack flows
        df_syn["Label"] = 1

        elapsed = time.time() - t0
        print(f"    Generated Raw Candidate Pool: {len(df_syn):,} synthetic flows across {len(quotas)} classes in {elapsed:.2f}s.\n")
        return df_syn

    def save_weights(self, filename: Optional[Union[str, Path]] = None) -> Path:
        """Saves generator weights and preprocessor state."""
        if filename is None:
            save_path = self.weights_dir / f"{self.model_type}_best_weights.pt"
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
            "preprocessor": self.preprocessor,
            "tvae_preprocessor": self.tvae_preprocessor,
            "diagnostics": self.training_diagnostics,
        }
        with open(save_path, "wb") as f:
            pickle.dump(state, f)

        self.best_weights_path = save_path
        print(f"    [Generators Checkpoint] Saved weights to: {save_path.name}")

        # Export individual per-class SDV TVAE models alongside the unified checkpoint
        if self.model_type == "tvae" and self.models:
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
        """Loads saved generator checkpoints."""
        load_path = Path(filepath) if filepath else self.best_weights_path
        if load_path is None or not load_path.exists():
            print(f"    [Warning] Checkpoint not found at {load_path}")
            return False

        with open(load_path, "rb") as f:
            state = pickle.load(f)

        if isinstance(state, dict) and "unified_model" in state:
            self.models = state.get("models", {})
            self.unified_model = state.get("unified_model", None)
            self.preprocessor = state.get("preprocessor", None)
            self.tvae_preprocessor = state.get("tvae_preprocessor", None)
            self.training_diagnostics = state.get("diagnostics", {})
        else:
            self.models = state

        print(f"    [Generators Checkpoint] Loaded weights from: {load_path.name}")
        return True


__all__ = [
    "TabularGenerators",
    "NetFlowPreprocessor",
    "TVAEPreprocessor",
    "CTGAN_DIMS",
    "TVAE_CODE_COLS",
    "TVAE_DIMS",
    "TVAE_DEFAULT_PARAMS",
    "CAT_CANDIDATES",
]


if __name__ == "__main__":
    print("Component 2 (tabular_generators.py) loaded successfully.")

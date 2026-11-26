#!/usr/bin/env python3
"""
CTGAN Tabular Deep Generative Model (Component 2 - CTGAN)

Derived from Minh's CTGAN training and tuning pipeline (`minh/src/ctgan_pipeline.py`,
`minh/src/fix_pipeline.py`, and `minh/notebooks/07_tune_ctgan.ipynb`):
- NetFlow-aware preprocessing (fit, transform, inverse):
  - Categorical handling with TOP_K common codes and rare value reconstruction.
  - Dropping constant and quasi-constant columns, restored on inverse.
  - Log1p transform on skewed numeric features (skew > 5), expm1 on inverse.
  - Post-generation validity constraints: clipping to real max, non-negative bounds,
    and rounding integer count columns (packets, bytes).
- Batch size and pac alignment (step = pac if pac % 2 == 0 else pac * 2).
- Conditional generation with rejection sampling (sample_by_class / sample_matched):
  Enforces exact class match, filters misconditioned flows, reports shortfalls.
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
    from ctgan import CTGAN
except ImportError:
    raise ImportError("ctgan package is required. Install via `pip install ctgan`.")

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.append(str(curr_dir))

from intra_attack_resampler import resolve_paths, DERIVED_FEATURES, recompute_derived_features

from sklearn.preprocessing import QuantileTransformer

# Candidate categorical / discrete code columns in NetFlow schemas
CAT_CANDIDATES = [
    "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE",
    "MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
]
ID_COLS = ["DNS_QUERY_ID"]
TOP_K_CATEGORICAL = 30
SKEW_LIMIT = 5.0
ZERO_INFLATED_MIN_RATE = 0.05

CTGAN_DIMS = {
    "256x256": (256, 256),
    "256x256x256": (256, 256, 256),
    "512x512": (512, 512),
}


class NetFlowPreprocessor:
    """
    Learns feature transformations, bounds, and categorical encodings strictly from real training data.
    Derived from `minh/src/fix_pipeline.py`.
    """

    def __init__(self, label_col: str = "Attack", top_k: int = TOP_K_CATEGORICAL, skew_limit: float = SKEW_LIMIT, random_state: int = 42):
        self.label_col = label_col
        self.top_k = top_k
        self.skew_limit = skew_limit
        self.random_state = random_state
        self.meta: Dict[str, Any] = {}

    @staticmethod
    def _apply_coupled_reparam(df: pd.DataFrame) -> pd.DataFrame:
        """Reparameterize coupled packet-length and TTL bounds into non-negative deltas."""
        out = df.copy()
        if "MIN_TTL" in out.columns and "MAX_TTL" in out.columns:
            out["MAX_TTL"] = (pd.to_numeric(out["MAX_TTL"], errors="coerce") - pd.to_numeric(out["MIN_TTL"], errors="coerce")).clip(lower=0.0)
        if "SHORTEST_FLOW_PKT" in out.columns and "LONGEST_FLOW_PKT" in out.columns:
            long_orig = pd.to_numeric(out["LONGEST_FLOW_PKT"], errors="coerce")
            short_orig = pd.to_numeric(out["SHORTEST_FLOW_PKT"], errors="coerce")
            out["LONGEST_FLOW_PKT"] = (long_orig - short_orig).clip(lower=0.0)
            if "MAX_IP_PKT_LEN" in out.columns:
                out["MAX_IP_PKT_LEN"] = (pd.to_numeric(out["MAX_IP_PKT_LEN"], errors="coerce") - long_orig).clip(lower=0.0)
        return out

    @staticmethod
    def _invert_coupled_reparam(df: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct coupled bounds and derived-zero consistency from generated deltas."""
        out = df.copy()
        if "MIN_TTL" in out.columns and "MAX_TTL" in out.columns:
            out["MAX_TTL"] = pd.to_numeric(out["MIN_TTL"], errors="coerce") + pd.to_numeric(out["MAX_TTL"], errors="coerce").clip(lower=0.0)
        if "SHORTEST_FLOW_PKT" in out.columns and "LONGEST_FLOW_PKT" in out.columns:
            out["LONGEST_FLOW_PKT"] = pd.to_numeric(out["SHORTEST_FLOW_PKT"], errors="coerce") + pd.to_numeric(out["LONGEST_FLOW_PKT"], errors="coerce").clip(lower=0.0)
            if "MAX_IP_PKT_LEN" in out.columns:
                out["MAX_IP_PKT_LEN"] = out["LONGEST_FLOW_PKT"] + pd.to_numeric(out["MAX_IP_PKT_LEN"], errors="coerce").clip(lower=0.0)
        if "SRC_TO_DST_AVG_THROUGHPUT" in out.columns and "IN_BYTES" in out.columns:
            out.loc[pd.to_numeric(out["IN_BYTES"], errors="coerce") <= 0, "SRC_TO_DST_AVG_THROUGHPUT"] = 0.0
        if "DST_TO_SRC_AVG_THROUGHPUT" in out.columns and "OUT_BYTES" in out.columns:
            out.loc[pd.to_numeric(out["OUT_BYTES"], errors="coerce") <= 0, "DST_TO_SRC_AVG_THROUGHPUT"] = 0.0
        return out

    def fit(self, df: pd.DataFrame) -> "NetFlowPreprocessor":
        p: Dict[str, Any] = {}
        derived_present = [c for c in DERIVED_FEATURES if c in df.columns]
        clean_df = self._apply_coupled_reparam(df.replace([np.inf, -np.inf], np.nan))
        if derived_present:
            clean_df = clean_df.drop(columns=derived_present)

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
        p["derived"] = derived_present
        p["cat"] = [c for c in p["cat"] if c not in p["drop"]]
        p["num"] = [c for c in clean_df.columns if c not in p["cat"] + list(p["drop"]) + [self.label_col, "Label"]]

        # Top K categories and rare values for reconstruction
        p["top"] = {c: set(clean_df[c].value_counts().index[:self.top_k]) for c in p["cat"]}
        p["rare_vals"] = {c: clean_df.loc[~clean_df[c].isin(p["top"][c]), c].dropna().to_numpy() for c in p["cat"]}

        p["median"] = clean_df[p["num"]].median().fillna(0.0)
        p["max"] = clean_df[p["num"]].max().fillna(1.0)
        p["min"] = clean_df[p["num"]].min().fillna(0.0)
        p["is_int"] = {c: bool(np.all(np.isclose(clean_df[c].dropna() % 1, 0))) for c in p["num"]}
        p["log"] = [c for c in p["num"] if clean_df[c].min() >= 0 and clean_df[c].skew() > self.skew_limit]

        # Zero-inflated continuous columns -> hurdle flag + positive-only QuantileTransformer
        zero_inflated = []
        pos_med = {}
        num_df = clean_df[p["num"]].fillna(p["median"]).copy()
        for c in p["num"]:
            if p["min"][c] >= 0:
                zero_rate = float((num_df[c] == 0).mean())
                if ZERO_INFLATED_MIN_RATE <= zero_rate <= 0.999:
                    zero_inflated.append(c)
                    pos_vals = num_df.loc[num_df[c] > 0, c]
                    med_p = float(pos_vals.median()) if len(pos_vals) > 0 else 0.0
                    pos_med[c] = med_p
                    num_df.loc[num_df[c] == 0, c] = med_p

        p["zero_inflated"] = zero_inflated
        p["pos_med"] = pos_med
        p["zero_flags"] = [f"{c}__is_zero" for c in zero_inflated]
        p["cat_all"] = p["cat"] + p["zero_flags"]

        if len(p["num"]) > 0 and len(num_df) >= 2:
            qt = QuantileTransformer(
                n_quantiles=min(1000, len(num_df)),
                output_distribution="normal",
                subsample=min(100_000, len(num_df)),
                random_state=self.random_state,
            )
            qt.fit(num_df)
            p["qt"] = qt
        else:
            p["qt"] = None

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
        clean_df = self._apply_coupled_reparam(df.replace([np.inf, -np.inf], np.nan))
        drop_cols = [c for c in list(p["drop"]) + p.get("derived", []) if c in clean_df.columns]
        out = clean_df.drop(columns=drop_cols).copy()

        for c in p["num"]:
            if c in out.columns:
                out[c] = out[c].fillna(p["median"][c])

        for c in p.get("zero_inflated", []):
            if c in out.columns:
                is_zero = (out[c] == 0)
                out[f"{c}__is_zero"] = np.where(is_zero, "1", "0")
                out.loc[is_zero, c] = p["pos_med"][c]

        if p.get("qt") is not None and len(p["num"]) > 0:
            num_cols_present = [c for c in p["num"] if c in out.columns]
            if len(num_cols_present) == len(p["num"]):
                out[p["num"]] = p["qt"].transform(out[p["num"]])
        else:
            for c in p["num"]:
                if c in out.columns and c in p["log"]:
                    out[c] = np.log1p(out[c].clip(lower=0))

        for c in p["cat"]:
            if c in out.columns:
                out[c] = np.where(out[c].isin(p["top"][c]), self._as_str(out[c]), "other")

        return out

    def inverse(self, syn: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
        p = self.meta
        rng = np.random.default_rng(seed)
        out = syn.copy()

        if p.get("qt") is not None and len(p["num"]) > 0:
            num_cols_present = [c for c in p["num"] if c in out.columns]
            if len(num_cols_present) == len(p["num"]):
                out[p["num"]] = p["qt"].inverse_transform(out[p["num"]])
        else:
            for c in p["num"]:
                if c in out.columns and c in p["log"]:
                    out[c] = np.expm1(out[c])

        for c in p.get("zero_inflated", []):
            flag_col = f"{c}__is_zero"
            if flag_col in out.columns and c in out.columns:
                out.loc[out[flag_col].astype(str) == "1", c] = 0.0
                out = out.drop(columns=[flag_col])

        for c in p["num"]:
            if c in out.columns:
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

        out = self._invert_coupled_reparam(out)
        out = recompute_derived_features(out)

        # Align columns to original order
        target_cols = [c for c in p["order"] if c in out.columns]
        extra_cols = [c for c in out.columns if c not in target_cols]
        return out[target_cols + extra_cols]


class CTGANGenerator:
    """
    Manager for CTGAN Generative Model for Network Intrusion Data.
    Derived from Minh's training and tuning architecture.

    Features:
      - NetFlow preprocessing & post-generation validity enforcement.
      - Fixed gradient update budget calculation (default: 42,000 steps).
      - Conditional generation with rejection sampling (`sample_by_class`).
      - Weight persistence & checkpoint storage.
    """

    def __init__(
        self,
        fixed_gradient_updates: int = 42000,
        random_state: int = 42,
        device: Optional[str] = None,
        weights_dir: Optional[Union[str, Path]] = None,
    ):
        self.model_type = "ctgan"
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
            self.weights_dir = final_dir / "generators/CTGAN/weights"
        else:
            self.weights_dir = Path(weights_dir).resolve()
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Any] = {}
        self.unified_model: Optional[Any] = None
        self.preprocessor: Optional[NetFlowPreprocessor] = None
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
        Trains CTGAN on attack flows.
        Supports both:
          - Unified generator (Minh's design): One generator conditioned on 'Attack' column.
          - Per-minority-class generators: Slices model per minority class.
        """
        import torch
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        t0 = time.time()
        target_classes = [cls for cls, n_syn in syn_deficit.items() if n_syn > 0]
        print(f"\n>>> [Generators Train] Training CTGAN ({'Unified' if unified_generator else 'Per-Class'})...")
        print(f"    Target Classes:          {target_classes}")
        print(f"    Fixed Gradient Updates:  {self.fixed_gradient_updates:,} steps")
        print(f"    Compute Device:          {self.device.upper()}")

        # Initialize and fit NetFlow preprocessor
        self.preprocessor = NetFlowPreprocessor(label_col="Attack", random_state=self.random_state)
        self.preprocessor.fit(train_attacks)
        transformed_attacks = self.preprocessor.transform(train_attacks)

        diagnostics = {}
        use_cuda = (self.device != "cpu")

        if unified_generator:
            # Train a unified generator across all attack flows
            n_samples = len(transformed_attacks)
            bs, resolved_pac = self._resolve_batch_size_and_pac(n_samples, batch_size, pac)
            calc_epochs = epochs if epochs is not None else self.calculate_epochs(n_samples, bs)

            discrete_cols = [c for c in self.preprocessor.meta.get("cat_all", self.preprocessor.meta["cat"]) if c in transformed_attacks.columns]
            if "Attack" not in discrete_cols and "Attack" in transformed_attacks.columns:
                discrete_cols.append("Attack")

            print(f"    - Fitting Unified CTGAN ({n_samples:,} flows, {calc_epochs} epochs, bs={bs}, pac={resolved_pac})...")

            clean_kwargs = {k: v for k, v in model_kwargs.items() if k not in ["epochs", "batch_size", "pac", "cuda", "verbose"]}
            if "generator_dim" in clean_kwargs and isinstance(clean_kwargs["generator_dim"], str):
                clean_kwargs["generator_dim"] = CTGAN_DIMS.get(clean_kwargs["generator_dim"], (256, 256))
            if "discriminator_dim" in clean_kwargs and isinstance(clean_kwargs["discriminator_dim"], str):
                clean_kwargs["discriminator_dim"] = CTGAN_DIMS.get(clean_kwargs["discriminator_dim"], (256, 256))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = CTGAN(
                    epochs=calc_epochs,
                    batch_size=bs,
                    pac=resolved_pac,
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
            self.unified_model = None
            self.models = {}
            for cls_name in target_classes:
                df_cls = transformed_attacks[transformed_attacks["Attack"] == cls_name].copy()
                n_samples = len(df_cls)
                if n_samples == 0:
                    continue

                bs, resolved_pac = self._resolve_batch_size_and_pac(n_samples, batch_size, pac)
                cls_epochs = epochs if epochs is not None else self.calculate_epochs(n_samples, bs)

                discrete_cols = [c for c in self.preprocessor.meta.get("cat_all", self.preprocessor.meta["cat"]) if c in df_cls.columns]
                t_cls = time.time()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = CTGAN(
                        epochs=cls_epochs,
                        batch_size=bs,
                        pac=resolved_pac,
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

    def get_generator_details(self) -> Dict[str, Any]:
        """Returns detailed CTGAN generator metadata, mode, and diagnostics."""
        return {
            "library": "ctgan.CTGAN",
            "mode": "unified" if self.unified_model is not None else "per_class",
            "diagnostics": self.training_diagnostics,
        }

    def sample(
        self,
        syn_deficit: Dict[str, int],
        multiplier: int = 10,
        max_rounds: int = 10,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Samples synthetic flows according to syn_deficit using conditional rejection sampling (`sample_by_class`).
        Applies inverse transformation with validity clipping and integer rounding.
        """
        print(f">>> [Generators Sample] Drawing candidate synthetic pool ({multiplier}x deficit quota)...")
        quotas = {cls: int(multiplier * n) for cls, n in syn_deficit.items() if n > 0}
        if not quotas:
            return pd.DataFrame()

        t0 = time.time()
        raw_parts = []

        if self.unified_model is not None:
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
            # Per-class generator sampling
            for cls_name, model in self.models.items():
                if cls_name not in quotas:
                    continue
                draw_count = quotas[cls_name]
                batch = model.sample(draw_count)
                batch["Attack"] = cls_name
                raw_parts.append(batch)

        if not raw_parts:
            return pd.DataFrame()

        df_raw = pd.concat(raw_parts, ignore_index=True)

        # Apply inverse transformation for validity (bounds clipping, expm1, integer rounding)
        if self.preprocessor is not None:
            df_syn = self.preprocessor.inverse(df_raw, seed=self.random_state)
        else:
            df_syn = df_raw

        # Ensure Label = 1 for all synthetic attack flows
        df_syn["Label"] = 1

        elapsed = time.time() - t0
        print(f"    Generated Raw Candidate Pool: {len(df_syn):,} synthetic flows across {len(quotas)} classes in {elapsed:.2f}s.\n")
        return df_syn

    def save_weights(self, filename: Optional[Union[str, Path]] = None) -> Path:
        """Saves CTGAN generator weights and NetFlowPreprocessor state."""
        if filename is None:
            save_path = self.weights_dir / "ctgan_best_weights.pt"
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
            "diagnostics": self.training_diagnostics,
        }
        with open(save_path, "wb") as f:
            pickle.dump(state, f)

        self.best_weights_path = save_path
        print(f"    [Generators Checkpoint] Saved weights to: {save_path.name}")
        return save_path

    def load_weights(self, filepath: Optional[Union[str, Path]] = None) -> bool:
        """Loads saved CTGAN generator checkpoints."""
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
            self.training_diagnostics = state.get("diagnostics", {})
        else:
            self.models = state

        print(f"    [Generators Checkpoint] Loaded weights from: {load_path.name}")
        return True


__all__ = [
    "CTGANGenerator",
    "NetFlowPreprocessor",
    "CTGAN_DIMS",
    "CAT_CANDIDATES",
    "ID_COLS",
    "TOP_K_CATEGORICAL",
    "SKEW_LIMIT",
]


if __name__ == "__main__":
    print("Component 2 (ctgan_generator.py) loaded successfully.")

#!/usr/bin/env python3
"""
Intra-Attack SMOTE-NC Resampler (Component 1)

Implements Steps 1 to 6 of the NF-UNSW-NB15-v3 preprocessing pipeline:
- Step 1: Pruning shortcut learning features (IPs, Ports, Timestamps) & deduplication.
- Step 2: Stratified 80/20 Holdout partitioning (20% pristine Holdout).
- Step 3: 70/10/20 Train/Val/Test partitioning of the 80% non-holdout dev set.
- Step 4: Disaggregating benign and attack flows for independent preprocessing.
- Step 5: Calculating Synthetic Deficit for minority attack classes (Research Problem 1):
    n_syn = max(0, t_c - n_c)
    t_c = max(n_c, min(t_max, y_max * n_c))
    Default: t_max = 10,000, y_max = 25
- Step 6: Scaled continuous normalization and SMOTE-NC oversampling.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Mitigate OpenMP conflicts across libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTENC


# ============================================================================
# Constants & Feature Group Definitions
# ============================================================================
SHORTCUT_COLS = [
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
]

CATEGORICAL_FEATURES = [
    "PROTOCOL",
    "L7_PROTO",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
]


# ============================================================================
# Directory Resolution Helper
# ============================================================================
def resolve_paths() -> Tuple[Path, Path, Path]:
    """
    Resolves repository root, raw dataset location, and export directories.
    """
    current_script_dir = Path(__file__).resolve().parent
    repo_roots = [
        current_script_dir.parent,
        current_script_dir.parents[1] if len(current_script_dir.parents) > 1 else current_script_dir,
        current_script_dir.parents[2] if len(current_script_dir.parents) > 2 else current_script_dir,
        current_script_dir,
        Path.cwd(),
        Path("/run/media/thanh/Data/Project A/duc_thanh"),
        Path("/run/media/thanh/Ventoy/Project A/duc_thanh"),
    ]

    base_dir = None
    for r in repo_roots:
        if (r / "datasets/NetFlow").exists():
            base_dir = r.resolve()
            break

    if base_dir is None:
        raise FileNotFoundError("Could not resolve workspace base directory with datasets/NetFlow!")

    raw_csv_path = base_dir / "datasets/NetFlow/NF-UNSW-NB15-v3/f7546561558c07c5_NFV3DATA-A11964_A11964/data/NF-UNSW-NB15-v3.csv"
    if not raw_csv_path.exists():
        found = list(base_dir.glob("**/NF-UNSW-NB15-v3.csv"))
        if not found:
            raise FileNotFoundError(f"Raw NF-UNSW-NB15-v3.csv not found under {base_dir}!")
        raw_csv_path = found[0]

    final_datasets_dir = base_dir / "datasets/final datasets"
    final_datasets_dir.mkdir(parents=True, exist_ok=True)

    return base_dir, raw_csv_path, final_datasets_dir


# ============================================================================
# Component 1 Class: IntraAttackResampler
# ============================================================================
class IntraAttackResampler:
    """
    Component 1: Intra-Attack SMOTE-NC Resampler.

    Implements Steps 4 to 6 of the NF-UNSW-NB15-v3 preprocessing pipeline:
      - Step 4: Disaggregate benign and attack flows.
      - Step 5: Synthetic Deficit calculation for minority attack classes.
      - Step 6: Continuous feature scaling and SMOTE-NC oversampling.
    """

    def __init__(
        self,
        t_max: int = 10000,
        y_max: int = 25,
        random_state: int = 42,
        scaler_type: str = "minmax",
        final_dir: Optional[Union[str, Path]] = None,
    ):
        self.t_max = t_max
        self.y_max = y_max
        self.random_state = random_state
        self.scaler_type = scaler_type.lower()

        # Initialize scaler
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        # Resolve paths if not explicitly provided
        if final_dir is not None:
            self.final_dir = Path(final_dir).resolve()
            self.base_dir = self.final_dir.parent.parent
        else:
            self.base_dir, _, self.final_dir = resolve_paths()

        # Destination export directories
        self.ids_multi_dir = self.final_dir / "IDS-multiclass-classification"
        self.gen_tvae_dir = self.final_dir / "generators/TVAE"
        self.gen_ctgan_dir = self.final_dir / "generators/CTGAN"
        self.dev_splits_dir = self.final_dir / "dev_splits"

        for d in [self.ids_multi_dir, self.gen_tvae_dir, self.gen_ctgan_dir, self.dev_splits_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Internal DuckDB connection for high-speed I/O
        self.con = duckdb.connect()
        self.con.execute("PRAGMA threads=4;")

        # State storage
        self.target_counts: Dict[str, int] = {}
        self.syn_deficit: Dict[str, int] = {}
        self.df_deficit_summary: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.cont_cols: List[str] = []

    def disaggregate_flows(self, train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Step 4: Disaggregates benign and attack flows from the training set.
        """
        print(">>> [Step 4/6] Disaggregating Benign and Attack Flows (Train Set)...")
        t0 = time.time()

        train_benign = train_df[train_df["Label"] == 0].copy().reset_index(drop=True)
        train_attacks = train_df[train_df["Label"] == 1].copy().reset_index(drop=True)

        print(f"    Train Benign Flows: {len(train_benign):,} ({len(train_benign) / len(train_df) * 100:.2f}%)")
        print(f"    Train Attack Flows: {len(train_attacks):,} ({len(train_attacks) / len(train_df) * 100:.2f}%)")
        print("    Class distribution of original train attacks:")
        for atk, cnt in train_attacks["Attack"].value_counts().items():
            print(f"      - {atk:<16}: {cnt:>6,} flows ({cnt / len(train_attacks) * 100:>5.2f}%)")

        print(f"    Step 4 Finished in {time.time() - t0:.2f}s\n")
        return train_benign, train_attacks

    def calculate_synthetic_deficit(
        self, train_attacks: pd.DataFrame
    ) -> Tuple[Dict[str, int], Dict[str, int], pd.DataFrame]:
        """
        Step 5: Calculates Synthetic Deficit for each attack class (Research Problem 1).
          n_syn = max(0, t_c - n_c)
          t_c = max(n_c, min(t_max, y_max * n_c))
        """
        print(">>> [Step 5/6] Calculating Synthetic Deficit for Attack Classes (Research Problem 1)...")
        t0 = time.time()

        attack_counts = train_attacks["Attack"].value_counts().to_dict()
        target_counts = {}
        syn_deficit = {}
        summary_rows = []

        for cls, n_c in attack_counts.items():
            scaled_c = int(self.y_max * n_c)
            t_c = max(n_c, min(self.t_max, scaled_c))
            n_syn = max(0, t_c - n_c)

            target_counts[cls] = t_c
            syn_deficit[cls] = n_syn

            multiplier = (t_c / n_c) if n_c > 0 else 0.0
            summary_rows.append({
                "Attack Class": cls,
                "Original (n_c)": n_c,
                "y_max * n_c": scaled_c,
                "Target (t_c)": t_c,
                "Synthetic Deficit (n_syn)": n_syn,
                "Multiplier": f"{multiplier:.2f}x",
                "Oversampled": "Yes" if n_syn > 0 else "No (Capped / Majority)",
            })

        df_summary = pd.DataFrame(summary_rows)
        self.target_counts = target_counts
        self.syn_deficit = syn_deficit
        self.df_deficit_summary = df_summary

        print("\n" + df_summary.to_string(index=False) + "\n")

        deficit_json_path = self.final_dir / "synthetic_deficit.json"
        deficit_export_payload = {
            "parameters": {"t_max": self.t_max, "y_max": self.y_max, "random_state": self.random_state},
            "original_counts": attack_counts,
            "target_counts": target_counts,
            "synthetic_deficit": syn_deficit,
        }
        with open(deficit_json_path, "w") as f:
            json.dump(deficit_export_payload, f, indent=2)
        print(f"    Exported Deficit Metadata: {deficit_json_path.name}")
        print(f"    Step 5 Finished in {time.time() - t0:.2f}s\n")

        return target_counts, syn_deficit, df_summary

    def resample_attacks(
        self,
        train_attacks: pd.DataFrame,
        dict_tc: Dict[str, int],
        syn_deficit: Dict[str, int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Step 6: Normalizes continuous samples first, passes dict_tc to SMOTE-NC,
        and returns both inverse-transformed (original scale) and normalized resampled attacks.
        No further post-hoc heuristic filtering is applied.
        """
        print(">>> [Step 6/6] Normalizing Samples & Resampling Attack Flows via SMOTE-NC...")
        t0 = time.time()

        feature_cols = [c for c in train_attacks.columns if c not in ["Label", "Attack"]]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
        cont_cols = [c for c in feature_cols if c not in cat_cols]
        cat_indices = [feature_cols.index(c) for c in cat_cols]

        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols

        print(f"    Feature Dimensionality:    {len(feature_cols)} total ({len(cont_cols)} continuous, {len(cat_cols)} nominal)")
        print(f"    Categorical Features (11): {cat_cols}")

        X_attack = train_attacks[feature_cols].copy()
        y_attack = train_attacks["Attack"].copy()

        # Sanitize continuous features
        for col in cont_cols:
            X_attack[col] = pd.to_numeric(X_attack[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            med = X_attack[col].median()
            X_attack[col] = X_attack[col].fillna(0.0 if np.isnan(med) else med)

        # Impute missing values in categorical features
        for col in cat_cols:
            if X_attack[col].isnull().any():
                mode_val = X_attack[col].mode().iloc[0] if not X_attack[col].mode().empty else 0
                X_attack[col] = X_attack[col].fillna(mode_val)

        # Step 6a: Normalize continuous features
        X_attack_norm = X_attack.copy()
        X_attack_norm[cont_cols] = self.scaler.fit_transform(X_attack[cont_cols])
        print(f"    Continuous Feature Scaling: {self.scaler.__class__.__name__} fitted strictly on training attacks")

        # Step 6b: Target sample counts per attack class
        attack_counts = y_attack.value_counts().to_dict()
        print(f"    Target t_c values for each attack class (SMOTE-NC sampling strategy):")
        for cls, t_c in dict_tc.items():
            n_c = attack_counts.get(cls, 0)
            n_syn = syn_deficit.get(cls, 0)
            status = f"(+{n_syn:,} synthetic flows)" if n_syn > 0 else "(no oversampling required)"
            print(f"      - {cls:<16}: {n_c:>6,} -> {t_c:>6,} {status}")

        oversampled_classes = [cls for cls, n_syn in syn_deficit.items() if n_syn > 0]
        min_minority_samples = min(attack_counts[cls] for cls in oversampled_classes) if oversampled_classes else 5
        k_neighbors = min(5, max(1, min_minority_samples - 1))
        print(f"    Configured SMOTE-NC k_neighbors = {k_neighbors} (min minority class size = {min_minority_samples})")

        # Step 6c: Resample with SMOTE-NC
        smotenc = SMOTENC(
            categorical_features=cat_indices,
            sampling_strategy=dict_tc,
            k_neighbors=k_neighbors,
            random_state=self.random_state,
        )
        X_res_norm, y_res = smotenc.fit_resample(X_attack_norm, y_attack)

        # Normalized DataFrame
        df_attacks_smotenc_norm = pd.DataFrame(X_res_norm, columns=feature_cols)
        df_attacks_smotenc_norm["Attack"] = y_res.values
        df_attacks_smotenc_norm["Label"] = 1

        # Original Scale DataFrame (inverse transformed)
        df_attacks_smotenc_orig = pd.DataFrame(X_res_norm, columns=feature_cols)
        df_attacks_smotenc_orig[cont_cols] = self.scaler.inverse_transform(df_attacks_smotenc_orig[cont_cols])

        for col in cat_cols:
            if col != "L7_PROTO":
                df_attacks_smotenc_orig[col] = df_attacks_smotenc_orig[col].round().astype(int)

        df_attacks_smotenc_orig["Attack"] = y_res.values
        df_attacks_smotenc_orig["Label"] = 1

        print(f"    SMOTE-NC Complete: {len(train_attacks):,} -> {len(df_attacks_smotenc_orig):,} attack flows")
        print(f"    Generated Synthetic Flows: {len(df_attacks_smotenc_orig) - len(train_attacks):,}")

        print(f"    Step 6 Finished in {time.time() - t0:.2f}s\n")
        return df_attacks_smotenc_orig, df_attacks_smotenc_norm

    def fit_resample(
        self,
        train_df: pd.DataFrame,
        export: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes Steps 4, 5, and 6 sequentially on the training set:
          1. Step 4: Disaggregate benign and attack flows.
          2. Step 5: Calculate synthetic deficit and dict of t_c values.
          3. Step 6: Normalize continuous features & resample attacks via SMOTE-NC.
          4. Combine benign flows with oversampled attacks to produce complete resampled training data.
        """
        start_time = time.time()
        print("=" * 85)
        print("Executing IntraAttackResampler (Steps 4 to 6)")
        print("=" * 85)

        train_benign, train_attacks = self.disaggregate_flows(train_df)
        dict_tc, syn_deficit, df_deficit_summary = self.calculate_synthetic_deficit(train_attacks)
        df_attacks_smotenc_orig, df_attacks_smotenc_norm = self.resample_attacks(
            train_attacks=train_attacks,
            dict_tc=dict_tc,
            syn_deficit=syn_deficit,
        )

        df_train_resampled = pd.concat([train_benign, df_attacks_smotenc_orig], ignore_index=True)

        if export:
            self._export_resampled_datasets(
                df_attacks_orig=df_attacks_smotenc_orig,
                df_attacks_norm=df_attacks_smotenc_norm,
                df_train_resampled=df_train_resampled,
            )

        elapsed = time.time() - start_time
        print(f"IntraAttackResampler (Steps 4-6) completed successfully in {elapsed:.2f}s!\n")

        return {
            "train_benign": train_benign,
            "train_attacks_raw": train_attacks,
            "df_attacks_smotenc": df_attacks_smotenc_orig,
            "df_attacks_smotenc_norm": df_attacks_smotenc_norm,
            "df_train_resampled": df_train_resampled,
            "dict_tc": dict_tc,
            "syn_deficit": syn_deficit,
            "deficit_summary": df_deficit_summary,
            "scaler": self.scaler,
        }

    def _export_resampled_datasets(
        self,
        df_attacks_orig: pd.DataFrame,
        df_attacks_norm: pd.DataFrame,
        df_train_resampled: pd.DataFrame,
    ):
        """Helper to export resampled datasets to target destination directories."""
        print(">>> Exporting Resampled Datasets...")
        export_targets = [
            self.ids_multi_dir / "unsw_nb15_attacks_smotenc.csv",
            self.gen_tvae_dir / "unsw_nb15_attacks_smotenc.csv",
            self.gen_ctgan_dir / "unsw_nb15_attacks_smotenc.csv",
        ]
        for dest in export_targets:
            dest_str = str(dest).replace("\\", "/")
            self.con.execute(f"COPY df_attacks_orig TO '{dest_str}' (HEADER, DELIMITER ',');")
            print(f"    Exported Attack Dataset:   {dest.name} ({dest.stat().st_size / (1024**2):.2f} MB)")

        dest_norm = self.ids_multi_dir / "unsw_nb15_attacks_smotenc_normalized.csv"
        dest_norm_str = str(dest_norm).replace("\\", "/")
        self.con.execute(f"COPY df_attacks_norm TO '{dest_norm_str}' (HEADER, DELIMITER ',');")
        print(f"    Exported Normalized Data: {dest_norm.name} ({dest_norm.stat().st_size / (1024**2):.2f} MB)")

        train_resampled_path = self.ids_multi_dir / "unsw_nb15_train_smotenc_resampled.csv"
        train_resampled_str = str(train_resampled_path).replace("\\", "/")
        self.con.execute(f"COPY df_train_resampled TO '{train_resampled_str}' (HEADER, DELIMITER ',');")
        print(f"    Exported Resampled Train:  {train_resampled_path.name} "
              f"({len(df_train_resampled):,} flows, {train_resampled_path.stat().st_size / (1024**2):.2f} MB)\n")


# ============================================================================
# Full Pipeline Execution (Steps 1 to 6) for Standalone CLI
# ============================================================================
def run_full_pipeline(
    t_max: int = 10000,
    y_max: int = 25,
    random_state: int = 42,
    scaler_type: str = "minmax",
    save_splits: bool = True,
):
    """
    Executes the entire end-to-end pipeline (Steps 1 to 6) in headless CLI mode.
    """
    start_total_time = time.time()
    base_dir, raw_csv_path, final_dir = resolve_paths()

    print("=" * 85)
    print("NF-UNSW-NB15-v3 Pipeline: Steps 1 to 6 (Standalone Execution)")
    print("=" * 85)

    con = duckdb.connect()
    con.execute("PRAGMA threads=6;")
    con.execute("PRAGMA memory_limit='10GB';")

    # Step 1: Shortcut Feature Pruning & Deduplication
    print(">>> [Step 1/6] Pruning Shortcut Learning Features & Deduplicating Dataset...")
    exclude_sql = f"EXCLUDE ({', '.join(SHORTCUT_COLS)})"
    raw_path_str = str(raw_csv_path).replace("\\", "/")
    con.execute(f"""
        CREATE OR REPLACE TABLE unsw_clean AS
        SELECT DISTINCT * {exclude_sql}
        FROM read_csv_auto('{raw_path_str}');
    """)
    df_clean = con.execute("SELECT * FROM unsw_clean;").df()

    # Step 2: Stratified Holdout Partitioning (80/20)
    print(">>> [Step 2/6] Extracting Pristine 20% Stratified Holdout Test Set...")
    dev_df, holdout_df = train_test_split(
        df_clean, test_size=0.20, random_state=random_state, stratify=df_clean["Attack"]
    )
    holdout_path = final_dir / "holdout_set/NF-UNSW-NB15-v3.csv"
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_path_str = str(holdout_path).replace("\\", "/")
    con.execute(f"COPY holdout_df TO '{holdout_path_str}' (HEADER, DELIMITER ',');")

    # Step 3: Train / Val / Test Split of 80% Dev Set (70/10/20)
    print(">>> [Step 3/6] Splitting 80% Non-Holdout Dev Set into Train/Val/Test (70/10/20)...")
    train_val_df, dev_test_df = train_test_split(
        dev_df, test_size=0.20, random_state=random_state, stratify=dev_df["Attack"]
    )
    train_df, dev_val_df = train_test_split(
        train_val_df, test_size=0.125, random_state=random_state, stratify=train_val_df["Attack"]
    )
    train_df = train_df.reset_index(drop=True)

    if save_splits:
        dev_splits_dir = final_dir / "dev_splits"
        dev_splits_dir.mkdir(parents=True, exist_ok=True)
        for name, split_df in [("train", train_df), ("val", dev_val_df), ("test", dev_test_df)]:
            p = dev_splits_dir / f"NF-UNSW-NB15-v3_{name}.csv"
            p_str = str(p).replace("\\", "/")
            con.execute(f"COPY split_df TO '{p_str}' (HEADER, DELIMITER ',');")

    # Steps 4 to 6 via IntraAttackResampler
    resampler = IntraAttackResampler(
        t_max=t_max,
        y_max=y_max,
        random_state=random_state,
        scaler_type=scaler_type,
        final_dir=final_dir,
    )
    resampler.fit_resample(train_df=train_df, export=True)

    total_elapsed = time.time() - start_total_time
    print(f"End-to-End Pipeline completed in {total_elapsed:.2f}s ({total_elapsed / 60:.2f} mins)!")


# ============================================================================
# CLI Entry Point
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="NF-UNSW-NB15-v3 Attack Resampling Pipeline (Steps 4-6 via Class, or Full Pipeline)"
    )
    parser.add_argument("--train-csv", type=str, default=None,
                        help="Path to pre-split train.csv. If provided, executes Steps 4-6 on this data directly.")
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Run end-to-end pipeline (Steps 1 to 6) from raw NetFlow CSV.")
    parser.add_argument("--t-max", type=int, default=10000, help="Maximum target sample ceiling (default: 10,000)")
    parser.add_argument("--y-max", type=int, default=25, help="Multiplication factor parameter (default: 25)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state seed (default: 42)")
    parser.add_argument("--scaler", type=str, choices=["minmax", "standard"], default="minmax",
                        help="Continuous feature normalization scaler (default: minmax)")
    parser.add_argument("--no-save-splits", action="store_true", help="Do not save train/val/test split CSVs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.train_csv:
        print(f"Loading provided train set: {args.train_csv}")
        df_train_in = pd.read_csv(args.train_csv)
        resampler_inst = IntraAttackResampler(
            t_max=args.t_max,
            y_max=args.y_max,
            random_state=args.random_state,
            scaler_type=args.scaler,
        )
        resampler_inst.fit_resample(train_df=df_train_in, export=True)
    elif args.full_pipeline:
        run_full_pipeline(
            t_max=args.t_max,
            y_max=args.y_max,
            random_state=args.random_state,
            scaler_type=args.scaler,
            save_splits=not args.no_save_splits,
        )
    else:
        print("IntraAttackResampler module loaded. Use --train-csv or --full-pipeline, or import as a class in notebooks.")


__all__ = [
    "IntraAttackResampler",
    "SHORTCUT_COLS",
    "CATEGORICAL_FEATURES",
    "resolve_paths",
    "run_full_pipeline",
]

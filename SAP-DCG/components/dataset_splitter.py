#!/usr/bin/env python3
"""
Dataset Splitting & Disaggregation Module.

Provides independent, modular dataset splitting capabilities:
  1. Splitting Benign and Attack flows (`split_benign_attack`).
  2. Stratified Partitioning with a single reproducible seed (`DatasetSplitter`):
     - Step 1: Shortcut feature pruning (IPs, Ports, Timestamps) & deduplication.
     - Step 2: 80/20 Stratified Holdout extraction (pristine evaluation benchmark).
     - Step 3: 70/10/20 Train/Val/Test partition of the 80% development set.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
from sklearn.model_selection import train_test_split

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.append(str(curr_dir))

from intra_attack_resampler import SHORTCUT_COLS, resolve_paths


# ============================================================================
# 1. Benign vs Attack Disaggregation Function
# ============================================================================
def split_benign_attack(
    df: pd.DataFrame,
    label_col: str = "Label",
    attack_col: str = "Attack",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits any dataset or partition into Benign and Attack subsets.

    Args:
        df: Input DataFrame.
        label_col: Column name containing binary indicator (0 = Benign, 1 = Attack).
        attack_col: Column name containing multiclass attack category.
        verbose: Whether to log flow counts and percentages.

    Returns:
        (benign_df, attack_df)
    """
    if label_col in df.columns:
        benign_mask = df[label_col] == 0
    elif attack_col in df.columns:
        benign_mask = df[attack_col].astype(str).str.lower() == "benign"
    else:
        raise ValueError(f"Neither '{label_col}' nor '{attack_col}' found in DataFrame columns.")

    benign_df = df[benign_mask].copy().reset_index(drop=True)
    attack_df = df[~benign_mask].copy().reset_index(drop=True)

    if verbose:
        total = len(df)
        b_cnt = len(benign_df)
        a_cnt = len(attack_df)
        b_pct = (b_cnt / total * 100) if total > 0 else 0.0
        a_pct = (a_cnt / total * 100) if total > 0 else 0.0
        print(f">>> [DatasetSplitter] Disaggregated Dataset ({total:,} flows):")
        print(f"      - Benign Flows: {b_cnt:,} ({b_pct:.2f}%)")
        print(f"      - Attack Flows: {a_cnt:,} ({a_pct:.2f}%)")
        if attack_col in attack_df.columns:
            print(f"      - Attack Classes Present: {attack_df[attack_col].nunique()} distinct classes")

    return benign_df, attack_df


# ============================================================================
# 2. Independent Dataset Splitter (Single Seed)
# ============================================================================
class DatasetSplitter:
    """
    Independent Dataset Partitioning Engine.
    Operates with a single reproducible seed for deterministic, stratified dataset splitting.
    """

    def __init__(
        self,
        random_state: int = 42,
        shortcut_cols: Optional[List[str]] = None,
        stratify_col: str = "Attack",
    ):
        self.random_state = random_state
        self.shortcut_cols = shortcut_cols if shortcut_cols is not None else SHORTCUT_COLS
        self.stratify_col = stratify_col

    def clean_dataset(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Prunes shortcut features (IPs, Ports, Timestamps) and drops exact duplicates.
        """
        t0 = time.time()
        initial_rows = len(df)
        cols_to_drop = [c for c in self.shortcut_cols if c in df.columns]

        df_clean = df.drop(columns=cols_to_drop)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

        if verbose:
            dropped_rows = initial_rows - len(df_clean)
            print(f">>> [DatasetSplitter] Clean & Deduplicate: {initial_rows:,} -> {len(df_clean):,} rows "
                  f"(dropped {len(cols_to_drop)} shortcut columns, {dropped_rows:,} duplicates in {time.time() - t0:.2f}s)")
        return df_clean

    def split_train_val_test_holdout(
        self,
        df: pd.DataFrame,
        holdout_ratio: float = 0.20,
        dev_test_ratio: float = 0.20,
        dev_val_ratio: float = 0.125,
        verbose: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Executes stratified holdout and development splits with a single seed:
          1. 80% Development / 20% Pristine Holdout (stratified by Attack).
          2. Dev partition split: 70% Train, 10% Val, 20% Test (stratified by Attack).

        Returns:
            Dictionary containing {"train": df_train, "val": df_val, "test": df_test, "holdout": df_holdout}
        """
        strat_col = self.stratify_col if self.stratify_col in df.columns else None

        # Step 1: Holdout Split (80% Dev, 20% Holdout)
        dev_df, holdout_df = train_test_split(
            df,
            test_size=holdout_ratio,
            random_state=self.random_state,
            stratify=df[strat_col] if strat_col else None,
        )

        # Step 2: Dev Split into (Train + Val) and Test
        train_val_df, test_df = train_test_split(
            dev_df,
            test_size=dev_test_ratio,
            random_state=self.random_state,
            stratify=dev_df[strat_col] if strat_col else None,
        )

        # Step 3: Train + Val Split into Train and Val (0.125 of 80% = 10% of dev)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=dev_val_ratio,
            random_state=self.random_state,
            stratify=train_val_df[strat_col] if strat_col else None,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        holdout_df = holdout_df.reset_index(drop=True)

        if verbose:
            total = len(df)
            print("=" * 80)
            print(f">>> [DatasetSplitter] Stratified Partitions Created (Seed={self.random_state}):")
            print("=" * 80)
            print(f"      - Total Flows:    {total:,}")
            print(f"      - Holdout Set:    {len(holdout_df):,} ({len(holdout_df) / total * 100:.2f}%)")
            print(f"      - Train Set:      {len(train_df):,} ({len(train_df) / total * 100:.2f}%)")
            print(f"      - Validation Set: {len(val_df):,} ({len(val_df) / total * 100:.2f}%)")
            print(f"      - Test Set:       {len(test_df):,} ({len(test_df) / total * 100:.2f}%)")
            print("=" * 80 + "\n")

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "holdout": holdout_df,
        }

    def save_splits(
        self,
        splits: Dict[str, pd.DataFrame],
        output_dir: Union[str, Path],
        prefix: str = "NF-UNSW-NB15-v3",
    ):
        """
        Exports split partitions to designated dev_splits and holdout_set folders.
        """
        out_path = Path(output_dir).resolve()
        dev_splits_dir = out_path / "dev_splits"
        holdout_dir = out_path / "holdout_set"
        dev_splits_dir.mkdir(parents=True, exist_ok=True)
        holdout_dir.mkdir(parents=True, exist_ok=True)

        if "holdout" in splits:
            h_path = holdout_dir / f"{prefix}.csv"
            splits["holdout"].to_csv(h_path, index=False)
            print(f"  [DatasetSplitter Export] Saved: {h_path.relative_to(out_path.parent)} ({len(splits['holdout']):,} flows)")

        for name in ["train", "val", "test"]:
            if name in splits:
                p = dev_splits_dir / f"{prefix}_{name}.csv"
                splits[name].to_csv(p, index=False)
                print(f"  [DatasetSplitter Export] Saved: {p.relative_to(out_path.parent)} ({len(splits[name]):,} flows)")


__all__ = [
    "split_benign_attack",
    "DatasetSplitter",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Dataset Splitting Module (Single Seed)")
    parser.add_argument("--input-csv", type=str, default=None, help="Path to input raw CSV file")
    parser.add_argument("--output-dir", type=str, default=None, help="Destination directory to save splits")
    parser.add_argument("--random-state", type=int, default=42, help="Reproducible random seed (default: 42)")
    args = parser.parse_args()

    base_dir, raw_csv, final_dir = resolve_paths()
    src = Path(args.input_csv) if args.input_csv else raw_csv
    dst = Path(args.output_dir) if args.output_dir else final_dir

    print(f"Loading dataset from: {src}")
    df_raw = pd.read_csv(src)

    splitter = DatasetSplitter(random_state=args.random_state)
    df_cleaned = splitter.clean_dataset(df_raw)
    splits = splitter.split_train_val_test_holdout(df_cleaned)
    splitter.save_splits(splits, output_dir=dst)
    print("Splitting completed successfully.")


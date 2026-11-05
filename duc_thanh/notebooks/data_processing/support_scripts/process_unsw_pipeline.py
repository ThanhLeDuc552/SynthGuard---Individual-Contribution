#!/usr/bin/env python3
"""
NF-UNSW-NB15-v3 Processing, Resampling, and Cross-Validation Pipeline.

Pipeline steps:
1. Pruning non-generalizable shortcut learning features (IPs, Ports, Timestamps)
   and deduplicating the dataset across all remaining features.
2. Stratified sampling (80/20) to extract a pristine Holdout test set (20%).
   All downstream steps strictly use the remaining non-holdout dev/training set (80%).
3. Disaggregating benign and attack flows for independent preprocessing.
4. Oversampling minority attack classes using SMOTE up to 10,000 samples.
5. Removing duplicates and technically invalid synthetic samples from the oversampled
   attack set -> generating the final dataset for TVAE, CTGAN, and IDS multiclass models.
6. Combining benign with natural original attack samples for binary IDS evaluation,
   and constructing 5-fold stratified train/test cross-validation splits (80/20).
7. Exporting and updating all final datasets under 'datasets/final datasets/'.
"""

import os
import sys
import time
from pathlib import Path

# Mitigate OpenMP conflicts across libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE


# ============================================================================
# Directory Resolution
# ============================================================================
def resolve_paths():
    current_script_dir = Path(__file__).resolve().parent
    repo_roots = [
        current_script_dir.parents[2],
        current_script_dir.parents[1],
        Path.cwd(),
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
        # Fallback search
        found = list(base_dir.glob("**/NF-UNSW-NB15-v3.csv"))
        if not found:
            raise FileNotFoundError(f"Raw NF-UNSW-NB15-v3.csv not found under {base_dir}!")
        raw_csv_path = found[0]

    final_datasets_dir = base_dir / "datasets/final datasets"
    final_datasets_dir.mkdir(parents=True, exist_ok=True)

    return base_dir, raw_csv_path, final_datasets_dir


# ============================================================================
# Main Processing Pipeline
# ============================================================================
def run_pipeline():
    start_total_time = time.time()
    base_dir, raw_csv_path, final_dir = resolve_paths()

    print("=" * 80)
    print("NF-UNSW-NB15-v3 Preprocessing, SMOTE Resampling & K-Fold Splitting Pipeline")
    print("=" * 80)
    print(f"Base Directory:          {base_dir}")
    print(f"Raw Input Dataset:       {raw_csv_path}")
    print(f"Final Datasets Directory:{final_dir}\n")

    # Destination directories
    holdout_dir = final_dir / "holdout_set"
    ids_multi_dir = final_dir / "IDS-multiclass-classification"
    gen_tvae_dir = final_dir / "generators/TVAE"
    gen_ctgan_dir = final_dir / "generators/CTGAN"
    ids_bin_dir = final_dir / "IDS-binary-classification_MIA"
    ids_bin_train_dir = ids_bin_dir / "train"
    ids_bin_test_dir = ids_bin_dir / "test"

    for d in [holdout_dir, ids_multi_dir, gen_tvae_dir, gen_ctgan_dir, ids_bin_train_dir, ids_bin_test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # DuckDB engine setup
    con = duckdb.connect()
    con.execute("PRAGMA threads=6;")
    con.execute("PRAGMA memory_limit='10GB';")
    con.execute("PRAGMA preserve_insertion_order=false;")

    # ------------------------------------------------------------------------
    # Step 1: Feature Pruning & Deduplication
    # ------------------------------------------------------------------------
    print(">>> [Step 1/7] Deduplicating & Pruning Shortcut Features...")
    t0 = time.time()
    shortcut_cols = [
        "IPV4_SRC_ADDR", "L4_SRC_PORT",
        "IPV4_DST_ADDR", "L4_DST_PORT",
        "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"
    ]
    exclude_sql = f"EXCLUDE ({', '.join(shortcut_cols)})"
    raw_path_str = str(raw_csv_path).replace("\\", "/")

    # Count raw records
    raw_total_count = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{raw_path_str}');").fetchone()[0]

    # Deduplicate via DuckDB
    con.execute(f"""
        CREATE OR REPLACE TABLE unsw_clean AS
        SELECT DISTINCT * {exclude_sql}
        FROM read_csv_auto('{raw_path_str}');
    """)
    dedup_total_count = con.execute("SELECT COUNT(*) FROM unsw_clean;").fetchone()[0]
    benign_dedup_count = con.execute("SELECT COUNT(*) FROM unsw_clean WHERE Label = 0;").fetchone()[0]
    attack_dedup_count = con.execute("SELECT COUNT(*) FROM unsw_clean WHERE Label = 1;").fetchone()[0]

    print(f"    Raw Flows:          {raw_total_count:,}")
    print(f"    Deduplicated Flows: {dedup_total_count:,} (Removed {raw_total_count - dedup_total_count:,} duplicates, "
          f"{((raw_total_count - dedup_total_count) / raw_total_count) * 100:.2f}%)")
    print(f"    Benign:             {benign_dedup_count:,} ({benign_dedup_count / dedup_total_count * 100:.2f}%)")
    print(f"    Attack:             {attack_dedup_count:,} ({attack_dedup_count / dedup_total_count * 100:.2f}%)")
    print(f"    Pruned Features (6):{shortcut_cols}")
    print(f"    Step 1 Finished in {time.time() - t0:.2f}s\n")

    # Fetch into memory
    df_clean = con.execute("SELECT * FROM unsw_clean;").df()
    feature_cols = [c for c in df_clean.columns if c not in ["Label", "Attack"]]
    print(f"    Remaining Features ({len(feature_cols)}): {feature_cols[:6]} ... {feature_cols[-4:]}")

    # ------------------------------------------------------------------------
    # Step 2: Stratified Sampling for Holdout Set (20%)
    # ------------------------------------------------------------------------
    print(">>> [Step 2/7] Generating 20% Stratified Holdout Partition...")
    t0 = time.time()
    train_val_df, holdout_df = train_test_split(
        df_clean,
        test_size=0.20,
        random_state=42,
        stratify=df_clean["Attack"]
    )
    train_val_df = train_val_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)

    print(f"    Non-Holdout Dev Set (80%): {len(train_val_df):,} flows")
    print(f"    Holdout Test Set    (20%): {len(holdout_df):,} flows")

    holdout_path = holdout_dir / "NF-UNSW-NB15-v3.csv"
    holdout_path_str = str(holdout_path).replace("\\", "/")
    con.execute(f"COPY holdout_df TO '{holdout_path_str}' (HEADER, DELIMITER ',');")
    print(f"    Materialized Holdout:      {holdout_path.name} ({holdout_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"    Step 2 Finished in {time.time() - t0:.2f}s\n")

    # ------------------------------------------------------------------------
    # Step 3: Split Benign and Attack Flows (from Non-Holdout Set)
    # ------------------------------------------------------------------------
    print(">>> [Step 3/7] Splitting Benign and Attack Subsets for Independent Preprocessing...")
    t0 = time.time()
    df_train_benign = train_val_df[train_val_df["Label"] == 0].copy().reset_index(drop=True)
    df_train_attacks = train_val_df[train_val_df["Label"] == 1].copy().reset_index(drop=True)

    print(f"    Non-Holdout Benign Flows: {len(df_train_benign):,}")
    print(f"    Non-Holdout Attack Flows: {len(df_train_attacks):,}")
    print("    Class distribution of original attacks:")
    for atk, cnt in df_train_attacks["Attack"].value_counts().items():
        print(f"      - {atk:<16}: {cnt:>6,} flows ({cnt / len(df_train_attacks) * 100:>5.2f}%)")
    print(f"    Step 3 Finished in {time.time() - t0:.2f}s\n")

    # ------------------------------------------------------------------------
    # Step 4: Oversample Attack Classes with SMOTE up to 10k Samples
    # ------------------------------------------------------------------------
    print(">>> [Step 4/7] Oversampling Attack Classes with SMOTE up to 10k Samples...")
    t0 = time.time()
    target_count = 10000

    # Sanitize continuous features for SMOTE (impute NaNs / infinities with median)
    X_attack = df_train_attacks[feature_cols].copy()
    for col in feature_cols:
        X_attack[col] = pd.to_numeric(X_attack[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = X_attack[col].median()
        X_attack[col] = X_attack[col].fillna(0.0 if np.isnan(med) else med)

    y_attack = df_train_attacks["Attack"].copy()
    class_counts = y_attack.value_counts()

    # Formulate SMOTE target strategy
    smote_strategy = {
        cls: target_count for cls, cnt in class_counts.items() if cnt < target_count
    }
    print(f"    Target ceiling per class: {target_count:,}")
    print(f"    Classes to be oversampled ({len(smote_strategy)}): {list(smote_strategy.keys())}")

    # Set k_neighbors adaptively based on the smallest class (e.g. Worms)
    min_samples = class_counts.min()
    k_neighbors = min(5, max(1, min_samples - 1))
    print(f"    Using k_neighbors = {k_neighbors} for SMOTE interpolation")

    nn_engine = NearestNeighbors(n_neighbors=k_neighbors, algorithm="kd_tree", n_jobs=1)
    smote_model = SMOTE(sampling_strategy=smote_strategy, random_state=42, k_neighbors=nn_engine)
    X_smote, y_smote = smote_model.fit_resample(X_attack, y_attack)

    df_smote = pd.DataFrame(X_smote, columns=feature_cols)
    df_smote["Attack"] = y_smote.values
    df_smote["Label"] = 1

    print(f"    SMOTE Complete: {len(df_train_attacks):,} -> {len(df_smote):,} attack flows")
    print(f"    Step 4 Finished in {time.time() - t0:.2f}s\n")

    # ------------------------------------------------------------------------
    # Step 5: Remove Duplicates & Technically Invalid Samples
    # ------------------------------------------------------------------------
    print(">>> [Step 5/7] Filtering Duplicates & Technically Invalid Samples from Oversampled Attacks...")
    t0 = time.time()

    valid_protocols = set(df_train_attacks["PROTOCOL"].unique())

    # Invariant checks:
    # 1. Non-negative values for all flow metrics
    neg_mask = (df_smote[feature_cols] < 0).any(axis=1)

    # 2. Zero-packet with non-zero bytes violations
    zero_out_pkts = (df_smote["OUT_PKTS"] == 0) & (df_smote["OUT_BYTES"] > 0)
    zero_in_pkts = (df_smote["IN_PKTS"] == 0) & (df_smote["IN_BYTES"] > 0)

    # 3. Protocol != 6 (TCP) but TCP flags present
    non_tcp_flags = (df_smote["PROTOCOL"] != 6) & (
        (df_smote["TCP_FLAGS"] > 0) |
        (df_smote["CLIENT_TCP_FLAGS"] > 0) |
        (df_smote["SERVER_TCP_FLAGS"] > 0)
    )

    # 4. Unknown/invalid protocol numbers
    invalid_proto = ~df_smote["PROTOCOL"].isin(valid_protocols)

    # 5. Packet length and TTL sanity invariants
    invalid_ip_len = df_smote["MIN_IP_PKT_LEN"] > df_smote["MAX_IP_PKT_LEN"]
    invalid_pkt_len = df_smote["SHORTEST_FLOW_PKT"] > df_smote["LONGEST_FLOW_PKT"]
    invalid_ttl = df_smote["MIN_TTL"] > df_smote["MAX_TTL"]

    invalid_mask = (
        neg_mask |
        zero_out_pkts |
        zero_in_pkts |
        non_tcp_flags |
        invalid_proto |
        invalid_ip_len |
        invalid_pkt_len |
        invalid_ttl
    )

    print(f"    Validity Violation Breakdown:")
    print(f"      - Negative values:               {neg_mask.sum():,}")
    print(f"      - OUT_PKTS == 0 & OUT_BYTES > 0: {zero_out_pkts.sum():,}")
    print(f"      - IN_PKTS == 0 & IN_BYTES > 0:   {zero_in_pkts.sum():,}")
    print(f"      - PROTOCOL != 6 & TCP flags > 0: {non_tcp_flags.sum():,}")
    print(f"      - Invalid protocol numbers:      {invalid_proto.sum():,}")
    print(f"      - Length / TTL logic violations: {(invalid_ip_len | invalid_pkt_len | invalid_ttl).sum():,}")
    print(f"      - Total Invalid Flows:           {invalid_mask.sum():,} ({invalid_mask.sum() / len(df_smote) * 100:.2f}%)")

    # Purge invalid flows
    df_smote_valid = df_smote[~invalid_mask].copy()

    # Deduplicate across feature columns
    dup_count = df_smote_valid.duplicated(subset=feature_cols).sum()
    df_clean_attacks_smote = df_smote_valid.drop_duplicates(subset=feature_cols).reset_index(drop=True)

    print(f"    Duplicate Flows Purged:            {dup_count:,}")
    print(f"    Final Clean Oversampled Attacks:   {len(df_clean_attacks_smote):,} flows")
    print("    Final Clean Attack distribution:")
    for atk, cnt in df_clean_attacks_smote["Attack"].value_counts().items():
        print(f"      - {atk:<16}: {cnt:>6,} flows")

    # Export to TVAE, CTGAN, and IDS Multiclass targets
    smote_export_filename = "unsw_nb15_attacks_smote_10k.csv"
    target_smote_paths = [
        ids_multi_dir / smote_export_filename,
        gen_tvae_dir / smote_export_filename,
        gen_ctgan_dir / smote_export_filename,
    ]

    for dest in target_smote_paths:
        dest_str = str(dest).replace("\\", "/")
        con.execute(f"COPY df_clean_attacks_smote TO '{dest_str}' (HEADER, DELIMITER ',');")
        print(f"    Exported: {dest.relative_to(base_dir)} ({dest.stat().st_size / (1024**2):.2f} MB)")

    print(f"    Step 5 Finished in {time.time() - t0:.2f}s\n")

    # ------------------------------------------------------------------------
    # Step 6: 5-Fold Stratified Cross-Validation for IDS Binary
    # ------------------------------------------------------------------------
    print(">>> [Step 6/7] Creating 5-Fold Stratified Train & Test Pairs (80/20) for IDS Binary...")
    t0 = time.time()

    # Combine benign with ORIGINAL attack flows (pristine natural non-holdout set)
    df_binary_pool = train_val_df.copy().reset_index(drop=True)
    print(f"    Binary Pool Size: {len(df_binary_pool):,} flows (Benign: {(df_binary_pool['Label'] == 0).sum():,}, "
          f"Attacks: {(df_binary_pool['Label'] == 1).sum():,})")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df_binary_pool, df_binary_pool["Attack"]), 1):
        fold_train = df_binary_pool.iloc[train_idx]
        fold_test = df_binary_pool.iloc[test_idx]

        train_file = ids_bin_train_dir / f"NF-UNSW-NB15-v3_fold_{fold_idx}.csv"
        test_file = ids_bin_test_dir / f"NF-UNSW-NB15-v3_fold_{fold_idx}.csv"

        train_path_str = str(train_file).replace("\\", "/")
        test_path_str = str(test_file).replace("\\", "/")

        con.execute(f"COPY fold_train TO '{train_path_str}' (HEADER, DELIMITER ',');")
        con.execute(f"COPY fold_test TO '{test_path_str}' (HEADER, DELIMITER ',');")

        train_mb = train_file.stat().st_size / (1024 ** 2)
        test_mb = test_file.stat().st_size / (1024 ** 2)

        fold_records.append({
            "Fold": f"Fold {fold_idx}",
            "Train Flows": f"{len(fold_train):,}",
            "Train Size (MB)": f"{train_mb:.2f}",
            "Test Flows": f"{len(fold_test):,}",
            "Test Size (MB)": f"{test_mb:.2f}",
            "Fold Total": f"{len(fold_train) + len(fold_test):,}"
        })
        print(f"    Fold {fold_idx}: Train={len(fold_train):,} ({train_mb:.2f} MB), "
              f"Test={len(fold_test):,} ({test_mb:.2f} MB)")

    print(f"    Step 6 Finished in {time.time() - t0:.2f}s\n")

    # ------------------------------------------------------------------------
    # Step 7: Final Catalog Verification & Summary
    # ------------------------------------------------------------------------
    print("=" * 80)
    print(">>> [Step 7/7] Catalog Verification of Exported Datasets in 'datasets/final datasets/'")
    print("=" * 80)

    summary_records = []
    for csv_file in sorted(final_dir.glob("**/*.csv")):
        size_mb = csv_file.stat().st_size / (1024 ** 2)
        csv_file_str = str(csv_file).replace("\\", "/")
        row_count = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{csv_file_str}');").fetchone()[0]
        summary_records.append({
            "Relative Path": str(csv_file.relative_to(final_dir)),
            "Flow Rows": f"{row_count:,}",
            "Size (MB)": f"{size_mb:.2f}"
        })

    df_summary = pd.DataFrame(summary_records)
    print(df_summary.to_string(index=False))

    total_elapsed = time.time() - start_total_time
    print(f"\nAll 7 pipeline stages completed successfully in {total_elapsed:.2f} seconds ({total_elapsed / 60:.2f} minutes)!")


if __name__ == "__main__":
    run_pipeline()

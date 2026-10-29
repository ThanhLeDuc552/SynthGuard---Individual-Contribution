"""
End-to-End Processing & Resampling Pipeline for NetFlow-v3 Intrusion Detection Datasets.

This script executes the complete data preparation pipeline:
1. Pruning non-generalizable shortcut features (IP addresses, L4 ports, timestamps).
2. Flow deduplication across remaining 47 features.
3. Feature normalization (Min-Max scaling to [0, 1] with numerical stabilization).
4. Resampling strategies:
   - NF-UNSW-NB15-v3: None (natural distribution)
   - NF-BoT-IoT-v3: C-VAE + RUS (downsample DoS/DDoS head classes; C-VAE oversample rare Theft)
   - NF-ToN-IoT-v3: C-VAE (C-VAE oversample rare ransomware, mitm, backdoor)
   - NF-CSE-CIC-IDS2018-v3: None (natural distribution)
5. Exporting all 4 final processed CSV files into 'NetFlow_IDS_final'.
"""

import os
import sys
import time
from pathlib import Path

# Allow duplicate OpenMP runtime initialization on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import duckdb
import numpy as np
import pandas as pd
import torch

# Add current folder to sys.path to load CVAEOversampler
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from cvae_oversampling import CVAEOversampler

# Setup directories
base_dir = Path.cwd()
possible_roots = [
    base_dir / 'duc_thanh/datasets/NetFlow',
    base_dir / 'datasets/NetFlow',
    Path('d:/Swinburne Courses/Project A/duc_thanh/datasets/NetFlow'),
    Path('d:/Swinburne Courses/Project A/datasets/NetFlow')
]

netflow_root = None
for p in possible_roots:
    if p.exists() and p.is_dir():
        netflow_root = p.resolve()
        break

if not netflow_root:
    raise FileNotFoundError("Could not find datasets/NetFlow directory!")

print(f"NetFlow Source Directory: {netflow_root}")

# Export directory setup
export_dirs = [
    netflow_root.parent / 'NetFlow_IDS_final',
    Path('d:/Swinburne Courses/Project A/duc_thanh/datasets/NetFlow_IDS_final'),
    Path('d:/Swinburne Courses/Project A/NetFlow_IDS_final')
]
export_dir = export_dirs[0]
export_dir.mkdir(parents=True, exist_ok=True)
print(f"Target Export Directory:  {export_dir.resolve()}\n")

# Initialize DuckDB connection
con = duckdb.connect()
con.execute("PRAGMA threads=6;")
con.execute("PRAGMA memory_limit='10GB';")
con.execute("PRAGMA preserve_insertion_order=false;")

# Non-generalizable shortcut features to eliminate
EXCLUDE_COLS = [
    'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT',
    'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS'
]
EXCLUDE_SQL = f"EXCLUDE ({', '.join(EXCLUDE_COLS)})"

dataset_files = {
    'unsw': list(netflow_root.glob('NF-UNSW-NB15-v3/*/data/NF-UNSW-NB15-v3.csv'))[0],
    'bot_iot': list(netflow_root.glob('NF-BoT-IoT-v3/*/data/NF-BoT-IoT-v3.csv'))[0],
    'cic_ids': list(netflow_root.glob('NF-CSE-CIC-IDS2018-v3/*/data/NF-CICIDS2018-v3.csv'))[0],
    'ton_iot': list(netflow_root.glob('NF-ToN-IoT-v3/*/data/NF-ToN-IoT-v3.csv'))[0],
}


def get_feature_columns(view_name: str) -> list:
    """Returns the 47 feature columns excluding Label and Attack."""
    cols = [r[0] for r in con.execute(f"DESCRIBE {view_name};").fetchall()]
    return [c for c in cols if c not in ['Label', 'Attack']]


def compute_min_max_expressions(table_name: str, feat_cols: list) -> list:
    """Computes finite min and max for each feature and generates SQL normalization expressions."""
    agg_exprs = [
        f"MIN(CASE WHEN isfinite({c}::DOUBLE) THEN {c}::DOUBLE ELSE NULL END) AS min_{c}, "
        f"MAX(CASE WHEN isfinite({c}::DOUBLE) THEN {c}::DOUBLE ELSE NULL END) AS max_{c}"
        for c in feat_cols
    ]
    min_max_df = con.execute(f"SELECT {', '.join(agg_exprs)} FROM {table_name};").df()
    
    scale_exprs = []
    for c in feat_cols:
        c_min = float(min_max_df[f"min_{c}"].iloc[0])
        c_max = float(min_max_df[f"max_{c}"].iloc[0])
        denom = c_max - c_min
        if denom <= 0 or not np.isfinite(denom):
            scale_exprs.append(f"0.0 AS {c}")
        else:
            scale_exprs.append(f"""
                CASE 
                    WHEN NOT isfinite({c}::DOUBLE) THEN 0.0
                    WHEN {c}::DOUBLE <= {c_min} THEN 0.0
                    WHEN {c}::DOUBLE >= {c_max} THEN 1.0
                    ELSE ROUND(({c}::DOUBLE - {c_min}) / {denom}, 6)
                END AS {c}
            """.strip())
    return scale_exprs


# ============================================================================
# 1. NF-UNSW-NB15-v3 Processing (Resampling: None)
# ============================================================================
def process_unsw():
    print("=" * 70)
    print(">>> [1/4] Processing NF-UNSW-NB15-v3 (Resampling: None)...")
    t0 = time.time()
    raw_path = str(dataset_files['unsw']).replace('\\', '/')
    
    # Deduplicate and drop shortcut features
    con.execute(f"""
        CREATE OR REPLACE VIEW unsw_clean AS
        SELECT DISTINCT * {EXCLUDE_SQL}
        FROM read_csv_auto('{raw_path}');
    """)
    
    feat_cols = get_feature_columns('unsw_clean')
    scale_exprs = compute_min_max_expressions('unsw_clean', feat_cols)
    
    con.execute(f"""
        CREATE OR REPLACE TABLE unsw_final AS
        SELECT 
            {', '.join(scale_exprs)},
            Label,
            Attack
        FROM unsw_clean;
    """)
    
    out_file = export_dir / 'NF-UNSW-NB15-v3.csv'
    out_path_sql = str(out_file).replace('\\', '/')
    con.execute(f"COPY unsw_final TO '{out_path_sql}' (HEADER, DELIMITER ',');")
    
    n_total = con.execute("SELECT COUNT(*) FROM unsw_final").fetchone()[0]
    size_mb = out_file.stat().st_size / (1024 ** 2)
    elapsed = time.time() - t0
    print(f"    Completed NF-UNSW-NB15-v3: {n_total:,} flows | {size_mb:.2f} MB | {elapsed:.2f}s")
    dist = con.execute("SELECT Attack, COUNT(*) as cnt FROM unsw_final GROUP BY Attack ORDER BY cnt DESC;").df()
    print("    Class distribution:\n", dist.to_string(index=False))


# ============================================================================
# 2. NF-BoT-IoT-v3 Processing (Resampling: C-VAE + RUS)
# ============================================================================
def process_bot_iot():
    print("\n" + "=" * 70)
    print(">>> [2/4] Processing NF-BoT-IoT-v3 (Resampling: C-VAE + RUS)...")
    t0 = time.time()
    raw_path = str(dataset_files['bot_iot']).replace('\\', '/')
    
    # 1. Deduplicate & drop shortcut features
    con.execute(f"""
        CREATE OR REPLACE VIEW bot_clean AS
        SELECT DISTINCT * {EXCLUDE_SQL}
        FROM read_csv_auto('{raw_path}');
    """)
    
    feat_cols = get_feature_columns('bot_clean')
    
    # 2. Random Under-Sampling (RUS) on majority attack classes (downsample DoS, DDoS, Recon to 25k each)
    con.execute("""
        CREATE OR REPLACE TABLE bot_rus AS
        (SELECT * FROM bot_clean WHERE Attack = 'DoS' USING SAMPLE 25000 (reservoir, 42))
        UNION ALL
        (SELECT * FROM bot_clean WHERE Attack = 'DDoS' USING SAMPLE 25000 (reservoir, 42))
        UNION ALL
        (SELECT * FROM bot_clean WHERE Attack = 'Reconnaissance' USING SAMPLE 25000 (reservoir, 42))
        UNION ALL
        (SELECT * FROM bot_clean WHERE Attack = 'Benign')
        UNION ALL
        (SELECT * FROM bot_clean WHERE Attack = 'Theft');
    """)
    
    rus_dist = con.execute("SELECT Attack, COUNT(*) as cnt FROM bot_rus GROUP BY Attack ORDER BY cnt DESC;").df()
    print("    After RUS Class Breakdown:\n", rus_dist.to_string(index=False))
    
    # 3. Train C-VAE on attack records to oversample minority Theft up to 20,000 samples
    print("    Training C-VAE on BoT-IoT attacks for minority 'Theft' oversampling...")
    df_atk_train = con.execute("SELECT * FROM bot_rus WHERE Label = 1;").df()
    
    cvae = CVAEOversampler(latent_dim=16, epochs=15, batch_size=128, beta=0.5, random_state=42)
    cvae.fit(df_atk_train, target_col='Attack', feature_cols=feat_cols, verbose=False)
    
    current_theft_cnt = (df_atk_train['Attack'] == 'Theft').sum()
    needed_theft = 20000 - current_theft_cnt
    print(f"    Synthesizing {needed_theft:,} C-VAE flows for 'Theft'...")
    df_syn_theft = cvae.generate_samples('Theft', needed_theft)
    df_syn_theft['Label'] = 1
    df_syn_theft['Attack'] = 'Theft'
    df_syn_theft = df_syn_theft[feat_cols + ['Label', 'Attack']]
    
    # Register synthetic samples in DuckDB
    con.register('df_syn_theft_view', df_syn_theft)
    
    con.execute("""
        CREATE OR REPLACE TABLE bot_resampled AS
        SELECT * FROM bot_rus
        UNION ALL BY NAME
        SELECT * FROM df_syn_theft_view;
    """)
    
    # 4. Feature Normalization (Min-Max Scaling to [0, 1])
    scale_exprs = compute_min_max_expressions('bot_resampled', feat_cols)
    con.execute(f"""
        CREATE OR REPLACE TABLE bot_final AS
        SELECT 
            {', '.join(scale_exprs)},
            Label,
            Attack
        FROM bot_resampled;
    """)
    
    out_file = export_dir / 'NF-BoT-IoT-v3.csv'
    out_path_sql = str(out_file).replace('\\', '/')
    con.execute(f"COPY bot_final TO '{out_path_sql}' (HEADER, DELIMITER ',');")
    
    n_total = con.execute("SELECT COUNT(*) FROM bot_final").fetchone()[0]
    size_mb = out_file.stat().st_size / (1024 ** 2)
    elapsed = time.time() - t0
    print(f"    Completed NF-BoT-IoT-v3: {n_total:,} flows | {size_mb:.2f} MB | {elapsed:.2f}s")
    dist = con.execute("SELECT Attack, COUNT(*) as cnt FROM bot_final GROUP BY Attack ORDER BY cnt DESC;").df()
    print("    Final Class distribution:\n", dist.to_string(index=False))


# ============================================================================
# 3. NF-ToN-IoT-v3 Processing (Resampling: C-VAE)
# ============================================================================
def process_ton_iot():
    print("\n" + "=" * 70)
    print(">>> [3/4] Processing NF-ToN-IoT-v3 (Resampling: C-VAE)...")
    t0 = time.time()
    raw_path = str(dataset_files['ton_iot']).replace('\\', '/')
    
    # 1. Deduplicate & drop shortcut features
    con.execute(f"""
        CREATE OR REPLACE VIEW ton_clean AS
        SELECT DISTINCT * {EXCLUDE_SQL}
        FROM read_csv_auto('{raw_path}');
    """)
    
    feat_cols = get_feature_columns('ton_clean')
    
    # Sample attacks to fit C-VAE model efficiently
    q_sample_attacks = """
        SELECT * FROM ton_clean 
        WHERE Label = 1
        USING SAMPLE 30000 (reservoir, 42);
    """
    df_ton_sample = con.execute(q_sample_attacks).df()
    # Ensure all rare classes are included in C-VAE training
    df_rare = con.execute("SELECT * FROM ton_clean WHERE Attack IN ('mitm', 'ransomware', 'Backdoor');").df()
    df_cvae_train = pd.concat([df_ton_sample, df_rare], ignore_index=True).drop_duplicates()
    
    print("    Training C-VAE on ToN-IoT attack classes...")
    cvae = CVAEOversampler(latent_dim=24, epochs=15, batch_size=128, beta=0.5, random_state=42)
    cvae.fit(df_cvae_train, target_col='Attack', feature_cols=feat_cols, verbose=False)
    
    # Oversample minority classes: ransomware up to 25k, mitm up to 25k, Backdoor up to 25k
    rare_targets = {'ransomware': 25000, 'mitm': 25000, 'Backdoor': 25000}
    current_counts = con.execute("""
        SELECT Attack, COUNT(*) as cnt 
        FROM ton_clean 
        WHERE Attack IN ('ransomware', 'mitm', 'Backdoor')
        GROUP BY Attack;
    """).df().set_index('Attack')['cnt'].to_dict()
    
    syn_dfs = []
    for cls_name, target_cnt in rare_targets.items():
        curr = current_counts.get(cls_name, 0)
        needed = target_cnt - curr
        if needed > 0:
            print(f"    Synthesizing {needed:,} C-VAE flows for '{cls_name}' (current: {curr:,} -> target: {target_cnt:,})...")
            syn = cvae.generate_samples(cls_name, needed)
            syn['Label'] = 1
            syn['Attack'] = cls_name
            syn = syn[feat_cols + ['Label', 'Attack']]
            syn_dfs.append(syn)
            
    df_syn_total = pd.concat(syn_dfs, ignore_index=True)
    con.register('df_ton_syn_view', df_syn_total)
    
    con.execute("""
        CREATE OR REPLACE TABLE ton_resampled AS
        SELECT * FROM ton_clean
        UNION ALL BY NAME
        SELECT * FROM df_ton_syn_view;
    """)
    
    # 2. Normalization
    scale_exprs = compute_min_max_expressions('ton_resampled', feat_cols)
    con.execute(f"""
        CREATE OR REPLACE TABLE ton_final AS
        SELECT 
            {', '.join(scale_exprs)},
            Label,
            Attack
        FROM ton_resampled;
    """)
    
    out_file = export_dir / 'NF-ToN-IoT-v3.csv'
    out_path_sql = str(out_file).replace('\\', '/')
    con.execute(f"COPY ton_final TO '{out_path_sql}' (HEADER, DELIMITER ',');")
    
    n_total = con.execute("SELECT COUNT(*) FROM ton_final").fetchone()[0]
    size_mb = out_file.stat().st_size / (1024 ** 2)
    elapsed = time.time() - t0
    print(f"    Completed NF-ToN-IoT-v3: {n_total:,} flows | {size_mb:.2f} MB | {elapsed:.2f}s")
    dist = con.execute("SELECT Attack, COUNT(*) as cnt FROM ton_final GROUP BY Attack ORDER BY cnt DESC;").df()
    print("    Final Class distribution:\n", dist.to_string(index=False))


# ============================================================================
# 4. NF-CSE-CIC-IDS2018-v3 Processing (Resampling: None)
# ============================================================================
def process_cic_ids():
    print("\n" + "=" * 70)
    print(">>> [4/4] Processing NF-CSE-CIC-IDS2018-v3 (Resampling: None)...")
    t0 = time.time()
    raw_path = str(dataset_files['cic_ids']).replace('\\', '/')
    
    # Deduplicate & drop shortcut features
    con.execute(f"""
        CREATE OR REPLACE VIEW cic_clean AS
        SELECT DISTINCT * {EXCLUDE_SQL}
        FROM read_csv_auto('{raw_path}');
    """)
    
    feat_cols = get_feature_columns('cic_clean')
    scale_exprs = compute_min_max_expressions('cic_clean', feat_cols)
    
    con.execute(f"""
        CREATE OR REPLACE TABLE cic_final AS
        SELECT 
            {', '.join(scale_exprs)},
            Label,
            Attack
        FROM cic_clean;
    """)
    
    out_file = export_dir / 'NF-CSE-CIC-IDS2018-v3.csv'
    out_path_sql = str(out_file).replace('\\', '/')
    print(f"    Exporting {out_file.name} to CSV...")
    con.execute(f"COPY cic_final TO '{out_path_sql}' (HEADER, DELIMITER ',');")
    
    n_total = con.execute("SELECT COUNT(*) FROM cic_final").fetchone()[0]
    size_mb = out_file.stat().st_size / (1024 ** 2)
    elapsed = time.time() - t0
    print(f"    Completed NF-CSE-CIC-IDS2018-v3: {n_total:,} flows | {size_mb:.2f} MB | {elapsed:.2f}s")
    dist = con.execute("SELECT Attack, COUNT(*) as cnt FROM cic_final GROUP BY Attack ORDER BY cnt DESC;").df()
    print("    Class distribution:\n", dist.to_string(index=False))


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    t_script = time.time()
    print("=" * 70)
    print("STARTING NETFLOW-V3 PROCESSING & RESAMPLING PIPELINE")
    print(f"Target Directory: {export_dir}")
    print("=" * 70)
    
    process_unsw()
    process_bot_iot()
    process_ton_iot()
    process_cic_ids()
    
    total_time = time.time() - t_script
    print("\n" + "=" * 70)
    print(f"ALL 4 DATASETS SUCCESSFULLY PROCESSED AND EXPORTED in {total_time:.2f} seconds!")
    print(f"Destination: {export_dir.resolve()}")
    print("=" * 70)
    
    # Print directory listing
    for f in export_dir.glob("*.csv"):
        print(f" - {f.name:<28}: {f.stat().st_size / (1024**2):,.2f} MB")


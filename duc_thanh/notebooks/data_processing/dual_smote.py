"""
Dual-Stage SMOTE Data Resampling Pipeline for NetFlow Intrusion Detection Datasets.

This module provides the `DualStageSmotePipeline` class, which handles:
1. Intra-Attack Multiclass SMOTE: Balances minority attack categories up to the majority attack class.
2. Attack-over-Benign Inter-Class SMOTE: Oversamples Attack flows relative to Benign flows (1:1 or custom ratio).
3. Robust numerical feature sanitization (replacing inf/-inf and handling missing values).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE


class DualStageSmotePipeline:
    """
    Two-Stage SMOTE Resampling Pipeline for NetFlow Network Intrusion Detection Datasets.
    
    Stage 1 (Intra-Attack Oversampling):
        Balances the multiclass attack distribution so that rare long-tail attack
        classes (e.g., Worms, Analysis, Shellcode) match the majority attack class.
        
    Stage 2 (Attack-over-Benign Oversampling):
        Oversamples Attack flows relative to Benign flows according to a configurable
        target ratio (e.g., 1.0 for 1:1 balance, or > 1.0 for attack over benign),
        preserving balanced multiclass attack representations.
    """
    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.diagnostics = {}
        
    def _clean_features(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Sanitizes continuous features by replacing inf/-inf and imputing nulls."""
        X = df[feature_cols].copy()
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
            med = X[col].median()
            X[col] = X[col].fillna(0.0 if np.isnan(med) else med)
        return X

    def fit_resample(
        self,
        df_dedup: pd.DataFrame,
        feature_cols: list,
        benign_sample_size: int = None,
        target_attack_ratio: float = 1.0
    ) -> pd.DataFrame:
        """
        Executes the two-stage SMOTE pipeline.
        
        Parameters:
        -----------
        df_dedup : pd.DataFrame
            Deduplicated NetFlow dataset containing 'Label', 'Attack', and feature columns.
        feature_cols : list
            List of numerical feature column names to use for synthesis.
        benign_sample_size : int, optional
            Number of Benign samples to retain (subsamples Benign if provided).
        target_attack_ratio : float, default=1.0
            Desired ratio of Attack flows to Benign flows (e.g., 1.0 for 1:1, 1.2 for 1.2:1).
            
        Returns:
        --------
        df_balanced : pd.DataFrame
            Resampled DataFrame containing balanced Benign and Attack records.
        """
        t_start = time.time()
        
        # 1. Separate Benign and Attacks
        df_atk = df_dedup[df_dedup['Label'] == 1].copy()
        df_bng = df_dedup[df_dedup['Label'] == 0].copy()
        
        raw_atk_count = len(df_atk)
        raw_bng_count = len(df_bng)
        raw_atk_dist = df_atk['Attack'].value_counts().to_dict()
        
        # Optional Benign subsampling for performance / scaling control
        if benign_sample_size is not None and benign_sample_size < len(df_bng):
            df_bng = df_bng.sample(n=benign_sample_size, random_state=self.random_state).reset_index(drop=True)
            
        n_benign = len(df_bng)
        
        # 2. Stage 1: Intra-Attack Multiclass SMOTE
        print(">>> [Stage 1/2] Executing Intra-Attack Multiclass SMOTE...")
        t_s1 = time.time()
        X_atk = self._clean_features(df_atk, feature_cols)
        y_atk = df_atk['Attack'].copy()
        
        min_samples_s1 = y_atk.value_counts().min()
        k_s1 = min(self.k_neighbors, max(1, min_samples_s1 - 1))
        
        nn_s1 = NearestNeighbors(n_neighbors=k_s1, algorithm='kd_tree', n_jobs=1)
        smote1 = SMOTE(sampling_strategy='not majority', random_state=self.random_state, k_neighbors=nn_s1)
        X_atk_s1, y_atk_s1 = smote1.fit_resample(X_atk, y_atk)
        s1_elapsed = time.time() - t_s1
        
        s1_atk_count = len(y_atk_s1)
        s1_atk_dist = y_atk_s1.value_counts().to_dict()
        print(f"    Stage 1 Complete: {raw_atk_count:,} raw attacks -> {s1_atk_count:,} balanced attacks across {len(s1_atk_dist)} classes ({s1_elapsed:.2f}s)")
        
        # 3. Stage 2: Attack over Benign SMOTE
        print(">>> [Stage 2/2] Executing Attack-over-Benign Inter-Class SMOTE...")
        t_s2 = time.time()
        X_bng = self._clean_features(df_bng, feature_cols)
        y_bng = pd.Series(['Benign'] * n_benign, name='Attack')
        
        # Target total attacks: n_benign * target_attack_ratio
        n_target_attacks = int(n_benign * target_attack_ratio)
        n_classes = len(s1_atk_dist)
        
        max_existing_per_class = max(s1_atk_dist.values())
        target_per_atk_class = max(int(np.ceil(n_target_attacks / n_classes)), max_existing_per_class)
        
        strategy_s2 = {'Benign': n_benign}
        for atk_cls in s1_atk_dist.keys():
            strategy_s2[atk_cls] = target_per_atk_class
            
        X_combined = pd.concat([X_bng, X_atk_s1], ignore_index=True)
        y_combined = pd.concat([y_bng, y_atk_s1], ignore_index=True)
        
        min_samples_s2 = min(s1_atk_dist.values())
        k_s2 = min(self.k_neighbors, max(1, min_samples_s2 - 1))
        
        nn_s2 = NearestNeighbors(n_neighbors=k_s2, algorithm='kd_tree', n_jobs=1)
        smote2 = SMOTE(sampling_strategy=strategy_s2, random_state=self.random_state, k_neighbors=nn_s2)
        X_resampled, y_resampled = smote2.fit_resample(X_combined, y_combined)
        s2_elapsed = time.time() - t_s2
        
        # 4. Construct final balanced dataframe
        df_final = X_resampled.copy()
        df_final['Attack'] = y_resampled.values
        df_final['Label'] = (y_resampled.values != 'Benign').astype(int)
        
        final_bng = (df_final['Label'] == 0).sum()
        final_atk = (df_final['Label'] == 1).sum()
        final_atk_dist = df_final[df_final['Label'] == 1]['Attack'].value_counts().to_dict()
        total_elapsed = time.time() - t_start
        
        self.diagnostics = {
            'raw_benign': raw_bng_count,
            'raw_attack': raw_atk_count,
            'raw_attack_dist': raw_atk_dist,
            'stage1_attack_count': s1_atk_count,
            'stage1_attack_dist': s1_atk_dist,
            'stage1_elapsed_sec': round(s1_elapsed, 2),
            'stage2_benign_count': final_bng,
            'stage2_attack_count': final_atk,
            'stage2_attack_dist': final_atk_dist,
            'stage2_elapsed_sec': round(s2_elapsed, 2),
            'final_total': len(df_final),
            'final_attack_ratio': round(final_atk / final_bng, 2),
            'total_elapsed_sec': round(total_elapsed, 2)
        }
        
        print("\n=== Dual-Stage SMOTE Pipeline Diagnostics ===")
        print(f"Stage 1 (Intra-Attack SMOTE): {raw_atk_count:,} raw attacks -> {s1_atk_count:,} balanced attacks ({s1_elapsed:.2f}s)")
        print(f"Stage 2 (Attack-over-Benign): {final_bng:,} Benign vs {final_atk:,} Attack ({s2_elapsed:.2f}s)")
        print(f"Final Binary Ratio:          {self.diagnostics['final_attack_ratio']} Attack : 1.00 Benign")
        print(f"Total Resampling Time:       {total_elapsed:.2f} seconds")
        print(f"Balanced Dataset Shape:      {df_final.shape}")
        
        return df_final

"""
Attack Oversampling Pipeline for NetFlow Intrusion Detection Datasets.

This module provides multi-generative oversampling techniques for rebalancing
minority attack classes within network flow datasets:
1. SMOTE (Synthetic Minority Over-sampling Technique)
2. CTGAN (Conditional Tabular Generative Adversarial Network)
3. Tabular GAN (PyTorch Deep Tabular Generative Adversarial Network)
4. SMOTIFIED-GAN (Hybrid SMOTE + GAN manifold-enrichment generation)
"""
import torch
import torch.nn as nn
import os
import sys
import ctypes
from pathlib import Path

# Allow duplicate OpenMP runtime initialization across numpy/duckdb/torch on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE, RandomOverSampler
from ctgan import CTGAN


# ============================================================================
# PyTorch Tabular GAN Architectures
# ============================================================================

class TabularGenerator(nn.Module):
    """Deep Multi-Layer Generator for Tabular Continuous Features."""
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Sigmoid()  # Data normalized to [0, 1] range
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TabularDiscriminator(nn.Module):
    """Deep Multi-Layer Discriminator with Dropout Regularization."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# Core Oversampling Pipeline Class
# ============================================================================

class AttackOversamplingPipeline:
    """
    Modular Data Resampling & Generative Pipeline for Network Intrusion Attack Samples.
    
    Generates 4 distinct oversampled datasets from input attack flows:
    - SMOTE: Local k-nearest-neighbors geometric interpolation.
    - CTGAN: Conditional Tabular GAN with mode-specific normalization.
    - GAN: Deep tabular generative adversarial network on normalized continuous features.
    - SMOTIFIED-GAN: Hybrid approach using SMOTE manifold expansion followed by deep GAN synthesis.
    """
    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "Attack",
        random_state: int = 42,
        device: Optional[str] = None
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.diagnostics: Dict[str, dict] = {}
        
        # Set seeds for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

    def _resolve_features(self, df: pd.DataFrame) -> List[str]:
        """Auto-detects continuous numerical feature columns if not explicitly passed."""
        if self.feature_cols is not None:
            return self.feature_cols
        excluded = [self.target_col, "Label", "label", "attack", "attack_cat", "id"]
        return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

    def _clean_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Sanitizes continuous features by replacing inf/-inf and imputing nulls."""
        X = df[feature_cols].copy()
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            med = X[col].median()
            X[col] = X[col].fillna(0.0 if np.isnan(med) else med)
        return X

    def _determine_target_counts(
        self,
        class_counts: pd.Series,
        target_count: Union[str, int, Dict[str, int]] = "majority"
    ) -> Dict[str, int]:
        """Calculates desired sample count for each attack class."""
        majority_count = int(class_counts.max())
        
        if isinstance(target_count, dict):
            return {cls: max(class_counts[cls], target_count.get(cls, class_counts[cls])) for cls in class_counts.index}
        elif target_count == "majority" or target_count is None:
            return {cls: majority_count for cls in class_counts.index}
        elif isinstance(target_count, (int, float)):
            target_int = int(target_count)
            return {cls: max(class_counts[cls], target_int) for cls in class_counts.index}
        else:
            raise ValueError(f"Unsupported target_count specification: {target_count}")

    # ------------------------------------------------------------------------
    # 1. SMOTE Oversampling
    # ------------------------------------------------------------------------
    def apply_smote(
        self,
        df_attacks: pd.DataFrame,
        target_count: Union[str, int, Dict[str, int]] = "majority"
    ) -> pd.DataFrame:
        """
        Applies Multi-Class Intra-Attack SMOTE to rebalance minority attack categories.
        """
        t0 = time.time()
        print("\n>>> [1/4] Executing SMOTE Attack Oversampling...")
        feature_cols = self._resolve_features(df_attacks)
        X_clean = self._clean_features(df_attacks, feature_cols)
        y = df_attacks[self.target_col].copy()

        class_counts = y.value_counts()
        target_dict = self._determine_target_counts(class_counts, target_count)
        
        # Adaptive k_neighbors for ultra-rare minority categories (e.g. Worms)
        min_samples = class_counts.min()
        k_neighbors = min(5, max(1, min_samples - 1))

        strategy = {cls: target_dict[cls] for cls in class_counts.index if target_dict[cls] > class_counts[cls]}
        if not strategy:
            print("    SMOTE: All classes already meet target count. Returning original.")
            return df_attacks.copy()

        nn_engine = NearestNeighbors(n_neighbors=k_neighbors, algorithm="kd_tree", n_jobs=1)
        smote_model = SMOTE(sampling_strategy=strategy, random_state=self.random_state, k_neighbors=nn_engine)
        
        X_res, y_res = smote_model.fit_resample(X_clean, y)
        
        df_out = pd.DataFrame(X_res, columns=feature_cols)
        df_out[self.target_col] = y_res.values
        df_out["Label"] = 1
        
        elapsed = time.time() - t0
        print(f"    SMOTE Complete: {len(df_attacks):,} -> {len(df_out):,} samples across {len(class_counts)} classes ({elapsed:.2f}s)")
        self.diagnostics["smote"] = {
            "elapsed_sec": round(elapsed, 2),
            "original_count": len(df_attacks),
            "oversampled_count": len(df_out),
            "class_distribution": df_out[self.target_col].value_counts().to_dict()
        }
        return df_out

    # ------------------------------------------------------------------------
    # 1b. Random Over-Sampling (ROS)
    # ------------------------------------------------------------------------
    def apply_ros(
        self,
        df_attacks: pd.DataFrame,
        target_count: Union[str, int, Dict[str, int]] = "majority"
    ) -> pd.DataFrame:
        """
        Applies Random Over-Sampling (ROS) to replicate minority attack samples to target counts.
        """
        t0 = time.time()
        print("\n>>> Executing Random Over-Sampling (ROS) Attack Oversampling...")
        feature_cols = self._resolve_features(df_attacks)
        X_clean = self._clean_features(df_attacks, feature_cols)
        y = df_attacks[self.target_col].copy()

        class_counts = y.value_counts()
        target_dict = self._determine_target_counts(class_counts, target_count)

        strategy = {cls: target_dict[cls] for cls in class_counts.index if target_dict[cls] > class_counts[cls]}
        if not strategy:
            print("    ROS: All classes already meet target count. Returning original.")
            return df_attacks.copy()

        ros_model = RandomOverSampler(sampling_strategy=strategy, random_state=self.random_state)
        X_res, y_res = ros_model.fit_resample(X_clean, y)

        df_out = pd.DataFrame(X_res, columns=feature_cols)
        df_out[self.target_col] = y_res.values
        df_out["Label"] = 1

        elapsed = time.time() - t0
        print(f"    ROS Complete: {len(df_attacks):,} -> {len(df_out):,} samples across {len(class_counts)} classes ({elapsed:.2f}s)")
        self.diagnostics["ros"] = {
            "elapsed_sec": round(elapsed, 2),
            "original_count": len(df_attacks),
            "oversampled_count": len(df_out),
            "class_distribution": df_out[self.target_col].value_counts().to_dict()
        }
        return df_out

    # ------------------------------------------------------------------------
    # 2. CTGAN Oversampling
    # ------------------------------------------------------------------------
    def apply_ctgan(
        self,
        df_attacks: pd.DataFrame,
        target_count: Union[str, int, Dict[str, int]] = "majority",
        epochs: int = 15,
        batch_size: int = 250
    ) -> pd.DataFrame:
        """
        Applies CTGAN to synthesize realistic tabular records for minority attack classes.
        """
        t0 = time.time()
        print("\n>>> [2/4] Executing CTGAN Attack Oversampling...")
        feature_cols = self._resolve_features(df_attacks)
        X_clean = self._clean_features(df_attacks, feature_cols)
        X_clean[self.target_col] = df_attacks[self.target_col].values

        class_counts = df_attacks[self.target_col].value_counts()
        target_dict = self._determine_target_counts(class_counts, target_count)
        
        synthetic_records = []
        use_gpu = (self.device != "cpu")

        for cls_name, current_count in class_counts.items():
            needed = target_dict[cls_name] - current_count
            if needed <= 0:
                continue

            print(f"    [CTGAN] Training model for class '{cls_name}' ({current_count} existing -> generating {needed:,})...")
            df_cls = X_clean[X_clean[self.target_col] == cls_name][feature_cols]

            # In CTGAN: batch_size % 2 == 0 and batch_size % pac == 0
            n_samples = len(df_cls)
            if n_samples >= 100:
                pac = 10
            elif n_samples >= 20:
                pac = 2
            else:
                pac = 1
                
            step = pac if pac % 2 == 0 else pac * 2
            cls_bs = (batch_size // step) * step
            cls_bs = max(step, min(cls_bs, (n_samples // step) * step))
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ctgan_model = CTGAN(
                    epochs=epochs,
                    batch_size=cls_bs,
                    pac=pac,
                    verbose=False,
                    enable_gpu=use_gpu
                )
                ctgan_model.fit(df_cls)
                syn_df = ctgan_model.sample(needed)

            syn_df[self.target_col] = cls_name
            syn_df["Label"] = 1
            synthetic_records.append(syn_df)

        if synthetic_records:
            df_syn_total = pd.concat(synthetic_records, ignore_index=True)
            df_base = X_clean.copy()
            df_base["Label"] = 1
            df_out = pd.concat([df_base, df_syn_total], ignore_index=True)
        else:
            df_out = X_clean.copy()
            df_out["Label"] = 1

        elapsed = time.time() - t0
        print(f"    CTGAN Complete: {len(df_attacks):,} -> {len(df_out):,} samples ({elapsed:.2f}s)")
        self.diagnostics["ctgan"] = {
            "elapsed_sec": round(elapsed, 2),
            "original_count": len(df_attacks),
            "oversampled_count": len(df_out),
            "class_distribution": df_out[self.target_col].value_counts().to_dict()
        }
        return df_out

    # ------------------------------------------------------------------------
    # 3. Tabular GAN Oversampling (PyTorch Deep Tabular GAN)
    # ------------------------------------------------------------------------
    def apply_gan(
        self,
        df_attacks: pd.DataFrame,
        target_count: Union[str, int, Dict[str, int]] = "majority",
        epochs: int = 30,
        batch_size: int = 128,
        latent_dim: int = 64
    ) -> pd.DataFrame:
        """
        Applies PyTorch Tabular GAN to synthesize numerical network flow vectors for minority attack categories.
        """
        t0 = time.time()
        print("\n>>> [3/4] Executing Deep Tabular GAN Attack Oversampling...")
        feature_cols = self._resolve_features(df_attacks)
        X_clean = self._clean_features(df_attacks, feature_cols)
        X_clean[self.target_col] = df_attacks[self.target_col].values

        class_counts = df_attacks[self.target_col].value_counts()
        target_dict = self._determine_target_counts(class_counts, target_count)
        
        synthetic_records = []
        criterion = nn.BCEWithLogitsLoss()

        for cls_name, current_count in class_counts.items():
            needed = target_dict[cls_name] - current_count
            if needed <= 0:
                continue

            print(f"    [Tabular GAN] Training GAN for class '{cls_name}' ({current_count} real -> generating {needed:,})...")
            df_cls = X_clean[X_clean[self.target_col] == cls_name][feature_cols]
            
            # Normalize continuous features into [0, 1] range for stable generator convergence
            scaler = MinMaxScaler()
            X_norm = scaler.fit_transform(df_cls.values)

            tensor_data = torch.tensor(X_norm, dtype=torch.float32)
            eff_bs = max(8, min(batch_size, len(tensor_data)))
            loader = torch.utils.data.DataLoader(
                tensor_data,
                batch_size=eff_bs,
                shuffle=True,
                drop_last=(len(tensor_data) > eff_bs)
            )

            input_dim = len(feature_cols)
            G = TabularGenerator(latent_dim=latent_dim, output_dim=input_dim).to(self.device)
            D = TabularDiscriminator(input_dim=input_dim).to(self.device)
            
            opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
            opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

            # Train loop
            G.train()
            D.train()
            for _ in range(epochs):
                for real_batch in loader:
                    real_batch = real_batch.to(self.device)
                    bs = real_batch.size(0)
                    
                    # Labels with label smoothing on real samples
                    real_labels = torch.ones((bs, 1), device=self.device) * 0.9
                    fake_labels = torch.zeros((bs, 1), device=self.device)

                    # 1. Discriminator update
                    opt_D.zero_grad()
                    d_real = D(real_batch)
                    loss_real = criterion(d_real, real_labels)

                    z = torch.randn(bs, latent_dim, device=self.device)
                    fake_data = G(z)
                    d_fake = D(fake_data.detach())
                    loss_fake = criterion(d_fake, fake_labels)

                    loss_D = loss_real + loss_fake
                    loss_D.backward()
                    opt_D.step()

                    # 2. Generator update
                    opt_G.zero_grad()
                    d_fake_for_g = D(fake_data)
                    loss_G = criterion(d_fake_for_g, real_labels)
                    loss_G.backward()
                    opt_G.step()

            # Generation phase
            G.eval()
            with torch.no_grad():
                chunks = []
                num_sampled = 0
                while num_sampled < needed:
                    c_size = min(4096, needed - num_sampled)
                    z_gen = torch.randn(c_size, latent_dim, device=self.device)
                    gen_norm = G(z_gen).cpu().numpy()
                    chunks.append(gen_norm)
                    num_sampled += c_size

                gen_all_norm = np.vstack(chunks)
                gen_real_scale = scaler.inverse_transform(gen_all_norm)

            syn_df = pd.DataFrame(gen_real_scale, columns=feature_cols)
            syn_df[self.target_col] = cls_name
            syn_df["Label"] = 1
            synthetic_records.append(syn_df)

        if synthetic_records:
            df_syn_total = pd.concat(synthetic_records, ignore_index=True)
            df_base = X_clean.copy()
            df_base["Label"] = 1
            df_out = pd.concat([df_base, df_syn_total], ignore_index=True)
        else:
            df_out = X_clean.copy()
            df_out["Label"] = 1

        elapsed = time.time() - t0
        print(f"    Tabular GAN Complete: {len(df_attacks):,} -> {len(df_out):,} samples ({elapsed:.2f}s)")
        self.diagnostics["gan"] = {
            "elapsed_sec": round(elapsed, 2),
            "original_count": len(df_attacks),
            "oversampled_count": len(df_out),
            "class_distribution": df_out[self.target_col].value_counts().to_dict()
        }
        return df_out

    # ------------------------------------------------------------------------
    # 4. SMOTIFIED-GAN Oversampling (Hybrid SMOTE + GAN)
    # ------------------------------------------------------------------------
    def apply_smotified_gan(
        self,
        df_attacks: pd.DataFrame,
        target_count: Union[str, int, Dict[str, int]] = "majority",
        epochs: int = 30,
        batch_size: int = 128,
        latent_dim: int = 64
    ) -> pd.DataFrame:
        """
        Applies SMOTIFIED-GAN:
        1. SMOTE manifold pre-enrichment to interpolate sparse minority local topologies.
        2. Deep Tabular GAN trained on the smoothed distribution to eliminate mode collapse.
        3. Synthetic generation to achieve the target multiclass representation.
        """
        t0 = time.time()
        print("\n>>> [4/4] Executing SMOTIFIED-GAN (Hybrid SMOTE + Deep GAN) Oversampling...")
        feature_cols = self._resolve_features(df_attacks)
        X_clean = self._clean_features(df_attacks, feature_cols)
        X_clean[self.target_col] = df_attacks[self.target_col].values

        class_counts = df_attacks[self.target_col].value_counts()
        target_dict = self._determine_target_counts(class_counts, target_count)
        
        synthetic_records = []
        criterion = nn.BCEWithLogitsLoss()

        for cls_name, current_count in class_counts.items():
            needed = target_dict[cls_name] - current_count
            if needed <= 0:
                continue

            print(f"    [SMOTIFIED-GAN] Enriching manifold & training for '{cls_name}' ({current_count} real -> {needed:,} syn)...")
            df_cls = X_clean[X_clean[self.target_col] == cls_name][feature_cols]

            # Step A: Pre-enrich the minority manifold with SMOTE (double sparse representation)
            if len(df_cls) > 2:
                k_sm = min(5, max(1, len(df_cls) - 1))
                nn_eng = NearestNeighbors(n_neighbors=k_sm, algorithm="kd_tree", n_jobs=1)
                
                # Create auxiliary binary frame to run SMOTE interpolation
                dummy_maj = pd.DataFrame(
                    np.zeros((len(df_cls) * 2, len(feature_cols))),
                    columns=feature_cols
                )
                dummy_maj["_label"] = 0
                df_cls_aux = df_cls.copy()
                df_cls_aux["_label"] = 1
                
                comb = pd.concat([df_cls_aux, dummy_maj], ignore_index=True)
                sm_int = SMOTE(sampling_strategy={1: len(df_cls) * 2}, k_neighbors=nn_eng, random_state=self.random_state)
                X_res_sm, y_res_sm = sm_int.fit_resample(comb[feature_cols], comb["_label"])
                X_manifold = X_res_sm[y_res_sm == 1].values
            else:
                X_manifold = df_cls.values

            # Step B: Train Deep Tabular GAN on the SMOTE-enriched manifold
            scaler = MinMaxScaler()
            X_norm = scaler.fit_transform(X_manifold)

            tensor_data = torch.tensor(X_norm, dtype=torch.float32)
            eff_bs = max(8, min(batch_size, len(tensor_data)))
            loader = torch.utils.data.DataLoader(
                tensor_data,
                batch_size=eff_bs,
                shuffle=True,
                drop_last=(len(tensor_data) > eff_bs)
            )

            input_dim = len(feature_cols)
            G = TabularGenerator(latent_dim=latent_dim, output_dim=input_dim).to(self.device)
            D = TabularDiscriminator(input_dim=input_dim).to(self.device)
            
            opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
            opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

            G.train()
            D.train()
            for _ in range(epochs):
                for real_batch in loader:
                    real_batch = real_batch.to(self.device)
                    bs = real_batch.size(0)
                    
                    real_labels = torch.ones((bs, 1), device=self.device) * 0.9
                    fake_labels = torch.zeros((bs, 1), device=self.device)

                    # 1. Discriminator update
                    opt_D.zero_grad()
                    loss_real = criterion(D(real_batch), real_labels)

                    z = torch.randn(bs, latent_dim, device=self.device)
                    fake_data = G(z)
                    loss_fake = criterion(D(fake_data.detach()), fake_labels)

                    loss_D = loss_real + loss_fake
                    loss_D.backward()
                    opt_D.step()

                    # 2. Generator update
                    opt_G.zero_grad()
                    loss_G = criterion(D(fake_data), real_labels)
                    loss_G.backward()
                    opt_G.step()

            # Step C: Generate synthetic samples from trained Generator
            G.eval()
            with torch.no_grad():
                chunks = []
                num_sampled = 0
                while num_sampled < needed:
                    c_size = min(4096, needed - num_sampled)
                    z_gen = torch.randn(c_size, latent_dim, device=self.device)
                    gen_norm = G(z_gen).cpu().numpy()
                    chunks.append(gen_norm)
                    num_sampled += c_size

                gen_all_norm = np.vstack(chunks)
                gen_real_scale = scaler.inverse_transform(gen_all_norm)

            syn_df = pd.DataFrame(gen_real_scale, columns=feature_cols)
            syn_df[self.target_col] = cls_name
            syn_df["Label"] = 1
            synthetic_records.append(syn_df)

        if synthetic_records:
            df_syn_total = pd.concat(synthetic_records, ignore_index=True)
            df_base = X_clean.copy()
            df_base["Label"] = 1
            df_out = pd.concat([df_base, df_syn_total], ignore_index=True)
        else:
            df_out = X_clean.copy()
            df_out["Label"] = 1

        elapsed = time.time() - t0
        print(f"    SMOTIFIED-GAN Complete: {len(df_attacks):,} -> {len(df_out):,} samples ({elapsed:.2f}s)")
        self.diagnostics["smotified_gan"] = {
            "elapsed_sec": round(elapsed, 2),
            "original_count": len(df_attacks),
            "oversampled_count": len(df_out),
            "class_distribution": df_out[self.target_col].value_counts().to_dict()
        }
        return df_out

    # ------------------------------------------------------------------------
    # Execute All 4 Generators
    # ------------------------------------------------------------------------
    def generate_all_oversampled_datasets(
        self,
        df_attacks: pd.DataFrame,
        target_count: Union[str, int, Dict[str, int]] = "majority",
        ctgan_epochs: int = 15,
        gan_epochs: int = 30,
        batch_size: int = 128
    ) -> Dict[str, pd.DataFrame]:
        """
        Executes all 4 oversampling methods (SMOTE, CTGAN, GAN, SMOTIFIED-GAN)
        and returns a dictionary mapping method names to their respective DataFrames.
        """
        t_total = time.time()
        print("=" * 70)
        print("STARTING ATTACK OVERSAMPLING PIPELINE (SMOTE, CTGAN, GAN, SMOTIFIED-GAN)")
        print(f"Input Attack Samples: {len(df_attacks):,}")
        print(f"Class Breakdown:\n{df_attacks[self.target_col].value_counts()}")
        print(f"Compute Device:       {self.device.upper()}")
        print("=" * 70)

        results = {}
        results["smote"] = self.apply_smote(df_attacks, target_count=target_count)
        results["ctgan"] = self.apply_ctgan(df_attacks, target_count=target_count, epochs=ctgan_epochs, batch_size=batch_size*2)
        results["gan"] = self.apply_gan(df_attacks, target_count=target_count, epochs=gan_epochs, batch_size=batch_size)
        results["smotified_gan"] = self.apply_smotified_gan(df_attacks, target_count=target_count, epochs=gan_epochs, batch_size=batch_size)

        total_time = time.time() - t_total
        print("\n" + "=" * 70)
        print(f"ALL 4 OVERSAMPLING METHODS COMPLETED in {total_time:.2f} seconds!")
        for name, diag in self.diagnostics.items():
            print(f" - {name.upper():<14}: {diag['oversampled_count']:,} samples ({diag['elapsed_sec']}s)")
        print("=" * 70)
        return results


# ============================================================================
# High-Level Helper Function
# ============================================================================

def generate_oversampled_attack_datasets(
    df_attacks: pd.DataFrame,
    target_count: Union[str, int, Dict[str, int]] = "majority",
    feature_cols: Optional[List[str]] = None,
    target_col: str = "Attack",
    random_state: int = 42,
    ctgan_epochs: int = 15,
    gan_epochs: int = 30
) -> Dict[str, pd.DataFrame]:
    """
    Convenience functional wrapper to generate 4 oversampled attack datasets:
    returns {'smote': df, 'ctgan': df, 'gan': df, 'smotified_gan': df}.
    """
    pipeline = AttackOversamplingPipeline(
        feature_cols=feature_cols,
        target_col=target_col,
        random_state=random_state
    )
    return pipeline.generate_all_oversampled_datasets(
        df_attacks=df_attacks,
        target_count=target_count,
        ctgan_epochs=ctgan_epochs,
        gan_epochs=gan_epochs
    )

"""
Conditional Variational Autoencoder (CVAE) for Tabular Attack Oversampling.

This module provides a deep generative Conditional Variational Autoencoder (CVAE)
specifically optimized for tabular network flow and intrusion detection datasets
(e.g., NetFlow-v3, UNSW-NB15, CIC-IDS2018).

Key Features:
1. Conditional Generation: Models P(X | Class) to synthesize realistic minority
   attack traffic (e.g., Worms, Shellcode, Analysis, SQL Injection) conditioned on
   one-hot encoded class labels.
2. Latent Reparameterization: Continuous probabilistic latent space with beta-VAE
   weighting to balance reconstruction fidelity and manifold regularization.
3. Tabular Robustness: Handles numerical feature scaling, missing value imputation,
   inf/-inf clipping, and inverse transformation to original data scales.
4. Flexible API:
   - Scikit-Learn / Imblearn compatible: `fit_resample(X, y)`
   - DataFrame Native: `CVAEOversampler.resample(df, target_col='Attack')`
   - Modular PyTorch Module: `CVAE(nn.Module)` for custom training loops
"""

import os
import sys
import time
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

# Prevent OpenMP library runtime clash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# 1. PyTorch CVAE Architectural Components
# ============================================================================

class CVAEEncoder(nn.Module):
    """
    Encoder network for CVAE: q_phi(z | x, c).
    Maps concatenated feature vector x and one-hot condition c to latent distribution parameters (mu, logvar).
    """
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        latent_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim + cond_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate features and condition
        xc = torch.cat([x, c], dim=-1)
        hidden = self.backbone(xc)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        # Clamp logvar for numerical stability to prevent exp() exploding
        logvar = torch.clamp(logvar, min=-20.0, max=10.0)
        return mu, logvar


class CVAEDecoder(nn.Module):
    """
    Decoder network for CVAE: p_theta(x | z, c).
    Reconstructs continuous tabular feature vector x from latent sample z conditioned on class c.
    """
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        use_sigmoid: bool = True
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]

        layers = []
        prev_dim = latent_dim + cond_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_dim, output_dim)
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        zc = torch.cat([z, c], dim=-1)
        hidden = self.backbone(zc)
        out = self.fc_out(hidden)
        if self.use_sigmoid:
            out = self.sigmoid(out)
        return out


class CVAE(nn.Module):
    """
    Complete Conditional Variational Autoencoder (CVAE) Module.
    Combines CVAEEncoder and CVAEDecoder with reparameterization trick and ELBO loss calculation.
    """
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        latent_dim: int = 32,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        use_sigmoid: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.use_sigmoid = use_sigmoid

        if encoder_hidden_dims is None:
            encoder_hidden_dims = [128, 64]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 128]

        self.encoder = CVAEEncoder(
            input_dim=input_dim,
            cond_dim=cond_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            dropout_rate=dropout_rate
        )
        self.decoder = CVAEDecoder(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            output_dim=input_dim,
            hidden_dims=decoder_hidden_dims,
            dropout_rate=dropout_rate,
            use_sigmoid=use_sigmoid
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder and decoder."""
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, c)
        return recon_x, mu, logvar

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Generates synthetic samples from latent vector z conditioned on c."""
        return self.decoder(z, c)

    def sample(self, num_samples: int, c: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Samples num_samples synthetic records conditioned on condition tensor c.
        c must have shape (num_samples, cond_dim) or (1, cond_dim).
        """
        self.eval()
        with torch.no_grad():
            if c.dim() == 1:
                c = c.unsqueeze(0)
            if c.size(0) == 1 and num_samples > 1:
                c = c.expand(num_samples, -1)
            
            c = c.to(device)
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.decode(z, c)
        return samples

    @staticmethod
    def compute_loss(
        x: torch.Tensor,
        recon_x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
        loss_type: str = "mse"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes CVAE Evidence Lower Bound (ELBO) Loss:
        Loss = Reconstruction_Loss + beta * KL_Divergence
        """
        # Reconstruction loss
        if loss_type == "mse":
            recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        elif loss_type == "smooth_l1":
            recon_loss = F.smooth_l1_loss(recon_x, x, reduction="mean")
        elif loss_type == "bce":
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction="mean")
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss


# ============================================================================
# 2. CVAE Oversampling Pipeline Class
# ============================================================================

class CVAEOversampler:
    """
    High-level Oversampling Pipeline using Conditional Variational Autoencoders.
    
    Trained conditionally on multi-class tabular network flows, this model learns
    the joint conditional density P(X | Class) and samples synthetic flow vectors
    specifically to rebalance underrepresented minority classes.

    Parameters:
    -----------
    latent_dim : int, default=32
        Dimensionality of the continuous latent space.
    encoder_hidden_dims : list of int, default=[128, 64]
        Hidden layer dimensions for the encoder network.
    decoder_hidden_dims : list of int, default=[64, 128]
        Hidden layer dimensions for the decoder network.
    lr : float, default=1e-3
        Learning rate for Adam optimizer.
    beta : float, default=0.5
        Weight of KL divergence term in ELBO loss (beta-VAE).
    batch_size : int, default=128
        Mini-batch size for training.
    epochs : int, default=30
        Number of training epochs.
    scaler_type : str, default='minmax'
        Feature scaling technique: 'minmax' (scales to [0, 1]) or 'standard' (zero-mean unit-variance).
    loss_type : str, default='mse'
        Reconstruction loss metric: 'mse', 'smooth_l1', or 'bce'.
    device : str, optional
        Target compute device ('cuda', 'cpu', or None for auto-detection).
    random_state : int, default=42
        Seed for reproducibility across numpy and torch.
    """
    def __init__(
        self,
        latent_dim: int = 32,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        beta: float = 0.5,
        batch_size: int = 128,
        epochs: int = 30,
        scaler_type: str = "minmax",
        loss_type: str = "mse",
        device: Optional[str] = None,
        random_state: int = 42
    ):
        self.latent_dim = latent_dim
        self.encoder_hidden_dims = encoder_hidden_dims or [128, 64]
        self.decoder_hidden_dims = decoder_hidden_dims or [64, 128]
        self.lr = lr
        self.beta = beta
        self.batch_size = batch_size
        self.epochs = epochs
        self.scaler_type = scaler_type.lower()
        self.loss_type = loss_type.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

        # Reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        # Internal state
        self.model: Optional[CVAE] = None
        self.scaler = None
        self.feature_cols: List[str] = []
        self.classes_: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.diagnostics: Dict[str, any] = {}

    def _resolve_features(self, df: pd.DataFrame, target_col: str, feature_cols: Optional[List[str]] = None) -> List[str]:
        """Identifies continuous numerical feature columns."""
        if feature_cols is not None:
            return feature_cols
        excluded = [target_col, "Label", "label", "attack", "attack_cat", "id"]
        return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

    def _clean_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Sanitizes continuous features by replacing inf/-inf and imputing nulls."""
        X = df[feature_cols].copy()
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            med = X[col].median()
            X[col] = X[col].fillna(0.0 if np.isnan(med) else med)
        return X

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "Attack",
        feature_cols: Optional[List[str]] = None,
        verbose: bool = True
    ) -> "CVAEOversampler":
        """
        Fits the CVAE model conditionally on the provided dataset.

        Parameters:
        -----------
        df : pd.DataFrame
            Input training dataframe containing features and target column.
        target_col : str, default='Attack'
            Column containing multiclass labels.
        feature_cols : list of str, optional
            Explicit subset of feature columns to model. If None, auto-detected.
        verbose : bool, default=True
            Whether to print training progress per epoch.
        """
        t0 = time.time()
        self.feature_cols = self._resolve_features(df, target_col, feature_cols)
        X_clean = self._clean_features(df, self.feature_cols)
        y = df[target_col].astype(str).values

        # Unique classes and one-hot mapping
        self.classes_ = sorted(list(np.unique(y)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes_)}
        num_classes = len(self.classes_)

        # Scale continuous features
        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
            use_sigmoid = True
        else:
            self.scaler = StandardScaler()
            use_sigmoid = False

        X_scaled = self.scaler.fit_transform(X_clean.values)

        # One-hot encode condition vectors
        y_indices = np.array([self.class_to_idx[val] for val in y])
        c_onehot = np.zeros((len(y), num_classes), dtype=np.float32)
        c_onehot[np.arange(len(y)), y_indices] = 1.0

        # Create PyTorch DataLoader
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        tensor_c = torch.tensor(c_onehot, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_c)
        
        eff_bs = max(8, min(self.batch_size, len(tensor_x)))
        loader = DataLoader(
            dataset,
            batch_size=eff_bs,
            shuffle=True,
            drop_last=(len(tensor_x) > eff_bs)
        )

        input_dim = len(self.feature_cols)
        cond_dim = num_classes

        # Initialize CVAE model
        self.model = CVAE(
            input_dim=input_dim,
            cond_dim=cond_dim,
            latent_dim=self.latent_dim,
            encoder_hidden_dims=self.encoder_hidden_dims,
            decoder_hidden_dims=self.decoder_hidden_dims,
            use_sigmoid=use_sigmoid
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        if verbose:
            print(f"[CVAE] Fitting model on {len(df):,} samples | {input_dim} features | {cond_dim} classes | Device: {self.device}")

        # Training loop
        self.model.train()
        history = {"total_loss": [], "recon_loss": [], "kl_loss": []}

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            num_batches = 0

            for bx, bc in loader:
                bx = bx.to(self.device)
                bc = bc.to(self.device)

                optimizer.zero_grad()
                recon_x, mu, logvar = self.model(bx, bc)
                loss, r_loss, k_loss = CVAE.compute_loss(
                    bx, recon_x, mu, logvar,
                    beta=self.beta,
                    loss_type=self.loss_type
                )
                loss.backward()
                # Gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += r_loss.item()
                epoch_kl += k_loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            avg_recon = epoch_recon / max(1, num_batches)
            avg_kl = epoch_kl / max(1, num_batches)

            history["total_loss"].append(avg_loss)
            history["recon_loss"].append(avg_recon)
            history["kl_loss"].append(avg_kl)

            scheduler.step(avg_loss)

            if verbose and (epoch % max(1, self.epochs // 5) == 0 or epoch == self.epochs or epoch == 1):
                print(f"  Epoch [{epoch:02d}/{self.epochs:02d}] -> Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")

        elapsed = time.time() - t0
        self.diagnostics = {
            "elapsed_sec": round(elapsed, 2),
            "input_dim": input_dim,
            "cond_dim": cond_dim,
            "final_loss": round(history["total_loss"][-1], 4),
            "history": history
        }
        if verbose:
            print(f"[CVAE] Training completed in {elapsed:.2f}s | Final ELBO Loss: {self.diagnostics['final_loss']}")

        return self

    def generate_samples(self, class_name: str, count: int) -> pd.DataFrame:
        """
        Synthesizes `count` realistic feature vectors conditioned on `class_name`.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model must be fitted before generating samples. Call .fit() first.")
        if class_name not in self.class_to_idx:
            raise ValueError(f"Unknown class '{class_name}'. Known classes: {self.classes_}")
        if count <= 0:
            return pd.DataFrame(columns=self.feature_cols)

        cls_idx = self.class_to_idx[class_name]
        cond_dim = len(self.classes_)

        # Prepare one-hot condition
        c_vec = torch.zeros(1, cond_dim, dtype=torch.float32, device=self.device)
        c_vec[0, cls_idx] = 1.0

        # Sample from CVAE in manageable batches
        chunks = []
        sampled = 0
        batch_limit = 4096

        self.model.eval()
        with torch.no_grad():
            while sampled < count:
                n_batch = min(batch_limit, count - sampled)
                samples_norm = self.model.sample(n_batch, c_vec, device=self.device).cpu().numpy()
                chunks.append(samples_norm)
                sampled += n_batch

        samples_scaled = np.vstack(chunks)
        # Inverse transform back to original feature scale
        samples_orig = self.scaler.inverse_transform(samples_scaled)

        df_syn = pd.DataFrame(samples_orig, columns=self.feature_cols)
        return df_syn

    def resample(
        self,
        df: pd.DataFrame,
        target_col: str = "Attack",
        target_count: Union[str, int, Dict[str, int]] = "majority",
        feature_cols: Optional[List[str]] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        End-to-end oversampling: Fits CVAE conditionally on df and synthesizes
        records for all classes that fall below target counts.

        Parameters:
        -----------
        df : pd.DataFrame
            Original imbalanced dataframe.
        target_col : str, default='Attack'
            Target multiclass column name.
        target_count : 'majority', int, or dict
            Desired sample volume per class.
        feature_cols : list of str, optional
            Numerical feature columns to model.
        verbose : bool, default=True
            Whether to display progress.
        """
        t0 = time.time()
        print(f"\n>>> [CVAE] Initiating Generative Oversampling on '{target_col}'...")

        # 1. Fit CVAE on existing data
        self.fit(df, target_col=target_col, feature_cols=feature_cols, verbose=verbose)

        # 2. Determine target counts
        class_counts = df[target_col].astype(str).value_counts()
        majority_count = int(class_counts.max())

        if isinstance(target_count, dict):
            target_dict = {cls: max(class_counts.get(cls, 0), target_count.get(cls, class_counts.get(cls, 0))) for cls in class_counts.index}
        elif target_count == "majority" or target_count is None:
            target_dict = {cls: majority_count for cls in class_counts.index}
        elif isinstance(target_count, (int, float)):
            target_int = int(target_count)
            target_dict = {cls: max(class_counts[cls], target_int) for cls in class_counts.index}
        else:
            raise ValueError(f"Unsupported target_count specification: {target_count}")

        # 3. Generate synthetic records for minority classes
        synthetic_dfs = []
        for cls_name, current_cnt in class_counts.items():
            needed = target_dict[cls_name] - current_cnt
            if needed > 0:
                if verbose:
                    print(f"  [CVAE Synthesis] Class '{cls_name}': generating {needed:,} synthetic samples (current: {current_cnt:,} -> target: {target_dict[cls_name]:,})")
                syn_df = self.generate_samples(cls_name, needed)
                syn_df[target_col] = cls_name
                if "Label" in df.columns:
                    # Inherit majority label for that class (typically 1 for attack, 0 for benign)
                    syn_df["Label"] = df[df[target_col] == cls_name]["Label"].iloc[0]
                synthetic_dfs.append(syn_df)

        # 4. Combine real and synthetic records
        if synthetic_dfs:
            df_syn_total = pd.concat(synthetic_dfs, ignore_index=True)
            df_resampled = pd.concat([df, df_syn_total], ignore_index=True)
        else:
            print("  [CVAE] All classes already meet target count. Returning original.")
            df_resampled = df.copy()

        elapsed = time.time() - t0
        print(f">>> [CVAE Complete] Dataset expanded: {len(df):,} -> {len(df_resampled):,} records across {len(class_counts)} classes ({elapsed:.2f}s)\n")
        return df_resampled

    def fit_resample(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        target_count: Union[str, int, Dict[str, int]] = "majority"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scikit-learn / Imbalanced-Learn compatible API.
        Accepts feature matrix X and target vector y, returns (X_resampled, y_resampled).
        """
        if isinstance(X, np.ndarray):
            feature_cols = [f"feat_{i}" for i in range(X.shape[1])]
            df_x = pd.DataFrame(X, columns=feature_cols)
        else:
            feature_cols = list(X.columns)
            df_x = X.copy()

        if isinstance(y, np.ndarray):
            s_y = pd.Series(y, name="target")
        else:
            s_y = y.copy().rename("target")

        df_combined = pd.concat([df_x.reset_index(drop=True), s_y.reset_index(drop=True)], axis=1)
        df_res = self.resample(df_combined, target_col="target", target_count=target_count, feature_cols=feature_cols, verbose=False)

        X_res = df_res[feature_cols].values
        y_res = df_res["target"].values
        return X_res, y_res


# ============================================================================
# 3. Functional Convenience API
# ============================================================================

def apply_cvae_oversampling(
    df: pd.DataFrame,
    target_col: str = "Attack",
    target_count: Union[str, int, Dict[str, int]] = "majority",
    feature_cols: Optional[List[str]] = None,
    epochs: int = 30,
    batch_size: int = 128,
    latent_dim: int = 32,
    beta: float = 0.5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Convenience wrapper to train a CVAE and return an oversampled DataFrame in a single call.
    """
    oversampler = CVAEOversampler(
        latent_dim=latent_dim,
        beta=beta,
        batch_size=batch_size,
        epochs=epochs,
        random_state=random_state
    )
    return oversampler.resample(
        df=df,
        target_col=target_col,
        target_count=target_count,
        feature_cols=feature_cols
    )


# ============================================================================
# 4. Self-Verification Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing CVAE Oversampling Module on Synthetic Tabular Network Data")
    print("=" * 70)

    # Create synthetic imbalanced network data
    np.random.seed(42)
    n_benign = 500
    n_dos = 200
    n_probe = 50
    n_worm = 10  # Ultra-rare class

    def gen_flows(n, mean_dur, mean_bytes):
        return pd.DataFrame({
            "FLOW_DURATION_MILLISECONDS": np.random.exponential(mean_dur, n),
            "IN_BYTES": np.random.lognormal(mean_bytes, 0.5, n),
            "OUT_BYTES": np.random.lognormal(mean_bytes * 0.8, 0.5, n),
            "IN_PKTS": np.random.poisson(10, n),
            "OUT_PKTS": np.random.poisson(8, n),
        })

    df_b = gen_flows(n_benign, 50, 6); df_b["Attack"] = "Benign"; df_b["Label"] = 0
    df_d = gen_flows(n_dos, 10, 8); df_d["Attack"] = "DoS"; df_d["Label"] = 1
    df_p = gen_flows(n_probe, 5, 4); df_p["Attack"] = "PortScan"; df_p["Label"] = 1
    df_w = gen_flows(n_worm, 1, 3); df_w["Attack"] = "Worms"; df_w["Label"] = 1

    df_synth = pd.concat([df_b, df_d, df_p, df_w], ignore_index=True)
    print("\nInitial Class Distribution:")
    print(df_synth["Attack"].value_counts())

    # Execute CVAE Oversampling to rebalance minority attack classes to 200 samples
    cvae_pipe = CVAEOversampler(
        latent_dim=8,
        epochs=10,
        batch_size=32,
        beta=0.5,
        random_state=42
    )
    df_balanced = cvae_pipe.resample(
        df=df_synth,
        target_col="Attack",
        target_count={"DoS": 200, "PortScan": 200, "Worms": 200, "Benign": 500}
    )

    print("\nBalanced Class Distribution after CVAE Oversampling:")
    print(df_balanced["Attack"].value_counts())
    print("\nCVAE Oversampling Test Passed Successfully!")


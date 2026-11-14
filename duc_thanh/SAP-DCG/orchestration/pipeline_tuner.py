#!/usr/bin/env python3
"""
Unified Pipeline Tuner for Network Intrusion Detection.

Derived from Minh's CTGAN tuning architecture (`minh/src/ctgan_pipeline.py`, `minh/src/fix_pipeline.py`,
`minh/notebooks/07_tune_ctgan.ipynb`) and Tuan's IDS model & quality evaluation (`tuan/ML-IDS/ids_pipeline.py`,
`tuan/ML-IDS/ml_ids_detect.py`):
  - Round 1 (Comp 1): Tuning SMOTE-NC multiplication factor (y_max).
  - Round 2 (Comp 2): Tuning Deep Generative Models (CTGAN / TVAE) via Optuna
    with Minh's exact search space (DIMS, pac, lr, steps, emb_dim) and dual evaluation
    (downstream IDS metric + label consistency & TSTR).
  - Round 3 (Comp 3): Tuning OOF Confidence Filter Threshold & ENN.
  - Final Round (Comp 4): Tuning Final IDS Model (Two-Tier XGBoost) Hyperparameters
    with Tuan's search space (weight_power, min_confidence, max_fpr, trees, depth, lr).
  - Full Sequential Pipeline: Coordinates multi-round execution with epsilon stopping criteria.
  - 5-Pillar Synthetic Quality Analysis: Coverage, Validity, Fidelity, Privacy, Utility.
"""

import os
import sys
import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import f1_score

try:
    import optuna
except ImportError:
    optuna = None

curr_dir = Path(__file__).resolve().parent
pkg_root = curr_dir.parent
comp_dir = pkg_root / "components"
for p in [curr_dir, pkg_root, comp_dir]:
    if p.exists() and str(p) not in sys.path:
        sys.path.append(str(p))

try:
    from components.intra_attack_resampler import resolve_paths, SHORTCUT_COLS, CATEGORICAL_FEATURES, IntraAttackResampler
    from components.tabular_generators import TabularGenerators, CTGAN_DIMS, TVAE_DIMS, TVAE_DEFAULT_PARAMS
    from components.synthetic_sample_filter import SyntheticSampleFilter
    from components.final_ids_model import FinalIDSModel, evaluate_synthetic_quality
    from components.dataset_splitter import split_benign_attack
except ImportError:
    from intra_attack_resampler import resolve_paths, SHORTCUT_COLS, CATEGORICAL_FEATURES, IntraAttackResampler
    from tabular_generators import TabularGenerators, CTGAN_DIMS, TVAE_DIMS, TVAE_DEFAULT_PARAMS
    from synthetic_sample_filter import SyntheticSampleFilter
    from final_ids_model import FinalIDSModel, evaluate_synthetic_quality
    try:
        from dataset_splitter import split_benign_attack
    except ImportError:
        def split_benign_attack(df, **kwargs):
            return df[df["Label"] == 0].copy().reset_index(drop=True), df[df["Label"] == 1].copy().reset_index(drop=True)


class PipelineTuner:
    """
    Unified Tuner consolidating hyperparameter optimization across all 4 pipeline components.
    Integrates Minh's CTGAN search space and Tuan's Two-Tier IDS tuning logic.
    """

    def __init__(
        self,
        t_max: int = 10000,
        random_state: int = 42,
        scaler_type: str = "minmax",
        metric_key: str = "macro_f1",
        epsilon: float = 0.002,
        final_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.t_max = t_max
        self.random_state = random_state
        self.scaler_type = scaler_type
        self.metric_key = metric_key
        self.epsilon = epsilon

        if final_dir is not None:
            self.final_dir = Path(final_dir).resolve()
            self.base_dir = self.final_dir.parent.parent
        else:
            self.base_dir, _, self.final_dir = resolve_paths()

        if output_dir is not None:
            self.output_dir = Path(output_dir).resolve()
        else:
            self.output_dir = self.final_dir / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tuning tracking states
        self.best_y_max: int = 25
        self.best_generator_params: Dict[str, Any] = {}
        self.best_filter_threshold: float = 0.6
        self.best_ids_params: Dict[str, Any] = {}
        self.tuning_history: Dict[str, Any] = {}

    # ========================================================================
    # Plotting Helpers for Tuning Curves
    # ========================================================================
    def _plot_tuning_curve(
        self,
        x_vals: List[Any],
        y_vals: List[float],
        best_x: Any,
        best_y: float,
        xlabel: str,
        ylabel: str,
        title: str,
        output_filename: str,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        Generates and exports publication-ready tuning curve charts with annotations.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"Warning: Matplotlib plot could not be created ({e}).")
            return None

        out_d = Path(output_dir) if output_dir else self.output_dir
        out_d.mkdir(parents=True, exist_ok=True)
        save_path = out_d / output_filename

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.plot(x_vals, y_vals, marker="o", color="#1f77b4", linewidth=2.0, markersize=7, label="Candidate Evaluation")

        # Highlight best point
        ax.scatter([best_x], [best_y], color="#d62728", s=130, zorder=5, label=f"Best: {best_x} ({best_y:.4f})")

        # Annotate points
        for x, y in zip(x_vals, y_vals):
            ax.annotate(
                f"{y:.4f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=9,
                color="#333333",
            )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, fontsize=10, fontweight="semibold")
        ax.set_ylabel(ylabel, fontsize=10, fontweight="semibold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True, loc="best")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

        print(f"  [Artifact Export] Saved Tuning Plot: {save_path.name}")
        return save_path

    def plot_combined_tuning_summary(
        self,
        r1_history: List[Dict[str, Any]],
        r3_history: List[Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """
        Generates a consolidated two-panel chart comparing Round 1 (y_max) and Round 3 (OOF Threshold) tuning.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"Warning: Matplotlib plot could not be created ({e}).")
            return None

        out_d = Path(output_dir) if output_dir else self.output_dir
        out_d.mkdir(parents=True, exist_ok=True)
        save_path = out_d / "tuning_curves_summary.png"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)

        # Panel 1: Round 1 (y_max)
        if r1_history:
            x1 = [h["y_max"] for h in r1_history]
            y1 = [h[self.metric_key] for h in r1_history]
            best_idx1 = int(np.argmax(y1))
            ax1.plot(x1, y1, marker="o", color="#1f77b4", linewidth=2.0, markersize=6)
            ax1.scatter([x1[best_idx1]], [y1[best_idx1]], color="#d62728", s=120, zorder=5, label=f"Best y_max={x1[best_idx1]} ({y1[best_idx1]:.4f})")
            for x, y in zip(x1, y1):
                ax1.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
            ax1.set_title("Round 1: SMOTE-NC $y_{\\max}$ Tuning", fontsize=11, fontweight="bold")
            ax1.set_xlabel("Multiplication Factor $y_{\\max}$", fontsize=10)
            ax1.set_ylabel(f"Val {self.metric_key.replace('_', ' ').title()}", fontsize=10)
            ax1.grid(True, linestyle="--", alpha=0.6)
            ax1.legend(loc="best")

        # Panel 2: Round 3 (threshold)
        if r3_history:
            x2 = [h["threshold"] for h in r3_history]
            y2 = [h[self.metric_key] for h in r3_history]
            best_idx2 = int(np.argmax(y2))
            ax2.plot(x2, y2, marker="s", color="#2ca02c", linewidth=2.0, markersize=6)
            ax2.scatter([x2[best_idx2]], [y2[best_idx2]], color="#d62728", s=120, zorder=5, label=f"Best θ={x2[best_idx2]} ({y2[best_idx2]:.4f})")
            for x, y in zip(x2, y2):
                ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
            ax2.set_title("Round 3: OOF Filter Threshold Tuning", fontsize=11, fontweight="bold")
            ax2.set_xlabel("OOF Confidence Gate Threshold $\\theta_{\\text{OOF}}$", fontsize=10)
            ax2.set_ylabel(f"Val {self.metric_key.replace('_', ' ').title()}", fontsize=10)
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.legend(loc="best")

        plt.suptitle("Sequential Pipeline Tuning Curves (Comp 1 & Comp 3)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

        print(f"  [Artifact Export] Saved Combined Tuning Summary Plot: {save_path.name}")
        return save_path

    # ========================================================================
    # 1. Round 1: Tune Intra-Attack Resampler (y_max)
    # ========================================================================
    def tune_resampler_ymax(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        ids_model: FinalIDSModel,
        candidate_ymax: Optional[List[int]] = None,
        use_optuna: bool = False,
        n_trials: int = 8,
        generator: Optional[TabularGenerators] = None,
        sample_filter: Optional[SyntheticSampleFilter] = None,
        batch_size: int = 200,
        epochs: Optional[int] = None,
        multiplier: int = 10,
        train_benign: Optional[pd.DataFrame] = None,
        train_attacks: Optional[pd.DataFrame] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Tunes the SMOTE-NC multiplication factor y_max in Component 1.
        The goal of tuning y_max is to tune the generator (CTGAN/TVAE) itself:
        For each candidate y_max:
          1. Resample train_df via SMOTE-NC -> df_train_resampled, syn_deficit.
          2. Train generator on resampled attack flows with fixed gradient updates.
          3. Generate candidate pool (multiplier * n_syn).
          4. Run candidate flows through filter and physical constraints / reverse transform.
          5. Evaluate downstream on val_df using FinalIDSModel (Macro F1).
        Runs all candidate cases without early stopping, then plots the tuning curve.
        """
        if candidate_ymax is None:
            candidate_ymax = [1, 5, 10, 15, 20, 25, 30, 40]

        if train_benign is None or train_attacks is None:
            train_benign, real_train_attacks = split_benign_attack(train_df, verbose=False)
        else:
            real_train_attacks = train_attacks

        print("=" * 80)
        print(">>> [Round 1] Tuning Component 1 (y_max) via Generator-in-the-Loop Evaluation")
        print("=" * 80)
        print(f"Candidates:       {candidate_ymax}")
        print(f"Ceiling t_max:     {self.t_max:,}")
        print(f"Generator Active:  {generator is not None}")
        print(f"Stopping Rule:    None (Evaluating all {len(candidate_ymax)} candidates)")
        print(f"Target Metric:    {self.metric_key}\n")

        history = []
        best_score = -1.0
        best_y = candidate_ymax[0]
        best_results = None
        best_candidate_pool = None
        best_clean_syn = None
        prev_score = -1.0

        def _evaluate_candidate(y_val: int, export: bool = False) -> Tuple[float, Dict[str, Any], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
            resampler = IntraAttackResampler(
                t_max=self.t_max,
                y_max=y_val,
                random_state=self.random_state,
                scaler_type=self.scaler_type,
                final_dir=self.final_dir,
            )
            out = resampler.fit_resample(train_df=train_df, export=export)

            cand_pool = None
            clean_syn = None

            if generator is not None:
                # Train generator on the SMOTE-NC pre-balanced attack data
                smote_attacks = out["df_train_resampled"][out["df_train_resampled"]["Label"] == 1].copy().reset_index(drop=True)
                feature_cols = [c for c in smote_attacks.columns if c not in ["Label", "Attack", "OOF_Confidence"]]
                cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

                generator.train(
                    train_attacks=smote_attacks,
                    syn_deficit=out["syn_deficit"],
                    feature_cols=feature_cols,
                    categorical_cols=cat_cols,
                    batch_size=batch_size,
                    epochs=epochs,
                )

                # Sample 10 * n_syn candidate pool
                cand_pool = generator.sample(syn_deficit=out["syn_deficit"], multiplier=multiplier)

                # Filter and physical constraint check
                if sample_filter is not None:
                    filter_res = sample_filter.filter_samples(
                        synthetic_df=cand_pool,
                        real_train_df=real_train_attacks,
                        syn_deficit=out["syn_deficit"],
                        oof_threshold=0.6,
                        run_enn=True,
                    )
                    clean_syn = filter_res["kept_df"]
                else:
                    clean_syn = cand_pool

                df_eval_train = pd.concat([train_benign, real_train_attacks, clean_syn], ignore_index=True)
            else:
                df_eval_train = out["df_train_resampled"]

            ids_model.train(df_eval_train, val_df=val_df)
            eval_metrics = ids_model.evaluate(val_df, split_name="val")
            val_score = eval_metrics.get(self.metric_key, 0.0)
            return val_score, out, cand_pool, clean_syn

        if use_optuna and optuna is not None:
            def objective(trial):
                y_val = trial.suggest_int("y_max", 5, 50, step=5)
                val_score, _, _, _ = _evaluate_candidate(y_val, export=False)
                return val_score

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_y = study.best_params["y_max"]
            best_score, best_results, best_candidate_pool, best_clean_syn = _evaluate_candidate(best_y, export=True)
            print(f"Optuna Best y_max: {best_y} with {self.metric_key} = {best_score:.4f}")

            self.best_y_max = best_y
            return {
                "best_y_max": self.best_y_max,
                "best_score": best_score,
                "best_results": best_results,
                "best_candidate_pool": best_candidate_pool,
                "best_clean_syn": best_clean_syn,
                "history": [{"best_y": best_y, "best_score": best_score}],
            }

        # Grid-search over all candidates (no early stopping)
        for idx, y_val in enumerate(candidate_ymax, 1):
            t_start = time.time()
            print(f"--- [Comp1 Tuning {idx}/{len(candidate_ymax)}] Testing y_max = {y_val} ---")

            score, out, cand_pool, clean_syn = _evaluate_candidate(y_val, export=False)
            elapsed = time.time() - t_start

            record = {
                "y_max": y_val,
                self.metric_key: score,
                "smote_resampled_rows": len(out["df_train_resampled"]),
                "candidate_pool_rows": len(cand_pool) if cand_pool is not None else 0,
                "filtered_syn_rows": len(clean_syn) if clean_syn is not None else 0,
                "elapsed_sec": round(elapsed, 2),
            }
            history.append(record)

            syn_info = f", Syn Kept: {len(clean_syn):,}" if clean_syn is not None else ""
            print(f"    Val {self.metric_key}: {score:.4f} (SMOTE Rows: {len(out['df_train_resampled']):,}{syn_info}) in {elapsed:.2f}s")

            if score > best_score:
                best_score = score
                best_y = y_val
                best_results = out
                best_candidate_pool = cand_pool
                best_clean_syn = clean_syn

            prev_score = score

        self.best_y_max = best_y
        print("=" * 80)
        print(f"Comp1 Tuning Finished: Best y_max = {self.best_y_max} ({self.metric_key}: {best_score:.4f})")
        print("=" * 80)

        # Generate Round 1 tuning curve plot
        plot_path = self._plot_tuning_curve(
            x_vals=[h["y_max"] for h in history],
            y_vals=[h[self.metric_key] for h in history],
            best_x=self.best_y_max,
            best_y=best_score,
            xlabel="SMOTE-NC Multiplication Factor ($y_{\\max}$)",
            ylabel=f"Validation {self.metric_key.replace('_', ' ').title()}",
            title=f"Round 1: IntraAttackResampler $y_{{\\max}}$ Tuning Curve (Best = {self.best_y_max})",
            output_filename="round1_ymax_tuning.png",
            output_dir=output_dir,
        )
        print("=" * 80 + "\n")

        return {
            "best_y_max": self.best_y_max,
            "best_score": best_score,
            "best_results": best_results,
            "best_candidate_pool": best_candidate_pool,
            "best_clean_syn": best_clean_syn,
            "history": history,
            "tuning_plot": str(plot_path) if plot_path else None,
        }

    # ========================================================================
    # 2. Round 2: Tune Tabular Generators (CTGAN / TVAE) via Minh's Space
    # ========================================================================
    def tune_generator(
        self,
        train_attacks: pd.DataFrame,
        syn_deficit: Dict[str, int],
        train_benign: pd.DataFrame,
        val_df: pd.DataFrame,
        generator: TabularGenerators,
        sample_filter: SyntheticSampleFilter,
        ids_model: FinalIDSModel,
        n_trials: int = 8,
        timeout: Optional[int] = None,
        multiplier: int = 10,
    ) -> Dict[str, Any]:
        """
        Tunes generator hyperparameters (CTGAN / TVAE) using Optuna with Minh's exact search space:
          - generator_dim & discriminator_dim from DIMS (256x256, 256x256x256, 512x512)
          - generator_lr, discriminator_lr in [1e-5, 1e-3] (log scale)
          - discriminator_steps in [1, 3, 5], log_frequency in [True, False]
          - pac in [1, 5, 10], embedding_dim in [64, 128, 256], batch_size in [100, 500, 1000]
        Evaluates each trial via downstream IDS performance on validation data,
        also tracking label consistency and TSTR macro-F1.
        """
        if optuna is None:
            raise ImportError("Optuna is required for tune_generator. Install via `pip install optuna`.")

        print("=" * 80)
        print(f">>> [Round 2] Tuning Component 2 ({generator.model_type.upper()}) via Optuna (Minh's Search Space)")
        print("=" * 80)
        print(f"Trials:       {n_trials}")
        print(f"Objective:    Maximize IDS Model {self.metric_key} on Validation Set\n")

        best_trial_score = -1.0
        best_trial_models = None

        def objective(trial):
            nonlocal best_trial_score, best_trial_models

            if generator.model_type == "ctgan":
                gen_lr = trial.suggest_float("generator_lr", 1e-5, 1e-3, log=True)
                disc_lr = trial.suggest_float("discriminator_lr", 1e-5, 1e-3, log=True)
                emb_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
                b_size = trial.suggest_categorical("batch_size", [100, 500, 1000])
                pac_val = trial.suggest_categorical("pac", [1, 5, 10])
                gen_dim_key = trial.suggest_categorical("generator_dim", list(CTGAN_DIMS.keys()))
                disc_dim_key = trial.suggest_categorical("discriminator_dim", list(CTGAN_DIMS.keys()))
                disc_steps = trial.suggest_categorical("discriminator_steps", [1, 3, 5])
                log_freq = trial.suggest_categorical("log_frequency", [True, False])
                kwargs = {
                    "generator_lr": gen_lr,
                    "discriminator_lr": disc_lr,
                    "embedding_dim": emb_dim,
                    "pac": pac_val,
                    "generator_dim": CTGAN_DIMS[gen_dim_key],
                    "discriminator_dim": CTGAN_DIMS[disc_dim_key],
                    "discriminator_steps": disc_steps,
                    "log_frequency": log_freq,
                }
            else:  # tvae (Do Minh Thanh's Optuna space from tvae_final_perclass_v2.ipynb)
                epochs_val = trial.suggest_categorical("epochs", [50, 100, 200, 300])
                b_size = trial.suggest_categorical("batch_size", [100, 250, 500, 1000])
                emb_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
                dims_val = trial.suggest_categorical("dims", list(TVAE_DIMS.keys()))
                l2scale = trial.suggest_float("l2scale", 1e-6, 1e-3, log=True)
                loss_factor = trial.suggest_categorical("loss_factor", [1, 2, 3, 5])
                pac_val = 1
                kwargs = {
                    "epochs": epochs_val,
                    "embedding_dim": emb_dim,
                    "dims": dims_val,
                    "l2scale": l2scale,
                    "loss_factor": loss_factor,
                }

            # Train generator with candidate hyperparameters
            generator.train(
                train_attacks=train_attacks,
                syn_deficit=syn_deficit,
                batch_size=b_size,
                pac=pac_val,
                unified_generator=(generator.model_type == "ctgan"),
                verbose=False,
                **kwargs,
            )

            # Sample candidate pool (10 * n_syn) via rejection/pool sampling
            df_candidate_pool = generator.sample(syn_deficit=syn_deficit, multiplier=multiplier)
            if df_candidate_pool.empty:
                return 0.0

            # Filter candidate pool using Comp3
            filter_res = sample_filter.filter_samples(
                synthetic_df=df_candidate_pool,
                real_train_df=train_attacks,
                syn_deficit=syn_deficit,
                oof_threshold=0.6,
                run_enn=True,
            )
            df_syn_clean = filter_res["kept_df"]

            # Train IDS model on augmented training set
            df_augmented_train = pd.concat([train_benign, train_attacks, df_syn_clean], ignore_index=True)
            ids_model.train(df_augmented_train, val_df=val_df)

            # Evaluate IDS model on validation set
            metrics = ids_model.evaluate(val_df, split_name="val")
            score = metrics.get(self.metric_key, 0.0)

            # Track user attributes
            trial.set_user_attr("e2e_macro_f1", metrics.get("e2e_macro_f1", score))
            trial.set_user_attr("t2_macro_f1", metrics.get("t2_macro_f1", score))
            trial.set_user_attr("dr", metrics.get("detection_rate", 0.0))
            trial.set_user_attr("far", metrics.get("false_alarm_rate", 0.0))

            if score > best_trial_score:
                best_trial_score = score
                best_trial_models = {
                    "models": copy.deepcopy(generator.models),
                    "unified_model": copy.deepcopy(generator.unified_model),
                    "preprocessor": copy.deepcopy(generator.preprocessor),
                    "tvae_preprocessor": copy.deepcopy(generator.tvae_preprocessor),
                }
                generator.save_weights("best_tuned_generator_weights.pt")

            return score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        if generator.model_type == "tvae":
            study.enqueue_trial(TVAE_DEFAULT_PARAMS)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        print("\n" + "=" * 80)
        print(f"Comp2 Optuna Tuning Finished: Best {self.metric_key} = {study.best_value:.4f}")
        print(f"Best Parameters: {study.best_params}")
        print("=" * 80 + "\n")

        self.best_generator_params = study.best_params
        if best_trial_models is not None:
            generator.models = best_trial_models["models"]
            generator.unified_model = best_trial_models["unified_model"]
            generator.preprocessor = best_trial_models["preprocessor"]
            generator.tvae_preprocessor = best_trial_models["tvae_preprocessor"]

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study,
        }

    # ========================================================================
    # 3. Round 3: Tune Synthetic Sample Filter (OOF Threshold & ENN)
    # ========================================================================
    def tune_filter_threshold(
        self,
        synthetic_pool: pd.DataFrame,
        real_train_attacks: pd.DataFrame,
        train_benign: pd.DataFrame,
        val_df: pd.DataFrame,
        sample_filter: SyntheticSampleFilter,
        ids_model: FinalIDSModel,
        syn_deficit: Dict[str, int],
        candidate_thresholds: Optional[List[float]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Tunes the OOF confidence gate threshold in Component 3.
        Evaluates each threshold via the IDS model on the validation set.
        Runs all candidate thresholds without early stopping, then plots the tuning curve.
        """
        if candidate_thresholds is None:
            candidate_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]

        print("=" * 80)
        print(">>> [Round 3] Tuning Component 3 (OOF Confidence Threshold) Based on IDS Model")
        print("=" * 80)
        print(f"Candidate Thresholds: {candidate_thresholds}")
        print(f"Stopping Rule:        None (Evaluating all {len(candidate_thresholds)} candidates)")

        best_score = -1.0
        best_thresh = candidate_thresholds[0]
        best_kept_df = pd.DataFrame()
        prev_score = -1.0
        history = []

        # Ensure OOF predictor and ENN are ready
        if not sample_filter.oof_models:
            sample_filter.fit_oof_predictor(real_train_attacks)
        if sample_filter.enn_engine is None:
            sample_filter.fit_enn(real_train_attacks)

        for idx, thresh in enumerate(candidate_thresholds, 1):
            t_start = time.time()
            print(f"--- [Comp3 Tuning {idx}/{len(candidate_thresholds)}] Testing Threshold = {thresh} ---")

            filter_res = sample_filter.filter_samples(
                synthetic_df=synthetic_pool,
                real_train_df=real_train_attacks,
                syn_deficit=syn_deficit,
                oof_threshold=thresh,
                run_enn=True,
            )
            df_syn_clean = filter_res["kept_df"]

            # Construct augmented training set
            df_train_augmented = pd.concat([train_benign, real_train_attacks, df_syn_clean], ignore_index=True)

            # Train IDS model
            ids_model.train(df_train_augmented, val_df=val_df)

            # Evaluate IDS model on val_df
            metrics = ids_model.evaluate(val_df, split_name="val")
            score = metrics.get(self.metric_key, 0.0)
            elapsed = time.time() - t_start

            record = {
                "threshold": thresh,
                self.metric_key: score,
                "macro_f1": metrics.get("macro_f1", score),
                "e2e_macro_f1": metrics.get("e2e_macro_f1", score),
                "balanced_acc": metrics.get("balanced_accuracy", 0.0),
                "kept_synthetic_flows": len(df_syn_clean),
                "elapsed_sec": round(elapsed, 2),
            }
            history.append(record)

            print(f"    Val {self.metric_key}: {score:.4f} (Kept Synthetic: {len(df_syn_clean):,}) in {elapsed:.2f}s")

            if score > best_score:
                best_score = score
                best_thresh = thresh
                best_kept_df = df_syn_clean

            prev_score = score

        self.best_filter_threshold = best_thresh
        print("\n" + "=" * 80)
        print(f"Comp3 Tuning Finished: Best Threshold = {self.best_filter_threshold} ({self.metric_key}: {best_score:.4f})")
        print("=" * 80)

        # Generate Round 3 tuning curve plot
        plot_path = self._plot_tuning_curve(
            x_vals=[h["threshold"] for h in history],
            y_vals=[h[self.metric_key] for h in history],
            best_x=self.best_filter_threshold,
            best_y=best_score,
            xlabel="OOF Confidence Gate Threshold ($\theta_{\\text{OOF}}$)",
            ylabel=f"Validation {self.metric_key.replace('_', ' ').title()}",
            title=f"Round 3: SyntheticSampleFilter Threshold Tuning Curve (Best = {self.best_filter_threshold})",
            output_filename="round3_threshold_tuning.png",
            output_dir=output_dir,
        )
        print("=" * 80 + "\n")

        return {
            "best_threshold": self.best_filter_threshold,
            "best_score": best_score,
            "best_kept_df": best_kept_df,
            "history": history,
            "tuning_plot": str(plot_path) if plot_path else None,
        }

    # ========================================================================
    # 4. Final Round: Tune Final IDS Model Hyperparameters via Tuan's Space
    # ========================================================================
    def tune_ids_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        ids_model: FinalIDSModel,
        n_trials: int = 10,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Tunes Final IDS Model (Two-Tier XGBoost) hyperparameters using Optuna with Tuan's search space:
          - n_estimators in [100, 200, 300]
          - max_depth in [4, 6, 8, 10]
          - learning_rate in [0.03, 0.08, 0.15, 0.2] (log scale)
          - subsample and colsample_bytree in [0.6, 0.8, 1.0]
          - weight_power in [0.25, 0.5, 0.75, 1.0]
          - min_confidence in [0.3, 0.4, 0.5]
          - max_fpr in [0.005, 0.01, 0.02]
        """
        print("=" * 80)
        print(">>> [Final Round] Tuning Component 4 (Final IDS Model Hyperparameters - Tuan's Space)")
        print("=" * 80)
        print(f"Trials:    {n_trials}")
        print(f"Objective: Maximize {self.metric_key} on Validation Set\n")

        best_score = -1.0
        best_params = copy.deepcopy(ids_model.params)
        best_cfg = {
            "weight_power": ids_model.weight_power,
            "min_confidence": ids_model.min_confidence,
            "max_fpr": ids_model.max_fpr,
        }

        if optuna is not None:
            def objective(trial):
                n_est = trial.suggest_categorical("n_estimators", [100, 200, 300])
                m_depth = trial.suggest_categorical("max_depth", [4, 6, 8, 10])
                lr = trial.suggest_float("learning_rate", 0.03, 0.2, log=True)
                subsamp = trial.suggest_categorical("subsample", [0.7, 0.8, 1.0])
                col_samp = trial.suggest_categorical("colsample_bytree", [0.7, 0.8, 1.0])
                w_pow = trial.suggest_categorical("weight_power", [0.25, 0.5, 0.75, 1.0])
                min_conf = trial.suggest_categorical("min_confidence", [0.3, 0.4, 0.5])
                m_fpr = trial.suggest_categorical("max_fpr", [0.005, 0.01, 0.02])

                temp_model = FinalIDSModel(
                    n_estimators=n_est,
                    max_depth=m_depth,
                    learning_rate=lr,
                    subsample=subsamp,
                    colsample_bytree=col_samp,
                    weight_power=w_pow,
                    min_confidence=min_conf,
                    max_fpr=m_fpr,
                    random_state=self.random_state,
                    n_jobs=-1,
                )
                temp_model.train(train_df, val_df=val_df)
                eval_metrics = temp_model.evaluate(val_df, split_name="val")
                return eval_metrics.get(self.metric_key, 0.0)

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, timeout=timeout)

            best_score = study.best_value
            for k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"]:
                if k in study.best_params:
                    best_params[k] = study.best_params[k]
            for k in ["weight_power", "min_confidence", "max_fpr"]:
                if k in study.best_params:
                    best_cfg[k] = study.best_params[k]

            print(f"Comp4 Tuning Best {self.metric_key}: {best_score:.4f} with params: {study.best_params}")
        else:
            print("Optuna not installed, using default tuned settings.")
            ids_model.train(train_df, val_df=val_df)
            best_score = ids_model.evaluate(val_df, split_name="val")[self.metric_key]

        # Re-train with optimal parameters
        ids_model.params = best_params
        ids_model.weight_power = best_cfg["weight_power"]
        ids_model.min_confidence = best_cfg["min_confidence"]
        ids_model.max_fpr = best_cfg["max_fpr"]
        ids_model.train(train_df, val_df=val_df)
        self.best_ids_params = {**best_params, **best_cfg}

        print("\n" + "=" * 80)
        print(f"Comp4 Final Optimization Complete: Best Val {self.metric_key} = {best_score:.4f}")
        print("=" * 80 + "\n")

        return {
            "best_params": self.best_ids_params,
            "best_score": best_score,
        }

    # ========================================================================
    # 5. Full Sequential Multi-Round Pipeline Execution
    # ========================================================================
    def run_sequential_tuning(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        holdout_df: Optional[pd.DataFrame] = None,
        generator_type: str = "ctgan",
        fixed_gradient_updates: int = 42000,
        candidate_ymax: Optional[List[int]] = None,
        candidate_thresholds: Optional[List[float]] = None,
        comp2_epochs: Optional[int] = None,
        comp4_trials: int = 6,
    ) -> Dict[str, Any]:
        """
        Executes the entire 4-round sequential tuning pipeline:
          - First round: Tune comp1 only based on comp4 evaluation.
          - Second round: Tune comp2 only based on comp4 evaluation.
          - Third round: Tune comp3 based on comp4 evaluation.
          - Final round: Tune comp4, report results on IDS performance,
            and complete 5-pillar synthetic data quality analysis.
        """
        print("\n" + "#" * 85)
        print("STARTING 4-ROUND SEQUENTIAL TUNING PIPELINE")
        print("#" * 85)

        # Initialize component instances
        ids_model = FinalIDSModel(random_state=self.random_state, n_jobs=-1)
        generators = TabularGenerators(
            model_type=generator_type,
            fixed_gradient_updates=fixed_gradient_updates,
            random_state=self.random_state,
            weights_dir=self.final_dir / f"generators/{generator_type.upper()}/weights",
        )
        sample_filter = SyntheticSampleFilter(
            random_state=self.random_state,
            default_oof_threshold=0.6,
            enn_neighbors=3,
        )

        # Disaggregate benign and attack flows upfront
        train_benign, real_train_attacks = split_benign_attack(train_df, verbose=True)

        # Pre-fit OOF predictor and ENN index on real training attack flows
        print(">>> [Pre-Training] Fitting OOF 5-Fold Predictor & ENN on Real Attack Training Set...")
        sample_filter.fit_oof_predictor(real_train_attacks)
        sample_filter.fit_enn(real_train_attacks)

        round_history = {}

        # --------------------------------------------------------------------
        # ROUND 1: Tune Comp1
        # --------------------------------------------------------------------
        print("\n" + "#" * 85)
        print("PIPELINE ROUND 1: TUNING COMP1 (y_max) BASED ON COMP4 EVALUATION")
        print("#" * 85)

        r1_res = self.tune_resampler_ymax(
            train_df=train_df,
            val_df=val_df,
            ids_model=ids_model,
            candidate_ymax=candidate_ymax,
            generator=generators,
            sample_filter=sample_filter,
            train_benign=train_benign,
            train_attacks=real_train_attacks,
            output_dir=self.output_dir,
        )
        best_train_resampled = r1_res["best_results"]["df_train_resampled"]
        best_syn_deficit = r1_res["best_results"]["syn_deficit"]

        round_history["round1"] = {
            "component": "comp1 (IntraAttackResampler)",
            "tuned_parameter": "y_max",
            "best_value": r1_res["best_y_max"],
            "best_score": r1_res["best_score"],
            "metric": self.metric_key,
            "details": r1_res["history"],
            "tuning_plot": r1_res.get("tuning_plot"),
        }

        # --------------------------------------------------------------------
        # ROUND 2: Tune Comp2
        # --------------------------------------------------------------------
        print("\n" + "#" * 85)
        print(f"PIPELINE ROUND 2: TUNING COMP2 ({generator_type.upper()}) BASED ON COMP4 EVALUATION")
        print("#" * 85)

        # Pass the SMOTE-NC pre-balanced attack dataset from Round 1 to CTGAN
        if best_train_resampled is not None and not best_train_resampled.empty:
            print(f">>> [Generators Train] Training {generator_type.upper()} on SMOTE-NC pre-balanced attack dataset from Round 1...")
            generator_train_attacks = best_train_resampled[best_train_resampled["Label"] == 1].copy().reset_index(drop=True)
        else:
            generator_train_attacks = real_train_attacks

        r1_score = r1_res["best_score"]

        # Train generator with fixed gradient updates (42k steps) using unified generator
        generators.train(
            train_attacks=generator_train_attacks,
            syn_deficit=best_syn_deficit,
            batch_size=500,
            epochs=comp2_epochs,
            unified_generator=True,
        )
        generators.save_weights("round2_best_weights.pt")

        # Sample 10 * n_syn candidate pool via rejection sampling
        df_candidate_pool = generators.sample(syn_deficit=best_syn_deficit, multiplier=10)

        # Baseline filter evaluation against genuine real attacks
        filter_res = sample_filter.filter_samples(
            synthetic_df=df_candidate_pool,
            real_train_df=real_train_attacks,
            syn_deficit=best_syn_deficit,
            oof_threshold=0.6,
            run_enn=True,
        )
        df_syn_clean = filter_res["kept_df"]

        df_train_augmented = pd.concat([train_benign, real_train_attacks, df_syn_clean], ignore_index=True)
        ids_model.train(df_train_augmented, val_df=val_df)
        val_metrics = ids_model.evaluate(val_df, split_name="val")
        r2_score = val_metrics.get(self.metric_key, 0.0)
        improvement = r2_score - r1_score

        round_history["round2"] = {
            "component": f"comp2 ({generator_type.upper()})",
            "fixed_gradient_updates": fixed_gradient_updates,
            "raw_candidate_pool": len(df_candidate_pool),
            "kept_after_default_filter": len(df_syn_clean),
            "val_score": r2_score,
            "improvement_over_r1": improvement,
            "metric": self.metric_key,
        }

        # --------------------------------------------------------------------
        # ROUND 3: Tune Comp3
        # --------------------------------------------------------------------
        print("\n" + "#" * 85)
        print("PIPELINE ROUND 3: TUNING COMP3 (SAMPLE FILTERING) BASED ON COMP4 EVALUATION")
        print("#" * 85)

        r3_res = self.tune_filter_threshold(
            synthetic_pool=df_candidate_pool,
            real_train_attacks=real_train_attacks,
            train_benign=train_benign,
            val_df=val_df,
            sample_filter=sample_filter,
            ids_model=ids_model,
            syn_deficit=best_syn_deficit,
            candidate_thresholds=candidate_thresholds,
            output_dir=self.output_dir,
        )
        best_clean_syn = r3_res["best_kept_df"]
        best_train_resampled = pd.concat([train_benign, real_train_attacks, best_clean_syn], ignore_index=True)

        round_history["round3"] = {
            "component": "comp3 (SyntheticSampleFilter)",
            "best_threshold": r3_res["best_threshold"],
            "best_score": r3_res["best_score"],
            "kept_synthetic_samples": len(best_clean_syn),
            "metric": self.metric_key,
            "details": r3_res["history"],
            "tuning_plot": r3_res.get("tuning_plot"),
        }

        # Export consolidated multi-panel tuning plot
        self.plot_combined_tuning_summary(
            r1_history=r1_res.get("history", []),
            r3_history=r3_res.get("history", []),
            output_dir=self.output_dir,
        )

        # --------------------------------------------------------------------
        # FINAL ROUND: Tune Comp4 & Comprehensive Reporting
        # --------------------------------------------------------------------
        print("\n" + "#" * 85)
        print("PIPELINE FINAL ROUND: OPTIMIZING COMP4 & COMPREHENSIVE BENCHMARK EVALUATION")
        print("#" * 85)

        comp4_tune_res = self.tune_ids_model(
            train_df=best_train_resampled,
            val_df=val_df,
            ids_model=ids_model,
            n_trials=comp4_trials,
        )

        print("\n>>> Computing Final Benchmark Evaluation Metrics across Partitions...")
        val_eval = ids_model.evaluate(val_df, split_name="val", verbose=True)
        test_eval = ids_model.evaluate(test_df, split_name="test", verbose=True)
        holdout_eval = ids_model.evaluate(holdout_df, split_name="holdout", verbose=True) if holdout_df is not None else None

        # 5-Pillar Synthetic Data Quality Evaluation (Tuan's ML-IDS methodology)
        eval_ref_test = holdout_df if holdout_df is not None else test_df
        synthetic_quality_report = evaluate_synthetic_quality(
            real_train=train_df,
            real_test=eval_ref_test,
            synthetic_df=best_clean_syn,
            seed=self.random_state,
        )

        # Distribution Distances
        dist_report = self._compute_distribution_distances(best_train_resampled, val_df)

        final_summary = {
            "final_params": comp4_tune_res["best_params"],
            "val_metrics": val_eval,
            "test_metrics": test_eval,
            "holdout_metrics": holdout_eval,
            "synthetic_quality": synthetic_quality_report,
            "distribution_distances": dist_report,
            "round_history": round_history,
            "best_train_resampled": best_train_resampled,
            "best_synthetic_flows": best_clean_syn,
        }
        return final_summary

    def _compute_distribution_distances(self, resampled_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates distribution divergence metrics between resampled synthetic flows
        and natural validation flow distributions.
        """
        print("\n>>> Computing Dataset Distribution Distances to Validation Distribution...")

        val_attacks = val_df[val_df["Label"] == 1].copy()
        feature_cols = [c for c in val_attacks.columns if c not in ["Label", "Attack", "OOF_Confidence"]]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
        cont_cols = [c for c in feature_cols if c not in cat_cols]

        resampled_attacks = resampled_df[resampled_df["Label"] == 1].copy()

        # Clean infinite/NaN values before computing distances
        val_clean = val_attacks[cont_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        val_clean = val_clean.fillna(val_clean.median().fillna(0.0))
        syn_clean = resampled_attacks[cont_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        syn_clean = syn_clean.fillna(syn_clean.median().fillna(0.0))

        # Wasserstein Distance per continuous attribute
        w_distances = {}
        for col in cont_cols:
            v_val = val_clean[col].values
            v_syn = syn_clean[col].values
            feat_range = max(1e-6, np.ptp(v_val))
            dist = wasserstein_distance(v_val, v_syn) / feat_range
            w_distances[col] = round(float(dist), 5)

        mean_w_dist = round(float(np.mean(list(w_distances.values()))), 5) if w_distances else 0.0

        # Centroid Euclidean Distance on standardized features
        val_mean = val_clean.mean()
        syn_mean = syn_clean.mean()
        val_std = val_clean.std().replace(0, 1.0)

        norm_diff = ((syn_mean - val_mean) / val_std).fillna(0.0)
        centroid_dist = round(float(np.sqrt(np.sum(norm_diff ** 2))), 4)

        report = {
            "mean_normalized_wasserstein_distance": mean_w_dist,
            "centroid_euclidean_distance": centroid_dist,
            "continuous_features_analyzed": len(cont_cols),
            "top_divergent_features": sorted(w_distances.items(), key=lambda x: x[1], reverse=True)[:5],
        }

        print(f"  Mean Normalized Wasserstein Distance: {mean_w_dist:.5f} (Lower = Closer to Natural Distribution)")
        print(f"  Centroid Standardized Euclidean Dist: {centroid_dist:.4f}")
        return report


__all__ = [
    "PipelineTuner",
]


if __name__ == "__main__":
    print("Pipeline Tuner (pipeline_tuner.py) loaded successfully.")

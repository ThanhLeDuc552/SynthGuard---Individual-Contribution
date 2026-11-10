#!/usr/bin/env python3
"""
Unified Pipeline Tuner for Network Intrusion Detection.

Consolidates all hyperparameter tuning logic previously scattered across components:
  - Round 1 (Comp 1): Tuning SMOTE-NC multiplication factor (y_max).
  - Round 2 (Comp 2): Tuning Deep Generative Models (CTGAN / TVAE) via Optuna.
  - Round 3 (Comp 3): Tuning OOF Confidence Filter Threshold & ENN.
  - Final Round (Comp 4): Tuning Final IDS Model (XGBoost) Hyperparameters.
  - Full Sequential Pipeline: Coordinates multi-round execution with epsilon stopping criteria.

All tuning evaluations strictly measure downstream validation performance of the
Final IDS Model (Comp 4), adhering to components.txt and dataset.txt.
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
    from components.tabular_generators import TabularGenerators
    from components.synthetic_sample_filter import SyntheticSampleFilter
    from components.final_ids_model import FinalIDSModel
except ImportError:
    from intra_attack_resampler import resolve_paths, SHORTCUT_COLS, CATEGORICAL_FEATURES, IntraAttackResampler
    from tabular_generators import TabularGenerators
    from synthetic_sample_filter import SyntheticSampleFilter
    from final_ids_model import FinalIDSModel


class PipelineTuner:
    """
    Unified Tuner consolidating hyperparameter optimization across all 4 pipeline components.
    """

    def __init__(
        self,
        t_max: int = 10000,
        random_state: int = 42,
        scaler_type: str = "minmax",
        metric_key: str = "macro_f1",
        epsilon: float = 0.002,
        final_dir: Optional[Union[str, Path]] = None,
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

        # Tuning tracking states
        self.best_y_max: int = 25
        self.best_generator_params: Dict[str, Any] = {}
        self.best_filter_threshold: float = 0.6
        self.best_ids_params: Dict[str, Any] = {}
        self.tuning_history: Dict[str, Any] = {}

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
    ) -> Dict[str, Any]:
        """
        Tunes the SMOTE-NC multiplication factor y_max in Component 1.
        Evaluates each candidate by training the IDS model and measuring validation performance.
        Stops early if consecutive improvement < epsilon.
        """
        if candidate_ymax is None:
            candidate_ymax = [5, 10, 15, 20, 25, 30, 40]

        print("=" * 80)
        print(">>> [Round 1] Tuning Component 1 (y_max) Based on IDS Model Validation")
        print("=" * 80)
        print(f"Candidates:   {candidate_ymax}")
        print(f"Ceiling t_max: {self.t_max:,}")
        print(f"Stopping eps: {self.epsilon} (improvement < epsilon -> proceed to next round)")
        print(f"Target Metric:{self.metric_key}\n")

        history = []
        best_score = -1.0
        best_y = candidate_ymax[0]
        best_results = None
        prev_score = -1.0

        if use_optuna and optuna is not None:
            def objective(trial):
                y_val = trial.suggest_int("y_max", 5, 50, step=5)
                resampler = IntraAttackResampler(
                    t_max=self.t_max,
                    y_max=y_val,
                    random_state=self.random_state,
                    scaler_type=self.scaler_type,
                    final_dir=self.final_dir,
                )
                out = resampler.fit_resample(train_df=train_df, export=False)
                ids_model.train(out["df_train_resampled"], val_df=val_df)
                eval_metrics = ids_model.evaluate(val_df, split_name="val")
                return eval_metrics.get(self.metric_key, 0.0)

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            best_y = study.best_params["y_max"]
            best_score = study.best_value
            print(f"Optuna Best y_max: {best_y} with {self.metric_key} = {best_score:.4f}")

            best_resampler = IntraAttackResampler(
                t_max=self.t_max,
                y_max=best_y,
                random_state=self.random_state,
                scaler_type=self.scaler_type,
                final_dir=self.final_dir,
            )
            best_results = best_resampler.fit_resample(train_df=train_df, export=True)
            self.best_y_max = best_y
            return {
                "best_y_max": best_y,
                "best_score": best_score,
                "best_results": best_results,
                "history": study.trials_dataframe().to_dict(orient="records"),
            }

        # Grid search over candidate_ymax
        for idx, y_val in enumerate(candidate_ymax, 1):
            t_start = time.time()
            print(f"--- [Comp1 Tuning {idx}/{len(candidate_ymax)}] Testing y_max = {y_val} ---")

            resampler = IntraAttackResampler(
                t_max=self.t_max,
                y_max=y_val,
                random_state=self.random_state,
                scaler_type=self.scaler_type,
                final_dir=self.final_dir,
            )
            out = resampler.fit_resample(train_df=train_df, export=False)

            # Train IDS model on resampled data
            ids_model.train(out["df_train_resampled"], val_df=val_df)

            # Evaluate IDS model on val_df
            eval_metrics = ids_model.evaluate(val_df, split_name="val")
            score = eval_metrics.get(self.metric_key, 0.0)
            macro_f1 = eval_metrics.get("macro_f1", score)
            bal_acc = eval_metrics.get("balanced_accuracy", 0.0)

            improvement = (score - best_score) if best_score > 0 else 0.0
            elapsed = time.time() - t_start

            record = {
                "y_max": y_val,
                self.metric_key: score,
                "macro_f1": macro_f1,
                "balanced_acc": bal_acc,
                "resampled_rows": len(out["df_train_resampled"]),
                "synthetic_rows": len(out["df_attacks_smotenc"]) - len(out["train_attacks_raw"]),
                "elapsed_sec": round(elapsed, 2),
            }
            history.append(record)

            print(f"    Val {self.metric_key}: {score:.4f} (Macro F1: {macro_f1:.4f}, Bal Acc: {bal_acc:.4f})")
            print(f"    Improvement over best: +{improvement:.4f} in {elapsed:.2f}s")

            if score > best_score:
                best_score = score
                best_y = y_val
                best_results = out

            # Convergence stopping condition (difference < epsilon)
            if prev_score > 0 and (score - prev_score) < self.epsilon and idx >= 3:
                print(f"\n[Stopping Condition Reached] Improvement ({score - prev_score:.5f}) < epsilon ({self.epsilon}).")
                print("Proceeding to second round...")
                break

            prev_score = score

        self.best_y_max = best_y
        print("=" * 80)
        print(f"Comp1 Tuning Finished: Best y_max = {self.best_y_max} ({self.metric_key}: {best_score:.4f})")
        print("=" * 80 + "\n")

        return {
            "best_y_max": self.best_y_max,
            "best_score": best_score,
            "best_results": best_results,
            "history": history,
        }

    # ========================================================================
    # 2. Round 2: Tune Tabular Generators (CTGAN / TVAE)
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
        Tunes generator hyperparameters (CTGAN / TVAE) using Optuna.
        Evaluates each trial end-to-end: generator training -> sampling (10 * n_syn) ->
        Comp3 filtering -> Comp4 IDS model training -> Comp4 validation performance.
        """
        if optuna is None:
            raise ImportError("Optuna is required for tune_generator. Install via `pip install optuna`.")

        print("=" * 80)
        print(f">>> [Round 2] Tuning Component 2 ({generator.model_type.upper()}) via Optuna")
        print("=" * 80)
        print(f"Trials:       {n_trials}")
        print(f"Objective:    Maximize IDS Model {self.metric_key} on Validation Set\n")

        feature_cols = [c for c in train_attacks.columns if c not in ["Label", "Attack"]]
        categorical_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
        best_trial_score = -1.0
        best_trial_models = None

        CTGAN_DIMS = {
            "256x256": (256, 256),
            "256x256x256": (256, 256, 256),
            "512x512": (512, 512),
        }

        def objective(trial):
            nonlocal best_trial_score, best_trial_models

            if generator.model_type == "ctgan":
                gen_lr = trial.suggest_float("generator_lr", 1e-5, 1e-3, log=True)
                disc_lr = trial.suggest_float("discriminator_lr", 1e-5, 1e-3, log=True)
                emb_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
                b_size = trial.suggest_categorical("batch_size", [100, 200, 500])
                gen_dim_key = trial.suggest_categorical("generator_dim", list(CTGAN_DIMS.keys()))
                disc_dim_key = trial.suggest_categorical("discriminator_dim", list(CTGAN_DIMS.keys()))
                disc_steps = trial.suggest_categorical("discriminator_steps", [1, 3, 5])
                log_freq = trial.suggest_categorical("log_frequency", [True, False])
                kwargs = {
                    "generator_lr": gen_lr,
                    "discriminator_lr": disc_lr,
                    "embedding_dim": emb_dim,
                    "generator_dim": CTGAN_DIMS[gen_dim_key],
                    "discriminator_dim": CTGAN_DIMS[disc_dim_key],
                    "discriminator_steps": disc_steps,
                    "log_frequency": log_freq,
                }
            else:  # tvae
                emb_dim = trial.suggest_categorical("embedding_dim", [64, 128])
                l2scale = trial.suggest_float("l2scale", 1e-6, 1e-4, log=True)
                b_size = trial.suggest_categorical("batch_size", [100, 200])
                kwargs = {
                    "embedding_dim": emb_dim,
                    "l2scale": l2scale,
                }

            # Train generator with candidate hyperparameters
            generator.train(
                train_attacks=train_attacks,
                syn_deficit=syn_deficit,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                batch_size=b_size,
                **kwargs,
            )

            # Sample candidate pool (10 * n_syn)
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

            if score > best_trial_score:
                best_trial_score = score
                best_trial_models = copy.deepcopy(generator.models)
                generator.save_weights("best_tuned_generator_weights.pt")

            return score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        print("\n" + "=" * 80)
        print(f"Comp2 Optuna Tuning Finished: Best {self.metric_key} = {study.best_value:.4f}")
        print(f"Best Parameters: {study.best_params}")
        print("=" * 80 + "\n")

        self.best_generator_params = study.best_params
        if best_trial_models is not None:
            generator.models = best_trial_models

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
    ) -> Dict[str, Any]:
        """
        Tunes the OOF confidence gate threshold in Component 3.
        Evaluates each threshold via the IDS model on the validation set.
        Stops early if consecutive improvement < epsilon.
        """
        if candidate_thresholds is None:
            candidate_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]

        print("=" * 80)
        print(">>> [Round 3] Tuning Component 3 (OOF Confidence Threshold) Based on IDS Model")
        print("=" * 80)
        print(f"Candidate Thresholds: {candidate_thresholds}")
        print(f"Stopping eps:         {self.epsilon}")

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

            # Stopping condition: improvement < epsilon
            if prev_score > 0 and (score - prev_score) < self.epsilon and idx >= 2:
                print(f"\n[Stopping Condition Reached] Improvement ({score - prev_score:.5f}) < epsilon ({self.epsilon}).")
                print("Proceeding to final round...")
                break

            prev_score = score

        self.best_filter_threshold = best_thresh
        print("\n" + "=" * 80)
        print(f"Comp3 Tuning Finished: Best Threshold = {self.best_filter_threshold} ({self.metric_key}: {best_score:.4f})")
        print("=" * 80 + "\n")

        return {
            "best_threshold": self.best_filter_threshold,
            "best_score": best_score,
            "best_kept_df": best_kept_df,
            "history": history,
        }

    # ========================================================================
    # 4. Final Round: Tune Final IDS Model Hyperparameters
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
        Tunes Final IDS Model (XGBoost) hyperparameters using Optuna.
        Re-fits the IDS model with optimal hyperparameters on the final resampled data.
        """
        print("=" * 80)
        print(">>> [Final Round] Tuning Component 4 (Final IDS Model Hyperparameters)")
        print("=" * 80)
        print(f"Trials:    {n_trials}")
        print(f"Objective: Maximize {self.metric_key} on Validation Set\n")

        best_score = -1.0
        best_params = copy.deepcopy(ids_model.params)

        if optuna is not None:
            def objective(trial):
                n_est = trial.suggest_int("n_estimators", 100, 300, step=50)
                m_depth = trial.suggest_int("max_depth", 4, 10, step=2)
                lr = trial.suggest_float("learning_rate", 0.03, 0.25, log=True)
                subsamp = trial.suggest_float("subsample", 0.6, 1.0, step=0.1)
                col_samp = trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1)

                trial_params = copy.deepcopy(ids_model.params)
                trial_params.update({
                    "n_estimators": n_est,
                    "max_depth": m_depth,
                    "learning_rate": lr,
                    "subsample": subsamp,
                    "colsample_bytree": col_samp,
                })

                temp_model = FinalIDSModel(**trial_params)
                temp_model.train(train_df)
                eval_metrics = temp_model.evaluate(val_df, split_name="val")
                return eval_metrics.get(self.metric_key, 0.0)

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, timeout=timeout)

            best_params.update(study.best_params)
            best_score = study.best_value
            print(f"Comp4 Tuning Best {self.metric_key}: {best_score:.4f} with params: {study.best_params}")
        else:
            print("Optuna not installed, using default tuned settings.")
            ids_model.train(train_df)
            best_score = ids_model.evaluate(val_df, split_name="val")[self.metric_key]

        # Re-train with optimal parameters
        ids_model.params = best_params
        ids_model.train(train_df)
        self.best_ids_params = best_params

        print("\n" + "=" * 80)
        print(f"Comp4 Final Optimization Complete: Best Val {self.metric_key} = {best_score:.4f}")
        print("=" * 80 + "\n")

        return {
            "best_params": best_params,
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
        Executes the entire 4-round sequential tuning pipeline as specified in components.txt:
          - First round: Tune comp1 only based on comp4 evaluation.
          - Second round: Tune comp2 only based on comp4 evaluation.
          - Third round: Tune comp3 based on comp4 evaluation.
          - Final round: Tune comp4, report results on IDS performance,
            and dataset distance to val set distribution.
        """
        print("\n" + "#" * 85)
        print("STARTING 4-ROUND SEQUENTIAL TUNING PIPELINE (components.txt)")
        print("#" * 85)

        # Initialize component instances
        ids_model = FinalIDSModel(random_state=self.random_state, n_jobs=4)
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
        }

        # --------------------------------------------------------------------
        # ROUND 2: Tune Comp2
        # --------------------------------------------------------------------
        print("\n" + "#" * 85)
        print(f"PIPELINE ROUND 2: TUNING COMP2 ({generator_type.upper()}) BASED ON COMP4 EVALUATION")
        print("#" * 85)

        train_benign = train_df[train_df["Label"] == 0].copy().reset_index(drop=True)
        train_attacks = train_df[train_df["Label"] == 1].copy().reset_index(drop=True)
        feature_cols = [c for c in train_attacks.columns if c not in ["Label", "Attack"]]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

        r1_score = r1_res["best_score"]

        # Train generator with fixed gradient updates (42k steps)
        generators.train(
            train_attacks=train_attacks,
            syn_deficit=best_syn_deficit,
            feature_cols=feature_cols,
            categorical_cols=cat_cols,
            batch_size=200,
            epochs=comp2_epochs,
        )
        generators.save_weights("round2_best_weights.pt")

        # Sample 10 * n_syn candidate pool
        df_candidate_pool = generators.sample(syn_deficit=best_syn_deficit, multiplier=10)

        # Baseline filter evaluation
        filter_res = sample_filter.filter_samples(
            synthetic_df=df_candidate_pool,
            real_train_df=train_attacks,
            syn_deficit=best_syn_deficit,
            oof_threshold=0.6,
            run_enn=True,
        )
        df_syn_clean = filter_res["kept_df"]

        df_train_augmented = pd.concat([train_benign, train_attacks, df_syn_clean], ignore_index=True)
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
            real_train_attacks=train_attacks,
            train_benign=train_benign,
            val_df=val_df,
            sample_filter=sample_filter,
            ids_model=ids_model,
            syn_deficit=best_syn_deficit,
            candidate_thresholds=candidate_thresholds,
        )
        best_clean_syn = r3_res["best_kept_df"]
        best_train_resampled = pd.concat([train_benign, train_attacks, best_clean_syn], ignore_index=True)

        round_history["round3"] = {
            "component": "comp3 (SyntheticSampleFilter)",
            "best_threshold": r3_res["best_threshold"],
            "best_score": r3_res["best_score"],
            "kept_synthetic_samples": len(best_clean_syn),
            "metric": self.metric_key,
            "details": r3_res["history"],
        }

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

        # Dataset Distance Calculation to Validation Set Distribution
        dist_report = self._compute_distribution_distances(best_train_resampled, val_df)

        final_summary = {
            "final_params": comp4_tune_res["best_params"],
            "val_metrics": val_eval,
            "test_metrics": test_eval,
            "holdout_metrics": holdout_eval,
            "distribution_distances": dist_report,
            "round_history": round_history,
            "best_train_resampled": best_train_resampled,
        }
        return final_summary

    def _compute_distribution_distances(self, resampled_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates distribution divergence metrics between the resampled synthetic attack flows
        and the natural validation attack flow distribution.
        """
        print("\n>>> Computing Dataset Distribution Distances to Validation Distribution...")

        val_attacks = val_df[val_df["Label"] == 1].copy()
        feature_cols = [c for c in val_attacks.columns if c not in ["Label", "Attack", "OOF_Confidence"]]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
        cont_cols = [c for c in feature_cols if c not in cat_cols]

        resampled_attacks = resampled_df[resampled_df["Label"] == 1].copy()

        # Wasserstein Distance per continuous attribute
        w_distances = {}
        for col in cont_cols:
            v_val = pd.to_numeric(val_attacks[col], errors="coerce").fillna(0.0).values
            v_syn = pd.to_numeric(resampled_attacks[col], errors="coerce").fillna(0.0).values
            feat_range = max(1e-6, np.ptp(v_val))
            dist = wasserstein_distance(v_val, v_syn) / feat_range
            w_distances[col] = round(float(dist), 5)

        mean_w_dist = round(float(np.mean(list(w_distances.values()))), 5)

        # Centroid Euclidean Distance on normalized features
        val_mean = val_attacks[cont_cols].mean()
        syn_mean = resampled_attacks[cont_cols].mean()
        val_std = val_attacks[cont_cols].std().replace(0, 1.0)

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


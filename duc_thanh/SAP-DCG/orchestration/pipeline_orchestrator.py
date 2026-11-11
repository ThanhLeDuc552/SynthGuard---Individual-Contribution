#!/usr/bin/env python3
"""
Sequential Tuning Pipeline Orchestrator for Network Intrusion Detection.

Implements the multi-round tuning pipeline specified in components.txt:
- Round 1: Tune IntraAttackResampler (y_max) based on FinalIDSModel evaluation.
  If improvement < epsilon -> Round 2.
- Round 2: Tune TabularGenerators (CTGAN / TVAE) based on FinalIDSModel evaluation.
  If improvement < epsilon -> Round 3.
- Round 3: Tune SyntheticSampleFilter (OOF gate & ENN) based on FinalIDSModel evaluation.
  If improvement < epsilon -> Final Round.
- Final Round: Tune FinalIDSModel hyperparameters, then report benchmark IDS performance,
  dataset distribution distance to validation set, and export finalized benchmark datasets.

NOTE: All rounds always evaluate downstream on FinalIDSModel,
since final IDS performance is the primary objective function.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# Mitigate OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

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
    from orchestration.pipeline_tuner import PipelineTuner
except ImportError:
    try:
        from .pipeline_tuner import PipelineTuner
    except ImportError:
        from pipeline_tuner import PipelineTuner
    from intra_attack_resampler import resolve_paths, SHORTCUT_COLS, CATEGORICAL_FEATURES, IntraAttackResampler
    from tabular_generators import TabularGenerators
    from synthetic_sample_filter import SyntheticSampleFilter
    from final_ids_model import FinalIDSModel


class SequentialTuningPipeline:
    """
    Master Orchestrator executing the 4-Round Sequential Tuning Pipeline.
    """

    def __init__(
        self,
        t_max: int = 10000,
        initial_ymax: int = 25,
        epsilon: float = 0.002,
        generator_type: str = "ctgan",
        fixed_gradient_updates: int = 42000,
        metric_key: str = "macro_f1",
        random_state: int = 42,
        final_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.t_max = t_max
        self.initial_ymax = initial_ymax
        self.epsilon = epsilon
        self.generator_type = generator_type.lower()
        self.fixed_gradient_updates = fixed_gradient_updates
        self.metric_key = metric_key
        self.random_state = random_state

        # Resolve output directories
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

        # Instantiate unified tuner and all 4 modular components
        self.tuner = PipelineTuner(
            t_max=self.t_max,
            random_state=self.random_state,
            metric_key=self.metric_key,
            epsilon=self.epsilon,
            final_dir=self.final_dir,
        )
        self.resampler = IntraAttackResampler(
            t_max=self.t_max,
            y_max=self.initial_ymax,
            random_state=self.random_state,
            final_dir=self.final_dir,
        )
        self.generators = TabularGenerators(
            model_type=self.generator_type,
            fixed_gradient_updates=self.fixed_gradient_updates,
            random_state=self.random_state,
            weights_dir=self.output_dir / "weights",
        )
        self.sample_filter = SyntheticSampleFilter(
            random_state=self.random_state,
            default_oof_threshold=0.6,
            enn_neighbors=3,
        )
        self.ids_model = FinalIDSModel(
            random_state=self.random_state,
            n_jobs=4,
        )

        # Pipeline progress tracking
        self.best_y_max: int = initial_ymax
        self.best_syn_deficit: Dict[str, int] = {}
        self.best_target_counts: Dict[str, int] = {}
        self.round_history: Dict[str, Any] = {}
        self.best_train_resampled: Optional[pd.DataFrame] = None
        self.best_synthetic_flows: Optional[pd.DataFrame] = None
        self.best_clean_syn: Optional[pd.DataFrame] = None
        self.best_params_payload: Dict[str, Any] = {}

    # ------------------------------------------------------------------------
    # Round 1: Tune Comp1 Only (y_max) Based on Comp4 Evaluation
    # ------------------------------------------------------------------------
    def run_round1_tune_comp1(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        candidate_ymax: Optional[List[int]] = None,
        use_optuna: bool = False,
        batch_size: int = 200,
        epochs: Optional[int] = None,
        multiplier: int = 10,
    ) -> Dict[str, Any]:
        """
        Round 1: Tunes IntraAttackResampler multiplication factor y_max.
        Tuning y_max tunes the generator (CTGAN/TVAE) itself:
        For each candidate y_max, the generator is trained on the resampled attack flows,
        generates 10 * n_syn candidate samples, passes through filtering, and evaluates
        downstream on FinalIDSModel validation performance.
        Advances to Round 2 when improvement < epsilon.
        """
        print("\n" + "#" * 85)
        print("PIPELINE ROUND 1: TUNING COMP1 (y_max) BASED ON COMP4 EVALUATION VIA GENERATOR")
        print("#" * 85)

        comp1_res = self.tuner.tune_resampler_ymax(
            train_df=train_df,
            val_df=val_df,
            ids_model=self.ids_model,
            candidate_ymax=candidate_ymax,
            use_optuna=use_optuna,
            generator=self.generators,
            sample_filter=self.sample_filter,
            batch_size=batch_size,
            epochs=epochs,
            multiplier=multiplier,
        )

        self.best_y_max = comp1_res["best_y_max"]
        self.best_train_resampled = comp1_res["best_results"]["df_train_resampled"]
        self.best_syn_deficit = comp1_res["best_results"]["syn_deficit"]
        self.best_target_counts = comp1_res["best_results"]["dict_tc"]
        if comp1_res.get("best_candidate_pool") is not None:
            self.best_synthetic_flows = comp1_res["best_candidate_pool"]

        # Checkpoint generator weights from the winning y_max
        round2_weights_path = self.output_dir / "round2_best_weights.pt"
        self.generators.save_weights(round2_weights_path)

        self.round_history["round1"] = {
            "component": "IntraAttackResampler",
            "tuned_parameter": "y_max",
            "best_value": self.best_y_max,
            "best_score": comp1_res["best_score"],
            "metric": self.metric_key,
            "details": comp1_res["history"],
        }
        return comp1_res

    # ------------------------------------------------------------------------
    # Round 2: Tune Generators (CTGAN / TVAE) Based on IDS Evaluation
    # ------------------------------------------------------------------------
    def run_round2_tune_comp2(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        n_trials: int = 4,
        batch_size: int = 200,
        epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Round 2: Trains TabularGenerators and evaluates based on FinalIDSModel.
        Trains with fixed 42k gradient updates.
        Samples 10 * n_syn candidate pool.
        Advances to Round 3 when improvement < epsilon.
        """
        print("\n" + "#" * 85)
        print(f"PIPELINE ROUND 2: TUNING GENERATORS ({self.generator_type.upper()}) BASED ON IDS EVALUATION")
        print("#" * 85)

        train_benign = train_df[train_df["Label"] == 0].copy().reset_index(drop=True)
        real_train_attacks = train_df[train_df["Label"] == 1].copy().reset_index(drop=True)

        # Pass the SMOTE-NC pre-balanced attack dataset from Round 1 to CTGAN
        if self.best_train_resampled is not None and not self.best_train_resampled.empty:
            print(f">>> [Generators Train] Training {self.generator_type.upper()} on SMOTE-NC pre-balanced attack dataset from Round 1 ({len(self.best_train_resampled):,} total flows)...")
            generator_train_attacks = self.best_train_resampled[self.best_train_resampled["Label"] == 1].copy().reset_index(drop=True)
        else:
            generator_train_attacks = real_train_attacks

        feature_cols = [c for c in generator_train_attacks.columns if c not in ["Label", "Attack", "OOF_Confidence"]]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

        # Check baseline score from Round 1
        r1_score = self.round_history.get("round1", {}).get("best_score", 0.0)

        # Train generator with fixed gradient updates (42,000 steps)
        print(f"Training {self.generator_type.upper()} with fixed gradient update budget: {self.fixed_gradient_updates:,} steps...")
        self.generators.train(
            train_attacks=generator_train_attacks,
            syn_deficit=self.best_syn_deficit,
            feature_cols=feature_cols,
            categorical_cols=cat_cols,
            batch_size=batch_size,
            epochs=epochs,
        )
        # Checkpoint generator weights directly to output_dir
        round2_weights_path = self.output_dir / "round2_best_weights.pt"
        self.generators.save_weights(round2_weights_path)

        # Sample 10 * n_syn candidate flows (Step 8)
        df_candidate_pool = self.generators.sample(syn_deficit=self.best_syn_deficit, multiplier=10)

        # Baseline evaluation with default filtering against genuine real attacks
        filter_res = self.sample_filter.filter_samples(
            synthetic_df=df_candidate_pool,
            real_train_df=real_train_attacks,
            syn_deficit=self.best_syn_deficit,
            oof_threshold=0.6,
            run_enn=True,
        )
        df_syn_clean = filter_res["kept_df"]

        # Train IDS model on augmented training set
        df_train_augmented = pd.concat([train_benign, real_train_attacks, df_syn_clean], ignore_index=True)
        self.ids_model.train(df_train_augmented, val_df=val_df)
        val_metrics = self.ids_model.evaluate(val_df, split_name="val")
        r2_score = val_metrics.get(self.metric_key, 0.0)

        improvement = r2_score - r1_score
        print(f"\n[Round 2 Result] Val {self.metric_key} = {r2_score:.4f} (Round 1 Baseline: {r1_score:.4f}, Improvement: +{improvement:.4f})")

        self.best_synthetic_flows = df_candidate_pool
        self.round_history["round2"] = {
            "component": f"TabularGenerators ({self.generator_type.upper()})",
            "fixed_gradient_updates": self.fixed_gradient_updates,
            "raw_candidate_pool": len(df_candidate_pool),
            "kept_after_default_filter": len(df_syn_clean),
            "val_score": r2_score,
            "improvement_over_r1": improvement,
            "metric": self.metric_key,
        }

        if abs(improvement) < self.epsilon:
            print(f"[Stopping Condition Reached] Improvement ({improvement:.5f}) < epsilon ({self.epsilon}).")
            print("Advancing to Round 3...")

        return self.round_history["round2"]

    # ------------------------------------------------------------------------
    # Round 3: Tune SyntheticSampleFilter (OOF Threshold & ENN) Based on IDS Model
    # ------------------------------------------------------------------------
    def run_round3_tune_comp3(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        candidate_thresholds: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Round 3: Tunes SyntheticSampleFilter threshold on the candidate pool.
        Evaluates each threshold via FinalIDSModel on val_df.
        Advances to Final Round when improvement < epsilon.
        """
        print("\n" + "#" * 85)
        print("PIPELINE ROUND 3: TUNING SAMPLE FILTER BASED ON IDS EVALUATION")
        print("#" * 85)

        train_benign = train_df[train_df["Label"] == 0].copy().reset_index(drop=True)
        train_attacks = train_df[train_df["Label"] == 1].copy().reset_index(drop=True)

        # If Round 2 did not generate a candidate pool, sample one now
        if self.best_synthetic_flows is None or self.best_synthetic_flows.empty:
            print("Sampling synthetic candidate pool for filter tuning...")
            self.best_synthetic_flows = self.generators.sample(syn_deficit=self.best_syn_deficit, multiplier=10)

        comp3_res = self.tuner.tune_filter_threshold(
            synthetic_pool=self.best_synthetic_flows,
            real_train_attacks=train_attacks,
            train_benign=train_benign,
            val_df=val_df,
            sample_filter=self.sample_filter,
            ids_model=self.ids_model,
            syn_deficit=self.best_syn_deficit,
            candidate_thresholds=candidate_thresholds,
        )

        best_clean_syn = comp3_res["best_kept_df"]
        self.best_clean_syn = best_clean_syn
        self.best_train_resampled = pd.concat([train_benign, train_attacks, best_clean_syn], ignore_index=True)

        # Export kept synthetic flows to output_dir
        if best_clean_syn is not None and not best_clean_syn.empty:
            kept_syn_path = self.output_dir / "synthetic_samples_kept.csv"
            best_clean_syn.to_csv(kept_syn_path, index=False)
            print(f"  [SampleFilter Checkpoint] Saved {len(best_clean_syn):,} filtered synthetic flows to: {kept_syn_path.name}")

        self.round_history["round3"] = {
            "component": "SyntheticSampleFilter",
            "best_threshold": comp3_res["best_threshold"],
            "best_score": comp3_res["best_score"],
            "kept_synthetic_samples": len(best_clean_syn),
            "metric": self.metric_key,
            "details": comp3_res["history"],
        }
        return comp3_res

    # ------------------------------------------------------------------------
    # Final Round: Tune FinalIDSModel & Comprehensive Reporting
    # ------------------------------------------------------------------------
    def run_final_round_tune_comp4(
        self,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        holdout_df: Optional[pd.DataFrame] = None,
        n_trials: int = 6,
    ) -> Dict[str, Any]:
        """
        Final Round: Optimizes FinalIDSModel hyperparameters on the best resampled dataset.
        Evaluates final performance across Validation, Test, and Holdout sets.
        Computes dataset distribution distance metrics between synthetic data and validation set.
        """
        print("\n" + "#" * 85)
        print("PIPELINE FINAL ROUND: OPTIMIZING FINAL IDS MODEL & BENCHMARK EVALUATION")
        print("#" * 85)

        if self.best_train_resampled is None:
            raise RuntimeError("No resampled training dataset available from previous rounds.")

        # Tune IDS hyperparameters using unified tuner
        comp4_tune_res = self.tuner.tune_ids_model(
            train_df=self.best_train_resampled,
            val_df=val_df,
            ids_model=self.ids_model,
            n_trials=n_trials,
        )

        # Checkpoint trained IDS model directly to output_dir
        ids_model_path = self.output_dir / "final_ids_model.joblib"
        self.ids_model.save_model(ids_model_path)

        # Comprehensive Evaluation across all benchmark partitions
        print("\n>>> Computing Final Benchmark Evaluation Metrics across Partitions...")
        val_eval = self.ids_model.evaluate(val_df, split_name="val", verbose=True)
        test_eval = self.ids_model.evaluate(test_df, split_name="test", verbose=True)

        holdout_eval = None
        if holdout_df is not None:
            holdout_eval = self.ids_model.evaluate(holdout_df, split_name="holdout", verbose=True)

        # Dataset Distance Calculation to Validation Set Distribution
        dist_report = self._compute_distribution_distances(val_df)

        # 5-Pillar Synthetic Data Quality Evaluation (Tuan's ML-IDS methodology)
        synthetic_quality_report = {}
        syn_for_eval = self.best_clean_syn if self.best_clean_syn is not None and not self.best_clean_syn.empty else self.best_synthetic_flows
        if syn_for_eval is not None and not syn_for_eval.empty:
            try:
                from components.final_ids_model import evaluate_synthetic_quality
            except ImportError:
                from final_ids_model import evaluate_synthetic_quality
            ref_eval_set = holdout_df if holdout_df is not None else test_df
            print("\n>>> Computing 5-Pillar Synthetic Data Quality Evaluation (Coverage, Validity, Fidelity, Privacy, Utility)...")
            synthetic_quality_report = evaluate_synthetic_quality(
                real_train=self.best_train_resampled,
                real_test=ref_eval_set,
                synthetic_df=syn_for_eval,
                seed=self.random_state,
            )

        # Compile and export best parameters across all rounds
        best_params_payload = {
            "pipeline_configuration": {
                "t_max": self.t_max,
                "generator_type": self.generator_type,
                "fixed_gradient_updates": self.fixed_gradient_updates,
                "epsilon": self.epsilon,
                "target_metric": self.metric_key,
                "random_state": self.random_state,
            },
            "round1_resampler_best_params": {
                "best_y_max": self.best_y_max,
                "target_counts": self.best_target_counts,
                "syn_deficit": self.best_syn_deficit,
                "best_score": self.round_history.get("round1", {}).get("best_score", None),
            },
            "round2_generator_best_params": {
                "generator_type": self.generator_type,
                "fixed_gradient_updates": self.fixed_gradient_updates,
                "weights_checkpoint": str(self.output_dir / "round2_best_weights.pt"),
                "diagnostics": getattr(self.generators, "training_diagnostics", {}),
                "val_score": self.round_history.get("round2", {}).get("val_score", None),
            },
            "round3_sample_filter_best_params": {
                "best_oof_threshold": self.round_history.get("round3", {}).get("best_threshold", 0.6),
                "enn_neighbors": 3,
                "kept_samples": self.round_history.get("round3", {}).get("kept_synthetic_samples", None),
                "best_score": self.round_history.get("round3", {}).get("best_score", None),
            },
            "final_round_ids_model_best_params": comp4_tune_res["best_params"],
        }
        self.best_params_payload = best_params_payload
        best_params_path = self.output_dir / "best_params.json"
        with open(best_params_path, "w") as f:
            json.dump(best_params_payload, f, indent=2)
        print(f"  [Artifact Export] Saved Best Hyperparameters: {best_params_path.name}")

        # Export final augmented datasets, json summary, and markdown evaluation report
        self._export_final_pipeline_outputs(
            val_eval=val_eval,
            test_eval=test_eval,
            holdout_eval=holdout_eval,
            dist_report=dist_report,
            synthetic_quality_report=synthetic_quality_report,
            best_params_payload=best_params_payload,
        )

        final_summary = {
            "final_params": comp4_tune_res["best_params"],
            "val_metrics": val_eval,
            "test_metrics": test_eval,
            "holdout_metrics": holdout_eval,
            "synthetic_quality": synthetic_quality_report,
            "distribution_distances": dist_report,
            "round_history": self.round_history,
            "best_params": best_params_payload,
        }
        return final_summary

    # ------------------------------------------------------------------------
    # Helper: Dataset Distribution Distance Metrics
    # ------------------------------------------------------------------------
    def _compute_distribution_distances(self, val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates distribution divergence metrics between the resampled synthetic attack flows
        and the natural validation attack flow distribution (Wasserstein distance and centroid distance).
        """
        print("\n>>> Computing Dataset Distribution Distances to Validation Distribution...")

        val_attacks = val_df[val_df["Label"] == 1].copy()
        feature_cols = [c for c in val_attacks.columns if c not in ["Label", "Attack", "OOF_Confidence"]]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
        cont_cols = [c for c in feature_cols if c not in cat_cols]

        resampled_attacks = self.best_train_resampled[self.best_train_resampled["Label"] == 1].copy()

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
            # Normalize by range to ensure comparability across features
            feat_range = max(1e-6, np.ptp(v_val))
            dist = wasserstein_distance(v_val, v_syn) / feat_range
            w_distances[col] = round(float(dist), 5)

        mean_w_dist = round(float(np.mean(list(w_distances.values()))), 5)

        # Centroid Euclidean Distance on normalized features
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

    # ------------------------------------------------------------------------
    # Helper: Export Final Pipeline Outputs
    # ------------------------------------------------------------------------
    def _export_final_pipeline_outputs(
        self,
        val_eval: Dict[str, Any],
        test_eval: Dict[str, Any],
        holdout_eval: Optional[Dict[str, Any]],
        dist_report: Dict[str, Any],
        synthetic_quality_report: Optional[Dict[str, Any]] = None,
        best_params_payload: Optional[Dict[str, Any]] = None,
    ):
        """Exports finalized datasets, metrics summaries, and diagnostic tables."""
        out_multi_dir = self.final_dir / "IDS-multiclass-classification"
        out_multi_dir.mkdir(parents=True, exist_ok=True)

        # Save final resampled training set
        if self.best_train_resampled is not None:
            final_train_path = out_multi_dir / "unsw_nb15_final_pipeline_train.csv"
            self.best_train_resampled.to_csv(final_train_path, index=False)

            dest_train_path = self.output_dir / "unsw_nb15_final_pipeline_train.csv"
            self.best_train_resampled.to_csv(dest_train_path, index=False)
            print(f"  [Artifact Export] Saved Final Resampled Train: {dest_train_path.name} ({len(self.best_train_resampled):,} flows)")

        # Compile JSON summary
        summary_payload = {
            "configuration": {
                "t_max": self.t_max,
                "best_y_max": self.best_y_max,
                "generator_type": self.generator_type,
                "fixed_gradient_updates": self.fixed_gradient_updates,
                "epsilon": self.epsilon,
                "target_metric": self.metric_key,
                "output_dir": str(self.output_dir),
            },
            "best_parameters": best_params_payload or self.best_params_payload,
            "tuning_rounds_summary": self.round_history,
            "validation_performance": {k: v for k, v in val_eval.items() if k != "confusion_matrix"},
            "test_performance": {k: v for k, v in test_eval.items() if k != "confusion_matrix"},
            "holdout_performance": {k: v for k, v in holdout_eval.items() if k != "confusion_matrix"} if holdout_eval else None,
            "synthetic_quality": synthetic_quality_report or {},
            "distribution_distances": dist_report,
        }

        # Write summary JSON to destined directory
        dest_json_path = self.output_dir / "pipeline_tuning_summary.json"
        with open(dest_json_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"  [Artifact Export] Saved Benchmark Summary JSON: {dest_json_path.name}")

        # Also write to final_dir for backward compatibility
        try:
            with open(self.final_dir / "pipeline_tuning_summary.json", "w") as f:
                json.dump(summary_payload, f, indent=2)
        except Exception:
            pass

        # Generate and export rich Markdown evaluation report
        self._export_markdown_report(
            val_eval=val_eval,
            test_eval=test_eval,
            holdout_eval=holdout_eval,
            dist_report=dist_report,
            synthetic_quality_report=synthetic_quality_report,
            best_params_payload=best_params_payload or self.best_params_payload,
        )

    # ------------------------------------------------------------------------
    # Helper: Export Comprehensive Markdown Evaluation Report
    # ------------------------------------------------------------------------
    def _export_markdown_report(
        self,
        val_eval: Dict[str, Any],
        test_eval: Dict[str, Any],
        holdout_eval: Optional[Dict[str, Any]],
        dist_report: Dict[str, Any],
        synthetic_quality_report: Optional[Dict[str, Any]],
        best_params_payload: Dict[str, Any],
    ) -> Path:
        """
        Exports a rich, human-readable Markdown evaluation report detailing:
        - Executive Summary & Experiment Setup
        - Overall Benchmark Performance (Macro F1, Detection Rate, FAR, MCC, Balanced Accuracy)
        - Per-Class Metrics Breakdown (Support, Precision, Recall, F1) across partitions
        - Confusion Matrices
        - 5-Pillar Synthetic Data Quality Benchmark
        - Feature Distribution Distances (Wasserstein & Centroid)
        - Optimal Hyperparameter Tables
        - Exported Artifacts Manifest
        """
        md_lines = []
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        md_lines.append("# SAP-DCG Pipeline Comprehensive Evaluation Report")
        md_lines.append("")
        md_lines.append(f"> **Generated:** {now_str}  ")
        md_lines.append(f"> **Destination Folder:** `{self.output_dir}`  ")
        md_lines.append(f"> **Generator Architecture:** `{self.generator_type.upper()}` ({self.fixed_gradient_updates:,} gradient updates)  ")
        md_lines.append(f"> **Target Class Ceiling ($t_{{\\max}}$):** `{self.t_max:,}` | **Convergence $\\epsilon$:** `{self.epsilon}` | **Optimization Metric:** `{self.metric_key}`  ")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        # 1. Overall Performance Comparison
        md_lines.append("## 1. Overall IDS Performance Benchmark")
        md_lines.append("")
        md_lines.append("Comparison of downstream Two-Tier IDS evaluation metrics across the validation, test, and holdout partitions:")
        md_lines.append("")

        has_holdout = holdout_eval is not None
        if has_holdout:
            md_lines.append("| Metric | Validation Set | Test Set | Holdout Set | Target Direction |")
            md_lines.append("| :--- | :---: | :---: | :---: | :---: |")
        else:
            md_lines.append("| Metric | Validation Set | Test Set | Target Direction |")
            md_lines.append("| :--- | :---: | :---: | :---: |")

        metric_rows = [
            ("Macro F1-Score", "macro_f1", "{:.4f}", "Higher is better (Max 1.0)"),
            ("Detection Rate (DR / Attack Recall)", "detection_rate", "{:.4f}", "Higher is better (Max 1.0)"),
            ("False Alarm Rate (FAR / Benign FPR)", "false_alarm_rate", "{:.4f}", "Lower is better (Min 0.0)"),
            ("Matthews Correlation (MCC)", "mcc", "{:.4f}", "Higher is better (Max 1.0)"),
            ("Balanced Accuracy", "balanced_accuracy", "{:.4f}", "Higher is better (Max 1.0)"),
            ("Overall Accuracy", "accuracy", "{:.4f}", "Higher is better (Max 1.0)"),
            ("Weighted F1-Score", "weighted_f1", "{:.4f}", "Higher is better (Max 1.0)"),
            ("Binary F1-Score (Attack vs Benign)", "binary_f1", "{:.4f}", "Higher is better (Max 1.0)"),
            ("Isolated Tier-2 Attack Macro F1", "t2_macro_f1", "{:.4f}", "Higher is better (Max 1.0)"),
            ("Rare Attack F1-Score", "t2_rare_f1", "{:.4f}", "Higher is better (Max 1.0)"),
            ("End-to-End Rare Attack Recall", "e2e_rare_recall", "{:.4f}", "Higher is better (Max 1.0)"),
            ("Unknown Attack Alert Rate", "unknown_rate", "{:.4f}", "Informational"),
        ]

        for label, key, fmt, target in metric_rows:
            v_val = fmt.format(val_eval.get(key, 0.0))
            v_test = fmt.format(test_eval.get(key, 0.0))
            if has_holdout:
                v_hold = fmt.format(holdout_eval.get(key, 0.0))
                md_lines.append(f"| **{label}** | `{v_val}` | `{v_test}` | `{v_hold}` | {target} |")
            else:
                md_lines.append(f"| **{label}** | `{v_val}` | `{v_test}` | {target} |")

        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        # 2. Per-Class Performance Breakdown
        md_lines.append("## 2. Per-Class Specific Performance Breakdown")
        md_lines.append("")
        md_lines.append("Detailed classification metrics (Precision, Recall, F1-Score, and Support) per network flow category:")
        md_lines.append("")

        splits_to_report = [("Validation Split", val_eval), ("Test Split", test_eval)]
        if has_holdout:
            splits_to_report.append(("Holdout Split", holdout_eval))

        for idx, (s_name, s_data) in enumerate(splits_to_report, 1):
            md_lines.append(f"### 2.{idx} {s_name}")
            md_lines.append("")
            md_lines.append("| Network Flow Class | Category Type | Support | Precision | Recall (DR) | F1-Score | Status |")
            md_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

            per_class = s_data.get("per_class", {})
            for c_name, c_m in per_class.items():
                is_benign = (c_name.lower() == "benign")
                is_rare = c_m.get("is_rare", False)
                if is_benign:
                    cat_type = "Benign Flow"
                    badge = "Normal"
                elif is_rare:
                    cat_type = "Rare Attack (Minority)"
                    badge = "⚠️ Rare Threat"
                else:
                    cat_type = "Frequent Attack"
                    badge = "Standard Threat"

                sup = f"{c_m.get('support', 0):,}"
                prec = f"{c_m.get('precision', 0.0) * 100:.2f}%"
                rec = f"{c_m.get('recall', 0.0) * 100:.2f}%"
                f1 = f"{c_m.get('f1', 0.0) * 100:.2f}%"
                md_lines.append(f"| **{c_name}** | {cat_type} | {sup} | {prec} | {rec} | **{f1}** | {badge} |")

            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

        # 3. Confusion Matrix (Test Split)
        md_lines.append("## 3. Confusion Matrix (Test Partition)")
        md_lines.append("")
        cm_df = test_eval.get("confusion_matrix", None)
        if cm_df is not None and isinstance(cm_df, pd.DataFrame):
            classes = list(cm_df.columns)
            hdr = "| True \\ Predicted | " + " | ".join(classes) + " |"
            sep = "| :--- | " + " | ".join([":---:"] * len(classes)) + " |"
            md_lines.append(hdr)
            md_lines.append(sep)
            for true_cls in cm_df.index:
                row_vals = [f"{int(cm_df.loc[true_cls, pred_cls]):,}" for pred_cls in classes]
                md_lines.append(f"| **{true_cls}** | " + " | ".join(row_vals) + " |")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

        # 4. 5-Pillar Synthetic Data Quality Evaluation
        md_lines.append("## 4. 5-Pillar Synthetic Data Quality Benchmark")
        md_lines.append("")
        md_lines.append("Evaluated on candidate synthetic flows prior to final augmentation:")
        md_lines.append("")
        if synthetic_quality_report:
            cov = synthetic_quality_report.get("coverage", {})
            val = synthetic_quality_report.get("validity", {})
            fid = synthetic_quality_report.get("fidelity", {})
            prv = synthetic_quality_report.get("privacy", {})
            utl = synthetic_quality_report.get("utility", {})

            md_lines.append("### Pillar 1: Coverage")
            md_lines.append(f"- **Classes Covered:** `{cov.get('classes_covered', 'N/A')}`")
            md_lines.append(f"- **Missing Classes:** `{cov.get('missing_classes', [])}`")
            md_lines.append("")

            md_lines.append("### Pillar 2: Validity")
            md_lines.append(f"- **Invalid Row Rate:** `{val.get('invalid_row_rate', 0.0) * 100:.2f}%` (Rows failing integrity rules)")
            md_lines.append(f"- **Negative Value Cell Rate:** `{val.get('neg_cell_rate', 0.0) * 100:.3f}%`")
            md_lines.append(f"- **Non-Integer Cell Rate in Count Columns:** `{val.get('nonint_cell_rate', 0.0) * 100:.3f}%`")
            md_lines.append(f"- **Over Maximum Bound Cell Rate:** `{val.get('over_max_cell_rate', 0.0) * 100:.3f}%`")
            md_lines.append("")

            md_lines.append("### Pillar 3: Fidelity")
            md_lines.append(f"- **Mean Kolmogorov-Smirnov (KS) Statistic:** `{fid.get('ks_mean', 'N/A')}` (Lower = Closer continuous distributions)")
            md_lines.append(f"- **Mean Spearman Correlation Difference:** `{fid.get('corr_diff_mean', 'N/A')}`")
            md_lines.append(f"- **Tail Divergence (q99 Log-Ratio):** `{fid.get('tail_q99_log_ratio', 'N/A')}`")
            md_lines.append("")

            md_lines.append("### Pillar 4: Privacy")
            md_lines.append(f"- **Exact Copy Rate:** `{prv.get('exact_copy_rate', 0.0) * 100:.3f}%`")
            md_lines.append(f"- **Distance to Closest Record (DCR) Train 5th Percentile:** `{prv.get('dcr_train_p5', 'N/A')}`")
            md_lines.append(f"- **Distance to Closest Record (DCR) Holdout 5th Percentile:** `{prv.get('dcr_holdout_p5', 'N/A')}`")
            md_lines.append(f"- **Closer to Train Rate:** `{prv.get('closer_to_train_rate', 0.0) * 100:.2f}%` (~50% indicates no training data memorization)")
            md_lines.append("")

            md_lines.append("### Pillar 5: Utility")
            md_lines.append(f"- **Baseline Real Train (TRTR) Macro F1:** `{utl.get('trtr_macro_f1', 'N/A')}`")
            md_lines.append(f"- **Synthetic Only (TSTR) Macro F1:** `{utl.get('tstr_macro_f1', 'N/A')}`")
            md_lines.append(f"- **Machine Learning Efficacy Ratio (TSTR / TRTR):** `{utl.get('ml_efficacy_ratio', 'N/A')}`")
            md_lines.append(f"- **Combined Real + Synthetic (TR+S) Macro F1:** `{utl.get('trs_macro_f1', 'N/A')}`")
            md_lines.append(f"- **Augmentation Gain (TR+S - TRTR):** `{utl.get('aug_gain_macro_f1', 'N/A')}`")
            md_lines.append(f"- **Rare Attack Augmentation Gain:** `{utl.get('aug_gain_rare_f1', 'N/A')}`")
            md_lines.append("")
        else:
            md_lines.append("*Synthetic quality evaluation was not executed or not available.*")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

        # 5. Dataset Distribution Distances
        md_lines.append("## 5. Dataset Distribution Distances to Validation Distribution")
        md_lines.append("")
        md_lines.append(f"- **Mean Normalized Wasserstein Distance:** `{dist_report.get('mean_normalized_wasserstein_distance', 'N/A')}` (Lower is closer)")
        md_lines.append(f"- **Standardized Centroid Euclidean Distance:** `{dist_report.get('centroid_euclidean_distance', 'N/A')}`")
        md_lines.append(f"- **Continuous Features Analyzed:** `{dist_report.get('continuous_features_analyzed', 'N/A')}`")
        md_lines.append("")
        top_div = dist_report.get("top_divergent_features", [])
        if top_div:
            md_lines.append("Top Divergent Continuous Features:")
            md_lines.append("| Feature | Normalized Wasserstein Distance |")
            md_lines.append("| :--- | :---: |")
            for feat, d_val in top_div:
                md_lines.append(f"| `{feat}` | {d_val:.5f} |")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

        # 6. Optimal Hyperparameters Discovered
        md_lines.append("## 6. Optimal Hyperparameters Summary")
        md_lines.append("")
        md_lines.append("### Round 1: Intra-Attack Resampler (SMOTE-NC)")
        r1_p = best_params_payload.get("round1_resampler_best_params", {})
        md_lines.append(f"- **Optimal $y_{{\\max}}$:** `{r1_p.get('best_y_max', 'N/A')}`")
        md_lines.append(f"- **Downstream Score:** `{r1_p.get('best_score', 'N/A')}`")
        md_lines.append(f"- **Synthetic Deficit Quota:** `{json.dumps(r1_p.get('syn_deficit', {}))}`")
        md_lines.append("")

        md_lines.append(f"### Round 2: Tabular Generators ({self.generator_type.upper()})")
        r2_p = best_params_payload.get("round2_generator_best_params", {})
        md_lines.append(f"- **Model Architecture:** `{r2_p.get('generator_type', self.generator_type).upper()}`")
        md_lines.append(f"- **Gradient Updates Budget:** `{r2_p.get('fixed_gradient_updates', self.fixed_gradient_updates):,}`")
        md_lines.append(f"- **Weights Checkpoint:** `{r2_p.get('weights_checkpoint', 'N/A')}`")
        md_lines.append(f"- **Downstream Score:** `{r2_p.get('val_score', 'N/A')}`")
        md_lines.append("")

        md_lines.append("### Round 3: Synthetic Sample Filter")
        r3_p = best_params_payload.get("round3_sample_filter_best_params", {})
        md_lines.append(f"- **Optimal OOF Confidence Threshold:** `{r3_p.get('best_oof_threshold', 'N/A')}`")
        md_lines.append(f"- **ENN Boundary Cleaner Neighbors:** `{r3_p.get('enn_neighbors', 3)}`")
        md_lines.append(f"- **Retained High-Fidelity Synthetic Flows:** `{r3_p.get('kept_samples', 'N/A')}`")
        md_lines.append(f"- **Downstream Score:** `{r3_p.get('best_score', 'N/A')}`")
        md_lines.append("")

        md_lines.append("### Final Round: Final IDS Model (Two-Tier Classifier)")
        r4_p = best_params_payload.get("final_round_ids_model_best_params", {})
        md_lines.append("```json")
        md_lines.append(json.dumps(r4_p, indent=2))
        md_lines.append("```")
        md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

        # 7. Artifact Manifest
        md_lines.append("## 7. Exported Artifacts Manifest")
        md_lines.append("")
        md_lines.append(f"All pipeline outputs and checkpoints have been exported to `{self.output_dir}`:")
        md_lines.append("")
        md_lines.append("| Artifact Filename | Type | Description |")
        md_lines.append("| :--- | :--- | :--- |")
        md_lines.append("| `evaluation_report.md` | Markdown Report | Comprehensive evaluation, per-class metrics, quality benchmarks |")
        md_lines.append("| `best_params.json` | JSON Config | All optimal hyperparameters across components 1-4 |")
        md_lines.append("| `pipeline_tuning_summary.json` | JSON Metrics | Complete structured numerical metrics across all splits and rounds |")
        md_lines.append(f"| `round2_best_weights.pt` | PyTorch Weights | Trained {self.generator_type.upper()} generator model checkpoint |")
        md_lines.append("| `final_ids_model.joblib` | Model State | Trained Two-Tier IDS detector (Tier-1 + Anomaly + Tier-2) |")
        md_lines.append("| `unsw_nb15_final_pipeline_train.csv` | Dataset CSV | Final augmented NetFlow training set with balanced minority flows |")
        md_lines.append("| `synthetic_samples_kept.csv` | Dataset CSV | Quality-filtered synthetic flows generated by the pipeline |")
        md_lines.append("")

        report_path = self.output_dir / "evaluation_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"  [Artifact Export] Saved Markdown Evaluation Report: {report_path.name}\n")
        return report_path

    # ------------------------------------------------------------------------
    # Master Sequential Pipeline Execution
    # ------------------------------------------------------------------------
    def run_pipeline(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        holdout_df: Optional[pd.DataFrame] = None,
        candidate_ymax: Optional[List[int]] = None,
        candidate_thresholds: Optional[List[float]] = None,
        comp2_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes the entire sequential pipeline across all 4 rounds:
          Round 1: Tune Comp1 (y_max)
          Round 2: Tune Comp2 (CTGAN/TVAE)
          Round 3: Tune Comp3 (OOF threshold & ENN)
          Final Round: Tune Comp4 (Final IDS Model) & Comprehensive Reporting
        """
        start_time = time.time()
        print("=" * 85)
        print("STARTING FULL SEQUENTIAL TUNING PIPELINE (ROUNDS 1 TO 4)")
        print("=" * 85)
        print(f"Train Flows:     {len(train_df):,}")
        print(f"Val Flows:       {len(val_df):,}")
        print(f"Test Flows:      {len(test_df):,}")
        print(f"Holdout Flows:   {len(holdout_df):,}" if holdout_df is not None else "Holdout: None")
        print(f"Target Ceiling:  t_max = {self.t_max:,}")
        print(f"Convergence eps: {self.epsilon}")
        print(f"Generator Type:  {self.generator_type.upper()} ({self.fixed_gradient_updates:,} gradient updates)")

        # Round 1
        self.run_round1_tune_comp1(train_df=train_df, val_df=val_df, candidate_ymax=candidate_ymax)

        # Round 2
        self.run_round2_tune_comp2(train_df=train_df, val_df=val_df, epochs=comp2_epochs)

        # Round 3
        self.run_round3_tune_comp3(train_df=train_df, val_df=val_df, candidate_thresholds=candidate_thresholds)

        # Final Round
        final_results = self.run_final_round_tune_comp4(
            val_df=val_df,
            test_df=test_df,
            holdout_df=holdout_df,
        )

        total_elapsed = time.time() - start_time
        print("=" * 85)
        print(f"Sequential Tuning Pipeline Successfully Completed in {total_elapsed:.2f}s ({total_elapsed / 60:.2f} mins)!")
        print("=" * 85 + "\n")
        return final_results


# ============================================================================
# CLI Entrypoint for Standalone Execution
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="NF-UNSW-NB15-v3 Sequential Multi-Round Tuning Pipeline Orchestrator")
    parser.add_argument("--train-csv", type=str, default=None, help="Path to training set CSV (defaults to dev_splits/NF-UNSW-NB15-v3_train.csv)")
    parser.add_argument("--val-csv", type=str, default=None, help="Path to validation set CSV")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to test set CSV")
    parser.add_argument("--holdout-csv", type=str, default=None, help="Path to holdout set CSV")
    parser.add_argument("--generator", type=str, choices=["ctgan", "tvae"], default="ctgan", help="Generator architecture for Comp2")
    parser.add_argument("--t-max", type=int, default=10000, help="Target sample ceiling per class (default: 10,000)")
    parser.add_argument("--epsilon", type=float, default=0.002, help="Stopping improvement epsilon between rounds")
    parser.add_argument("--fixed-updates", type=int, default=42000, help="Fixed gradient update budget for Comp2 (default: 42,000)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        "--dest-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Destination directory to export best parameters, weights, and evaluation reports (default: <dataset_dir>/outputs)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir, _, final_dir = resolve_paths()

    # Resolve dataset paths if not provided
    train_path = Path(args.train_csv) if args.train_csv else final_dir / "dev_splits/NF-UNSW-NB15-v3_train.csv"
    val_path = Path(args.val_csv) if args.val_csv else final_dir / "dev_splits/NF-UNSW-NB15-v3_val.csv"
    test_path = Path(args.test_csv) if args.test_csv else final_dir / "dev_splits/NF-UNSW-NB15-v3_test.csv"
    holdout_path = Path(args.holdout_csv) if args.holdout_csv else final_dir / "holdout_set/NF-UNSW-NB15-v3.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        print(f"Error: Required dataset splits not found under {final_dir / 'dev_splits'}.")
        print("Please execute Steps 1 to 3 in `nf_unsw_resampling_v2.ipynb` or run `intra_attack_resampler.py --full-pipeline` first.")
        sys.exit(1)

    print(f"Loading dataset splits:")
    print(f"  Train:   {train_path}")
    print(f"  Val:     {val_path}")
    print(f"  Test:    {test_path}")
    print(f"  Holdout: {holdout_path if holdout_path.exists() else 'None'}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    df_holdout = pd.read_csv(holdout_path) if holdout_path.exists() else None

    orchestrator = SequentialTuningPipeline(
        t_max=args.t_max,
        epsilon=args.epsilon,
        generator_type=args.generator,
        fixed_gradient_updates=args.fixed_updates,
        random_state=args.random_state,
        final_dir=final_dir,
        output_dir=args.output_dir,
    )

    print(f"Destination Output Folder: {orchestrator.output_dir}\n")

    orchestrator.run_pipeline(
        train_df=df_train,
        val_df=df_val,
        test_df=df_test,
        holdout_df=df_holdout,
    )

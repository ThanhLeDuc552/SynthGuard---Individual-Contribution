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
            weights_dir=self.final_dir / f"generators/{self.generator_type.upper()}/weights",
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

    # ------------------------------------------------------------------------
    # Round 1: Tune Comp1 Only (y_max) Based on Comp4 Evaluation
    # ------------------------------------------------------------------------
    def run_round1_tune_comp1(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        candidate_ymax: Optional[List[int]] = None,
        use_optuna: bool = False,
    ) -> Dict[str, Any]:
        """
        Round 1: Tunes y_max in Comp1.
        Evaluates each candidate via Comp4 on val_df.
        Advances to Round 2 when improvement < epsilon.
        """
        print("\n" + "#" * 85)
        print("PIPELINE ROUND 1: TUNING COMP1 (y_max) BASED ON COMP4 EVALUATION")
        print("#" * 85)

        comp1_res = self.tuner.tune_resampler_ymax(
            train_df=train_df,
            val_df=val_df,
            ids_model=self.ids_model,
            candidate_ymax=candidate_ymax,
            use_optuna=use_optuna,
        )

        self.best_y_max = comp1_res["best_y_max"]
        self.best_train_resampled = comp1_res["best_results"]["df_train_resampled"]
        self.best_syn_deficit = comp1_res["best_results"]["syn_deficit"]
        self.best_target_counts = comp1_res["best_results"]["dict_tc"]

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
        train_attacks = train_df[train_df["Label"] == 1].copy().reset_index(drop=True)
        feature_cols = [c for c in train_attacks.columns if c not in ["Label", "Attack"]]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

        # Check baseline score from Round 1
        r1_score = self.round_history.get("round1", {}).get("best_score", 0.0)

        # Train generator with fixed gradient updates (42,000 steps)
        print(f"Training {self.generator_type.upper()} with fixed gradient update budget: {self.fixed_gradient_updates:,} steps...")
        self.generators.train(
            train_attacks=train_attacks,
            syn_deficit=self.best_syn_deficit,
            feature_cols=feature_cols,
            categorical_cols=cat_cols,
            batch_size=batch_size,
            epochs=epochs,
        )
        self.generators.save_weights("round2_best_weights.pt")

        # Sample 10 * n_syn candidate flows (Step 8)
        df_candidate_pool = self.generators.sample(syn_deficit=self.best_syn_deficit, multiplier=10)

        # Baseline evaluation with default filtering
        filter_res = self.sample_filter.filter_samples(
            synthetic_df=df_candidate_pool,
            real_train_df=train_attacks,
            syn_deficit=self.best_syn_deficit,
            oof_threshold=0.6,
            run_enn=True,
        )
        df_syn_clean = filter_res["kept_df"]

        # Train IDS model on augmented training set
        df_train_augmented = pd.concat([train_benign, train_attacks, df_syn_clean], ignore_index=True)
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
        self.best_train_resampled = pd.concat([train_benign, train_attacks, best_clean_syn], ignore_index=True)

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

        # Comprehensive Evaluation across all benchmark partitions
        print("\n>>> Computing Final Benchmark Evaluation Metrics across Partitions...")
        val_eval = self.ids_model.evaluate(val_df, split_name="val", verbose=True)
        test_eval = self.ids_model.evaluate(test_df, split_name="test", verbose=True)

        holdout_eval = None
        if holdout_df is not None:
            holdout_eval = self.ids_model.evaluate(holdout_df, split_name="holdout", verbose=True)

        # Dataset Distance Calculation to Validation Set Distribution
        dist_report = self._compute_distribution_distances(val_df)

        # Export final augmented datasets & reports
        self._export_final_pipeline_outputs(val_eval, test_eval, holdout_eval, dist_report)

        final_summary = {
            "final_params": comp4_tune_res["best_params"],
            "val_metrics": val_eval,
            "test_metrics": test_eval,
            "holdout_metrics": holdout_eval,
            "distribution_distances": dist_report,
            "round_history": self.round_history,
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

        # Wasserstein Distance per continuous attribute
        w_distances = {}
        for col in cont_cols:
            v_val = pd.to_numeric(val_attacks[col], errors="coerce").fillna(0.0).values
            v_syn = pd.to_numeric(resampled_attacks[col], errors="coerce").fillna(0.0).values
            # Normalize by range to ensure comparability across features
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

    # ------------------------------------------------------------------------
    # Helper: Export Final Pipeline Outputs
    # ------------------------------------------------------------------------
    def _export_final_pipeline_outputs(
        self,
        val_eval: Dict[str, Any],
        test_eval: Dict[str, Any],
        holdout_eval: Optional[Dict[str, Any]],
        dist_report: Dict[str, Any],
    ):
        """Exports finalized datasets, metrics summaries, and diagnostic tables."""
        out_multi_dir = self.final_dir / "IDS-multiclass-classification"
        out_multi_dir.mkdir(parents=True, exist_ok=True)

        # Save final resampled training set
        if self.best_train_resampled is not None:
            final_train_path = out_multi_dir / "unsw_nb15_final_pipeline_train.csv"
            self.best_train_resampled.to_csv(final_train_path, index=False)
            print(f"  Exported Final Resampled Train: {final_train_path.name} ({len(self.best_train_resampled):,} flows)")

        # Compile JSON summary
        summary_payload = {
            "configuration": {
                "t_max": self.t_max,
                "best_y_max": self.best_y_max,
                "generator_type": self.generator_type,
                "fixed_gradient_updates": self.fixed_gradient_updates,
                "epsilon": self.epsilon,
                "target_metric": self.metric_key,
            },
            "tuning_rounds_summary": self.round_history,
            "validation_performance": {k: v for k, v in val_eval.items() if k != "confusion_matrix"},
            "test_performance": {k: v for k, v in test_eval.items() if k != "confusion_matrix"},
            "holdout_performance": {k: v for k, v in holdout_eval.items() if k != "confusion_matrix"} if holdout_eval else None,
            "distribution_distances": dist_report,
        }

        report_json_path = self.final_dir / "pipeline_tuning_summary.json"
        with open(report_json_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"  Exported Pipeline Benchmark Summary: {report_json_path.name}\n")

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
        print("Please execute Steps 1 to 3 in `nf_unsw_resampling_v2.ipynb` or run `new_nf_unsw_pipeline.py --full-pipeline` first.")
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
    )

    orchestrator.run_pipeline(
        train_df=df_train,
        val_df=df_val,
        test_df=df_test,
        holdout_df=df_holdout,
    )

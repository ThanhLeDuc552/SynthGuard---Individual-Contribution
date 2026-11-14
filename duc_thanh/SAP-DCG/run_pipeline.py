#!/usr/bin/env python3
"""
Master Execution Script for the 4-Round Sequential Tuning Pipeline.

Implements the complete pipeline workflow defined in components.txt and docs/fixes.txt:
- Data Splitting & Disaggregation: Uses DatasetSplitter to disaggregate Benign and Attack flows
  upfront, and fits the SyntheticSampleFilter (OOF 5-fold predictor & ENN index) immediately.
- Round 1: Exhaustively tunes IntraAttackResampler (SMOTE-NC y_max in [1, 5, 10, 15, 20, 25, 30, 40])
  based on FinalIDSModel evaluation; plots y_max tuning curve.
- Round 2: Tunes TabularGenerators (CTGAN / TVAE) based on FinalIDSModel evaluation with fixed 42k gradient updates.
- Round 3: Exhaustively tunes SyntheticSampleFilter (OOF threshold in [0.4, 0.5, 0.6, 0.7, 0.8] & ENN n=3)
  based on FinalIDSModel evaluation; plots threshold tuning curve and consolidated multi-panel summary.
- Final Round: Tunes FinalIDSModel (Two-Tier IDS Model with custom rare-boost class weighting equation)
  on best augmented data, then reports final metrics across Val, Test, and Holdout sets, along with
  dataset distance to the validation distribution.

All optimal hyperparameters, model weights, resampled datasets, tuning plots, metrics JSON, and a comprehensive
Markdown evaluation report (`evaluation_report.md`) are exported to the designated destination folder.

Usage:
    python run_pipeline.py --help
    python run_pipeline.py --generator ctgan --output-dir outputs/run_experiment_1
"""

import sys
from pathlib import Path

# Add SAP-DCG root and subdirectories to sys.path
curr_dir = Path(__file__).resolve().parent
for p in [curr_dir, curr_dir / "components", curr_dir / "orchestration"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from orchestration.pipeline_orchestrator import SequentialTuningPipeline, parse_args
    from components.intra_attack_resampler import resolve_paths
except ImportError:
    from pipeline_orchestrator import SequentialTuningPipeline, parse_args
    from intra_attack_resampler import resolve_paths
import pandas as pd


def main():
    args = parse_args()
    base_dir, _, final_dir = resolve_paths()

    train_path = Path(args.train_csv) if args.train_csv else final_dir / "dev_splits/NF-UNSW-NB15-v3_train.csv"
    val_path = Path(args.val_csv) if args.val_csv else final_dir / "dev_splits/NF-UNSW-NB15-v3_val.csv"
    test_path = Path(args.test_csv) if args.test_csv else final_dir / "dev_splits/NF-UNSW-NB15-v3_test.csv"
    holdout_path = Path(args.holdout_csv) if args.holdout_csv else final_dir / "holdout_set/NF-UNSW-NB15-v3.csv"

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        print(f"Error: Required dataset splits not found under {final_dir / 'dev_splits'}.")
        print("Please execute Steps 1 to 3 in `nf_unsw_resampling_v2.ipynb` or run `intra_attack_resampler.py --full-pipeline` first.")
        sys.exit(1)

    print("=" * 85)
    print("NF-UNSW-NB15-v3 Master Pipeline (components.txt)")
    print("=" * 85)
    print(f"Loading partitions from: {final_dir}")
    print(f"  Train:   {train_path.name}")
    print(f"  Val:     {val_path.name}")
    print(f"  Test:    {test_path.name}")
    print(f"  Holdout: {holdout_path.name if holdout_path.exists() else 'None'}\n")

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

    print(f"Destination Output Directory: {orchestrator.output_dir}\n")

    orchestrator.run_pipeline(
        train_df=df_train,
        val_df=df_val,
        test_df=df_test,
        holdout_df=df_holdout,
    )


if __name__ == "__main__":
    main()

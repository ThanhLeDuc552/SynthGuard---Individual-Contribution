"""
Component Modules for NF-UNSW-NB15-v3 Data Processing & Resampling.

Components:
- Component 1: IntraAttackResampler (intra_attack_resampler.py)
- Component 2: TabularGenerators (tabular_generators.py)
- Component 3: SyntheticSampleFilter (synthetic_sample_filter.py)
- Component 4: FinalIDSModel (final_ids_model.py)
"""

from .dataset_splitter import (
    DatasetSplitter,
    split_benign_attack,
)
from .intra_attack_resampler import (
    IntraAttackResampler,
    SHORTCUT_COLS,
    CATEGORICAL_FEATURES,
    resolve_paths,
    run_full_pipeline,
)
from .tabular_generators import (
    TabularGenerators,
    NetFlowPreprocessor,
    CTGAN_DIMS,
    CAT_CANDIDATES,
)
from .synthetic_sample_filter import SyntheticSampleFilter
from .final_ids_model import (
    FinalIDSModel,
    Preprocessor,
    AnomalyDetector,
    evaluate_synthetic_quality,
)

__all__ = [
    "DatasetSplitter",
    "split_benign_attack",
    "IntraAttackResampler",
    "TabularGenerators",
    "NetFlowPreprocessor",
    "CTGAN_DIMS",
    "CAT_CANDIDATES",
    "SyntheticSampleFilter",
    "FinalIDSModel",
    "Preprocessor",
    "AnomalyDetector",
    "evaluate_synthetic_quality",
    "SHORTCUT_COLS",
    "CATEGORICAL_FEATURES",
    "resolve_paths",
    "run_full_pipeline",
]


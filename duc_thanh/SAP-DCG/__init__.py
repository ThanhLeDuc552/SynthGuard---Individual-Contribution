"""
SAP-DCG: Sequential Adaptive Processing & Deep Conditional Generative Framework.

Unified package providing modular intrusion detection resampling, deep generative modeling,
protocol-aware sample filtering, final IDS classification, and multi-round hyperparameter tuning.
"""

from .components import (
    IntraAttackResampler,
    TabularGenerators,
    NetFlowPreprocessor,
    SyntheticSampleFilter,
    FinalIDSModel,
    Preprocessor,
    AnomalyDetector,
    evaluate_synthetic_quality,
    SHORTCUT_COLS,
    CATEGORICAL_FEATURES,
    resolve_paths,
    run_full_pipeline,
)
from .orchestration import (
    PipelineTuner,
    SequentialTuningPipeline,
    parse_args,
)

__all__ = [
    # Core Components
    "IntraAttackResampler",
    "TabularGenerators",
    "NetFlowPreprocessor",
    "SyntheticSampleFilter",
    "FinalIDSModel",
    "Preprocessor",
    "AnomalyDetector",
    "evaluate_synthetic_quality",
    # Orchestration & Tuning
    "PipelineTuner",
    "SequentialTuningPipeline",
    "parse_args",
    # Constants & Helpers
    "SHORTCUT_COLS",
    "CATEGORICAL_FEATURES",
    "resolve_paths",
    "run_full_pipeline",
]

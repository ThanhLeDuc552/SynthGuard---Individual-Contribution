"""
SAP-DCG: Sequential Adaptive Processing & Deep Conditional Generative Framework.

Unified package providing modular intrusion detection resampling, deep generative modeling,
protocol-aware sample filtering, final IDS classification, and multi-round hyperparameter tuning.
"""

from .components import (
    IntraAttackResampler,
    TabularGenerators,
    SyntheticSampleFilter,
    FinalIDSModel,
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
    "SyntheticSampleFilter",
    "FinalIDSModel",
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

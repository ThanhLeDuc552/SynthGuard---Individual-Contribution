#!/usr/bin/env python3
"""
Tabular Deep Generative Models Facade (Component 2)

CTGAN and TVAE implementations have been split into dedicated modules:
- `components/ctgan_generator.py`: `CTGANGenerator`, `NetFlowPreprocessor`, `CTGAN_DIMS`, `CAT_CANDIDATES`
- `components/tvae_generator.py`: `TVAEGenerator`, `TVAEPreprocessor`, `TVAE_CODE_COLS`, `TVAE_DIMS`, `TVAE_DEFAULT_PARAMS`

This module re-exports those symbols and provides `TabularGenerators` as a
backward-compatible factory dispatching to `CTGANGenerator` or `TVAEGenerator`.
"""

import sys
from pathlib import Path
from typing import Optional, Union

curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.append(str(curr_dir))

from ctgan_generator import (
    CTGANGenerator,
    NetFlowPreprocessor,
    CTGAN_DIMS,
    CAT_CANDIDATES,
    ID_COLS,
    TOP_K_CATEGORICAL,
    SKEW_LIMIT,
)
from tvae_generator import (
    TVAEGenerator,
    TVAEPreprocessor,
    TVAE_CODE_COLS,
    TVAE_DIMS,
    TVAE_DEFAULT_PARAMS,
)


class TabularGenerators:
    """
    Backward-compatible factory for CTGANGenerator and TVAEGenerator.
    Instantiating `TabularGenerators(model_type=...)` returns a `CTGANGenerator`
    or `TVAEGenerator` instance directly.
    """

    def __new__(
        cls,
        model_type: str = "ctgan",
        fixed_gradient_updates: int = 42000,
        random_state: int = 42,
        device: Optional[str] = None,
        weights_dir: Optional[Union[str, Path]] = None,
    ):
        kind = model_type.lower()
        if kind == "ctgan":
            return CTGANGenerator(
                fixed_gradient_updates=fixed_gradient_updates,
                random_state=random_state,
                device=device,
                weights_dir=weights_dir,
            )
        if kind == "tvae":
            return TVAEGenerator(
                fixed_gradient_updates=fixed_gradient_updates,
                random_state=random_state,
                device=device,
                weights_dir=weights_dir,
            )
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'ctgan' or 'tvae'.")


__all__ = [
    "TabularGenerators",
    "CTGANGenerator",
    "TVAEGenerator",
    "NetFlowPreprocessor",
    "TVAEPreprocessor",
    "CTGAN_DIMS",
    "TVAE_CODE_COLS",
    "TVAE_DIMS",
    "TVAE_DEFAULT_PARAMS",
    "CAT_CANDIDATES",
    "ID_COLS",
    "TOP_K_CATEGORICAL",
    "SKEW_LIMIT",
]


if __name__ == "__main__":
    print("Component 2 (tabular_generators.py) loaded successfully.")

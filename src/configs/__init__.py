"""
Backwards-compatible exports for config dataclasses.

Dataclasses now live in `configs.schema` so that parsing/IO concerns can be kept separate
from the config schema definition.
"""

from configs.schema import (  # noqa: F401
    AugmentationConfig,
    ClusterConfig,
    Config,
    Configs,
    EvaluationConfig,
    TrainingConfig,
)

__all__ = [
    "Config",
    "AugmentationConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ClusterConfig",
    "Configs",
]
import torch
from pathlib import Path

from src.config import (
    Config,
    AugmentationConfig,
    TrainingConfig,
    EvaluationConfig,
    ClusterConfig,
    Configs,
)
from src.utils.yaml_loader import load_yaml

def load_config(config_path: Path) -> Configs:
    path = Path(config_path)
    data = load_yaml(path)

    config_data = dict(data["config"])
    if config_data.get("device") is None:
        config_data["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    augmentation_data = dict(data["augmentation"])
    augmentation_data["scale"] = tuple(augmentation_data["scale"])
    augmentation_data["mean"] = tuple(augmentation_data["mean"])
    augmentation_data["std"] = tuple(augmentation_data["std"])

    config = Config(**config_data)
    augmentation_config = AugmentationConfig(**augmentation_data)
    training_config = TrainingConfig(**data["training"])
    evaluation_config = EvaluationConfig(**data["evaluation"])
    cluster_config = ClusterConfig(**data["cluster"])

    return Configs(config, augmentation_config, training_config, evaluation_config, cluster_config)
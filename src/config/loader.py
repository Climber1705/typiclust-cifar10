import yaml
import torch
from typing import Union
from pathlib import Path

from .schema import (
    Config,
    PathsConfig,
    TrainingConfig,
    EvaluationConfig,
    AugmentationConfig,
    ClusterConfig,
)

def load_config(path: Union[str, Path]) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        seed=raw["seed"],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),

        paths=PathsConfig(
            data_directory=raw["paths"]["data_directory"],
            model_directory=raw["paths"]["model_directory"],
            representation_model_name=raw["paths"]["representation_model_name"],
        ),

        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),

        augmentation=AugmentationConfig(
            num_views=raw["augmentation"]["num_views"],
            size=raw["augmentation"]["size"],
            scale=tuple(raw["augmentation"]["scale"]),
            p_random_horizontal_flip=raw["augmentation"]["p_random_horizontal_flip"],
            brightness=raw["augmentation"]["brightness"],
            contrast=raw["augmentation"]["contrast"],
            saturation=raw["augmentation"]["saturation"],
            hue=raw["augmentation"]["hue"],
            p_color_jitter=raw["augmentation"]["p_color_jitter"],
            p_grayscale=raw["augmentation"]["p_grayscale"],
            mean=tuple(raw["augmentation"]["mean"]),
            std=tuple(raw["augmentation"]["std"]),
        ),

        cluster=ClusterConfig(**raw["cluster"]),
    )
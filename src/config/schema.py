import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PathsConfig:
    data_directory: str
    model_directory: str
    representation_model_name: str

@dataclass
class AugmentationConfig:
    num_views: int
    size: int
    scale: Tuple[float, float]
    p_random_horizontal_flip: float
    brightness: float
    contrast: float
    saturation: float
    hue: float
    p_color_jitter: float
    p_grayscale: float
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float

    warmup_epochs: int
    start_factor: float
    end_factor: float

@dataclass
class EvaluationConfig:
    epochs: int
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    nesterov: bool

@dataclass
class ClusterConfig:
    max_clusters: int
    B: int
    min_cluster_size: int

@dataclass
class Config:
    seed: int
    device: torch.device
    paths: PathsConfig
    augmentation: AugmentationConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    cluster: ClusterConfig
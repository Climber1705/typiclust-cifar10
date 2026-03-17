import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    seed: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_directory: str = "../data"
    model_directory: str = "../models"
    representation_model_name: str = "simclr_cifar10_dev.pth"


@dataclass
class AugmentationConfig:
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
class Configs:
    config: Config
    augmentation: AugmentationConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    cluster: ClusterConfig


import torch
import torchvision.transforms as transforms
from typing import List

from src.configs import AugmentationConfig

class StochasticDataAugmentation:
    def __init__(self, base_transform: transforms.Compose, num_views: int = 2) -> None:
        self.base_transform = base_transform
        self.num_views = num_views

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [self.base_transform(x) for _ in range(self.num_views)]

def get_contrastive_base_transform(config: AugmentationConfig) -> StochasticDataAugmentation:
    contrastive_base_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=config.size,
            scale=config.scale
        ),
        transforms.RandomHorizontalFlip(
            p=config.p_random_horizontal_flip
        ),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=config.brightness,
                contrast=config.contrast,
                saturation=config.saturation,
                hue=config.hue
            )
        ], p=config.p_color_jitter),
        transforms.RandomGrayscale(
            p=config.p_grayscale
        ),  
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.mean,
            std=config.std
        ),
    ])

    return StochasticDataAugmentation(
        contrastive_base_transform,
        num_views=2
    )

def get_embedding_transform(config: AugmentationConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.mean,
            std=config.std
        ),
    ])

def get_classifier_train_transform(config: AugmentationConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(config.size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.mean, 
            std=config.std
        ),
    ])
import torchvision.transforms as transforms

from src.config import AugmentationConfig
from src.data.augmentations import StochasticDataAugmentation

def get_contrastive_base_transform(augmentation_config: AugmentationConfig) -> StochasticDataAugmentation:
    base_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=augmentation_config.size,
            scale=augmentation_config.scale
        ),
        transforms.RandomHorizontalFlip(
            p=augmentation_config.p_random_horizontal_flip
        ),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=augmentation_config.brightness,
                contrast=augmentation_config.contrast,
                saturation=augmentation_config.saturation,
                hue=augmentation_config.hue
            )
        ], p=augmentation_config.p_color_jitter),
        transforms.RandomGrayscale(
            p=augmentation_config.p_grayscale
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=augmentation_config.mean,
            std=augmentation_config.std
            ),
        ])
    return StochasticDataAugmentation(base_transform, num_views=augmentation_config.num_views)

def get_embedding_transform(augmentation_config: AugmentationConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=augmentation_config.mean,
            std=augmentation_config.std
        ),
    ])

def get_classifier_train_transform(augmentation_config: AugmentationConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(augmentation_config.size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=augmentation_config.mean, 
            std=augmentation_config.std
        ),
    ])
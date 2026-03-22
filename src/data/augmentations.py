import torch
import torchvision.transforms as transforms
from typing import List

class StochasticDataAugmentation:

    def __init__(self, base_transform: transforms.Compose, num_views: int = 2) -> None:
        self.base_transform = base_transform
        self.num_views = num_views

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [self.base_transform(x) for _ in range(self.num_views)]
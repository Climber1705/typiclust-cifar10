import torch
import torchvision

from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple
from PIL import Image

class SimCLRDataset(Dataset):
    def __init__(self, root: str, train: bool = True, download: bool = True, transform: Optional[Callable] = None, num_views: int = 2) -> None:
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform
        self.num_views = num_views

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[index]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        views = [self.transform(image) for _ in range(self.num_views)]
        return views, label
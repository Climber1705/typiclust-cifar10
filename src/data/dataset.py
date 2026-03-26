import torchvision
from torch.utils.data import Dataset
from typing import Callable, Tuple
from PIL import Image

class SimCLRDataset(Dataset):
    def __init__(self, root: str, transform: Callable, train: bool = True, download: bool = True) -> None:
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        image, label = self.dataset[index]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.transform(image), label
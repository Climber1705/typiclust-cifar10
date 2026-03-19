from typing import List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from src.config import EvaluationConfig
from src.utils.training_utils import build_resnet18_classifier
from src.utils import build_loader

def train_classifier(
    config: EvaluationConfig,
    labeled_indices: List[int],
    train_dataset: Dataset,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 64,
    model_override: Optional[nn.Module] = None
) -> nn.Module:
    classifier = model_override if model_override else build_resnet18_classifier(device=device)
    
    subset = Subset(train_dataset, labeled_indices)
    loader = build_loader(
        dataset=subset,
        batch_size=min(batch_size, len(labeled_indices)),
        shuffle=True,
        drop_last=False
    )

    optimizer = optim.SGD(
        classifier.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    classifier.train()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(classifier(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return classifier
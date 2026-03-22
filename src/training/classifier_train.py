import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
from typing import List, Optional
from torch.utils.data import Dataset

from src.config import EvaluationConfig
from src.model import build_resnet18_classifier
from src.utils import build_loader

def train_classifier(
    config: EvaluationConfig,
    labeled_indices: List[int],
    train_dataset: Dataset,
    device: torch.device,
    model_override: Optional[nn.Module] = None
) -> nn.Module:
    classifier = model_override if model_override else build_resnet18_classifier(device=device)
    
    subset = Subset(train_dataset, labeled_indices)
    loader = build_loader(
        dataset=subset,
        batch_size=min(config.batch_size, len(labeled_indices)),
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    classifier.train()
    for _ in range(config.epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(classifier(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return classifier
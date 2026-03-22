import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def evaluate_classifier(
    classifier: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> float:
    classifier.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total
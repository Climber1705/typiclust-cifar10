import torch
import torchvision
import torch.nn as nn

def build_resnet18_classifier(
    device: torch.device,
    num_classes: int = 10
) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    return model

def add_dropout_to_resnet(model: nn.Module, p: float = 0.5) -> nn.Module:
    for _, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            for i, m in enumerate(module):
                if isinstance(m, nn.ReLU):
                    module[i] = nn.Sequential(m, nn.Dropout(p=p))
    return model
import torch
import torch.nn as nn
import torchvision

def build_resnet18_classifier(
    device: torch.device,
    num_classes: int = 10
) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    return model

def add_dropout_to_resnet(model: nn.Module, p: float = 0.5) -> nn.Module:
    for module in model.modules():
        if isinstance(module, torchvision.models.resnet.BasicBlock):
            module.relu = nn.Sequential(module.relu, nn.Dropout(p=p))
    return model
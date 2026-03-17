import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class SimCLRModel(nn.Module):
    def __init__(self, feature_dimension: int = 128) -> None:
        super().__init__()
        self.base_encoder = torchvision.models.resnet18(weights=None)
        in_features = self.base_encoder.fc.in_features
        self.base_encoder.fc = nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, feature_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.base_encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.normalize(self.base_encoder(x), p=2, dim=1)

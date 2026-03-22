import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# SimCLR Model
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

# SimCLR Loss
class SimCLRLoss(nn.Module):
    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)

        z = F.normalize(z, dim=1)
        similarity = torch.matmul(z, z.T) / self.temperature
        
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        similarity.masked_fill_(mask, -float('inf'))

        positives = torch.cat([
            torch.diag(similarity, N),
            torch.diag(similarity, -N)
        ])

        loss = -positives + torch.logsumexp(similarity, dim=1)
        return loss.mean()
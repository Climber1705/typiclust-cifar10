import torch
import torch.nn as nn
import torch.nn.functional as F

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
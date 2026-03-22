import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_simclr(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler,
    device: torch.device,
    epochs: int,
) -> None:
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for views, _ in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x_i, x_j = views[0].to(device), views[1].to(device)
            z_i, z_j = model(x_i), model(x_j)

            loss = criterion(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        n = len(loader)
        print(
            f"Epoch {epoch + 1} | "
            f"Loss: {total_loss / n:7.4f} | "
            f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}"
        )

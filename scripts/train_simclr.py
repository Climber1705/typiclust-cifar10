import torch
import argparse
import torch.optim as optim
from pathlib import Path

from src.models import SimCLRModel, SimCLRLoss
from src.datasets import SimCLRDataset, get_contrastive_base_transform
from src.training import train_simclr
from src.utils import load_config, build_loader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the config file"
    )
    return parser.parse_args()

def train_simclr_pipeline(args):
    config = load_config(args.config)
    device = torch.device(config.config.device)
    print(f"Using device: {device}")

    model = SimCLRModel().to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.training.lr,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config.training.epochs,
    )
    criterion = SimCLRLoss()

    train_dataset = SimCLRDataset(
        root=config.config.data_directory,
        train=True,
        transform=get_contrastive_base_transform(config.augmentation)
    )
    train_loader = build_loader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True
    )

    train_simclr(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=config.training.epochs
    )

    model_path = Path(config.config.model_directory) / config.config.simclr_model_name
    torch.save(model.state_dict(), str(model_path))


if __name__ == "__main__":
    args = parse_args()
    train_simclr_pipeline(args)
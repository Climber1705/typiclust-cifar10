import torch
import argparse
import torch.optim as optim
from pathlib import Path

from src.config import load_config
from src.data import SimCLRDataset, get_contrastive_base_transform
from src.model import SimCLRModel, SimCLRLoss
from src.training import train_simclr
from src.utils import build_loader, set_seed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the config file"
    )
    return parser.parse_args()

def train_simclr_pipeline(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parent.parent
    
    config = load_config(args.config)

    set_seed(config.seed)

    data_directory = (project_root / config.paths.data_directory).resolve()
    model_directory = (project_root / config.paths.model_directory).resolve()
    
    device = torch.device(config.device)
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

    data_directory.mkdir(parents=True, exist_ok=True)
    train_dataset = SimCLRDataset(
        root=str(data_directory),
        transform=get_contrastive_base_transform(config.augmentation),
        train=True,
        download=True,
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

    model_directory.mkdir(parents=True, exist_ok=True)
    model_path = model_directory / config.paths.representation_model_name
    torch.save(model.state_dict(), str(model_path))


if __name__ == "__main__":
    args = parse_args()
    train_simclr_pipeline(args)
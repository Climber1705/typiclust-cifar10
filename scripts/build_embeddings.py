import torch
import argparse
import torchvision
import numpy as np
from pathlib import Path

from src.config import load_config
from src.data import get_embedding_transform
from src.model import load_model
from src.embeddings import build_embeddings
from src.utils import build_loader, set_seed, get_directories

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_B10.yaml",
        help="Path to the config file"
    )
    return parser.parse_args()

def build_embeddings_pipeline(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = torch.device(config.device)

    data_directory, model_directory = get_directories(config)
    model_path = model_directory / config.paths.representation_model_name

    model = load_model(model_path, device=device)

    set_seed(config.seed)

    embedding_dataset = torchvision.datasets.CIFAR10(
        root=str(data_directory),
        train=True,
        download=True,
        transform=get_embedding_transform(config.augmentation)
    )
    embedding_loader = build_loader(
        dataset=embedding_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=False
    )

    all_embeddings, all_labels = build_embeddings(
        model=model,
        loader=embedding_loader,
        device=device
    )

    np.save(data_directory / "embeddings.npy", all_embeddings)
    np.save(data_directory / "labels.npy", all_labels)

    print(f"Embeddings: {all_embeddings.shape}, Labels: {all_labels.shape}")

if __name__ == "__main__":
    args = parse_args()
    build_embeddings_pipeline(args)
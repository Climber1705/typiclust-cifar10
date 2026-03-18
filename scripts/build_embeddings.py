import torch
import argparse
import torchvision
import numpy as np
from pathlib import Path

from src.datasets import get_embedding_transform
from src.training import build_embeddings
from src.utils import load_config, load_model, build_loader

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the config file"
    )
    return parser.parse_args()

def build_embeddings_pipeline(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    model = load_model(config.config.model_directory / config.config.representation_model_name)
    device = torch.device(config.config.device)
    
    embedding_dataset = torchvision.datasets.CIFAR10(
        root=config.config.data_directory,
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

    np.save(Path(config.config.data_directory) / "embeddings.npy", all_embeddings)
    np.save(Path(config.config.data_directory) / "labels.npy", all_labels)

    print(f"Embeddings: {all_embeddings.shape}, Labels: {all_labels.shape}")

if __name__ == "__main__":
    args = parse_args()
    build_embeddings_pipeline(args)
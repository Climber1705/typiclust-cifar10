import torch
import torchvision
import argparse
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset

from src.active_learning import State, build_strategies, query_strategy
from src.config import Config, load_config
from src.data import get_embedding_transform, get_classifier_train_transform
from src.training import train_classifier
from src.evaluation import evaluate_classifier, save_results
from src.utils import set_seed, build_loader, get_directories

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the config file"
    )
    return parser.parse_args()

def build_datasets(config: Config) -> Tuple[Dataset, Dataset, Dataset]:
    data_directory, _ = get_directories(config)

    inference_dataset = torchvision.datasets.CIFAR10(
        root=str(data_directory),
        train=True,
        download=True,
        transform=get_embedding_transform(config.augmentation)
    )

    full_train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_directory),
        train=True,
        download=True,
        transform=get_classifier_train_transform(config.augmentation)
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_directory),
        train=False,
        download=True,
        transform=get_embedding_transform(config.augmentation)
    )

    return inference_dataset, full_train_dataset, test_dataset

def run_active_learning_pipeline(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = torch.device(config.device)

    data_directory, _ = get_directories(config)

    set_seed(config.seed)

    inference_dataset, full_train_dataset, test_dataset = build_datasets(config)
    test_loader = build_loader(
        dataset=test_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
    )

    STRATEGIES = build_strategies(config)

    num_rounds = 6
    num_repetitions = 10

    all_embeddings = np.load(data_directory / "embeddings.npy")

    all_results = {name: [] for name in STRATEGIES}

    for repetition_idx in range(num_repetitions):
        print(f"Repetition {repetition_idx + 1}/{num_repetitions}")

        strategies = {name: fn() for name, fn in STRATEGIES.items()}
        labeled_sets = {name: set() for name in strategies}
        accuracies = {name: [] for name in strategies}

        for round_idx in range(num_rounds):
            print(f"=== Round {round_idx + 1}/{num_rounds} (B={config.cluster.B}) ===")

            for name, strategy in strategies.items():
                state = State(
                    labeled=labeled_sets[name],
                    dataset=inference_dataset,
                    embeddings=all_embeddings,
                    model=None,
                )

                new_indices = query_strategy(name, strategy, state, config, full_train_dataset, device)
                labeled_sets[name].update(new_indices)

                evaluation_model = train_classifier(
                    config=config.evaluation,
                    labeled_indices=list(labeled_sets[name]),
                    train_dataset=full_train_dataset,
                    device=device,
                )
                accuracy = evaluate_classifier(evaluation_model, test_loader, device)
                accuracies[name].append(accuracy)
                print(f"  {name:12s} | labeled={len(labeled_sets[name]):4d} | accuracy={accuracy:.4f}")

        for name in strategies:
            all_results[name].append(accuracies[name])
        
    save_results(all_results, data_directory, budget_per_round=config.cluster.B)

if __name__ == "__main__":
    args = parse_args()
    run_active_learning_pipeline(args)
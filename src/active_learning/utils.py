import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from typing import List, Set
from torch.utils.data import Dataset, Subset

from src.utils import build_loader

def get_unlabeled_pool(dataset_size: int, labeled: Set[int]) -> List[int]:
    return list(set(range(dataset_size)) - labeled)

def get_softmax_scores(
    model: nn.Module,
    dataset: Dataset,
    indices: List[int],
    device: torch.device,
    batch_size: int = 256
) -> np.ndarray:

    subset = Subset(dataset, indices)
    loader = build_loader(
        dataset=subset,
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    probabilities = []
    with torch.no_grad():
        for images, _ in loader:
            probabilities.append(F.softmax(model(images.to(device)), dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)

def get_dropout_predictions(
    model: nn.Module,
    dataset: Dataset,
    indices: List[int],
    device: torch.device,
    passes: int = 10,
    batch_size: int = 256
) -> np.ndarray:

    model.train()
    subset = Subset(dataset, indices)
    loader = build_loader(
        dataset=subset,
        batch_size=batch_size,
        shuffle=False
    )

    all_probabilities = []

    for _ in range(passes):
        probabilities_list = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = model(x)
                probabilities = torch.softmax(logits, dim=1)
                probabilities_list.append(probabilities.cpu().numpy())

        all_probabilities.append(np.concatenate(probabilities_list))

    return np.stack(all_probabilities)

def get_gradient_embeddings(
    model: nn.Module,
    dataset: Dataset,
    indices: List[int],
    device: torch.device,
    batch_size: int = 256
) -> np.ndarray:

    if not isinstance(model, torchvision.models.ResNet):
        raise TypeError("Expects a ResNet Model")

    model.eval()
    subset = Subset(dataset, indices)
    loader = build_loader(
        dataset=subset,
        batch_size=batch_size,
        shuffle=False
    )

    embeddings = []

    backbone = nn.Sequential(*list(model.children())[:-1]).to(device)

    with torch.no_grad():

        for x, _ in loader:
            x = x.to(device)
            features = torch.flatten(backbone(x), 1)

            logits = model.fc(features)
            probabilities = torch.softmax(logits, dim=1)

            predictions = probabilities.argmax(dim=1)

            one_hot = torch.zeros_like(probabilities)
            one_hot.scatter_(1, predictions.unsqueeze(1), 1)

            gradient = probabilities - one_hot

            gradient_embeddings = torch.einsum("bi,bj->bij", gradient, features)
            gradient_embeddings = gradient_embeddings.reshape(gradient_embeddings.size(0), -1)
            embeddings.append(gradient_embeddings.cpu().numpy())

    return np.concatenate(embeddings)

def kmeans_plus_plus_seeding(
    embeddings: np.ndarray,
    budget: int
) -> List[int]:
    n = len(embeddings)
    first = np.random.randint(n)
    centers = [first]

    for _ in range(budget - 1):
        distances = pairwise_distances(embeddings, embeddings[centers])
        min_distances = distances.min(axis=1) ** 2

        probabilities = min_distances / min_distances.sum()
        next_center = np.random.choice(n, p=probabilities)
        centers.append(next_center)

    return centers
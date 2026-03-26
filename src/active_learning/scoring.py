import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset


def typicality(embeddings: np.ndarray, k: int = 20) -> np.ndarray:
    k = min(k, len(embeddings) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    mean_distance = distances[:, 1:].mean(axis=1)
    return 1.0 / np.clip(mean_distance, 1e-8, None)


def model_uncertainty(model: nn.Module, dataset: Dataset, unlabeled_indices: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []

    with torch.no_grad():
        for idx in unlabeled_indices:
            sample = dataset[idx]
            x = sample[0] if isinstance(sample, (tuple, list)) else sample
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            logits = model(x.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=-1).squeeze()
            top2 = probs.topk(2).values
            scores.append(1.0 - (top2[0] - top2[1]).item())

    return np.array(scores, dtype=np.float32)


def cluster_disagreement_uncertainty(embeddings: np.ndarray, cluster_labels: np.ndarray, k: int = 20) -> np.ndarray:
    k = min(k, len(embeddings) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    return np.array(
        [np.mean(cluster_labels[nbrs] != cluster_labels[i]) for i, nbrs in enumerate(indices[:, 1:])],
        dtype=np.float32
    )


def hybrid_score(
    typicality_scores: np.ndarray,
    uncertainty_scores: np.ndarray,
    lam: float,
) -> np.ndarray:
    return lam * minmax(typicality_scores) + (1.0 - lam) * minmax(uncertainty_scores)


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)
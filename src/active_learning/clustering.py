import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def cluster(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    if n_clusters <= 50:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    else:
        km = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=42)
    return km.fit_predict(embeddings)


def uncovered_clusters(cluster_labels: np.ndarray, labeled_indices: np.ndarray, n_clusters: int) -> list[int]:
    labeled_clusters = set(cluster_labels[labeled_indices])
    return [c for c in range(n_clusters) if c not in labeled_clusters]


def cluster_count(n_labeled: int, budget: int, max_clusters: int) -> int:
    return max(min(n_labeled + budget, max_clusters), budget)
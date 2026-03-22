import torch
import numpy as np
from typing import List, Union
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, MiniBatchKMeans

from src.active_learning.base import Strategy
from src.active_learning.state import State


class TypiclustStrategy(Strategy):
    def __init__(
        self,
        max_clusters: int = 500,
        min_cluster_size: int = 5,
        seed: int = 42,
    ) -> None:
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.seed = seed

    def _get_clustering_model(self, n_clusters: int) -> Union[KMeans, MiniBatchKMeans]:
        if n_clusters <= 50:
            return KMeans(n_clusters=n_clusters, n_init=10, random_state=self.seed)
        return MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=1024, n_init=10, random_state=self.seed
        )

    def _compute_typicality(self, cluster_embeddings: np.ndarray, K: int = 20) -> np.ndarray:
        cluster_size = cluster_embeddings.shape[0]
        K = min(K, cluster_size - 1)
        if K <= 0:
            return np.ones(cluster_size, dtype=float)

        nn_model = NearestNeighbors(n_neighbors=K + 1, metric="euclidean")
        nn_model.fit(cluster_embeddings)
        distances, _ = nn_model.kneighbors(cluster_embeddings)

        mean_distances = distances[:, 1:].mean(axis=1)
        return 1.0 / (mean_distances + 1e-8)

    def _eligible_clusters(
        self,
        cluster_labels: np.ndarray,
        labeled_mask: np.ndarray,   # bool array of length N
    ) -> List[int]:
        cluster_ids = [c for c in np.unique(cluster_labels) if c >= 0]

        labeled_counts = {c: int(labeled_mask[cluster_labels == c].sum()) for c in cluster_ids}
        cluster_sizes = {c: int((cluster_labels == c).sum()) for c in cluster_ids
}

        min_labeled = min(labeled_counts.values())

        eligible = [
            c for c in cluster_ids
            if labeled_counts[c] == min_labeled
            and cluster_sizes[c] >= self.min_cluster_size
        ]

        if not eligible:
            eligible = [c for c in cluster_ids if labeled_counts[c] == min_labeled]

        return eligible

    def query(self, state: State, budget: int, device: torch.device) -> List[int]:
        embeddings: np.ndarray = state.embeddings
        n = len(embeddings)

        labeled_mask = np.zeros(n, dtype=bool)
        labeled_mask[list(state.labeled)] = True

        n_clusters = min(len(state.labeled) + budget, self.max_clusters)
        cluster_labels: np.ndarray = self._get_clustering_model(n_clusters).fit_predict(
            embeddings
        )

        valid_labels = cluster_labels[cluster_labels >= 0]
        cluster_sizes = np.bincount(valid_labels, minlength=n_clusters)

        cluster_to_indices: dict[int, List[int]] = {}
        for idx, c in enumerate(cluster_labels):
            if c >= 0:
                cluster_to_indices.setdefault(c, []).append(idx)

        selected: List[int] = []

        for _ in range(budget):
            eligible = self._eligible_clusters(cluster_labels, labeled_mask)
            if not eligible:
                break

            chosen_cluster = max(eligible, key=lambda c: cluster_sizes[c])

            members = np.array(cluster_to_indices[chosen_cluster])
            unlabeled_members = members[~labeled_mask[members]]

            if len(unlabeled_members) == 0:
                cluster_labels[cluster_labels == chosen_cluster] = -1
                continue

            scores = self._compute_typicality(embeddings[unlabeled_members])
            chosen_idx = int(unlabeled_members[np.argmax(scores)])

            selected.append(chosen_idx)
            labeled_mask[chosen_idx] = True

        return selected
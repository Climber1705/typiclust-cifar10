from typing import List
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from typing import Set, Union

from src.config import State
from src.strategies.base import Strategy

class TypiclustStrategy(Strategy):
    def __init__(self, seed: int = 42, max_clusters: int = 500, min_cluster_size: int = 5) -> None:
        self.seed = seed
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size

    def compute_typicality(self, cluster_embeddings: np.ndarray, K: int = 20) -> np.ndarray:
        cluster_size = cluster_embeddings.shape[0]
        K = min(K, cluster_size - 1)
        if K <= 0:
            return np.ones(cluster_size, dtype=float)

        nn_model = NearestNeighbors(n_neighbors=K + 1, metric="euclidean")
        nn_model.fit(cluster_embeddings)
        distances, _ = nn_model.kneighbors(cluster_embeddings)

        mean_distances = np.mean(distances[:, 1:], axis=1)
        return 1.0 / (mean_distances + 1e-8)

    def get_clustering_model(self, n_clusters: int) -> Union[KMeans, MiniBatchKMeans]:
        if n_clusters <= 50:
            return KMeans(n_clusters=n_clusters, n_init=10, random_state=self.seed)
        return MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, n_init=10, random_state=self.seed)

    def eligible_clusters(
        self,
        cluster_labels: np.ndarray,
        labeled_set: Set[int]
    ) -> List[int]:
        cluster_ids = np.unique(cluster_labels)
        cluster_sizes = {}
        cluster_labeled_counts = {}

        for cluster_id in cluster_ids:
            if cluster_id < 0:
                continue

            members = np.where(cluster_labels == cluster_id)[0]
            cluster_sizes[cluster_id] = len(members)
            cluster_labeled_counts[cluster_id] = np.sum(np.isin(members, list(labeled_set)))

        min_labeled = min(cluster_labeled_counts.values())

        eligible = [
            cluster_id for cluster_id in cluster_ids
            if cluster_id >= 0
            and cluster_labeled_counts[cluster_id] == min_labeled
            and cluster_sizes[cluster_id] >= self.min_cluster_size
        ]

        if not eligible:
            eligible = [
                cluster_id for cluster_id in cluster_ids
                if cluster_id >= 0 and cluster_labeled_counts[cluster_id] == min_labeled
            ]

        return eligible
    
    def select_largest_cluster(self, cluster_labels: np.ndarray, eligible: List[int]) -> int:
        return max(eligible, key=lambda cluster: np.sum(cluster_labels == cluster))

    def query(self, state: State, budget: int) -> List[int]:
        embeddings = state.embeddings
        labeled_set = state.labeled

        n_clusters = min(len(labeled_set) + budget, self.max_clusters)
        clustering_model = self.get_clustering_model(n_clusters)
        cluster_labels = clustering_model.fit_predict(embeddings)

        new_indices = []
        remaining = budget
        temp_labels = cluster_labels.copy()
        temp_labeled = set(labeled_set)

        while remaining > 0:
            eligible = self.eligible_clusters(temp_labels, temp_labeled)
            if not eligible:
                break

            selected_cluster = self.select_largest_cluster(temp_labels, eligible)
            
            cluster_indices = np.where(temp_labels == selected_cluster)[0]
            cluster_indices = [idx for idx in cluster_indices if idx not in temp_labeled and idx not in new_indices]

            if len(cluster_indices) == 0:
                continue

            cluster_indices = np.array(cluster_indices)
            cluster_embeddings = embeddings[cluster_indices]

            typicality_scores = self.compute_typicality(cluster_embeddings, K=20)
            best_local_idx = np.argmax(typicality_scores)
            chosen_idx = int(cluster_indices[best_local_idx])

            new_indices.append(chosen_idx)
            temp_labeled.add(chosen_idx)
            remaining -= 1

        return new_indices
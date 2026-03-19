import numpy as np
from typing import List
from sklearn.metrics import pairwise_distances

from src.config import State
from src.strategies.base import Strategy
from src.active_learning import get_unlabeled_pool


class CoreSetStrategy(Strategy):
    def query(self, state: State, budget: int) -> List[int]:
        labeled = state.labeled
        unlabeled = get_unlabeled_pool(len(state.dataset), labeled)
        embeddings = state.embeddings

        labeled_embeddings = embeddings[list(labeled)]
        unlabeled_embeddings = embeddings[unlabeled]

        min_distances = pairwise_distances(
            unlabeled_embeddings, 
            labeled_embeddings
        ).min(axis=1)

        chosen_indices = []

        for _ in range(budget):
            best = np.argmax(min_distances)
            chosen_indices.append(best)

            new_distances = pairwise_distances(
                unlabeled_embeddings,
                unlabeled_embeddings[best].reshape(1, -1)
            ).flatten()
            min_distances = np.minimum(min_distances, new_distances)

        return [unlabeled[index] for index in chosen_indices]
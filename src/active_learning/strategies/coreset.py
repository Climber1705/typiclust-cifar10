import torch
import numpy as np
from typing import List
from sklearn.metrics import pairwise_distances

from src.active_learning.base import Strategy
from src.active_learning.state import State

class CoreSetStrategy(Strategy):
    def query(self, state: State, budget: int, device: torch.device) -> List[int]:
        embeddings = state.embeddings
        labeled = list(state.labeled)
        unlabeled = list(set(range(len(embeddings))) - set(labeled))

        labeled_embeddings = embeddings[labeled]
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
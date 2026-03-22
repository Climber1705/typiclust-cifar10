import torch
import numpy as np
from typing import List
from sklearn.metrics import pairwise_distances

from src.active_learning.base import Strategy
from src.active_learning.state import State
from src.active_learning.utils import get_unlabeled_pool, get_gradient_embeddings, kmeans_plus_plus_seeding

class BADGEStrategy(Strategy):
    def query(self, state: State, budget: int, device: torch.device) -> List[int]:
        model = state.model
        dataset = state.dataset
        labeled = state.labeled

        pool = get_unlabeled_pool(len(dataset), labeled)

        gradient_embeddings = get_gradient_embeddings(
            model=model,
            dataset=dataset,
            indices=pool,
            device=device
        )

        chosen = kmeans_plus_plus_seeding(gradient_embeddings, budget)
        return [pool[index] for index in chosen]
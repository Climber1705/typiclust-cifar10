import numpy as np
from typing import List

from src.config import State
from src.strategies.base import Strategy
from src.active_learning import get_unlabeled_pool, get_softmax_scores

class MarginStrategy(Strategy):
    def query(self, state: State, budget: int) -> List[int]:
        model = state.model
        dataset = state.dataset
        labeled = state.labeled
        device = state.device
        
        pool = get_unlabeled_pool(len(dataset), labeled)
        probabilities = get_softmax_scores(model, dataset, pool, device)
        sorted_probabilities = np.sort(probabilities, axis=1)

        margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
        chosen = np.argsort(margins)[:budget]
        return [pool[index] for index in chosen]
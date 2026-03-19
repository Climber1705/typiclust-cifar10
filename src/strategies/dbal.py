import numpy as np
from typing import List

from src.config import State
from src.strategies.base import Strategy
from src.al import get_unlabeled_pool, get_dropout_predictions

class DBALStrategy(Strategy):
    def query(self, state: State, budget: int) -> List[int]:
        model   = state.model
        dataset = state.dataset
        labeled = state.labeled
        device = state.device

        pool = get_unlabeled_pool(len(dataset), labeled)
        predictions = get_dropout_predictions(model, dataset, pool, device)

        mean_probabilities = predictions.mean(axis=0)
        entropy = -np.sum(mean_probabilities * np.log(mean_probabilities + 1e-12), axis=1)

        chosen = np.argsort(-entropy)[:budget]
        return [pool[index] for index in chosen]
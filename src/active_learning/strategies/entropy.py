import torch
import numpy as np
from typing import List

from src.active_learning.base import Strategy
from src.active_learning.state import State
from src.active_learning.utils import get_unlabeled_pool, get_softmax_scores

class EntropyStrategy(Strategy):
    def query(self, state: State, budget: int, device: torch.device) -> List[int]:
        model = state.model
        dataset = state.dataset
        labeled = state.labeled

        pool = get_unlabeled_pool(len(dataset), labeled)
        probabilities = get_softmax_scores(model, dataset, pool, device)

        entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
        chosen = np.argsort(-entropy)[:budget]
        return [pool[index] for index in chosen]
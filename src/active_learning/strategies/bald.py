import torch
import numpy as np
from typing import List

from src.active_learning.base import Strategy
from src.active_learning.state import State
from src.active_learning.utils import get_unlabeled_pool, get_dropout_predictions

class BALDStrategy(Strategy):
    def query(self, state: State, budget: int, device: torch.device) -> List[int]:
        model = state.model
        dataset = state.dataset
        labeled = state.labeled

        pool = get_unlabeled_pool(len(dataset), labeled)
        predictions = get_dropout_predictions(model, dataset, pool, device)

        mean_probabilities = predictions.mean(axis=0)
        entropy_mean = -np.sum(mean_probabilities * np.log(mean_probabilities + 1e-12), axis=1)
        entropy_samples = -np.sum(predictions * np.log(predictions + 1e-12), axis=2)
        
        mean_entropy = entropy_samples.mean(axis=0)
        bald_score = entropy_mean - mean_entropy
        chosen = np.argsort(-bald_score)[:budget]

        return [pool[index] for index in chosen]
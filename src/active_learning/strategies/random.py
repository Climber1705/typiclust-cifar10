import torch
import random
from typing import List

from src.active_learning.base import Strategy
from src.active_learning.state import State
from src.active_learning.utils import get_unlabeled_pool

class RandomStrategy(Strategy):
    def query(self, state: State, budget: int, device: torch.device) -> List[int]:
        dataset = state.dataset
        labeled = state.labeled

        pool = get_unlabeled_pool(len(dataset), labeled)
        return random.sample(pool, budget)
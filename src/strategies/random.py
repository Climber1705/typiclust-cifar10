import random
from typing import List

from src.config import State
from src.active_learning import get_unlabeled_pool
from src.strategies.base import Strategy

class RandomStrategy(Strategy):
    def query(self, state: State, budget: int) -> List[int]:
        dataset = state.dataset
        labeled = state.labeled

        pool = get_unlabeled_pool(len(dataset), labeled)
        return random.sample(pool, budget)
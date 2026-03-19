from typing import List

from src.config import State
from src.strategies.base import Strategy
from src.al import get_unlabeled_pool, get_gradient_embeddings, kmeans_plus_plus_seeding


class BADGEStrategy(Strategy):
    def query(self, state: State, budget: int) -> List[int]:
        model = state.model
        dataset = state.dataset
        labeled = state.labeled
        device = state.device
        
        pool = get_unlabeled_pool(len(dataset), labeled)

        gradient_embeddings = get_gradient_embeddings(
            model=model,
            dataset=dataset,
            indices=pool,
            device=device
        )

        chosen = kmeans_plus_plus_seeding(gradient_embeddings, budget)
        return [pool[index] for index in chosen]
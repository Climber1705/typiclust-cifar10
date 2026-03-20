from typing import Dict

from src.config import Configs
from src.strategies.base import Strategy
from src.strategies.typiclust import TypiclustStrategy
from src.strategies.random import RandomStrategy
from src.strategies.uncertainty import UncertaintyStrategy
from src.strategies.margin import MarginStrategy
from src.strategies.entropy import EntropyStrategy
from src.strategies.dbal import DBALStrategy
from src.strategies.coreset import CoreSetStrategy
from src.strategies.bald import BALDStrategy
from src.strategies.badge import BADGEStrategy

def build_strategies(config: Configs) -> Dict[str, Strategy]:
    return {
        "TPC_RP": TypiclustStrategy(
            seed=config.config.seed,
            max_clusters=config.cluster.max_clusters,
            min_cluster_size=config.cluster.min_cluster_size
        ),
        "Random": RandomStrategy(),
        "Uncertainty": UncertaintyStrategy(),
        "Margin": MarginStrategy(),
        "Entropy": EntropyStrategy(),
        "DBAL": DBALStrategy(),
        "CoreSet": CoreSetStrategy(),
        "BALD": BALDStrategy(),
        "BADGE": BADGEStrategy(),
    }
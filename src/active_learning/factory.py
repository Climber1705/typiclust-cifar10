from src.active_learning.strategies import (
    BADGEStrategy,
    BALDStrategy,
    CoreSetStrategy,
    DBALStrategy,
    EntropyStrategy,
    MarginStrategy,
    RandomStrategy,
    TypiclustStrategy,
    UncertaintyStrategy,
)


def build_strategies(config):
    return {
        "TPC_RP": lambda: TypiclustStrategy(
            seed=config.seed,
            max_clusters=config.cluster.max_clusters,
            min_cluster_size=config.cluster.min_cluster_size,
        ),
        "Random": lambda: RandomStrategy(),
        "Uncertainty": lambda: UncertaintyStrategy(),
        "Margin": lambda: MarginStrategy(),
        "Entropy": lambda: EntropyStrategy(),
        "DBAL": lambda: DBALStrategy(),
        "CoreSet": lambda: CoreSetStrategy(),
        "BALD": lambda: BALDStrategy(),
        "BADGE": lambda: BADGEStrategy(),
    }
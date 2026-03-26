from src.active_learning.strategies import (
    BADGEStrategy,
    BALDStrategy,
    BudgetAdaptiveTypiClust,
    CoreSetStrategy,
    DBALStrategy,
    EntropyStrategy,
    MarginStrategy,
    RandomStrategy,
    TypiclustStrategy,
    UncertaintyStrategy,
)

n_classes = 10
k = 20
steepness = 0.1
default_factor = 5.0

def build_strategies(config):
    return {
        "BudgetAdaptiveTypiClust": lambda: BudgetAdaptiveTypiClust(
            n_classes=n_classes,
            k=k,
            max_clusters=config.cluster.max_clusters,
            steepness=steepness,
            default_factor=default_factor,
        ),
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
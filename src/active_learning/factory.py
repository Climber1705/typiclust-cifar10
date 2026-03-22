from src.active_learning.strategies import *

def build_strategies(config):
    return {
        "TPC_RP": lambda: TypiclustStrategy(
            seed=config.seed,
            max_clusters=config.cluster.max_clusters,
            min_cluster_size=config.cluster.min_cluster_size,
        ),
        "Random": RandomStrategy,
        #"Uncertainty": UncertaintyStrategy,
        #"Margin": MarginStrategy,
        #"Entropy": EntropyStrategy,
       # "DBAL": DBALStrategy,
        #"CoreSet": CoreSetStrategy,
        #"BALD": BALDStrategy,
        #"BADGE": BADGEStrategy,
    }
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path

from src.config import load_config

NAME_COLOURS = {
    "TPC_RP": "blue",           
    "Random": "black",
    "Uncertainty": "red",
    "Margin": "orange",
    "Entropy": "saddlebrown",   
    "DBAL": "deeppink",         
    "CoreSet": "yellowgreen",   
    "BALD": "limegreen",        
    "BADGE": "olive",           
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the config file"
    )
    return parser.parse_args()

def plot_results(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data_directory = Path(config.paths.data_directory)
    num_rounds = 10

    df = pd.read_csv(data_directory / "al_results.csv")
    budgets = [config.cluster.B * (i + 1) for i in range(num_rounds)]

    plt.figure(figsize=(10, 6))

    for name in NAME_COLOURS.keys():
        if name not in set(df["strategy"].unique()):
            continue
        strat_df = df[df["strategy"] == name].sort_values("round")

        means = strat_df["mean"].to_numpy()
        std_errors = strat_df["std_error"].to_numpy()

        x = budgets[: len(means)]
        plt.plot(x, means, marker="o", label=name, color=NAME_COLOURS[name])
        plt.fill_between(x, means - std_errors, means + std_errors, alpha=0.2, color=NAME_COLOURS[name])

    plt.xlabel("Cumulative Budget")
    plt.ylabel("Test Accuracy (%)")
    plt.title("CIFAR-10: Low Budget Active Learning (TPC_RP vs Baselines)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../figures/al_results.png", dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    plot_results(args)
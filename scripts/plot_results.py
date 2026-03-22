import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path

from src.config import load_config
from src.utils import get_directories

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
    project_root = Path(__file__).resolve().parent.parent
    data_directory = project_root / config.paths.data_directory
    figures_directory = project_root / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_directory / "al_results.csv")

    plt.figure(figsize=(10, 6))

    for name in NAME_COLOURS.keys():
        if name not in set(df["strategy"].unique()):
            continue
        strat_df = df[df["strategy"] == name].sort_values("round")

        budgets = strat_df["B"].to_numpy()
        means = strat_df["mean"].to_numpy()
        std_errors = strat_df["std_error"].to_numpy()

        plt.plot(budgets, means, marker="o", label=name, color=NAME_COLOURS[name])
        plt.fill_between(budgets, means - std_errors, means + std_errors, alpha=0.2, color=NAME_COLOURS[name])

    plt.xlabel("Cumulative Budget (B)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("CIFAR-10: Low Budget Active Learning (TPC_RP vs Baselines)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_directory / "al_results.png", dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    plot_results(args)
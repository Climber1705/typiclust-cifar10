import argparse
import pandas as pd
from pathlib import Path

from src.config import load_config
from src.evaluation import (
    plot_active_learning_results,
    plot_final_accuracy_results,
    plot_phase_transition_results,
)

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

    plot_active_learning_results(df, figures_directory / "al_results.png")
    plot_final_accuracy_results(df, figures_directory / "final_accuracy_results.png")
    plot_phase_transition_results(df, figures_directory / "phase_transition_results.png")

if __name__ == "__main__":
    args = parse_args()
    plot_results(args)
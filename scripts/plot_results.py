import argparse
import pandas as pd

from src.config import load_config
from src.evaluation import (
    plot_active_learning_results,
    plot_final_accuracy_results,
    plot_phase_transition_results,
)
from src.utils import get_project_root, get_directories

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_B10.yaml",
        help="Path to the config file"
    )
    return parser.parse_args()

def plot_results(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    project_root = get_project_root()
    data_directory, _ = get_directories(config)
    figures_directory = project_root / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_directory / f"al_results_{config.cluster.B}.csv")

    plot_active_learning_results(df, figures_directory / f"al_results_{config.cluster.B}.png")
    plot_final_accuracy_results(df, figures_directory / f"final_accuracy_results_{config.cluster.B}.png")
    plot_phase_transition_results(df, figures_directory / f"phase_transition_results_{config.cluster.B}.png")

if __name__ == "__main__":
    args = parse_args()
    plot_results(args)
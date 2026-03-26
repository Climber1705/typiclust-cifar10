import numpy as np
import pandas as pd

from typing import Dict, Any
from pathlib import Path

def save_results(all_results: Dict[str, Any], data_directory: Path, budget_per_round: int = 10) -> None:
    rows = []

    for name, runs in all_results.items():
        accuracies = np.array(runs)
        means = accuracies.mean(axis=0) * 100
        std_errors = accuracies.std(axis=0) / np.sqrt(len(runs)) * 100

        for i, (mean, std_error) in enumerate(zip(means, std_errors)):
            rows.append({
                "strategy": name,
                "round": i + 1,
                "B": budget_per_round * (i + 1),
                "mean": round(mean, 4),
                "std_error": round(std_error, 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(data_directory / f"al_results_{budget_per_round}.csv", index=False)
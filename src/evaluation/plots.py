import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

NAME_COLOURS = {
    "BudgetAdaptiveTypiClust": "purple",
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

def plot_active_learning_results(df: pd.DataFrame, save_path: Path) -> None:
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
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

def plot_final_accuracy_results(df: pd.DataFrame, save_path: Path) -> None:
    final = df.loc[df.groupby("strategy")["round"].idxmax()]
    final = final.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(final["strategy"], final["mean"], 
                yerr=final["std_error"], capsize=4,
                color=["#2196F3" if "TPC" in s else "#BDBDBD" for s in final["strategy"]])
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Accuracy (%)")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

def plot_phase_transition_results(df: pd.DataFrame, save_path: Path) -> None:
    typiclust = df[df["strategy"] == "TPC_RP"][["B", "mean"]].rename(columns={"mean": "tpc_mean"})
    fig, ax = plt.subplots(figsize=(7, 4))
    for strategy in df["strategy"].unique():
        if strategy == "TPC_RP":
            continue
        merged = df[df["strategy"] == strategy][["B", "mean"]].merge(typiclust, on="B")
        merged["diff"] = merged["mean"] - merged["tpc_mean"]
        ax.plot(merged["B"], merged["diff"], marker='o', label=strategy, markersize=3)

    ax.axhline(0, color='black', linewidth=1, linestyle='--', label="TypiClust baseline")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Labeled Examples (Budget)")
    ax.set_ylabel("Accuracy Difference vs. TypiClust (%)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
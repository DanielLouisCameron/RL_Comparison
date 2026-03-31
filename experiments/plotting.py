import os

import pandas as pd
import matplotlib.pyplot as plt

from helpers.logger import Logger

logger = Logger("plotting")

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def load_results() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_DIR, "evaluation_results.csv"))


def plot_roi_vs_noise(df: pd.DataFrame):
    """ROI vs training noise for each algorithm (mean +/- std across seeds)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        grouped = sub.groupby("train_noise")["roi"]
        means, stds = grouped.mean(), grouped.std()
        ax.errorbar(
            means.index, means.values, yerr=stds.values,
            marker="o", capsize=5, label=algo,
        )

    ax.set_xlabel("Training Noise Level")
    ax.set_ylabel("ROI")
    ax.set_title("Return on Investment vs Training Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roi_vs_noise.png"), dpi=150)
    plt.close()


def plot_sharpe_vs_noise(df: pd.DataFrame):
    """Sharpe ratio vs training noise for each algorithm."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        grouped = sub.groupby("train_noise")["sharpe_ratio"]
        means, stds = grouped.mean(), grouped.std()
        ax.errorbar(
            means.index, means.values, yerr=stds.values,
            marker="s", capsize=5, label=algo,
        )

    ax.set_xlabel("Training Noise Level")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio vs Training Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sharpe_vs_noise.png"), dpi=150)
    plt.close()


def plot_drawdown_vs_noise(df: pd.DataFrame):
    """Max drawdown vs training noise for each algorithm."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        grouped = sub.groupby("train_noise")["max_drawdown"]
        means, stds = grouped.mean(), grouped.std()
        ax.errorbar(
            means.index, means.values, yerr=stds.values,
            marker="^", capsize=5, label=algo,
        )

    ax.set_xlabel("Training Noise Level")
    ax.set_ylabel("Max Drawdown")
    ax.set_title("Max Drawdown vs Training Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "drawdown_vs_noise.png"), dpi=150)
    plt.close()


def plot_roi_bar_chart(df: pd.DataFrame):
    """Grouped bar chart of average ROI per algorithm at each noise level."""
    pivot = df.pivot_table(
        values="roi", index="train_noise", columns="algorithm", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax, width=0.7)
    ax.set_xlabel("Training Noise Level")
    ax.set_ylabel("Average ROI")
    ax.set_title("Average ROI by Algorithm and Noise Level")
    ax.legend(title="Algorithm")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roi_bar_chart.png"), dpi=150)
    plt.close()


def plot_final_value_heatmap(df: pd.DataFrame):
    """Heatmap of average final portfolio value."""
    pivot = df.pivot_table(
        values="final_value", index="algorithm", columns="train_noise", aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Training Noise Level")
    ax.set_title("Average Final Portfolio Value ($)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(
                j, i, f"${pivot.values[i, j]:,.0f}",
                ha="center", va="center", fontsize=10, fontweight="bold",
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "final_value_heatmap.png"), dpi=150)
    plt.close()


def generate_all_plots():
    """Generate every comparison plot from saved evaluation_results.csv."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = load_results()

    if df.empty:
        logger.error("No evaluation results to plot. Run training and evaluation first.")
        return

    plot_roi_vs_noise(df)
    logger.info("Saved roi_vs_noise.png")

    plot_sharpe_vs_noise(df)
    logger.info("Saved sharpe_vs_noise.png")

    plot_drawdown_vs_noise(df)
    logger.info("Saved drawdown_vs_noise.png")

    plot_roi_bar_chart(df)
    logger.info("Saved roi_bar_chart.png")

    plot_final_value_heatmap(df)
    logger.info("Saved final_value_heatmap.png")

    logger.info(f"All plots saved to {PLOTS_DIR}/")

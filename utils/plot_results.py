import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_eval_metrics(metrics_root: str = "results/eval_metrics") -> pd.DataFrame:
    metrics_root = Path(metrics_root)
    rows = []

    for file in metrics_root.rglob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
        rows.append(data)

    if not rows:
        raise ValueError(f"No evaluation JSON files found under {metrics_root}")

    return pd.DataFrame(rows)


def save_bar_chart(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    ylabel: str,
):
    summary = (
        df.groupby(["group", "run_type"], as_index=False)[value_col]
        .mean()
        .sort_values(["group", "run_type"])
    )

    pivot = summary.pivot(index="group", columns="run_type", values=value_col)

    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Stock Group")
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def save_per_stock_curves(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        run_type = row["run_type"].lower()
        group = row["group"]
        seed = row["seed"]
        per_stock_results = row.get("per_stock_results", [])

        if not per_stock_results:
            continue

        for stock_result in per_stock_results:
            symbol = stock_result.get("symbol", "UNKNOWN")
            portfolio_values = stock_result.get("portfolio_values", [])

            if not portfolio_values:
                continue

            plt.figure(figsize=(10, 5))
            plt.plot(portfolio_values)
            plt.title(f"{row['run_type']} | {group} | seed {seed} | {symbol}")
            plt.xlabel("Step")
            plt.ylabel("Portfolio Value")
            plt.tight_layout()

            filename = f"{run_type}_{group}_seed_{seed}_{symbol.lower()}_curve.png"
            plt.savefig(output_dir / filename)
            plt.close()


def plot_all_results(results_dir):
    plots_dir = Path(f"{results_dir}/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_eval_metrics(f"{results_dir}/eval_metrics")

    save_bar_chart(
        df,
        value_col="final_value",
        output_path=plots_dir / "final_value_by_group.png",
        title="Average Final Portfolio Value by Model and Group",
        ylabel="Final Portfolio Value",
    )

    save_bar_chart(
        df,
        value_col="total_return_pct",
        output_path=plots_dir / "roi_by_group.png",
        title="Average ROI by Model and Group",
        ylabel="ROI (%)",
    )

    save_bar_chart(
        df,
        value_col="sharpe_ratio",
        output_path=plots_dir / "sharpe_by_group.png",
        title="Average Sharpe Ratio by Model and Group",
        ylabel="Sharpe Ratio",
    )

    save_bar_chart(
        df,
        value_col="max_drawdown",
        output_path=plots_dir / "max_drawdown_by_group.png",
        title="Average Max Drawdown by Model and Group",
        ylabel="Max Drawdown",
    )

    save_per_stock_curves(df, plots_dir / "portfolio_curves")

    print(f"Plots saved to {plots_dir}")


import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from agents.a2c_agent import A2CAgent
from agents.base_agent import BaseAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from utils.logger import get_logger
from utils.seed import set_seeds

logger = get_logger(__name__)


def load_group_data(group_name: str, split: str, data_dir: Path) -> list[pd.DataFrame]:
    """
    Load all CSVs for a group and split.
    """
    group_dir = data_dir / group_name

    files = sorted(group_dir.glob(f"*_{split}.csv"))

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        symbol = file.stem.replace(f"_{split}", "").upper()
        df["symbol"] = symbol
        dfs.append(df)

    return dfs


def get_agent_class(agent_name: str):
    if agent_name == "ppo":
        return PPOAgent
    if agent_name == "dqn":
        return DQNAgent
    if agent_name == "a2c":
        return A2CAgent
    raise ValueError(f"Unsupported agent: {agent_name}")


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run(config: dict, results_dir: Path = Path("results")):
    """
    Evaluate all saved models for:
        groups x agents x seeds
    """
    results_dir = Path(results_dir)
    models_dir = results_dir / "models"
    eval_metrics_dir = results_dir / "eval_metrics"

    eval_metrics_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(config["paths"]["data_dir"])
    groups = config["experiment"]["groups"]
    seeds = config["experiment"]["seeds"]
    agents_to_run = ["ppo", "dqn", "a2c"]

    total_runs = len(groups) * len(agents_to_run) * len(seeds)
    current_run = 0

    for group_name in groups:
        test_dfs = load_group_data(group_name, "test", data_dir)

        logger.info(
            f"Loaded group '{group_name}' | {len(test_dfs)} test datasets"
        )

        for agent_name in agents_to_run:
            AgentClass = get_agent_class(agent_name)

            for seed in seeds:
                current_run += 1
                logger.info(
                    f"[{current_run}/{total_runs}] "
                    f"Evaluating {agent_name.upper()} | group={group_name} | seed={seed}"
                )

                set_seeds(seed)

                model_path = models_dir / group_name / agent_name / f"seed_{seed}"
                model_zip_path = Path(str(model_path) + ".zip")

                model = AgentClass.load(str(model_path))

                metrics = BaseAgent.evaluate(
                    model=model,
                    test_dfs=test_dfs,
                    config=config,
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                metrics_path = (
                    eval_metrics_dir
                    / group_name
                    / agent_name
                    / f"seed_{seed}_{timestamp}.json"
                )

                result = {
                    "run_type": agent_name.upper(),
                    "group": group_name,
                    "seed": seed,
                    "timestamp": datetime.now().isoformat(),
                    "final_value": round(metrics["final_value"], 2),
                    "roi": round(metrics["roi"], 6),
                    "total_return_pct": round(metrics["total_return_pct"], 4),
                    "sharpe_ratio": round(metrics["sharpe_ratio"], 4),
                    "max_drawdown": round(metrics["max_drawdown"], 4),
                    "total_reward": round(metrics["total_reward"], 4),
                    "avg_steps": metrics["avg_steps"],
                    "num_stocks_evaluated": metrics["num_stocks_evaluated"],
                    "per_stock_results": metrics["per_stock_results"],
                    "model_path": str(model_zip_path),
                }

                save_json(result, metrics_path)

                logger.info(
                    f"Evaluated model: {model_zip_path} | "
                    f"Final value: ${metrics['final_value']:,.2f} | "
                    f"ROI: {metrics['total_return_pct']:.2f}% | "
                    f"Stocks: {metrics['num_stocks_evaluated']}"
                )

    logger.info("All evaluation runs complete.")
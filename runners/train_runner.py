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
    agent_name = agent_name.lower()

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
    Run training for:
        groups x agents x seeds
    """
    results_dir = Path(results_dir)
    models_dir = results_dir / "models"
    metrics_dir = results_dir / "metrics"

    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(config["paths"]["data_dir"])
    groups = config["experiment"]["groups"]
    seeds = config["experiment"]["seeds"]
    timesteps = config["experiment"]["train_timesteps"]

    agents_to_run = [a.lower() for a in config["experiment"]["agents"]]

    total_runs = len(groups) * len(agents_to_run) * len(seeds)
    current_run = 0

    for group_name in groups:
        train_dfs = load_group_data(group_name, "train", data_dir)
        test_dfs = load_group_data(group_name, "test", data_dir)

        logger.info(
            f"Loaded group '{group_name}' | "
            f"{len(train_dfs)} train datasets | {len(test_dfs)} test datasets"
        )

        for agent_name in agents_to_run:
            AgentClass = get_agent_class(agent_name)

            for seed in seeds:
                current_run += 1
                logger.info(
                    f"[{current_run}/{total_runs}] "
                    f"Training {agent_name.upper()} | group={group_name} | seed={seed}"
                )

                set_seeds(seed)

                agent_instance = AgentClass(
                    config=config,
                    seed=seed,
                    train_dfs=train_dfs,
                )

                agent_instance.train(timesteps)

                model_dir = models_dir / group_name / agent_name
                model_dir.mkdir(parents=True, exist_ok=True)

                model_path = agent_instance.save(str(model_dir))

                metrics = BaseAgent.evaluate(
                    model=agent_instance.model,
                    test_dfs=test_dfs,
                    config=config,
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                metrics_path = (
                    metrics_dir
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
                    "model_path": model_path,
                }

                save_json(result, metrics_path)

                logger.info(
                    f"Saved model: {model_path} | "
                    f"Final value: ${metrics['final_value']:,.2f} | "
                    f"ROI: {metrics['total_return_pct']:.2f}%"
                )

    logger.info("All training runs complete.")
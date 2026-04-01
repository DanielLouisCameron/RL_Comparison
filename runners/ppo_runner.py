import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from agents.ppo_agent import PPOAgent
from agents.base_agent import BaseAgent
from utils.logger import get_logger
from utils.seed import set_seeds

logger = get_logger(__name__)

MODELS_DIR = "results/models"


def train(config: dict, results_dir: Path = Path("results")):
    """Train PPO across all noise levels and seeds, evaluate each, save JSON results."""
    symbol = config["symbol"].lower()
    processed_dir = Path(config["paths"]["processed_dir"])

    train_df = pd.read_csv(processed_dir / f"{symbol}_train.csv")
    test_df = pd.read_csv(processed_dir / f"{symbol}_test.csv")
    logger.info(f"Loaded data: {len(train_df)} train rows, {len(test_df)} test rows")

    noise_levels = config["experiment"]["noise_levels"]
    seeds = config["experiment"]["seeds"]
    timesteps = config["experiment"]["train_timesteps"]
    initial_cash = config["portfolio"]["initial_cash"]

    results_dir = Path(results_dir)
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    total = len(noise_levels) * len(seeds)
    current = 0

    for noise in noise_levels:
        for seed in seeds:
            current += 1
            logger.info(f"[{current}/{total}] Training PPO | noise={noise:.2f} | seed={seed}")

            set_seeds(seed)
            agent = PPOAgent(config=config, noise=noise, seed=seed, train_df=train_df)
            agent.train(timesteps)
            agent.save(str(models_dir))

            model = agent.model
            metrics = BaseAgent.evaluate(model, test_df, config, noise=0.0)

            result = {
                "run_type": "PPO",
                "symbol": symbol.upper(),
                "seed": seed,
                "noise_level": noise,
                "timestamp": datetime.now().isoformat(),
                "initial_cash": initial_cash,
                "final_value": round(metrics["final_value"], 2),
                "total_return_pct": round(metrics["total_return_pct"], 4),
                "sharpe_ratio": round(metrics["sharpe_ratio"], 4),
                "max_drawdown": round(metrics["max_drawdown"], 4),
                "steps": metrics["steps"],
                "portfolio_values": metrics["portfolio_values"],
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"ppo_{symbol}_noise{noise:.2f}_seed{seed}_{timestamp}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            logger.info(f"  Final value: ${metrics['final_value']:,.2f} | ROI: {metrics['total_return_pct']:.2f}%")
            logger.info(f"  Results saved: {output_path}")

    logger.info("PPO training complete.")


def evaluate(config: dict, results_dir: Path = Path("results")):
    """Load saved PPO models and evaluate on test data."""
    symbol = config["symbol"].lower()
    processed_dir = Path(config["paths"]["processed_dir"])
    test_df = pd.read_csv(processed_dir / f"{symbol}_test.csv")

    noise_levels = config["experiment"]["noise_levels"]
    seeds = config["experiment"]["seeds"]
    initial_cash = config["portfolio"]["initial_cash"]

    results_dir = Path(results_dir)
    models_dir = results_dir / "models"

    for noise in noise_levels:
        for seed in seeds:
            filename = f"PPO-noise={noise:.2f}-seed_{seed}"
            model_path = str(models_dir / filename)

            if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
                logger.warning(f"Skipping {filename} (not found)")
                continue

            logger.info(f"Evaluating {filename}")
            set_seeds(seed)
            model = PPOAgent.load(model_path)
            metrics = BaseAgent.evaluate(model, test_df, config, noise=0.0)

            result = {
                "run_type": "PPO",
                "symbol": symbol.upper(),
                "seed": seed,
                "noise_level": noise,
                "timestamp": datetime.now().isoformat(),
                "initial_cash": initial_cash,
                "final_value": round(metrics["final_value"], 2),
                "total_return_pct": round(metrics["total_return_pct"], 4),
                "sharpe_ratio": round(metrics["sharpe_ratio"], 4),
                "max_drawdown": round(metrics["max_drawdown"], 4),
                "steps": metrics["steps"],
                "portfolio_values": metrics["portfolio_values"],
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"ppo_{symbol}_noise{noise:.2f}_seed{seed}_{timestamp}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            logger.info(f"  Final value: ${metrics['final_value']:,.2f} | Saved: {output_path}")

    logger.info("PPO evaluation complete.")

import os

import pandas as pd

from configs.config import NOISE_LEVELS, SEEDS, TRAIN_TIMESTEPS
from agents.base_agent import BaseAgent
from agents.ppo_agent import PPOAgent
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent
from helpers.logger import Logger

logger = Logger("experiment")

RESULTS_DIR = "results"

AGENTS = [PPOAgent, DQNAgent, A2CAgent]


def train_all():
    """Train every agent x noise level x seed combination."""
    train_df = pd.read_csv("data/processed/aapl_train.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total = len(AGENTS) * len(NOISE_LEVELS) * len(SEEDS)
    current = 0

    for agent_cls in AGENTS:
        for noise in NOISE_LEVELS:
            for seed in SEEDS:
                current += 1
                logger.info(
                    f"[{current}/{total}] Training {agent_cls.name} | noise={noise:.2f} | seed={seed}"
                )
                agent = agent_cls(noise=noise, seed=seed, train_df=train_df)
                agent.train(TRAIN_TIMESTEPS)
                path = agent.save(RESULTS_DIR)
                logger.info(f"  -> Saved to {path}")

    logger.info("All training complete.")


def evaluate_all() -> pd.DataFrame:
    """Load every trained model, evaluate on the clean test set, and save results."""
    test_df = pd.read_csv("data/processed/aapl_test.csv")
    results = []

    for agent_cls in AGENTS:
        for noise in NOISE_LEVELS:
            for seed in SEEDS:
                filename = f"{agent_cls.name}-noise={noise:.2f}-seed_{seed}"
                model_path = os.path.join(RESULTS_DIR, filename)

                if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
                    logger.warning(f"Skipping {filename} (model file not found)")
                    continue

                logger.info(f"Evaluating {filename}")
                model = agent_cls.load(model_path)
                metrics = BaseAgent.evaluate(model, test_df, noise=0.0)

                results.append({
                    "algorithm": agent_cls.name,
                    "train_noise": noise,
                    "seed": seed,
                    "final_value": metrics["final_value"],
                    "roi": metrics["roi"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "total_reward": metrics["total_reward"],
                })

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    return results_df


def run_full_experiment() -> pd.DataFrame:
    """Run the complete pipeline: train all models, then evaluate them."""
    logger.info("=== Starting full experiment ===")
    train_all()
    logger.info("=== Training complete — starting evaluation ===")
    results_df = evaluate_all()
    logger.info("=== Experiment complete ===")
    return results_df

import os

import numpy as np
import pandas as pd

from environment.trading_env import TradingEnvironment
from configs.config import PORTFOLIO_CASH


class BaseAgent:
    """Base wrapper around a Stable Baselines 3 algorithm for the trading environment."""

    name: str = ""
    algo_cls = None

    def __init__(self, noise: float, seed: int, train_df: pd.DataFrame, **kwargs):
        self.noise = noise
        self.seed = seed
        self.env = TradingEnvironment(df=train_df, noise=noise)
        self.model = self.algo_cls(
            "MlpPolicy", self.env, seed=seed, verbose=0, **kwargs
        )

    def train(self, timesteps: int):
        self.model.learn(total_timesteps=timesteps)

    def save(self, results_dir: str) -> str:
        os.makedirs(results_dir, exist_ok=True)
        filename = f"{self.name}-noise={self.noise:.2f}-seed_{self.seed}"
        path = os.path.join(results_dir, filename)
        self.model.save(path)
        return path

    @classmethod
    def load(cls, path: str):
        return cls.algo_cls.load(path)

    @staticmethod
    def evaluate(model, test_df: pd.DataFrame, noise: float = 0.0) -> dict:
        """Run a trained model on test data and return performance metrics."""
        env = TradingEnvironment(df=test_df, noise=noise)
        obs, info = env.reset()

        total_reward = 0.0
        portfolio_values = [info["portfolio_value"]]
        actions_taken = []

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            portfolio_values.append(info["portfolio_value"])
            actions_taken.append(info["action_taken"])
            done = terminated or truncated

        portfolio_values = np.array(portfolio_values)
        daily_returns = np.diff(portfolio_values) / (portfolio_values[:-1] + 1e-8)

        peak = np.maximum.accumulate(portfolio_values)
        drawdowns = (peak - portfolio_values) / (peak + 1e-8)

        return {
            "final_value": portfolio_values[-1],
            "roi": (portfolio_values[-1] - PORTFOLIO_CASH) / PORTFOLIO_CASH,
            "sharpe_ratio": np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252),
            "max_drawdown": np.max(drawdowns),
            "total_reward": total_reward,
            "portfolio_values": portfolio_values.tolist(),
            "actions": actions_taken,
        }

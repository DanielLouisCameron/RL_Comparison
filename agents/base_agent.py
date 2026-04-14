import os

import numpy as np
import pandas as pd

from environment.trading_env import TradingEnvironment


class BaseAgent:
    """Base wrapper around a Stable Baselines 3 algorithm for the trading environment."""

    name: str = ""
    algo_cls = None

    def __init__(self, config: dict, seed: int, train_dfs: list[pd.DataFrame], **kwargs):
        self.config = config
        self.seed = seed
        self.env = TradingEnvironment(dfs=train_dfs, config=config)
        self.model = self.algo_cls(
            "MlpPolicy", self.env, seed=seed, verbose=0, **kwargs
        )

    def train(self, timesteps: int):
        self.model.learn(total_timesteps=timesteps)

    def save(self, results_dir: str) -> str:
        os.makedirs(results_dir, exist_ok=True)
        filename = f"seed_{self.seed}"
        path = os.path.join(results_dir, filename)
        self.model.save(path)
        return path

    @classmethod
    def load(cls, path: str):
        return cls.algo_cls.load(path)

    @staticmethod
    def evaluate(model, test_dfs: list[pd.DataFrame], config: dict) -> dict:
        """
        Evaluate the model on ALL stocks in test_dfs and average the scalar metrics.
        """
        per_stock_results = []

        for df in test_dfs:
            env = TradingEnvironment(dfs=[df], config=config)
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

            initial_cash = config["portfolio"]["initial_cash"]

            sharpe = 0.0
            if len(daily_returns) > 0 and np.std(daily_returns) > 1e-8:
                sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))

            per_stock_results.append({
                "symbol": info.get("symbol", "unknown"),
                "final_value": float(portfolio_values[-1]),
                "roi": float((portfolio_values[-1] - initial_cash) / initial_cash),
                "total_return_pct": float((portfolio_values[-1] - initial_cash) / initial_cash * 100),
                "sharpe_ratio": sharpe,
                "max_drawdown": float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0,
                "total_reward": float(total_reward),
                "steps": len(actions_taken),
                "portfolio_values": portfolio_values.tolist(),
                "actions": actions_taken,
            })

        total_stocks = len(per_stock_results)

        return {
            "final_value": sum(r["final_value"] for r in per_stock_results) / total_stocks,
            "roi": sum(r["roi"] for r in per_stock_results) / total_stocks,
            "total_return_pct": sum(r["total_return_pct"] for r in per_stock_results) / total_stocks,
            "sharpe_ratio": sum(r["sharpe_ratio"] for r in per_stock_results) / total_stocks,
            "max_drawdown": sum(r["max_drawdown"] for r in per_stock_results) / total_stocks,
            "total_reward": sum(r["total_reward"] for r in per_stock_results) / total_stocks,
            "avg_steps": int(sum(r["steps"] for r in per_stock_results) / total_stocks),
            "num_stocks_evaluated": total_stocks,
            "per_stock_results": per_stock_results,
        }
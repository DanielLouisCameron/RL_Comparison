from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd


class TradingEnvironment(gym.Env):
    """
    3-action trading environment.

    Actions:
        0 = 0% invested
        1 = 50% invested
        2 = 100% invested

    Reward:
        position_fraction * next-step return
    """

    metadata = {"render_modes": []}

    def __init__(self, dfs: list[pd.DataFrame], config: dict = None):
        super().__init__()

        if not dfs:
            raise ValueError("Need stock data.")

        self.dfs = [df.reset_index(drop=True) for df in dfs]
        self.current_df_idx = -1
        self.df = self.dfs[0]

        self.initial_cash = float(config["portfolio"]["initial_cash"]) if config else 10000.0

        self.feature_cols = [
            "daily_return",
            "momentum_5",
            "momentum_20",
            "ma_20_ratio",
            "ma_5_20_spread",
            "volatility_20",
        ]

        self.position_levels = [0.0, 0.5, 1.0]

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_cols),),
            dtype=np.float32,
        )

        self.counter = 0
        self.position = 0.0
        self.portfolio_value = self.initial_cash

    def _get_observation(self):
        row = self.df.iloc[self.counter]
        return np.array([float(row[col]) for col in self.feature_cols], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.current_df_idx = (self.current_df_idx + 1) % len(self.dfs)
        self.df = self.dfs[self.current_df_idx]

        self.counter = 0
        self.position = 0.0
        self.portfolio_value = self.initial_cash

        current_price = float(self.df.iloc[self.counter]["close"])

        obs = self._get_observation()
        info = {
            "portfolio_value": float(self.portfolio_value),
            "cash": float(self.portfolio_value),
            "stock_amt": 0.0,
            "stock_price": current_price,
            "roi": 0.0,
            "action_taken": None,
            "did_trade": False,
            "trade_value": 0.0,
            "cost_paid": 0.0,
            "entry_price": 0.0,
            "in_position": False,
            "symbol": self.df["symbol"].iloc[0],
        }
        return obs, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        prev_position = self.position
        self.position = self.position_levels[int(action)]

        current_price = float(self.df.iloc[self.counter]["close"])
        self.counter += 1

        terminated = self.counter >= len(self.df) - 1
        truncated = False

        next_price = float(self.df.iloc[self.counter]["close"])
        price_return = (next_price - current_price) / current_price

        reward = float(self.position * price_return)

        self.portfolio_value *= (1 + reward)

        did_trade = self.position != prev_position

        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_observation()

        roi = (self.portfolio_value - self.initial_cash) / self.initial_cash

        cash_value = self.portfolio_value * (1 - self.position)

        info = {
            "portfolio_value": float(self.portfolio_value),
            "action_taken": int(action),
            "symbol": self.df["symbol"].iloc[0] if "symbol" in self.df.columns else "unknown",
        }

        return obs, float(reward), terminated, truncated, info
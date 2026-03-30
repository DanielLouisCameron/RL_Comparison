from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd

from configs import config


class TradingEnvironment(gym.Env):
    def __init__(self, df: pd.DataFrame, noise: float = 0.0):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.noise = noise

        # 0 = sell all
        # 1 = sell half
        # 2 = hold
        # 3 = buy half
        # 4 = buy all
        # 0 - sell all, 1 - sell half, 2 - hold, 3 - buy half, 4 - buy all
        self.action_space = gym.spaces.Discrete(5)

        self.feature_cols = [
            "close",
            "daily_return",
            "ma_5",
            "ma_20",
            "ma_60",
            "volatility_20",
            "momentum_5",
        ]

        obs_dim = len(self.feature_cols) + 2  # + stock_amt, cash_ratio
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.counter = 0
        self.cash = float(config.PORTFOLIO_CASH)
        self.stock_amt = 0.0
        self.stock_price = float(self.df.iloc[0]["close"])
        self.portfolio_value = self.cash

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.counter]

        obs = np.array(
            [
                row["close"],
                row["daily_return"],
                row["ma_5"],
                row["ma_20"],
                row["ma_60"],
                row["volatility_20"],
                row["momentum_5"],
                self.stock_amt,
                self.cash / max(self.portfolio_value, 1e-8),
            ],
            dtype=np.float32,
        )

        if self.noise > 0:
            noise_vec = np.random.normal(0, self.noise, size=7).astype(np.float32)
            obs[:7] += noise_vec

        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.counter = 0
        self.cash = float(config.PORTFOLIO_CASH)
        self.stock_amt = 0.0
        self.stock_price = float(self.df.iloc[0]["close"])
        self.portfolio_value = self.cash

        obs = self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "stock_amt": self.stock_amt,
            "stock_price": self.stock_price,
        }
        return obs, info

    def step(self, action: int):
        
        current_price = float(self.df.iloc[self.counter]["close"])
        old_value = self.cash + self.stock_amt * current_price

        if action == 0:
            # sell all
            shares_to_sell = self.stock_amt
            self.cash += shares_to_sell * current_price
            self.stock_amt = 0.0

        elif action == 1:
            # sell half
            shares_to_sell = 0.5 * self.stock_amt
            self.stock_amt -= shares_to_sell
            self.cash += shares_to_sell * current_price

        elif action == 2:
            # hold
            pass

        elif action == 3:
            # buy half
            amount_to_spend = 0.5 * self.cash
            shares_to_buy = amount_to_spend / current_price
            self.stock_amt += shares_to_buy
            self.cash -= amount_to_spend

        elif action == 4:
            # buy all
            amount_to_spend = self.cash
            shares_to_buy = amount_to_spend / current_price
            self.stock_amt += shares_to_buy
            self.cash = 0.0

        else:
            raise ValueError("Invalid action. Must be between 0 and 4.")

        self.counter += 1

        terminated = self.counter >= len(self.df) - 1
        truncated = False

        self.stock_price = float(self.df.iloc[self.counter]["close"])
        self.portfolio_value = self.cash + self.stock_amt * self.stock_price

        reward = (self.portfolio_value / old_value) - 1.0

        obs = self._get_observation()

        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "stock_amt": self.stock_amt,
            "stock_price": self.stock_price,
            "roi": (self.portfolio_value - float(config.PORTFOLIO_CASH)) / float(config.PORTFOLIO_CASH),
            "action_taken": action,
        }

        return obs, reward, terminated, truncated, info
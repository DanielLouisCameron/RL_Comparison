from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd


class TradingEnvironment(gym.Env):
    """
    Gymnasium-compatible trading environment.

    Accepts a list of dataframes for one stock group.
    Each episode randomly selects one dataframe (stock).
    """

    def __init__(self, dfs: list[pd.DataFrame], config: dict, noise: float = 0.0):
        super().__init__()

        if not dfs:
            raise ValueError("TradingEnvironment requires at least one dataframe.")

        self.dfs = [df.reset_index(drop=True) for df in dfs]
        self.df = self.dfs[0]

        self.noise = noise
        self.initial_cash = float(config["portfolio"]["initial_cash"])
        self.transaction_cost = float(config["portfolio"]["transaction_cost"])

        # 0 - sell all, 1 - sell half, 2 - hold, 3 - buy half, 4 - buy all
        self.action_space = gym.spaces.Discrete(5)

        self.feature_cols = [
            "daily_return",
            "ma_5",
            "ma_20",
            "ma_60",
            "volatility_20",
            "momentum_5",
        ]

        obs_dim = len(self.feature_cols) + 2  # stock_value_ratio, cash_ratio
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.counter = 0
        self.cash = self.initial_cash
        self.stock_amt = 0.0
        self.stock_price = float(self.df.iloc[0]["close"])
        self.portfolio_value = self.cash
        self.current_symbol = "unknown"

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.counter]

        stock_value_ratio = (self.stock_amt * self.stock_price) / max(self.portfolio_value, 1e-8)
        cash_ratio = self.cash / max(self.portfolio_value, 1e-8)

        obs = np.array(
            [row[col] for col in self.feature_cols] + [stock_value_ratio, cash_ratio],
            dtype=np.float32,
        )

        if self.noise > 0:
            noise_vec = np.random.normal(0, self.noise, size=len(self.feature_cols)).astype(np.float32)
            obs[: len(self.feature_cols)] += noise_vec

        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        idx = self.np_random.integers(0, len(self.dfs))
        self.df = self.dfs[idx]

        if "symbol" in self.df.columns:
            self.current_symbol = str(self.df.iloc[0]["symbol"])
        else:
            self.current_symbol = "unknown"

        self.counter = 0
        self.cash = self.initial_cash
        self.stock_amt = 0.0
        self.stock_price = float(self.df.iloc[0]["close"])
        self.portfolio_value = self.cash

        obs = self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "stock_amt": self.stock_amt,
            "stock_price": self.stock_price,
            "symbol": self.current_symbol,
        }
        return obs, info

    def step(self, action: int):
        current_price = float(self.df.iloc[self.counter]["close"])
        old_value = self.cash + self.stock_amt * current_price

        if action == 0:
            self.cash += self.stock_amt * current_price
            self.stock_amt = 0.0

        elif action == 1:
            shares_to_sell = 0.5 * self.stock_amt
            self.stock_amt -= shares_to_sell
            self.cash += shares_to_sell * current_price

        elif action == 2:
            pass

        elif action == 3:
            amount_to_spend = 0.5 * self.cash
            self.stock_amt += amount_to_spend / current_price
            self.cash -= amount_to_spend

        elif action == 4:
            self.stock_amt += self.cash / current_price
            self.cash = 0.0

        else:
            raise ValueError(f"Invalid action {action}. Must be between 0 and 4.")

        self.counter += 1
        terminated = self.counter >= len(self.df) - 1
        truncated = False

        if not terminated:
            self.stock_price = float(self.df.iloc[self.counter]["close"])
        else:
            self.stock_price = current_price

        self.portfolio_value = self.cash + self.stock_amt * self.stock_price

        transaction_cost = self.transaction_cost if action != 2 else 0.0
        reward = (self.portfolio_value / max(old_value, 1e-8)) - 1.0 - transaction_cost

        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "stock_amt": self.stock_amt,
            "stock_price": self.stock_price,
            "roi": (self.portfolio_value - self.initial_cash) / self.initial_cash,
            "action_taken": action,
            "symbol": self.current_symbol,
        }

        return obs, reward, terminated, truncated, info
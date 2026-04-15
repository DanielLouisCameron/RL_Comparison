from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd


class TradingEnvironment(gym.Env):
    """
    Gymnasium-compatible trading environment.

    Each episode randomly selects one dataframe (stock) from the provided list.

    Actions:
        0 = sell all
        1 = hold
        2 = buy all

    Reward:
        - step reward = log portfolio return
        - optional hold penalty
        - optional invalid action penalty
        - optional terminal bonus based on final ROI
    """

    metadata = {"render_modes": []}

    def __init__(self, dfs: list[pd.DataFrame], config: dict, noise: float = 0.0):
        super().__init__()

        if not dfs:
            raise ValueError("TradingEnvironment requires at least one dataframe.")

        self.dfs = [df.reset_index(drop=True) for df in dfs]
        self.df = self.dfs[0]

        self.noise = float(noise)

        portfolio_cfg = config["portfolio"]

        self.initial_cash = float(portfolio_cfg["initial_cash"])
        self.transaction_cost = float(portfolio_cfg["transaction_cost"])

        self.hold_penalty = float(portfolio_cfg.get("hold_penalty", 0.0))
        self.invalid_action_penalty = float(portfolio_cfg.get("invalid_action_penalty", 0.0))
        self.terminal_bonus_weight = float(portfolio_cfg.get("terminal_bonus_weight", 0.0))

        # 0 = sell all, 1 = hold, 2 = buy all
        self.action_space = gym.spaces.Discrete(3)

        self.feature_cols = [
            "daily_return",
            "ma_5",
            "ma_20",
            "ma_60",
            "volatility_20",
            "momentum_5",
        ]

        # market features + portfolio state
        # stock_value_ratio, cash_ratio, in_position, position_return
        obs_dim = len(self.feature_cols) + 4
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
        self.portfolio_value = self.initial_cash
        self.current_symbol = "unknown"
        self.entry_price = 0.0

    def _get_portfolio_value(self, price: float) -> float:
        return float(self.cash + self.stock_amt * price)

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.counter]

        portfolio_value = max(self.portfolio_value, 1e-8)
        stock_value = self.stock_amt * self.stock_price

        stock_value_ratio = stock_value / portfolio_value
        cash_ratio = self.cash / portfolio_value
        in_position = 1.0 if self.stock_amt > 1e-12 else 0.0

        position_return = 0.0
        if in_position > 0.0 and self.entry_price > 0.0:
            position_return = (self.stock_price - self.entry_price) / self.entry_price

        obs = np.array(
            [float(row[col]) for col in self.feature_cols]
            + [stock_value_ratio, cash_ratio, in_position, position_return],
            dtype=np.float32,
        )

        if self.noise > 0.0:
            noise_vec = np.random.normal(
                0.0,
                self.noise,
                size=len(self.feature_cols),
            ).astype(np.float32)
            obs[: len(self.feature_cols)] += noise_vec

        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        idx = int(self.np_random.integers(0, len(self.dfs)))
        self.df = self.dfs[idx]

        if "symbol" in self.df.columns:
            self.current_symbol = str(self.df.iloc[0]["symbol"])
        else:
            self.current_symbol = "unknown"

        self.counter = 0
        self.cash = self.initial_cash
        self.stock_amt = 0.0
        self.stock_price = float(self.df.iloc[0]["close"])
        self.portfolio_value = self.initial_cash
        self.entry_price = 0.0

        obs = self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "stock_amt": self.stock_amt,
            "stock_price": self.stock_price,
            "roi": 0.0,
            "action_taken": None,
            "did_trade": False,
            "invalid_action": False,
            "trade_value": 0.0,
            "cost_paid": 0.0,
            "entry_price": self.entry_price,
            "in_position": False,
            "symbol": self.current_symbol,
        }
        return obs, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, 1, 2].")

        current_price = float(self.df.iloc[self.counter]["close"])
        old_value = self._get_portfolio_value(current_price)

        did_trade = False
        invalid_action = False
        trade_value = 0.0
        cost_paid = 0.0

        # 0 = sell all
        if action == 0:
            if self.stock_amt > 1e-12:
                gross_value = self.stock_amt * current_price
                cost_paid = gross_value * self.transaction_cost
                net_value = gross_value - cost_paid

                self.cash += net_value
                self.stock_amt = 0.0
                self.entry_price = 0.0

                did_trade = True
                trade_value = gross_value
            else:
                invalid_action = True

        # 1 = hold
        elif action == 1:
            pass

        # 2 = buy all
        elif action == 2:
            if self.cash > 1e-12:
                # Reserve transaction cost so cash does not go negative
                budget = self.cash / (1.0 + self.transaction_cost)
                cost_paid = budget * self.transaction_cost
                shares_bought = budget / max(current_price, 1e-8)

                self.stock_amt += shares_bought
                self.cash -= (budget + cost_paid)

                # numerical cleanup
                if abs(self.cash) < 1e-10:
                    self.cash = 0.0

                self.entry_price = current_price
                did_trade = True
                trade_value = budget
            else:
                invalid_action = True

        # advance time
        self.counter += 1
        terminated = self.counter >= len(self.df) - 1
        truncated = False

        if not terminated:
            next_price = float(self.df.iloc[self.counter]["close"])
        else:
            next_price = current_price

        self.stock_price = next_price
        self.portfolio_value = self._get_portfolio_value(self.stock_price)

        # Step reward aligned with compounded growth
        reward = np.log((self.portfolio_value + 1e-8) / (old_value + 1e-8))

        # Optional shaping
        if action == 1:
            reward -= self.hold_penalty

        if invalid_action:
            reward -= self.invalid_action_penalty

        # Optional terminal bonus for final ROI
        final_roi = (self.portfolio_value - self.initial_cash) / max(self.initial_cash, 1e-8)
        if terminated and self.terminal_bonus_weight != 0.0:
            reward += self.terminal_bonus_weight * final_roi

        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_observation()

        info = {
            "portfolio_value": float(self.portfolio_value),
            "cash": float(self.cash),
            "stock_amt": float(self.stock_amt),
            "stock_price": float(self.stock_price),
            "roi": float(final_roi),
            "action_taken": int(action),
            "did_trade": bool(did_trade),
            "invalid_action": bool(invalid_action),
            "trade_value": float(trade_value),
            "cost_paid": float(cost_paid),
            "entry_price": float(self.entry_price),
            "in_position": bool(self.stock_amt > 1e-12),
            "symbol": self.current_symbol,
        }

        return obs, float(reward), terminated, truncated, info
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd


class TradingEnvironment(gym.Env):
    """
    Trading environment v2.

    Key changes from v1:
    - Observation is a rolling window of past rows
    - Action means target portfolio allocation, not buy/hold/sell
    - Reward is step log-return of total portfolio value
    - Optional terminal bonus encourages final ROI

    Actions:
        action i -> target allocation in self.position_levels[i]

    Example:
        self.position_levels = [0.0, 0.5, 1.0]
        0 = 0% invested
        1 = 50% invested
        2 = 100% invested
    """

    metadata = {"render_modes": []}

    def __init__(self, dfs: list[pd.DataFrame], config: dict, noise: float = 0.0):
        super().__init__()

        if not dfs:
            raise ValueError("TradingEnvironmentV2 requires at least one dataframe.")

        self.dfs = [df.reset_index(drop=True) for df in dfs]
        self.df = self.dfs[0]

        self.noise = float(noise)

        portfolio_cfg = config["portfolio"]
        env_cfg = config.get("environment", {})

        self.initial_cash = float(portfolio_cfg["initial_cash"])
        self.transaction_cost = float(portfolio_cfg["transaction_cost"])
        self.terminal_bonus_weight = float(portfolio_cfg.get("terminal_bonus_weight", 0.0))

        self.window_size = int(env_cfg.get("window_size", 30))
        self.position_levels = env_cfg.get("position_levels", [0.0, 0.5, 1.0])

        if not self.position_levels:
            raise ValueError("position_levels must not be empty.")

        self.action_space = gym.spaces.Discrete(len(self.position_levels))

        self.feature_cols = [
            "daily_return",
            "ma_5",
            "ma_20",
            "ma_60",
            "volatility_20",
            "momentum_5",
        ]

        # market features + portfolio state
        # [stock_value_ratio, cash_ratio, current_position, position_return]
        self.row_feature_dim = len(self.feature_cols) + 4
        obs_dim = self.window_size * self.row_feature_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.counter = self.window_size - 1
        self.cash = self.initial_cash
        self.stock_amt = 0.0
        self.stock_price = float(self.df.iloc[self.counter]["close"])
        self.portfolio_value = self.initial_cash
        self.current_symbol = "unknown"
        self.entry_price = 0.0

    def _get_portfolio_value(self, price: float) -> float:
        return float(self.cash + self.stock_amt * price)

    def _get_position_ratio(self, price: float) -> float:
        portfolio_value = self._get_portfolio_value(price)
        if portfolio_value <= 1e-8:
            return 0.0
        stock_value = self.stock_amt * price
        return float(stock_value / portfolio_value)

    def _build_row_features(self, row_idx: int) -> list[float]:
        row = self.df.iloc[row_idx]

        portfolio_value = max(self.portfolio_value, 1e-8)
        stock_value = self.stock_amt * self.stock_price

        stock_value_ratio = stock_value / portfolio_value
        cash_ratio = self.cash / portfolio_value
        current_position = stock_value_ratio

        position_return = 0.0
        if self.stock_amt > 1e-12 and self.entry_price > 1e-12:
            position_return = (self.stock_price - self.entry_price) / self.entry_price

        return [float(row[col]) for col in self.feature_cols] + [
            stock_value_ratio,
            cash_ratio,
            current_position,
            position_return,
        ]

    def _get_observation(self) -> np.ndarray:
        start_idx = self.counter - self.window_size + 1
        end_idx = self.counter + 1

        obs_rows = [self._build_row_features(i) for i in range(start_idx, end_idx)]
        obs = np.array(obs_rows, dtype=np.float32)

        if self.noise > 0.0:
            noise = np.random.normal(
                0.0, self.noise, size=obs[:, : len(self.feature_cols)].shape
            ).astype(np.float32)
            obs[:, : len(self.feature_cols)] += noise

        return obs.flatten()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        idx = int(self.np_random.integers(0, len(self.dfs)))
        self.df = self.dfs[idx]

        if len(self.df) < self.window_size + 1:
            raise ValueError(
                f"Dataframe too short for window_size={self.window_size}. "
                f"Need at least {self.window_size + 1} rows, got {len(self.df)}."
            )

        if "symbol" in self.df.columns:
            self.current_symbol = str(self.df.iloc[0]["symbol"])
        else:
            self.current_symbol = "unknown"

        self.counter = self.window_size - 1
        self.cash = self.initial_cash
        self.stock_amt = 0.0
        self.stock_price = float(self.df.iloc[self.counter]["close"])
        self.portfolio_value = self.initial_cash
        self.entry_price = 0.0

        obs = self._get_observation()
        info = {
            "portfolio_value": float(self.portfolio_value),
            "cash": float(self.cash),
            "stock_amt": float(self.stock_amt),
            "stock_price": float(self.stock_price),
            "roi": 0.0,
            "action_taken": None,
            "target_position": None,
            "actual_position": 0.0,
            "did_trade": False,
            "trade_value": 0.0,
            "cost_paid": 0.0,
            "symbol": self.current_symbol,
        }
        return obs, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in range action_space.")

        current_price = float(self.df.iloc[self.counter]["close"])
        old_value = self._get_portfolio_value(current_price)

        target_position = float(self.position_levels[int(action)])
        current_position = self._get_position_ratio(current_price)

        target_stock_value = target_position * old_value
        current_stock_value = self.stock_amt * current_price
        delta_stock_value = target_stock_value - current_stock_value

        did_trade = False
        trade_value = 0.0
        cost_paid = 0.0

        # Rebalance toward target allocation
        if abs(delta_stock_value) > 1e-10:
            did_trade = True
            trade_value = abs(delta_stock_value)
            cost_paid = trade_value * self.transaction_cost

            if delta_stock_value > 0:
                # Buy additional stock
                max_affordable_trade = self.cash / (1.0 + self.transaction_cost)
                actual_trade_value = min(delta_stock_value, max_affordable_trade)

                if actual_trade_value > 1e-12:
                    shares_bought = actual_trade_value / max(current_price, 1e-8)
                    actual_cost = actual_trade_value * self.transaction_cost

                    self.stock_amt += shares_bought
                    self.cash -= (actual_trade_value + actual_cost)

                    # update entry price as weighted average when increasing
                    old_shares = self.stock_amt - shares_bought
                    if self.stock_amt > 1e-12:
                        if old_shares <= 1e-12:
                            self.entry_price = current_price
                        else:
                            old_cost_basis = old_shares * self.entry_price
                            new_cost_basis = shares_bought * current_price
                            self.entry_price = (old_cost_basis + new_cost_basis) / self.stock_amt

                    trade_value = actual_trade_value
                    cost_paid = actual_cost
                else:
                    did_trade = False
                    trade_value = 0.0
                    cost_paid = 0.0

            else:
                # Sell stock
                desired_sell_value = abs(delta_stock_value)
                max_sell_value = current_stock_value
                actual_trade_value = min(desired_sell_value, max_sell_value)

                if actual_trade_value > 1e-12:
                    shares_sold = actual_trade_value / max(current_price, 1e-8)
                    shares_sold = min(shares_sold, self.stock_amt)

                    gross_value = shares_sold * current_price
                    actual_cost = gross_value * self.transaction_cost
                    net_value = gross_value - actual_cost

                    self.stock_amt -= shares_sold
                    self.cash += net_value

                    if self.stock_amt <= 1e-12:
                        self.stock_amt = 0.0
                        self.entry_price = 0.0

                    trade_value = gross_value
                    cost_paid = actual_cost
                else:
                    did_trade = False
                    trade_value = 0.0
                    cost_paid = 0.0

        # numerical cleanup
        if abs(self.cash) < 1e-10:
            self.cash = 0.0

        # Move forward one bar
        self.counter += 1
        terminated = self.counter >= len(self.df) - 1
        truncated = False

        if not terminated:
            next_price = float(self.df.iloc[self.counter]["close"])
        else:
            next_price = current_price

        self.portfolio_value = self._get_portfolio_value(self.stock_price)

        # Main reward: log-return of total portfolio value
        reward = np.log((self.portfolio_value + 1e-8) / (old_value + 1e-8))

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
            "target_position": float(target_position),
            "actual_position": float(self._get_position_ratio(self.stock_price)),
            "did_trade": bool(did_trade),
            "trade_value": float(trade_value),
            "cost_paid": float(cost_paid),
            "symbol": self.current_symbol,
        }

        return obs, float(reward), terminated, truncated, info
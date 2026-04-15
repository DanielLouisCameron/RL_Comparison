from stable_baselines3.common.callbacks import BaseCallback


class TradingMetricsCallback(BaseCallback):
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            env = self.training_env

            # unwrap Monitor if needed
            if hasattr(env, "env"):
                raw_env = env.env
            else:
                raw_env = env

            portfolio_value = getattr(raw_env, "portfolio_value", None)
            cash = getattr(raw_env, "cash", None)
            position = getattr(raw_env, "position", None)
            num_trades = getattr(raw_env, "num_trades", None)
            initial_cash = getattr(raw_env, "initial_cash", None)

            if portfolio_value is not None:
                self.logger.record("custom/portfolio_value", float(portfolio_value))

            if portfolio_value is not None and initial_cash is not None:
                roi = (portfolio_value - initial_cash) / initial_cash
                self.logger.record("custom/roi", float(roi))

            if cash is not None:
                self.logger.record("custom/cash", float(cash))

            if position is not None:
                self.logger.record("custom/position", float(position))

            if num_trades is not None:
                self.logger.record("custom/num_trades", float(num_trades))

            self.logger.dump(self.num_timesteps)

        return True
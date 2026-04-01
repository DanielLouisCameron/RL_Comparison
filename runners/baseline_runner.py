import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from agents.baseline_agent import BuyAndHoldAgent
from environment.trading_env import TradingEnvironment
from utils.logger import get_logger
from utils.seed import set_seeds

logger = get_logger(__name__)


def run(config: dict, results_dir: Path = Path("results")):
    seed = config["experiment"]["seeds"][0]
    set_seeds(seed)
    logger.info(f"Seed set to {seed}")

    symbol = config["symbol"].lower()
    processed_dir = Path(config["paths"]["processed_dir"])
    test_path = processed_dir / f"{symbol}_test.csv"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}. Run --mode data first."
        )

    df = pd.read_csv(test_path)
    logger.info(f"Loaded test data: {len(df)} rows ({symbol.upper()})")

    env = TradingEnvironment(df=df, config=config)
    agent = BuyAndHoldAgent()

    obs, info = env.reset()
    done = False
    step = 0
    portfolio_values = [info["portfolio_value"]]

    while not done:
        action = agent.act(step)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        portfolio_values.append(info["portfolio_value"])

    initial_cash = config["portfolio"]["initial_cash"]
    final_value = portfolio_values[-1]
    total_return_pct = (final_value - initial_cash) / initial_cash * 100

    results = {
        "run_type": "baseline_buy_and_hold",
        "symbol": symbol.upper(),
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "initial_cash": initial_cash,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return_pct, 4),
        "steps": step,
        "portfolio_values": portfolio_values,
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"baseline_{symbol}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Initial cash:  ${initial_cash:,.2f}")
    logger.info(f"Final value:   ${final_value:,.2f}")
    logger.info(f"Total return:  {total_return_pct:.2f}%")
    logger.info(f"Results saved: {output_path}")

    return results

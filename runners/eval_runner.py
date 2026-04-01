from pathlib import Path

from runners import ppo_runner, dqn_runner, a2c_runner
from utils.logger import get_logger

logger = get_logger(__name__)


def run(config: dict, results_dir: Path = Path("results"), agent: str = None):
    """
    Evaluate saved RL models. If agent is specified, evaluate only that one.
    Otherwise evaluate all three (PPO, DQN, A2C).
    """
    runners = {
        "ppo": ("PPO", ppo_runner.evaluate),
        "dqn": ("DQN", dqn_runner.evaluate),
        "a2c": ("A2C", a2c_runner.evaluate),
    }

    if agent:
        key = agent.lower()
        if key not in runners:
            raise ValueError(f"Unknown agent '{agent}'. Choose from: {list(runners.keys())}")
        name, eval_fn = runners[key]
        logger.info(f"=== Evaluating {name} ===")
        eval_fn(config, results_dir)
    else:
        for key, (name, eval_fn) in runners.items():
            logger.info(f"=== Evaluating {name} ===")
            eval_fn(config, results_dir)

    logger.info("=== All evaluation complete ===")

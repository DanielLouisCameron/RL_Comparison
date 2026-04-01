from pathlib import Path

from runners import ppo_runner, dqn_runner, a2c_runner
from utils.logger import get_logger

logger = get_logger(__name__)


def run(config: dict, results_dir: Path = Path("results"), agent: str = None):
    """
    Train RL agents. If agent is specified, train only that one.
    Otherwise train all three (PPO, DQN, A2C).
    """
    runners = {
        "ppo": ("PPO", ppo_runner.train),
        "dqn": ("DQN", dqn_runner.train),
        "a2c": ("A2C", a2c_runner.train),
    }

    if agent:
        key = agent.lower()
        if key not in runners:
            raise ValueError(f"Unknown agent '{agent}'. Choose from: {list(runners.keys())}")
        name, train_fn = runners[key]
        logger.info(f"=== Training {name} ===")
        train_fn(config, results_dir)
    else:
        for key, (name, train_fn) in runners.items():
            logger.info(f"=== Training {name} ===")
            train_fn(config, results_dir)

    logger.info("=== All training complete ===")

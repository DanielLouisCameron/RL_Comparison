import argparse
from pathlib import Path

from utils.config_validator import load_and_validate_config
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RL Trading Agent Comparison")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON file (e.g. data_config.json)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["data", "baseline", "train", "eval"],
        help="data: download dataset | baseline: run buy-and-hold | train: train RL agents | eval: evaluate saved models",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        choices=["ppo", "dqn", "a2c"],
        help="Run only a specific agent (for train/eval modes). Omit to run all.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save run artifacts (default: results/)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    results_dir = Path(args.results_dir)

    logger.info(f"Mode:        {args.mode}")
    logger.info(f"Config:      {config_path}")
    logger.info(f"Results dir: {results_dir}")

    config = load_and_validate_config(config_path)
    logger.info("Config validated successfully")

    if args.mode == "data":
        from data.create_dataset import create_dataset
        create_dataset(config_path)

    elif args.mode == "baseline":
        from runners import baseline_runner
        baseline_runner.run(config=config, results_dir=results_dir)

    elif args.mode == "train":
        from runners import train_runner
        train_runner.run(config=config, results_dir=results_dir, agent=args.agent)

    elif args.mode == "eval":
        from runners import eval_runner
        eval_runner.run(config=config, results_dir=results_dir, agent=args.agent)


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RL Trading Agent Comparison")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON file (e.g. experiment_config.json)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["data", "train", "eval", "plot"],
        help="data: download dataset | train: train RL agents | eval: evaluate saved models | plot: graph results",
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
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Mode:        {args.mode}")
    logger.info(f"Config:      {config_path}")
    logger.info(f"Results dir: {results_dir}")

    with open(config_path, "r") as f:
        config = json.load(f)

    if args.mode == "data":
        from data.create_dataset import create_dataset
        create_dataset(config_path)

    elif args.mode == "train":
        from runners import train_runner
        train_runner.run(config=config, results_dir=results_dir)

    elif args.mode == "eval":
        from runners import eval_runner
        eval_runner.run(config=config, results_dir=results_dir)
    elif args.mode == "plot":
        from utils.plot_results import plot_all_results
        plot_all_results(results_dir=results_dir)

if __name__ == "__main__":
    main()
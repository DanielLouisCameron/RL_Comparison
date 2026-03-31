import argparse

from data.create_dataset import main as create_dataset
from experiments.experiment import train_all, evaluate_all, run_full_experiment
from experiments.plotting import generate_all_plots

parser = argparse.ArgumentParser(description="RL Trading Agent Comparison")
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["train", "experiment", "eval", "data"],
    help="train: train all models | experiment: train + evaluate + plot | eval: evaluate saved models + plot | data: download & process dataset",
)
parser.add_argument("--symbol", type=str, default="AAPL")

args = parser.parse_args()

if args.mode == "data":
    create_dataset(stock_symbol=args.symbol)

elif args.mode == "train":
    train_all()

elif args.mode == "experiment":
    run_full_experiment()
    generate_all_plots()

elif args.mode == "eval":
    evaluate_all()
    generate_all_plots()

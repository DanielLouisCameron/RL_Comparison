import argparse
from data.create_dataset import main as create_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["train", "experiment", "eval", "data"]
)
parser.add_argument("--symbol", type=str, default="AAPL")

args = parser.parse_args()

if args.mode == "data":
    create_dataset(stock_symbol=args.symbol)
    
elif args.mode == "train":
    # Train all models
    pass
elif args.mode == "experiment":
    # run all experiments
    pass

elif args.mode == "eval":
    # evaluate trained models
    pass

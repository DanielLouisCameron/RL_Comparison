import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["train", "experiment", "eval"]
)
args = parser.parse_args()

if args.mode == "train":
    # Train all models
    pass
elif args.mode == "experiment":
    # run all experiments
    pass

elif args.mode == "eval":
    # evaluate trained models
    pass

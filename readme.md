# RL Comparison - Quick Overview

## Setup

```bash
python -m venv venv
source venv/bin/activate        # on windows: venv\Scripts\Activate
pip install -r requirements.txt
```

## How to Run

Everything goes through `main.py` with a config file and a mode:

```bash
python main.py --config CONFIG_NAME.json --mode MODE
```

### Modes

| Mode | What it does |
|------|-------------|
| `data` | Downloads stock data and splits into train/validation/test |
| `train` | Trains PPO, DQN, and A2C across all stock groups (e.g., large-cap, mid-cap, small-cap) and seeds |
| `eval` | Loads already-trained models and re-evaluates them on test data |
| `plot` | Plots graphs of all evals focusing on portfolio values and overarching trends |

### Run order

```bash
# 1. Get the data first (only need to do this once)
python main.py --config data_config.json --mode data

# 2. (Optional) Start tensorboard for training insights
tensorboard --logdir results/tb_logs

# 3. Train all agents (this takes a while with 100k timesteps)
python main.py --config experiment_config.json --mode train

# 4. Re-evaluate saved models without retraining
python main.py --config experiment_config.json --mode eval

# 5. Plot output
python main.py --config experiment_config.json --mode plot

```

You will be able to see training insights from tensorboard on http://localhost:6006

## Project structure

```
agents/           - agent classes (base + PPO, DQN, A2C, baseline)
data/             - dataset download + feature engineering
environment/      - the gym trading environment
results/          - Where all results are saved
runners/          - run logic for each agent (train + eval), one file per agent
utils/            - logger, config validation, seed setting

```

### Results Dreakdown

```
models/        - Saved trained models for each group/model/seed run
eval_metrics/  - Evaluation results produced by running test data split on trained models
plots/         - Generated graphs
```


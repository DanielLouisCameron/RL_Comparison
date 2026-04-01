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
python main.py --config data_config.json --mode MODE
```

### Modes

| Mode | What it does |
|------|-------------|
| `data` | Downloads stock data (NVDA by default) and creates train/val/test splits in `data_storage/` |
| `baseline` | Runs the buy-and-hold baseline on test data, saves results JSON |
| `train` | Trains PPO, DQN, A2C across all noise levels and seeds, saves models + result JSONs |
| `eval` | Loads already-trained models and re-evaluates them on test data |

### Run order

```bash
# 1. Get the data first (only need to do this once)
python main.py --config data_config.json --mode data

# 2. Run baseline
python main.py --config data_config.json --mode baseline

# 3. Train all agents (this takes a while with 100k timesteps)
python main.py --config data_config.json --mode train

# 4. Or train just one agent
python main.py --config data_config.json --mode train --agent ppo
python main.py --config data_config.json --mode train --agent dqn
python main.py --config data_config.json --mode train --agent a2c

# 5. Re-evaluate saved models without retraining
python main.py --config data_config.json --mode eval
```

## Config (`data_config.json`)

All experiment parameters live here. Key things you might want to change:

- `symbol` - stock ticker (currently `NVDA`)
- `start_date` / `end_date` - date range for data
- `experiment.train_timesteps` - how long to train (currently `100000`)
- `experiment.noise_levels` - observation noise levels we test (`[0.0, 0.05, 0.1, 0.3]`)
- `experiment.seeds` - random seeds for reproducibility (`[1, 2, 3]`)
- `portfolio.initial_cash` - starting cash (`10000`)
- `portfolio.transaction_cost` - cost per trade (`0.001`)

## Where stuff goes

- **Models** get saved to `results/models/` (e.g. `PPO-noise=0.05-seed_1`)
- **Result JSONs** get saved to `results/` (e.g. `ppo_nvda_noise0.05_seed1_20260401_120000.json`)
- **Data** lives in `data_storage/raw/` and `data_storage/processed/`

## Project structure

```
agents/           - agent classes (base + PPO, DQN, A2C, baseline)
runners/          - run logic for each agent (train + eval), one file per agent
environment/      - the gym trading environment
data/             - dataset download + feature engineering
utils/            - logger, config validation, seed setting
```

# RL Project 4453

## Development

run this in root dir

```bash
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install -r requirements.txt
```

## To run:
python main.py --mode MODE
MODE can be train, expirment (default), eval or data
for data you can add --symbol STOCK_SYMBOL
it defaults to AAPL


## Rough Idea

3 diff agents, trained on same dataset at 3 seeds for each noise level

Gonna have smth like:
PPO training:

ppo_train(noise=0 seed=1)
ppo_train(noise=0 seed=2)
ppo_train(noise=0 seed=3)

ppo_train(noise=0.05 seed=1)
ppo_train(noise=0.05 seed=2)
ppo_train(noise=0.05 seed=3)

cont for all noises

Then do same for DQN and A2C


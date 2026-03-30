import pandas as pd

from environment.trading_env import TradingEnvironment


def main():
    df = pd.read_csv("data/processed/aapl_train.csv")

    env = TradingEnvironment(df, noise=0.0)

    obs, info = env.reset()
    print("Initial obs:", obs)
    print("Initial info:", info)
    print("Action space:", env.action_space)
    print("Observation shape:", obs.shape)

    done = False
    step_num = 0

    while not done and step_num < 10:
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"\nStep {step_num + 1}")
        print("Action:", action)
        print("Reward:", reward)
        print("Portfolio value:", info["portfolio_value"])
        print("Cash:", info["cash"])
        print("Stock amt:", info["stock_amt"])
        print("Stock price:", info["stock_price"])
        print("Done:", done)

        step_num += 1


if __name__ == "__main__":
    main()
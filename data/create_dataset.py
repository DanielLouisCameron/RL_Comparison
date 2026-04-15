import json
from pathlib import Path

import pandas as pd
import yfinance as yf


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def download_data(symbol, start, end):    
    df = yf.download(symbol, start=start, end=end, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data for symbol {symbol}")

    df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]

    return df


def add_features(df, feature_cfg):
    df = df.copy()

    df["daily_return"] = df["close"].pct_change()

    # Momentum
    df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1

    # MA features
    ma_5 = df["close"].rolling(5).mean()
    ma_20 = df["close"].rolling(20).mean()

    df["ma_20_ratio"] = ma_20 / df["close"] - 1
    df["ma_5_20_spread"] = ma_5 / ma_20 - 1

    # Volatility
    df["volatility_20"] = df["daily_return"].rolling(20).std()

    return df.dropna().reset_index(drop=True)


def split_dataset(df, split_cfg):
    n = len(df)

    train_end = int(n * split_cfg["train_ratio"])
    val_end = int(n * (split_cfg["train_ratio"] + split_cfg["val_ratio"]))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test


def create_dataset(config_path):
    cfg = load_config(config_path)

    symbols = cfg["symbols"]

    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    for size, ticker_list in symbols.items():
        size_dir = Path("data") / size
        size_dir.mkdir(parents=True, exist_ok=True)


        for symbol in ticker_list:
            try:
                if cfg.get("data_source") == "synthetic":
                    df = generate_synthetic_data(
                        symbol=symbol,
                        start_date=cfg["start_date"],
                        n_steps=cfg["synthetic"]["n_steps"],
                        start_price=cfg["synthetic"]["start_price"],
                        seed=cfg["synthetic"]["seed"],
                    )
                else:
                    df = download_data(symbol, cfg["start_date"], cfg["end_date"])

                # df = download_data(symbol, cfg["start_date"], cfg["end_date"]) is old
                df.to_csv(raw_dir / f"{symbol.lower()}_raw.csv", index=False)

                df = add_features(df, cfg["features"])
                train, val, test = split_dataset(df, cfg["split"])
                feature_cols = [
                    "daily_return",
                    "momentum_5",
                    "momentum_20",
                    "ma_20_ratio",
                    "ma_5_20_spread",
                    "volatility_20",
                ]

                stats = fit_normalizer(train, feature_cols)
                train = apply_normalizer(train, stats)
                val = apply_normalizer(val, stats)
                test = apply_normalizer(test, stats)

                train.to_csv(size_dir / f"{symbol.lower()}_train.csv", index=False)
                val.to_csv(size_dir / f"{symbol.lower()}_val.csv", index=False)
                test.to_csv(size_dir / f"{symbol.lower()}_test.csv", index=False)

                print(f"Dataset created for {symbol}")

            except Exception as e:
                print(f"Failed for {symbol}: {e}")



import numpy as np
import pandas as pd

def fit_normalizer(train_df, feature_cols):
    stats = {}

    for col in feature_cols:
        mean = train_df[col].mean()
        std = train_df[col].std()

        if pd.isna(std) or std < 1e-8:
            std = 1.0

        stats[col] = {"mean": float(mean), "std": float(std)}

    return stats


def apply_normalizer(df, stats):
    df = df.copy()

    for col, s in stats.items():
        df[col] = (df[col] - s["mean"]) / s["std"]

    return df


def generate_synthetic_data(
    symbol: str,
    start_date: str,
    n_steps: int = 1000,
    start_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for debugging a trading RL environment.

    Supported symbols:
    - SYNTH_UP      : upward drift
    - SYNTH_DOWN    : downward drift
    - SYNTH_MR      : mean-reverting prices
    - SYNTH_CYCLE   : cyclical / oscillating
    - SYNTH_REGIME  : changing market regimes
    - anything else : random walk
    """
    rng = np.random.default_rng(seed + abs(hash(symbol)) % 10_000)
    dates = pd.bdate_range(start=start_date, periods=n_steps)

    if symbol == "SYNTH_MR":
        prices = [start_price]
        kappa = 0.15   # pull back toward mean
        sigma = 1.5
        mu = start_price

        for _ in range(1, n_steps):
            prev = prices[-1]
            next_price = prev + kappa * (mu - prev) + rng.normal(0, sigma)
            prices.append(max(1.0, next_price))

        close = np.array(prices)

    else:
        returns = np.zeros(n_steps)

        if symbol == "SYNTH_UP":
            returns = rng.normal(loc=0.001, scale=0.015, size=n_steps)

        elif symbol == "SYNTH_DOWN":
            returns = rng.normal(loc=-0.001, scale=0.015, size=n_steps)

        elif symbol == "SYNTH_CYCLE":
            t = np.arange(n_steps)
            returns = (
                0.01 * np.sin(2 * np.pi * t / 30)
                + rng.normal(loc=0.0, scale=0.01, size=n_steps)
            )

        elif symbol == "SYNTH_REGIME":
            q = n_steps // 4
            returns = np.concatenate(
                [
                    rng.normal(0.001, 0.01, q),                 # uptrend
                    rng.normal(0.0, 0.005, q),                  # flat
                    rng.normal(-0.001, 0.015, q),               # downtrend
                    rng.normal(0.0, 0.03, n_steps - 3 * q),     # volatile
                ]
            )

        else:
            # fallback: random walk
            returns = rng.normal(loc=0.0, scale=0.02, size=n_steps)

        close = start_price * np.cumprod(1 + returns)
        close = np.maximum(close, 1.0)

    open_ = close * (1 + rng.normal(0, 0.002, n_steps))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, n_steps))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, n_steps))
    volume = rng.integers(100_000, 1_000_000, n_steps)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    return df
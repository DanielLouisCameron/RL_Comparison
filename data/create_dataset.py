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

    df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1

    ma_5 = df["close"].rolling(5).mean()
    ma_20 = df["close"].rolling(20).mean()

    df["ma_20_ratio"] = ma_20 / df["close"] - 1
    df["ma_5_20_spread"] = ma_5 / ma_20 - 1

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

                df = download_data(symbol, cfg["start_date"], cfg["end_date"])
                df.to_csv(raw_dir / f"{symbol.lower()}_raw.csv", index=False)

                df = add_features(df, cfg["features"])
                train, val, test = split_dataset(df, cfg["split"])

                train.to_csv(size_dir / f"{symbol.lower()}_train.csv", index=False)
                val.to_csv(size_dir / f"{symbol.lower()}_val.csv", index=False)
                test.to_csv(size_dir / f"{symbol.lower()}_test.csv", index=False)

                print(f"Dataset created for {symbol}")

            except Exception as e:
                print(f"Failed for {symbol}: {e}")


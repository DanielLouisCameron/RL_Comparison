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

    for w in feature_cfg["ma_windows"]:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()

    df[f"volatility_{feature_cfg['volatility_window']}"] = df["daily_return"].rolling(
        feature_cfg["volatility_window"]
    ).std()

    df[f"momentum_{feature_cfg['momentum_window']}"] = (
        df["close"] / df["close"].shift(feature_cfg["momentum_window"]) - 1
    )

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

    symbol = cfg["symbol"]

    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = download_data(symbol, cfg["start_date"], cfg["end_date"])

    df.to_csv(raw_dir / f"{symbol.lower()}_raw.csv", index=False)

    df = add_features(df, cfg["features"])

    train, val, test = split_dataset(df, cfg["split"])

    train.to_csv(processed_dir / f"{symbol.lower()}_train.csv", index=False)
    val.to_csv(processed_dir / f"{symbol.lower()}_val.csv", index=False)
    test.to_csv(processed_dir / f"{symbol.lower()}_test.csv", index=False)

    print(f"Dataset created for {symbol}")

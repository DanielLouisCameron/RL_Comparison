from pathlib import Path
import pandas as pd
import yfinance as yf
from helpers.logger import Logger

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def download_data(symbol: str, start: str = "2025-01-01", end: str = "2026-01-01") -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError("No data for symbol %s", symbol)
    
    # flatten multiIndex return
    df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["daily_return"] = df["close"].pct_change()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_60"] = df["close"].rolling(60).mean()
    df["volatility_20"] = df["daily_return"].rolling(20).std()
    df["momentum_5"] = df["close"] / df["close"].shift(5) - 1

    df = df.dropna().reset_index(drop=True)
    return df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def main(stock_symbol):
    symbol = stock_symbol
    logger = Logger("dataset")

    df = download_data(symbol=symbol)
    raw_path = RAW_DIR / f"{symbol.lower()}_raw.csv"
    df.to_csv(raw_path, index=False)

    featured_df = add_features(df)

    train_df, val_df, test_df = split_dataset(featured_df)

    train_df.to_csv(PROCESSED_DIR / f"{symbol.lower()}_train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / f"{symbol.lower()}_val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / f"{symbol.lower()}_test.csv", index=False)

    logger.info("Saved new dataset")
    


if __name__ == "__main__":
    main()
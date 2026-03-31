import json
from pathlib import Path


REQUIRED_KEYS = {
    "symbol": str,
    "start_date": str,
    "end_date": str,
    "paths": {
        "raw_dir": str,
        "processed_dir": str,
    },
    "features": {
        "ma_windows": list,
        "volatility_window": int,
        "momentum_window": int,
    },
    "split": {
        "train_ratio": float,
        "val_ratio": float,
        "test_ratio": float,
    },
    "portfolio": {
        "initial_cash": (int, float),
        "transaction_cost": float,
    },
    "experiment": {
        "seeds": list,
        "noise_levels": list,
        "train_timesteps": int,
    },
}


def _validate_section(section: dict, schema: dict, path: str = ""):
    for key, expected_type in schema.items():
        full_key = f"{path}.{key}" if path else key

        if key not in section:
            raise KeyError(f"Missing required config key: '{full_key}'")

        value = section[key]

        if isinstance(expected_type, dict):
            if not isinstance(value, dict):
                raise TypeError(f"Config key '{full_key}' must be a dict, got {type(value).__name__}")
            _validate_section(value, expected_type, path=full_key)
        else:
            types = expected_type if isinstance(expected_type, tuple) else (expected_type,)
            if not isinstance(value, types):
                expected_names = " or ".join(t.__name__ for t in types)
                raise TypeError(
                    f"Config key '{full_key}' must be {expected_names}, got {type(value).__name__}"
                )


def _validate_split_ratios(config: dict):
    split = config["split"]
    total = split["train_ratio"] + split["val_ratio"] + split["test_ratio"]
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.4f} "
            f"(train={split['train_ratio']}, val={split['val_ratio']}, test={split['test_ratio']})"
        )


def load_and_validate_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    _validate_section(config, REQUIRED_KEYS)
    _validate_split_ratios(config)

    return config
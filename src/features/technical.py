from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_OHLCV_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
)

FEATURE_COLUMNS = (
    "return_1d",
    "return_5d",
    "close_to_sma_5",
    "close_to_sma_20",
    "volatility_10d",
    "rsi_14",
    "volume_change_1d",
    "volume_to_sma_20",
    "intraday_range_pct",
    "close_location",
)

FORWARD_RETURN_COLUMN = "forward_return_1d"
TARGET_COLUMN = "target_next_up"


def _validate_ohlcv(history: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy historical OHLCV data before feature generation."""
    if not isinstance(history, pd.DataFrame):
        raise TypeError("Historical data must be a pandas DataFrame.")

    if history.empty:
        raise ValueError("Historical data cannot be empty.")

    missing_columns = [
        column
        for column in REQUIRED_OHLCV_COLUMNS
        if column not in history.columns
    ]

    if missing_columns:
        missing_values = ", ".join(missing_columns)
        raise ValueError(
            "Historical data is missing required columns: "
            f"{missing_values}."
        )

    if history.index.has_duplicates:
        raise ValueError("Historical data cannot contain duplicate dates.")

    validated = history.copy().sort_index()
    required_columns = list(REQUIRED_OHLCV_COLUMNS)

    validated[required_columns] = validated[required_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    invalid_columns = [
        column
        for column in required_columns
        if validated[column].isna().any()
    ]

    if invalid_columns:
        invalid_values = ", ".join(invalid_columns)
        raise ValueError(
            "Historical data contains missing or invalid values in: "
            f"{invalid_values}."
        )

    if (validated["High"] < validated["Low"]).any():
        raise ValueError("Historical data contains High values below Low values.")

    if (validated["Close"] <= 0).any():
        raise ValueError("Historical data contains non-positive Close values.")

    if (validated["Volume"] < 0).any():
        raise ValueError("Historical data contains negative Volume values.")

    return validated


def _calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate a simple rolling RSI from closing prices."""
    price_change = close.diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.rolling(
        window=window,
        min_periods=window,
    ).mean()

    average_loss = losses.rolling(
        window=window,
        min_periods=window,
    ).mean()

    relative_strength = average_gain.div(average_loss)
    rsi = 100 - (100 / (1 + relative_strength))

    flat_prices = (average_gain == 0) & (average_loss == 0)
    only_gains = (average_gain > 0) & (average_loss == 0)
    only_losses = (average_gain == 0) & (average_loss > 0)

    rsi = rsi.mask(flat_prices, 50.0)
    rsi = rsi.mask(only_gains, 100.0)
    rsi = rsi.mask(only_losses, 0.0)

    return rsi


def build_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Add technical features and next-session labels to OHLCV history."""
    features = _validate_ohlcv(history)

    close = features["Close"]
    high = features["High"]
    low = features["Low"]
    volume = features["Volume"]

    return_1d = close.pct_change(fill_method=None)

    features["return_1d"] = return_1d
    features["return_5d"] = close.pct_change(
        periods=5,
        fill_method=None,
    )

    sma_5 = close.rolling(window=5, min_periods=5).mean()
    sma_20 = close.rolling(window=20, min_periods=20).mean()

    features["close_to_sma_5"] = close.div(sma_5).sub(1)
    features["close_to_sma_20"] = close.div(sma_20).sub(1)

    features["volatility_10d"] = return_1d.rolling(
        window=10,
        min_periods=10,
    ).std(ddof=0)

    features["rsi_14"] = _calculate_rsi(close)

    features["volume_change_1d"] = volume.pct_change(fill_method=None)

    volume_sma_20 = volume.rolling(
        window=20,
        min_periods=20,
    ).mean()

    features["volume_to_sma_20"] = volume.div(volume_sma_20).sub(1)

    intraday_range = high.sub(low)

    features["intraday_range_pct"] = intraday_range.div(close)

    features["close_location"] = (
        close.sub(low)
        .div(intraday_range)
        .where(intraday_range.ne(0), 0.5)
    )

    forward_return = close.shift(-1).div(close).sub(1)

    features[FORWARD_RETURN_COLUMN] = forward_return

    target = pd.Series(
        pd.NA,
        index=features.index,
        dtype="Int64",
    )

    valid_target_rows = forward_return.notna()

    target.loc[valid_target_rows] = (
        forward_return.loc[valid_target_rows] > 0
    ).astype("int64")

    features[TARGET_COLUMN] = target

    features = features.replace([np.inf, -np.inf], np.nan)

    return features


def prepare_model_dataset(history: pd.DataFrame) -> pd.DataFrame:
    """Return rows with complete features and a known next-day target."""
    features = build_feature_frame(history)

    required_columns = [
        *FEATURE_COLUMNS,
        FORWARD_RETURN_COLUMN,
        TARGET_COLUMN,
    ]

    model_dataset = features.dropna(
        subset=required_columns,
    ).copy()

    if model_dataset.empty:
        raise ValueError(
            "Not enough historical data to create a complete model dataset."
        )

    model_dataset[TARGET_COLUMN] = model_dataset[TARGET_COLUMN].astype("int64")

    return model_dataset
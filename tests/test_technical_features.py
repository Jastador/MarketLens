import pandas as pd
import pytest

from src.features.technical import (
    FEATURE_COLUMNS,
    FORWARD_RETURN_COLUMN,
    TARGET_COLUMN,
    build_feature_frame,
    prepare_model_dataset,
)


def _history_from_closes(closes: list[float]) -> pd.DataFrame:
    """Create predictable fake OHLCV data for feature tests."""
    dates = pd.date_range(
        "2026-01-02",
        periods=len(closes),
        freq="B",
    )

    return pd.DataFrame(
        {
            "Open": [close - 0.5 for close in closes],
            "High": [close + 1.0 for close in closes],
            "Low": [close - 1.0 for close in closes],
            "Close": closes,
            "Volume": [
                1_000_000 + index * 10_000
                for index in range(len(closes))
            ],
        },
        index=dates,
    )


def _sample_history() -> pd.DataFrame:
    """Create enough rows for all rolling technical features."""
    closes = [100.0 + index for index in range(40)]
    return _history_from_closes(closes)


def test_build_feature_frame_creates_expected_columns():
    """Feature output should include all model inputs and future labels."""
    history = _sample_history()

    features = build_feature_frame(history)

    expected_columns = {
        *FEATURE_COLUMNS,
        FORWARD_RETURN_COLUMN,
        TARGET_COLUMN,
    }

    assert expected_columns.issubset(features.columns)
    assert features.index.equals(history.index)
    assert pd.isna(features[TARGET_COLUMN].iloc[-1])


def test_target_uses_the_next_closing_price():
    """The next-day target must come from tomorrow's close, not today's."""
    history = _history_from_closes([100.0, 105.0, 95.0, 95.0])

    features = build_feature_frame(history)

    assert features[FORWARD_RETURN_COLUMN].iloc[0] == pytest.approx(0.05)
    assert features[TARGET_COLUMN].iloc[0] == 1
    assert features[TARGET_COLUMN].iloc[1] == 0
    assert features[TARGET_COLUMN].iloc[2] == 0
    assert pd.isna(features[TARGET_COLUMN].iloc[3])


def test_prepare_model_dataset_removes_incomplete_rows():
    """Training data should contain no missing model features or targets."""
    dataset = prepare_model_dataset(_sample_history())

    assert not dataset[list(FEATURE_COLUMNS)].isna().any().any()
    assert dataset[TARGET_COLUMN].isin([0, 1]).all()
    assert pd.api.types.is_integer_dtype(dataset[TARGET_COLUMN])
    assert len(dataset) < len(_sample_history())


def test_build_feature_frame_rejects_missing_required_columns():
    """Missing OHLCV inputs should fail with a clear message."""
    history = _sample_history().drop(columns="Volume")

    with pytest.raises(ValueError, match="missing required columns"):
        build_feature_frame(history)


def test_build_feature_frame_does_not_change_input_data():
    """Feature engineering must not mutate the caller's original history."""
    history = _sample_history()
    original_history = history.copy(deep=True)

    build_feature_frame(history)

    pd.testing.assert_frame_equal(history, original_history)
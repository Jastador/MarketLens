import pandas as pd
import pytest

from src.data.yfinance_client import (
    MarketDataError,
    YFinanceClient,
    normalize_ticker,
)


def _sample_history() -> pd.DataFrame:
    """Create fake market data so tests do not need the internet."""
    dates = pd.date_range("2026-01-02", periods=2, freq="B")

    return pd.DataFrame(
        {
            "Volume": [1_000_000, 1_200_000],
            "Close": [102.0, 105.0],
            "Adj Close": [101.5, 104.5],
            "High": [103.0, 106.0],
            "Open": [100.0, 103.0],
            "Low": [99.0, 101.0],
        },
        index=dates,
    )


def test_normalize_ticker_removes_spaces_and_uppercases():
    """Ticker symbols should be trimmed and converted to uppercase."""
    assert normalize_ticker(" reliance.ns ") == "RELIANCE.NS"


def test_normalize_ticker_rejects_empty_values():
    """Blank ticker values should fail before any provider call."""
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_ticker("   ")


def test_get_history_downloads_and_standardizes_data(monkeypatch):
    """Downloaded data should have one predictable OHLCV column order."""
    captured_arguments = {}

    def fake_download(**kwargs):
        captured_arguments.update(kwargs)
        return _sample_history()

    monkeypatch.setattr(
        "src.data.yfinance_client.yf.download",
        fake_download,
    )

    history = YFinanceClient().get_history(
        " reliance.ns ",
        period="1y",
        interval="1d",
    )

    assert captured_arguments["tickers"] == "RELIANCE.NS"
    assert captured_arguments["period"] == "1y"
    assert captured_arguments["interval"] == "1d"
    assert captured_arguments["auto_adjust"] is False
    assert list(history.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]
    assert history.index.name == "Date"
    assert history.loc[pd.Timestamp("2026-01-02"), "Close"] == 102.0


def test_get_history_rejects_empty_provider_response(monkeypatch):
    """Empty results should produce a clear MarketDataError."""

    def fake_download(**kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(
        "src.data.yfinance_client.yf.download",
        fake_download,
    )

    with pytest.raises(MarketDataError, match="No historical data"):
        YFinanceClient().get_history("INVALID")


def test_get_history_rejects_unsupported_interval():
    """Intraday data is intentionally not enabled in the first version."""
    with pytest.raises(ValueError, match="Unsupported interval"):
        YFinanceClient().get_history("RELIANCE.NS", interval="5m")
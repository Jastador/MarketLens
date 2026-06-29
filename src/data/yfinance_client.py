from __future__ import annotations

import pandas as pd
import yfinance as yf

from src.config import Settings, load_settings


REQUIRED_HISTORY_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
)

SUPPORTED_INTERVALS = frozenset({"1d", "1wk", "1mo"})


class MarketDataError(RuntimeError):
    """Raised when market data cannot be downloaded or validated."""


def normalize_ticker(ticker: str) -> str:
    """Clean a user-supplied ticker symbol for use with yfinance."""
    if not isinstance(ticker, str):
        raise TypeError("Ticker must be a string.")

    normalized_ticker = ticker.strip().upper()

    if not normalized_ticker:
        raise ValueError("Ticker cannot be empty.")

    return normalized_ticker


class YFinanceClient:
    """Fetch and standardize historical OHLCV data through yfinance."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

    def get_history(
        self,
        ticker: str,
        *,
        period: str | None = None,
        interval: str | None = None,
    ) -> pd.DataFrame:
        """Download validated historical data for one ticker."""
        symbol = normalize_ticker(ticker)
        requested_period = period or self.settings.default_period
        requested_interval = interval or self.settings.default_interval

        if requested_interval not in SUPPORTED_INTERVALS:
            supported_values = ", ".join(sorted(SUPPORTED_INTERVALS))
            raise ValueError(
                f"Unsupported interval {requested_interval!r}. "
                f"Supported intervals: {supported_values}."
            )

        try:
            history = yf.download(
                tickers=symbol,
                period=requested_period,
                interval=requested_interval,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                multi_level_index=False,
                timeout=15,
            )
        except Exception as error:
            raise MarketDataError(
                f"Could not download historical data for {symbol}."
            ) from error

        if history is None or history.empty:
            raise MarketDataError(
                f"No historical data was returned for {symbol}. "
                "Check the ticker symbol, exchange suffix, and date range."
            )

        return self._standardize_history(history, symbol)

    @staticmethod
    def _standardize_history(
        history: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """Return historical data with one predictable OHLCV schema."""
        standardized = history.copy()

        if isinstance(standardized.columns, pd.MultiIndex):
            standardized.columns = standardized.columns.get_level_values(0)

        missing_columns = [
            column
            for column in REQUIRED_HISTORY_COLUMNS
            if column not in standardized.columns
        ]

        if missing_columns:
            missing_values = ", ".join(missing_columns)
            raise MarketDataError(
                f"Historical data for {symbol} is missing columns: "
                f"{missing_values}."
            )

        standardized = standardized.loc[:, REQUIRED_HISTORY_COLUMNS].copy()
        standardized = standardized.sort_index()
        standardized.index.name = "Date"

        return standardized
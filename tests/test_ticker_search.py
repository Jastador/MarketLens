import pytest

from src.data.ticker_search import (
    TickerSearchError,
    YFinanceTickerSearch,
    normalize_search_query,
)


def test_normalize_search_query_removes_outer_spaces():
    """Company-name searches should keep their text but lose outer spaces."""
    assert normalize_search_query(" Tata Motors ") == "Tata Motors"


def test_normalize_search_query_rejects_blank_query():
    """Blank searches should fail before contacting Yahoo Finance."""
    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_search_query("   ")


def test_search_returns_clean_deduplicated_results(monkeypatch):
    """Provider results should be standardized and duplicate symbols removed."""
    captured_arguments = {}

    class FakeSearch:
        def __init__(self, **kwargs):
            captured_arguments.update(kwargs)
            self.quotes = [
                {
                    "symbol": "RELIANCE.NS",
                    "shortname": "Reliance Industries Limited",
                    "quoteType": "EQUITY",
                    "exchDisp": "NSE",
                    "region": "in",
                    "currency": "inr",
                },
                {
                    "symbol": "reliance.ns",
                    "shortname": "Duplicate Reliance Result",
                    "quoteType": "EQUITY",
                    "exchDisp": "NSE",
                    "region": "in",
                    "currency": "inr",
                },
                {
                    "symbol": "RELIANCE.BO",
                    "longname": "Reliance Industries Limited",
                    "quoteType": "EQUITY",
                    "exchange": "BSE",
                    "region": "in",
                    "currency": "inr",
                },
            ]

    monkeypatch.setattr(
        "src.data.ticker_search.yf.Search",
        FakeSearch,
    )

    results = YFinanceTickerSearch().search(" reliance ", max_results=5)

    assert captured_arguments["query"] == "reliance"
    assert captured_arguments["max_results"] == 5
    assert captured_arguments["news_count"] == 0
    assert [result.symbol for result in results] == [
        "RELIANCE.NS",
        "RELIANCE.BO",
    ]
    assert results[0].name == "Reliance Industries Limited"
    assert results[0].exchange == "NSE"
    assert results[0].quote_type == "EQUITY"
    assert results[0].region == "IN"
    assert results[0].currency == "INR"


def test_search_rejects_invalid_result_count():
    """The search result count should stay within safe project limits."""
    with pytest.raises(ValueError, match="between 1 and 25"):
        YFinanceTickerSearch().search("Reliance", max_results=0)


def test_search_wraps_provider_errors(monkeypatch):
    """Provider errors should become clear MarketLens-specific errors."""

    def failing_search(**kwargs):
        raise ConnectionError("Provider unavailable")

    monkeypatch.setattr(
        "src.data.ticker_search.yf.Search",
        failing_search,
    )

    with pytest.raises(TickerSearchError, match="Could not search"):
        YFinanceTickerSearch().search("Reliance")
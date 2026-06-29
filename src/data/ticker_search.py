from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yfinance as yf


class TickerSearchError(RuntimeError):
    """Raised when ticker search cannot be completed."""


@dataclass(frozen=True, slots=True)
class TickerSearchResult:
    """A clean, provider-independent ticker search result."""

    symbol: str
    name: str
    quote_type: str
    exchange: str
    region: str
    currency: str


def normalize_search_query(query: str) -> str:
    """Clean a company name or ticker query before sending it to a provider."""
    if not isinstance(query, str):
        raise TypeError("Search query must be a string.")

    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError("Search query cannot be empty.")

    return normalized_query


def _read_text(record: dict[str, Any], key: str) -> str:
    """Read one text value safely from a provider response."""
    value = record.get(key, "")

    if value is None:
        return ""

    return str(value).strip()


class YFinanceTickerSearch:
    """Search Yahoo Finance for ticker symbols and company names."""

    def search(
        self,
        query: str,
        *,
        max_results: int = 8,
    ) -> list[TickerSearchResult]:
        """Return clean and deduplicated ticker search results."""
        normalized_query = normalize_search_query(query)

        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise TypeError("max_results must be an integer.")

        if not 1 <= max_results <= 25:
            raise ValueError("max_results must be between 1 and 25.")

        try:
            response = yf.Search(
                query=normalized_query,
                max_results=max_results,
                news_count=0,
                lists_count=0,
                include_cb=False,
                timeout=15,
                raise_errors=True,
            )
            raw_quotes = response.quotes or []
        except Exception as error:
            raise TickerSearchError(
                f"Could not search for ticker results for {normalized_query!r}."
            ) from error

        results = self._standardize_quotes(raw_quotes)
        return self._rank_exact_symbol_matches(results, normalized_query)

    @staticmethod
    def _standardize_quotes(
        raw_quotes: list[dict[str, Any]],
    ) -> list[TickerSearchResult]:
        """Convert Yahoo Finance quote dictionaries into stable app objects."""
        results: list[TickerSearchResult] = []
        seen_symbols: set[str] = set()

        for quote in raw_quotes:
            if not isinstance(quote, dict):
                continue

            symbol = _read_text(quote, "symbol").upper()

            if not symbol or symbol in seen_symbols:
                continue

            seen_symbols.add(symbol)

            name = (
                _read_text(quote, "longname")
                or _read_text(quote, "shortname")
                or symbol
            )

            results.append(
                TickerSearchResult(
                    symbol=symbol,
                    name=name,
                    quote_type=_read_text(quote, "quoteType").upper() or "UNKNOWN",
                    exchange=(
                        _read_text(quote, "exchDisp")
                        or _read_text(quote, "exchange")
                        or "UNKNOWN"
                    ),
                    region=_read_text(quote, "region").upper() or "UNKNOWN",
                    currency=_read_text(quote, "currency").upper() or "UNKNOWN",
                )
            )

        return results

    @staticmethod
    def _rank_exact_symbol_matches(
        results: list[TickerSearchResult],
        query: str,
    ) -> list[TickerSearchResult]:
        """Place an exact ticker-symbol match above broader company matches."""
        normalized_query = query.upper()

        return sorted(
            results,
            key=lambda result: result.symbol != normalized_query,
        )
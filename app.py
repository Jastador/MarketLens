from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.ticker_search import (
    TickerSearchError,
    TickerSearchResult,
    YFinanceTickerSearch,
)
from src.data.yfinance_client import (
    MarketDataError,
    YFinanceClient,
    normalize_ticker,
)


st.set_page_config(
    page_title="MarketLens AI",
    page_icon="📈",
    layout="wide",
)


def results_to_dataframe(
    results: list[TickerSearchResult],
) -> pd.DataFrame:
    """Convert ticker-search objects into a dashboard table."""
    return pd.DataFrame(
        [
            {
                "Symbol": result.symbol,
                "Company": result.name,
                "Exchange": result.exchange,
                "Type": result.quote_type,
                "Region": result.region,
                "Currency": result.currency,
            }
            for result in results
        ]
    )


def build_candlestick_chart(
    history: pd.DataFrame,
    symbol: str,
) -> go.Figure:
    """Create an interactive candlestick chart from OHLCV data."""
    chart = go.Figure(
        data=[
            go.Candlestick(
                x=history.index,
                open=history["Open"],
                high=history["High"],
                low=history["Low"],
                close=history["Close"],
                name=symbol,
            )
        ]
    )

    chart.update_layout(
        title=f"{symbol} Historical Price",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=550,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )

    return chart


def clear_loaded_history() -> None:
    """Remove old market data when the selected instrument changes."""
    for key in ("history", "history_symbol", "history_currency"):
        st.session_state.pop(key, None)


def clear_selected_instrument() -> None:
    """Remove the current selected instrument and its loaded data."""
    for key in (
        "selected_symbol",
        "selected_name",
        "selected_currency",
    ):
        st.session_state.pop(key, None)

    clear_loaded_history()


def select_instrument(
    symbol: str,
    name: str,
    currency: str,
) -> None:
    """Store one active instrument for the shared history panel."""
    previous_symbol = st.session_state.get("selected_symbol")

    st.session_state["selected_symbol"] = symbol
    st.session_state["selected_name"] = name
    st.session_state["selected_currency"] = currency

    if previous_symbol != symbol:
        clear_loaded_history()


def format_price(currency: str, value: float) -> str:
    """Format a price without pretending an unknown currency is known."""
    if currency == "UNKNOWN":
        return f"{value:,.2f}"

    return f"{currency} {value:,.2f}"


def render_market_overview(
    history: pd.DataFrame,
    symbol: str,
    currency: str,
) -> None:
    """Render key metrics, chart, volume, and recent historical candles."""
    latest_close = float(history["Close"].iloc[-1])
    first_close = float(history["Close"].iloc[0])
    period_return = ((latest_close / first_close) - 1) * 100
    average_volume = float(history["Volume"].mean())

    st.divider()
    st.subheader(f"{symbol} Overview")

    metric_one, metric_two, metric_three = st.columns(3)

    metric_one.metric(
        "Latest close",
        format_price(currency, latest_close),
    )
    metric_two.metric(
        "Period return",
        f"{period_return:.2f}%",
    )
    metric_three.metric(
        "Average volume",
        f"{average_volume:,.0f}",
    )

    st.plotly_chart(
        build_candlestick_chart(history, symbol),
        width="stretch",
        config={"displaylogo": False},
    )

    st.subheader("Volume")
    st.bar_chart(history["Volume"])

    st.subheader("Latest candles")
    st.dataframe(
        history.tail(20),
        width="stretch",
    )


st.title("MarketLens AI")
st.caption(
    "Explainable market research for Indian and global instruments. "
    "For educational and research use only, not financial advice."
)

st.divider()

direct_column, search_column = st.columns(2)

with direct_column:
    st.subheader("Use an exact ticker")
    st.caption(
        "Best for known symbols such as RELIANCE.NS, TCS.NS, "
        "AAPL, TSLA, or 7203.T."
    )

    with st.form("direct_ticker_form"):
        direct_ticker = st.text_input(
            "Exact ticker symbol",
            placeholder="Example: RELIANCE.NS",
        )
        direct_submitted = st.form_submit_button("Use exact ticker")

    if direct_submitted:
        try:
            symbol = normalize_ticker(direct_ticker)

            st.session_state.pop("ticker_results", None)
            select_instrument(
                symbol=symbol,
                name=f"Direct ticker: {symbol}",
                currency="UNKNOWN",
            )

            st.success(f"Selected exact ticker: {symbol}")

        except (TypeError, ValueError) as error:
            st.error(str(error))

with search_column:
    st.subheader("Search by company")
    st.caption("Useful when you know the company name but not its ticker.")

    with st.form("ticker_search_form"):
        query = st.text_input(
            "Search company or ticker",
            placeholder="Examples: Reliance, Tata Motors, Tesla, AAPL",
        )
        search_submitted = st.form_submit_button("Search instruments")

    if search_submitted:
        try:
            results = YFinanceTickerSearch().search(query)

            if not results:
                st.session_state.pop("ticker_results", None)
                st.warning("No instruments were found for that search.")
            else:
                st.session_state["ticker_results"] = results
                clear_selected_instrument()

        except (TickerSearchError, TypeError, ValueError) as error:
            st.error(str(error))

results = st.session_state.get("ticker_results", [])

if results:
    st.divider()
    st.subheader("Search results")
    st.caption("Click one row to choose an instrument.")

    selection_event = st.dataframe(
        results_to_dataframe(results),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="ticker_results_table",
    )

    selected_rows = selection_event.selection.rows

    if selected_rows:
        selected_result = results[selected_rows[0]]

        select_instrument(
            symbol=selected_result.symbol,
            name=selected_result.name,
            currency=selected_result.currency,
        )

        st.success(
            f"Selected: {selected_result.symbol} | "
            f"{selected_result.name} | {selected_result.exchange}"
        )
    else:
        st.info("Choose one instrument by clicking its row in the table.")

selected_symbol = st.session_state.get("selected_symbol")

if not selected_symbol:
    st.info("Search for a company or enter an exact ticker to begin.")
    st.stop()

selected_name = st.session_state["selected_name"]
selected_currency = st.session_state["selected_currency"]

st.divider()
st.subheader("Load historical market data")
st.caption(f"Selected instrument: {selected_symbol} | {selected_name}")

with st.form("history_form"):
    first_column, second_column = st.columns(2)

    with first_column:
        period = st.selectbox(
            "Historical period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=4,
        )

    with second_column:
        interval = st.selectbox(
            "Candle interval",
            options=["1d", "1wk", "1mo"],
            index=0,
        )

    history_submitted = st.form_submit_button("Load market data")

if history_submitted:
    try:
        history = YFinanceClient().get_history(
            selected_symbol,
            period=period,
            interval=interval,
        )

        st.session_state["history"] = history
        st.session_state["history_symbol"] = selected_symbol
        st.session_state["history_currency"] = selected_currency

    except (MarketDataError, ValueError) as error:
        st.error(str(error))

history = st.session_state.get("history")
history_symbol = st.session_state.get("history_symbol")

if history is not None and history_symbol == selected_symbol:
    history_currency = st.session_state.get(
        "history_currency",
        "UNKNOWN",
    )

    render_market_overview(
        history=history,
        symbol=selected_symbol,
        currency=history_currency,
    )
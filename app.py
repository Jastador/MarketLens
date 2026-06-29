from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.ticker_search import (
    TickerSearchError,
    TickerSearchResult,
    YFinanceTickerSearch,
)
from src.data.yfinance_client import MarketDataError, YFinanceClient


st.set_page_config(
    page_title="MarketLens AI",
    page_icon="📈",
    layout="wide",
)


def results_to_dataframe(
    results: list[TickerSearchResult],
) -> pd.DataFrame:
    """Convert ticker-search objects into a table for the dashboard."""
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


st.title("MarketLens AI")
st.caption(
    "Explainable market research for Indian and global instruments. "
    "For educational and research use only, not financial advice."
)

st.divider()

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
            st.warning("No instruments were found for that search.")
        else:
            st.session_state["ticker_results"] = results
            st.session_state.pop("history", None)
            st.session_state.pop("history_symbol", None)

    except (TickerSearchError, TypeError, ValueError) as error:
        st.error(str(error))

results = st.session_state.get("ticker_results", [])

if results:
    st.subheader("Search results")
    st.caption("Click one row to choose an instrument.")

    results_table = results_to_dataframe(results)

    selection_event = st.dataframe(
        results_table,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="ticker_results_table",
    )

    selected_rows = selection_event.selection.rows

    if not selected_rows:
        st.info("Choose one instrument by clicking its row in the table.")
        st.stop()

    selected_result = results[selected_rows[0]]

    st.success(
        f"Selected: {selected_result.symbol} | "
        f"{selected_result.name} | {selected_result.exchange}"
    )

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
                selected_result.symbol,
                period=period,
                interval=interval,
            )

            st.session_state["history"] = history
            st.session_state["history_symbol"] = selected_result.symbol
            st.session_state["history_currency"] = selected_result.currency

        except (MarketDataError, ValueError) as error:
            st.error(str(error))

    history = st.session_state.get("history")
    history_symbol = st.session_state.get("history_symbol")

    if history is not None and history_symbol == selected_result.symbol:
        currency = st.session_state.get("history_currency", "UNKNOWN")

        latest_close = float(history["Close"].iloc[-1])
        first_close = float(history["Close"].iloc[0])
        period_return = ((latest_close / first_close) - 1) * 100
        average_volume = float(history["Volume"].mean())

        st.divider()
        st.subheader(f"{selected_result.symbol} Overview")

        metric_one, metric_two, metric_three = st.columns(3)

        metric_one.metric(
            "Latest close",
            f"{currency} {latest_close:,.2f}",
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
            build_candlestick_chart(history, selected_result.symbol),
            use_container_width=True,
            config={"displaylogo": False},
        )

        st.subheader("Volume")
        st.bar_chart(history["Volume"])

        st.subheader("Latest candles")
        st.dataframe(
            history.tail(20),
            use_container_width=True,
        )
else:
    st.info("Search for a company or ticker to begin.")
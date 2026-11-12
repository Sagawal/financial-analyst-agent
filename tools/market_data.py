import json
import yfinance as yf

from langchain_core.tools import tool


SUPPORTED_TICKERS = {
    "AAPL",
    "MSFT",
    "NVDA"
}


def validate_ticker(ticker: str):
    ticker = ticker.upper().strip()

    if ticker not in SUPPORTED_TICKERS:
        raise ValueError(
            f"Unsupported ticker: {ticker}. "
            f"Supported tickers: {', '.join(SUPPORTED_TICKERS)}"
        )

    return ticker


@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest available stock price, previous close and currency."""

    try:
        ticker = validate_ticker(ticker)
        stock = yf.Ticker(ticker)

        history = stock.history(
            period="5d",
            interval="1d",
            auto_adjust=True
        )

        if history.empty:
            return "No stock price data available."

        latest = history.iloc[-1]

        previous_close = (
            float(history.iloc[-2]["Close"])
            if len(history) >= 2 else None
        )

        result = {
            "ticker": ticker,
            "currency": stock.fast_info.get("currency"),
            "latest_available_close": round(float(latest["Close"]), 2),
            "previous_close": (
                round(previous_close, 2)
                if previous_close is not None else None
            ),
            "day_high": round(float(latest["High"]), 2),
            "day_low": round(float(latest["Low"]), 2),
            "date": latest.name.strftime("%Y-%m-%d")
        }

        return json.dumps(result, default=str)

    except Exception as e:
        return f"Unable to retrieve stock price: {str(e)}"


@tool
def get_historical_stock_data(
    ticker: str,
    period: str = "1mo"
) -> str:
    """
    Get historical daily stock prices.
    Supported periods: 1mo, 3mo, 6mo, 1y, 2y, 5y.
    """

    try:
        ticker = validate_ticker(ticker)

        allowed_periods = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]

        if period not in allowed_periods:
            return f"Unsupported period. Choose from: {allowed_periods}"

        stock = yf.Ticker(ticker)

        history = stock.history(
            period=period,
            interval="1d",
            auto_adjust=True
        )

        if history.empty:
            return "No historical stock data available."

        # Return the most recent 10 trading days
        history = history.tail(10)

        records = []

        for date, row in history.iterrows():
            records.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"])
            })

        result = {
            "ticker": ticker,
            "period": period,
            "currency": stock.fast_info.get("currency"),
            "historical_data": records
        }

        return json.dumps(result, default=str)

    except Exception as e:
        return f"Unable to retrieve historical data: {str(e)}"


@tool
def get_company_market_info(ticker: str) -> str:
    """Get company information including sector, industry, market capitalization and currency."""

    try:
        ticker = validate_ticker(ticker)

        stock = yf.Ticker(ticker)
        info = stock.info

        result = {
            "ticker": ticker,
            "company_name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency"),
            "website": info.get("website")
        }

        return json.dumps(result, default=str)

    except Exception as e:
        return f"Unable to retrieve company information: {str(e)}"
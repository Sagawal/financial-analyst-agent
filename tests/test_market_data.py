
import json
from unittest.mock import patch

import pandas as pd
import pytest

from tools.market_data import (
    get_stock_price,
    get_historical_stock_data,
    get_company_market_info,
)


@pytest.fixture
def mock_history():
    dates = pd.to_datetime([
        "2026-11-11",
        "2026-11-12",
        "2026-11-13",
    ])

    return pd.DataFrame(
        {
            "Open": [100.0, 105.0, 110.0],
            "High": [108.0, 112.0, 115.0],
            "Low": [98.0, 103.0, 108.0],
            "Close": [105.0, 110.0, 112.5],
            "Volume": [1000, 1200, 1500],
        },
        index=dates,
    )


# ---------------- Stock Price Tests ----------------

@patch("tools.market_data.yf.Ticker")
def test_get_stock_price(mock_ticker, mock_history):
    stock = mock_ticker.return_value
    stock.history.return_value = mock_history
    stock.fast_info = {"currency": "USD"}

    result = get_stock_price.invoke({"ticker": "AAPL"})
    data = json.loads(result)

    assert data["ticker"] == "AAPL"
    assert data["currency"] == "USD"
    assert data["latest_available_close"] == 112.5
    assert data["previous_close"] == 110.0
    assert data["day_high"] == 115.0
    assert data["day_low"] == 108.0
    assert data["date"] == "2026-11-13"


@patch("tools.market_data.yf.Ticker")
def test_get_stock_price_with_lowercase_ticker(mock_ticker, mock_history):
    stock = mock_ticker.return_value
    stock.history.return_value = mock_history
    stock.fast_info = {"currency": "USD"}

    result = get_stock_price.invoke({"ticker": "aapl"})
    data = json.loads(result)

    assert data["ticker"] == "AAPL"
    mock_ticker.assert_called_once_with("AAPL")


@patch("tools.market_data.yf.Ticker")
def test_get_stock_price_empty_history(mock_ticker):
    stock = mock_ticker.return_value
    stock.history.return_value = pd.DataFrame()

    result = get_stock_price.invoke({"ticker": "AAPL"})

    assert result == "No stock price data available."


def test_get_stock_price_invalid_ticker():
    result = get_stock_price.invoke({"ticker": "INVALID"})

    assert result.startswith("Unable to retrieve stock price:")
    assert "Unsupported ticker" in result


@patch("tools.market_data.yf.Ticker")
def test_get_stock_price_api_error(mock_ticker):
    mock_ticker.side_effect = Exception("API unavailable")

    result = get_stock_price.invoke({"ticker": "AAPL"})

    assert "API unavailable" in result


# ---------------- Historical Data Tests ----------------

@patch("tools.market_data.yf.Ticker")
def test_get_historical_stock_data(mock_ticker, mock_history):
    stock = mock_ticker.return_value
    stock.history.return_value = mock_history
    stock.fast_info = {"currency": "USD"}

    result = get_historical_stock_data.invoke({
        "ticker": "NVDA",
        "period": "1mo",
    })

    data = json.loads(result)

    assert data["ticker"] == "NVDA"
    assert data["period"] == "1mo"
    assert data["currency"] == "USD"
    assert len(data["historical_data"]) == 3

    latest = data["historical_data"][-1]

    assert latest["date"] == "2026-11-13"
    assert latest["close"] == 112.5
    assert latest["volume"] == 1500


def test_get_historical_stock_data_unsupported_period():
    result = get_historical_stock_data.invoke({
        "ticker": "AAPL",
        "period": "10y",
    })

    assert "Unsupported period" in result


def test_get_historical_stock_data_invalid_ticker():
    result = get_historical_stock_data.invoke({
        "ticker": "INVALID",
        "period": "1mo",
    })

    assert result.startswith("Unable to retrieve historical data:")
    assert "Unsupported ticker" in result


@patch("tools.market_data.yf.Ticker")
def test_get_historical_stock_data_empty_history(mock_ticker):
    stock = mock_ticker.return_value
    stock.history.return_value = pd.DataFrame()

    result = get_historical_stock_data.invoke({
        "ticker": "AAPL",
        "period": "1mo",
    })

    assert result == "No historical stock data available."


@patch("tools.market_data.yf.Ticker")
def test_get_historical_stock_data_api_error(mock_ticker):
    mock_ticker.side_effect = Exception("API unavailable")

    result = get_historical_stock_data.invoke({
        "ticker": "AAPL",
        "period": "1mo",
    })

    assert "API unavailable" in result


# ---------------- Company Information Tests ----------------

@patch("tools.market_data.yf.Ticker")
def test_get_company_market_info(mock_ticker):
    stock = mock_ticker.return_value
    stock.info = {
        "longName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 3000000000000,
        "currency": "USD",
        "website": "https://www.apple.com",
    }

    result = get_company_market_info.invoke({"ticker": "AAPL"})
    data = json.loads(result)

    assert data["ticker"] == "AAPL"
    assert data["company_name"] == "Apple Inc."
    assert data["sector"] == "Technology"
    assert data["industry"] == "Consumer Electronics"
    assert data["market_cap"] == 3000000000000
    assert data["currency"] == "USD"


def test_get_company_market_info_invalid_ticker():
    result = get_company_market_info.invoke({"ticker": "INVALID"})

    assert result.startswith("Unable to retrieve company information:")
    assert "Unsupported ticker" in result


@patch("tools.market_data.yf.Ticker")
def test_get_company_market_info_api_error(mock_ticker):
    stock = mock_ticker.return_value
    stock.info = property(lambda self: None)

    # Simulate an exception while accessing company information.
    type(stock).info = property(
        lambda self: (_ for _ in ()).throw(Exception("API unavailable"))
    )

    result = get_company_market_info.invoke({"ticker": "AAPL"})

    assert "API unavailable" in result

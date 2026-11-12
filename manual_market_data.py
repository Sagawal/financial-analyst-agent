from tools.market_data import (
    get_stock_price,
    get_historical_stock_data,
    get_company_market_info
)

print("\n--- STOCK PRICE ---")
print(get_stock_price.invoke({"ticker": "AAPL"}))

print("\n--- HISTORICAL DATA ---")
print(
    get_historical_stock_data.invoke({
        "ticker": "NVDA",
        "period": "1mo"
    })
)

print("\n--- COMPANY INFORMATION ---")
print(get_company_market_info.invoke({"ticker": "MSFT"}))
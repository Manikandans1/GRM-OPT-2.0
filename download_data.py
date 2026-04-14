import yfinance as yf
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

markets = {
    "nifty50": "^NSEI",
    "sp500": "^GSPC",
    "dax40": "^GDAXI"
}

start = "2010-01-01"
end = "2024-12-31"

for name, ticker in markets.items():
    print(f"Downloading {name}...")
    df = yf.download(ticker, start=start, end=end)

    df = df[['Close']]
    df = df.resample('M').last()
    df.reset_index(inplace=True)

    df.to_csv(f"data/{name}.csv", index=False)

print("All datasets saved successfully.")
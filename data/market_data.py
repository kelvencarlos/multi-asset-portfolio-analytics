import pandas as pd
import yfinance as yf

def get_price_dataframe(symbol, period="2y", interval="1d"):
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"Nao foi possivel obter dados para {symbol} via Yahoo Finance.")

    close_data = data["Close"]
    # Em algumas versoes o yfinance retorna coluna Close como DataFrame (2D).
    if isinstance(close_data, pd.DataFrame):
        close_series = close_data.iloc[:, 0]
    else:
        close_series = close_data

    close_series = pd.to_numeric(close_series, errors="coerce").dropna()
    if close_series.empty:
        raise ValueError(f"Nao ha serie de fechamento valida para {symbol}.")

    df = pd.DataFrame({"date": close_series.index, "close": close_series.values})

    return df.set_index("date")
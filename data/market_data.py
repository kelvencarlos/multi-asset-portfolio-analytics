from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import pandas as pd
import yfinance as yf


PERIOD_TO_BUSINESS_DAYS = {
    "1mo": 22,
    "3mo": 66,
    "6mo": 132,
    "1y": 252,
    "2y": 504,
    "5y": 1260,
    "10y": 2520,
}

CDI_PROXY_ANNUAL_RATE = 0.11


def _period_to_business_days(period: str) -> int:
    return PERIOD_TO_BUSINESS_DAYS.get(period, 504)


def _build_cdi_proxy(period: str, annual_rate: float = CDI_PROXY_ANNUAL_RATE) -> pd.DataFrame:
    # Proxy determinístico simplificado para representar caixa pós-fixado no app.
    # Não replica o CDI oficial diário e existe apenas para fins de simulação.
    business_days = _period_to_business_days(period)
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=business_days)
    daily_rate = (1 + annual_rate) ** (1 / 252) - 1
    levels = 100 * (1 + daily_rate) ** pd.RangeIndex(start=0, stop=len(dates), step=1)
    return pd.DataFrame({"close": levels}, index=dates)


def _extract_price_series(data: pd.DataFrame, symbol: str) -> pd.Series:
    # Preferimos série ajustada para cálculo de retornos históricos quando disponível.
    # Em ativos sem ajuste (ex.: alguns câmbios/índices), voltamos ao fechamento padrão.
    for column_name in ("Adj Close", "Close"):
        if isinstance(data.columns, pd.MultiIndex):
            if column_name not in data.columns.get_level_values(0):
                continue
            column_data = data.xs(column_name, axis=1, level=0)
        else:
            if column_name not in data.columns:
                continue
            column_data = data[column_name]

        if isinstance(column_data, pd.DataFrame):
            series = column_data.iloc[:, 0]
        else:
            series = column_data

        series = pd.to_numeric(series, errors="coerce").dropna()
        if not series.empty:
            return series

    raise ValueError(
        f"Não há série de preços válida para {symbol} (colunas esperadas: 'Adj Close' ou 'Close')."
    )


def get_price_dataframe(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    if symbol == "CDI_PROXY":
        return _build_cdi_proxy(period)

    data = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"Não foi possível obter dados para {symbol} via Yahoo Finance.")

    close_series = _extract_price_series(data, symbol)

    df = pd.DataFrame({"date": close_series.index, "close": close_series.values})
    return df.set_index("date")


def validate_tickers(symbols: Iterable[str], period: str = "1mo") -> Tuple[List[str], Dict[str, str]]:
    valid: List[str] = []
    invalid: Dict[str, str] = {}

    for symbol in symbols:
        try:
            frame = get_price_dataframe(symbol, period=period)
            if frame.empty:
                invalid[symbol] = "série vazia"
            else:
                valid.append(symbol)
        except Exception as exc:  # noqa: BLE001
            invalid[symbol] = str(exc)

    return valid, invalid

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional

import requests


BASE_URL = "https://finnhub.io/api/v1"


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _get_api_key() -> Optional[str]:
    _load_env_file()
    return os.getenv("FINNHUB_API_KEY")


def get_candles(symbol: str, days: int = 365) -> Dict[str, object]:
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("Defina FINNHUB_API_KEY no ambiente ou no arquivo .env com uma chave válida da Finnhub.")

    url = f"{BASE_URL}/stock/candle"
    params = {
        "symbol": symbol,
        "resolution": "D",
        "from": int(time.time()) - days * 86400,
        "to": int(time.time()),
        "token": api_key,
    }

    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 403:
        raise ValueError(
            "A Finnhub recusou a requisição (403). Verifique se a FINNHUB_API_KEY é válida e se o plano permite este endpoint."
        )

    response.raise_for_status()
    payload = response.json()

    if payload.get("s") == "no_data":
        raise ValueError(f"Não há dados disponíveis para o ativo {symbol}.")

    if payload.get("s") not in (None, "ok"):
        detail = payload.get("error", "erro desconhecido")
        raise ValueError(f"Erro retornado pela Finnhub para {symbol}: {detail}")

    return payload

from __future__ import annotations

from typing import Dict, List, Union


DEFAULT_ASSET_CLASS = "Alternativos"


ASSET_CATALOG: Dict[str, Dict[str, str]] = {
    "BOVA11.SA": {
        "display_name": "ETF iShares Ibovespa (BOVA11)",
        "selector_label": "ETF iShares Ibovespa (BOVA11.SA)",
        "class_name": "Ações Brasil",
        "category": "Brasil",
    },
    "IRFM11.SA": {
        "display_name": "ETF IRF-M 1+ (IRFM11)",
        "selector_label": "ETF IRF-M 1+ (IRFM11.SA)",
        "class_name": "Renda Fixa",
        "category": "Brasil",
    },
    "IMAB11.SA": {
        "display_name": "ETF IMA-B (IMAB11)",
        "selector_label": "ETF IMA-B (IMAB11.SA)",
        "class_name": "Renda Fixa",
        "category": "Brasil",
    },
    "CDI_PROXY": {
        "display_name": "CDI proxy simplificado (11% a.a.)",
        "selector_label": "CDI proxy simplificado (CDI_PROXY)",
        "class_name": "Renda Fixa",
        "category": "Brasil",
    },
    "IVVB11.SA": {
        "display_name": "ETF S&P 500 (IVVB11)",
        "selector_label": "ETF S&P 500 (IVVB11.SA)",
        "class_name": "Ações EUA",
        "category": "Exterior",
    },
    "QQQ": {
        "display_name": "Invesco QQQ Trust (Nasdaq 100)",
        "selector_label": "Invesco QQQ Trust (QQQ)",
        "class_name": "Ações EUA",
        "category": "Exterior",
    },
    "USDBRL=X": {
        "display_name": "Dólar comercial (USD/BRL)",
        "selector_label": "Dólar comercial (USDBRL=X)",
        "class_name": "Dólar",
        "category": "Exterior",
    },
    "KNRI11.SA": {
        "display_name": "FII Kinea Renda Imobiliária (KNRI11)",
        "selector_label": "FII Kinea Renda Imobiliária (KNRI11.SA)",
        "class_name": "Alternativos",
        "category": "Alternativos",
    },
    "XFIX11.SA": {
        "display_name": "ETF IFIX (XFIX11)",
        "selector_label": "ETF IFIX (XFIX11.SA)",
        "class_name": "Alternativos",
        "category": "Alternativos",
    },
    "BTC-USD": {
        "display_name": "Bitcoin em dólar (BTC-USD)",
        "selector_label": "Bitcoin em dólar (BTC-USD)",
        "class_name": "Alternativos",
        "category": "Alternativos",
    },
}


STRESS_CLASS_CATALOG: Dict[str, Dict[str, Union[int, str]]] = {
    "Ações Brasil": {
        "slider_label": "Choque Ações Brasil (%)",
        "default_shock_pct": -30,
    },
    "Ações EUA": {
        "slider_label": "Choque Ações EUA (%)",
        "default_shock_pct": -20,
    },
    "Renda Fixa": {
        "slider_label": "Choque Juros (%)",
        "default_shock_pct": 5,
    },
    "Dólar": {
        "slider_label": "Choque Dólar (%)",
        "default_shock_pct": 15,
    },
    "Alternativos": {
        "slider_label": "Choque Alternativos (%)",
        "default_shock_pct": -10,
    },
}


def ordered_asset_symbols() -> List[str]:
    return list(ASSET_CATALOG.keys())


def ordered_stress_classes() -> List[str]:
    return list(STRESS_CLASS_CATALOG.keys())


def asset_display_name(symbol: str) -> str:
    meta = ASSET_CATALOG.get(symbol, {})
    return str(meta.get("display_name", symbol))


def asset_selector_label(symbol: str) -> str:
    meta = ASSET_CATALOG.get(symbol, {})
    return str(meta.get("selector_label", symbol))


def asset_class(symbol: str) -> str:
    meta = ASSET_CATALOG.get(symbol, {})
    return str(meta.get("class_name", DEFAULT_ASSET_CLASS))


def stress_slider_label(class_name: str) -> str:
    class_meta = STRESS_CLASS_CATALOG.get(class_name, {})
    return str(class_meta.get("slider_label", class_name))


def stress_default_shock_pct(class_name: str) -> int:
    class_meta = STRESS_CLASS_CATALOG.get(class_name, {})
    value = class_meta.get("default_shock_pct", 0)
    return int(value)


def validate_catalog_integrity() -> List[str]:
    issues: List[str] = []
    required_asset_fields = ("display_name", "selector_label", "class_name", "category")

    for symbol, meta in ASSET_CATALOG.items():
        for field in required_asset_fields:
            if not str(meta.get(field, "")).strip():
                issues.append(f"Ativo {symbol}: campo obrigatório ausente ({field}).")

        class_name = str(meta.get("class_name", ""))
        if class_name and class_name not in STRESS_CLASS_CATALOG:
            issues.append(
                f"Ativo {symbol}: classe '{class_name}' não mapeada em STRESS_CLASS_CATALOG."
            )

    for class_name, class_meta in STRESS_CLASS_CATALOG.items():
        if not str(class_meta.get("slider_label", "")).strip():
            issues.append(f"Classe {class_name}: slider_label obrigatório está vazio.")
        if not isinstance(class_meta.get("default_shock_pct"), int):
            issues.append(f"Classe {class_name}: default_shock_pct deve ser inteiro.")

    return issues

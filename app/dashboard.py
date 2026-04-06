import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Garantir imports absolutos mesmo quando o Streamlit e executado de outro diretorio.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.market_data import get_price_dataframe, validate_tickers
from core.portfolio import calculate_returns, portfolio_return, cumulative_return
from core.risk import volatility, drawdown, risk_contribution
from core.scenarios import stress_test


def load_local_css(css_file: Path):
    import hashlib
    css_content = css_file.read_text(encoding='utf-8')
    css_hash = hashlib.md5(css_content.encode()).hexdigest()[:8]
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    # Força invalidação de cache do navegador
    st.markdown(f"<!-- CSS Version: {css_hash} -->", unsafe_allow_html=True)



ASSET_CATALOG = {
    "BOVA11.SA": {"nome": "BOVA11", "classe": "Ações Brasil", "categoria": "Brasil"},
    "IRFM11.SA": {"nome": "IRFM11", "classe": "Renda Fixa", "categoria": "Brasil"},
    "IMAB11.SA": {"nome": "IMAB11", "classe": "Renda Fixa", "categoria": "Brasil"},
    "CDI_PROXY": {"nome": "CDI (proxy)", "classe": "Renda Fixa", "categoria": "Brasil"},
    "IVVB11.SA": {"nome": "IVVB11", "classe": "Ações EUA", "categoria": "Exterior"},
    "QQQ": {"nome": "NASDAQ (QQQ)", "classe": "Ações EUA", "categoria": "Exterior"},
    "USDBRL=X": {"nome": "Dólar (USD/BRL)", "classe": "Dólar", "categoria": "Exterior"},
    "KNRI11.SA": {"nome": "KNRI11", "classe": "Alternativos", "categoria": "Alternativos"},
    "XFIX11.SA": {"nome": "XFIX11", "classe": "Alternativos", "categoria": "Alternativos"},
    "BTC-USD": {"nome": "Bitcoin", "classe": "Alternativos", "categoria": "Alternativos"},
}

PRESETS = {
    "Carteira Brasil": {
        "BOVA11.SA": 25,
        "IRFM11.SA": 35,
        "IMAB11.SA": 20,
        "CDI_PROXY": 20,
    },
    "Carteira Global": {
        "IVVB11.SA": 45,
        "QQQ": 20,
        "USDBRL=X": 15,
        "IRFM11.SA": 20,
    },
    "Carteira Balanceada": {
        "BOVA11.SA": 20,
        "IVVB11.SA": 25,
        "IRFM11.SA": 25,
        "IMAB11.SA": 15,
        "USDBRL=X": 10,
        "KNRI11.SA": 5,
    },
    "Custom": {},
}

BENCHMARK_DEFAULT = {
    "IVVB11.SA": 50,
    "IRFM11.SA": 50,
}


def _clean_title(symbol: str) -> str:
    return ASSET_CATALOG.get(symbol, {}).get("nome", symbol)


def _normalize_weights(raw_weights: np.ndarray) -> np.ndarray:
    total = raw_weights.sum()
    if total <= 0:
        return raw_weights
    return raw_weights / total


def _portfolio_profile(ann_vol: float, max_dd: float) -> str:
    if ann_vol <= 0.08 and max_dd >= -0.08:
        return "Conservador"
    if ann_vol <= 0.16 and max_dd >= -0.18:
        return "Moderado"
    return "Agressivo"


def _mean_abs_correlation(returns: pd.DataFrame) -> float:
    if returns.shape[1] <= 1:
        return 0.0
    corr = returns.corr().abs()
    corr_array = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_array, np.nan)
    return float(np.nanmean(corr_array))


def _build_insights(
    ann_vol: float,
    max_dd: float,
    mean_corr: float,
    max_rc_symbol: str,
    max_rc_value: float,
    weights_by_asset: Dict[str, float],
) -> List[str]:
    messages: List[str] = []
    equity_weight = sum(
        w
        for symbol, w in weights_by_asset.items()
        if ASSET_CATALOG.get(symbol, {}).get("classe") in {"Ações Brasil", "Ações EUA"}
    )
    dolar_weight = sum(
        w for symbol, w in weights_by_asset.items() if ASSET_CATALOG.get(symbol, {}).get("classe") == "Dólar"
    )

    if equity_weight > 0.55:
        messages.append("Carteira apresenta concentração relevante em renda variável.")
    if max_dd < -0.20:
        messages.append("Drawdown máximo superior ao esperado para perfil conservador.")
    if mean_corr < 0.35 and equity_weight <= 0.60:
        messages.append("Boa diversificação entre classes de ativos e geografias.")
    if dolar_weight > 0.10:
        messages.append("Exposição relevante ao risco de variação cambial.")
    if max_rc_value > 0.35:
        messages.append(
            f"Estrutura de alocação indica concentração de risco marginal em {_clean_title(max_rc_symbol)}."
        )
    if ann_vol < 0.10 and max_dd > -0.12:
        messages.append("Perfil de risco histórico alinhado com mandato defensivo.")

    return messages[:4] if messages else ["Estrutura de risco sem alertas criticos no periodo analisado."]


@st.cache_data(show_spinner=False)
def _get_valid_universe(symbols):
    return validate_tickers(symbols, period="1mo")


@st.cache_data(show_spinner=False)
def _load_close_prices(symbols, period):
    frame = pd.DataFrame()
    for symbol in symbols:
        frame[symbol] = get_price_dataframe(symbol, period=period)["close"]

    frame = frame.dropna(how="all")
    return frame.dropna()


st.set_page_config(page_title="Análise de Portfólio", layout="wide")
load_local_css(Path(__file__).with_name("styles.css"))

st.markdown(
    """
    <div class='topbar'>
        <div>
            <div class='eyebrow'>Plataforma de Monitoramento de Carteiras</div>
            <h1 class='app-title'>Painel de Risco e Performance</h1>
            <div class='app-subtitle'>Análise institucional de risco, cenários e recomendações para assessoria de investimentos.</div>
        </div>
        <div class='topbar-tag'>Comitê de Investimentos</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Cenário considerado: ambiente de juros elevados, incerteza macroeconômica global e necessidade de preservação de capital por meio de diversificação internacional."
)

period_options = {
    "6 meses": "6mo",
    "1 ano": "1y",
    "2 anos": "2y",
    "5 anos": "5y",
}

valid_symbols, invalid_symbols = _get_valid_universe(tuple(ASSET_CATALOG.keys()))
asset_universe = [symbol for symbol in ASSET_CATALOG.keys() if symbol in valid_symbols]

if invalid_symbols:
    st.warning(
        "Alguns ativos foram removidos automaticamente por indisponibilidade no yfinance: "
        + ", ".join(sorted(invalid_symbols.keys()))
    )

st.sidebar.markdown("<div class='sidebar-section-title'>Parâmetros</div>", unsafe_allow_html=True)
st.sidebar.selectbox("Horizonte de análise", list(period_options.keys()), index=2, key="period_label", help="Período analisado para cálculo de retornos e risco")
period_label = st.session_state["period_label"]

preset_label = st.sidebar.selectbox("Predefinição de carteira", list(PRESETS.keys()), index=2)
preset_weights = {
    symbol: weight for symbol, weight in PRESETS[preset_label].items() if symbol in asset_universe
}
preset_assets = list(preset_weights.keys()) if preset_weights else asset_universe[:4]

selected_assets = st.sidebar.multiselect(
    "Universo de ativos",
    options=asset_universe,
    default=preset_assets,
    format_func=lambda x: f"{_clean_title(x)} ({x})",
)

if not selected_assets:
    st.warning("Selecione ao menos um ativo para compor a carteira analisada.")
    st.stop()

st.sidebar.markdown("<div style='font-size: 0.68rem; font-weight: 700; color: rgba(11, 31, 58, 0.60); margin: 0.06rem 0 0.16rem 0;'>Pesos da Carteira</div>", unsafe_allow_html=True)


def _slider_default(symbol: str, fallback: int = 0) -> int:
    if preset_weights:
        return int(preset_weights.get(symbol, fallback))
    return fallback


with st.sidebar.container():
    raw_weights = []
    for symbol in selected_assets:
        slider_default = _slider_default(symbol, int(100 / len(selected_assets)))
        raw_weights.append(
            st.slider(
                f"{_clean_title(symbol)} ({symbol})",
                min_value=0,
                max_value=100,
                value=slider_default,
                step=1,
                key=f"w_{symbol}",
            )
        )

if st.sidebar.button("Resetar pesos", help="Restaura os pesos padrão da predefinição selecionada"):
    for symbol in selected_assets:
        st.session_state[f"w_{symbol}"] = _slider_default(symbol, int(100 / len(selected_assets)))
    st.rerun()

raw_weights = np.array(raw_weights, dtype=float)
if raw_weights.sum() <= 0:
    st.sidebar.error("Erro: carteira sem alocação. Ajuste os pesos para prosseguir.")
    st.error("A carteira deve ter ao menos uma alocação positiva. Defina os pesos dos ativos.")
    st.stop()

if not np.isclose(raw_weights.sum(), 100):
    st.sidebar.warning("Os pesos serão normalizados para 100%.")

weights = _normalize_weights(raw_weights)
weights_by_asset = {selected_assets[idx]: float(weights[idx]) for idx in range(len(selected_assets))}
st.sidebar.markdown(
    f"<div class='sidebar-note'>Soma dos pesos (normalizada): <strong>{weights.sum() * 100:.0f}%</strong></div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div class='sidebar-section-title'>Carteira B (Benchmark)</div>", unsafe_allow_html=True)
benchmark_assets = st.sidebar.multiselect(
    "Ativos da Carteira B",
    options=asset_universe,
    default=[symbol for symbol in BENCHMARK_DEFAULT.keys() if symbol in asset_universe],
    key="benchmark_assets",
    format_func=lambda x: f"{_clean_title(x)} ({x})",
)

if not benchmark_assets:
    st.sidebar.error("Selecione ao menos um ativo para a Carteira B (Benchmark).")
    st.stop()

benchmark_raw = []
for symbol in benchmark_assets:
    benchmark_raw.append(
        st.sidebar.slider(
            f"Referência: {_clean_title(symbol)}",
            min_value=0,
            max_value=100,
            value=int(BENCHMARK_DEFAULT.get(symbol, int(100 / len(benchmark_assets)))),
            step=1,
            key=f"wb_{symbol}",
        )
    )

benchmark_raw = np.array(benchmark_raw, dtype=float)
if benchmark_raw.sum() <= 0:
    st.sidebar.error("A Carteira B (Benchmark) deve ter ao menos uma alocação positiva.")
    st.stop()

benchmark_weights = _normalize_weights(benchmark_raw)

st.sidebar.markdown("<div class='sidebar-section-title'>Teste de Estresse por Classe</div>", unsafe_allow_html=True)
stress_defaults = {
    "Ações Brasil": -30,
    "Ações EUA": -20,
    "Renda Fixa": 5,
    "Dólar": 15,
    "Alternativos": -10,
}
class_shocks = {}
for class_name in ["Ações Brasil", "Ações EUA", "Renda Fixa", "Dólar", "Alternativos"]:
    display_name = f"Choque {class_name} (%)" if class_name != "Renda Fixa" else "Choque Juros (%)"
    class_shocks[class_name] = (
        st.sidebar.slider(display_name, -60, 40, stress_defaults[class_name], key=f"shock_{class_name}")
        / 100
    )

stress_vector = np.array([
    class_shocks[ASSET_CATALOG.get(symbol, {}).get("classe", "Alternativos")] for symbol in selected_assets
])
stress_impact = stress_test(weights, stress_vector)


try:
    all_required_assets = tuple(sorted(set(selected_assets) | set(benchmark_assets)))
    data = _load_close_prices(all_required_assets, period_options[period_label])
    if data.empty:
        raise ValueError("Sem dados após alinhar as séries dos ativos selecionados.")
except Exception as exc:
    st.error(f"Erro ao carregar os dados de mercado: {exc}")
    st.info("Verifique conectividade e disponibilidade dos dados na fonte Yahoo Finance.")
    st.stop()

returns = calculate_returns(data)
returns_a = returns[selected_assets]
returns_b = returns[benchmark_assets]

port_ret_a = portfolio_return(returns_a, weights)
port_ret_b = portfolio_return(returns_b, benchmark_weights)

cum_ret_a = cumulative_return(port_ret_a)
cum_ret_b = cumulative_return(port_ret_b)
dd_a = drawdown(cum_ret_a)
dd_b = drawdown(cum_ret_b)

rc = risk_contribution(returns_a, weights)

last_return = (1 + port_ret_a.iloc[-1]) - 1 if not port_ret_a.empty else 0
acc_return_a = cum_ret_a.iloc[-1] - 1 if not cum_ret_a.empty else 0
acc_return_b = cum_ret_b.iloc[-1] - 1 if not cum_ret_b.empty else 0
vol_a = volatility(port_ret_a)
vol_b = volatility(port_ret_b)
max_dd_a = dd_a.min() if not dd_a.empty else 0
max_dd_b = dd_b.min() if not dd_b.empty else 0

assets_last_return = pd.Series(dtype=float)
for symbol in selected_assets:
    assets_last_return.loc[symbol] = data[symbol].pct_change().dropna().iloc[-1] if len(data[symbol].pct_change().dropna()) else 0

top_asset = assets_last_return.idxmax() if not assets_last_return.empty else "N/A"
top_asset_value = assets_last_return.max() if not assets_last_return.empty else 0
weak_asset = assets_last_return.idxmin() if not assets_last_return.empty else "N/A"
weak_asset_value = assets_last_return.min() if not assets_last_return.empty else 0

narrative_status = "com desempenho positivo recente" if last_return >= 0 else "com desempenho negativo recente"

summary_col1, summary_col2, summary_col3, summary_col4, summary_col5, summary_col6 = st.columns(6)
summary_col1.metric("Volatilidade A", f"{vol_a:.2%}")
summary_col2.metric("Volatilidade B", f"{vol_b:.2%}")
summary_col3.metric("Drawdown máximo A", f"{max_dd_a:.2%}")
summary_col4.metric("Drawdown máximo B", f"{max_dd_b:.2%}")
summary_col5.metric("Impacto no cenário atual", f"{stress_impact:.2%}")
summary_col6.metric("Retorno recente", f"{last_return:.2%}")

st.markdown(
    f"<div class='single-insight'>Comparação de retorno acumulado: <strong>Carteira A {acc_return_a:.2%}</strong> vs <strong>Carteira B {acc_return_b:.2%}</strong></div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div class='single-insight'>Melhor desempenho recente: <strong>{_clean_title(top_asset)} ({top_asset_value:.2%})</strong> | Pior desempenho recente: <strong>{_clean_title(weak_asset)} ({weak_asset_value:.2%})</strong></div>",
    unsafe_allow_html=True,
)

rc_df = pd.DataFrame({"Ativo": selected_assets, "Contribuição": rc})
rc_total = rc_df["Contribuição"].sum()
if rc_total != 0:
    rc_df["Contribuição"] = rc_df["Contribuição"] / rc_total

rc_df = rc_df.sort_values("Contribuição", ascending=False)
rc_df["AtivoLabel"] = rc_df["Ativo"].apply(_clean_title)
major_risk_symbol = rc_df.iloc[0]["Ativo"] if not rc_df.empty else "N/A"
major_risk_value = float(rc_df.iloc[0]["Contribuição"]) if not rc_df.empty else 0.0
rc_df["Destaque"] = np.where(rc_df["Ativo"] == major_risk_symbol, "Maior risco", "Demais")

indexed_nav_a = 100 * (cum_ret_a / cum_ret_a.iloc[0]) if not cum_ret_a.empty else cum_ret_a
indexed_nav_b = 100 * (cum_ret_b / cum_ret_b.iloc[0]) if not cum_ret_b.empty else cum_ret_b

comparison_nav = pd.DataFrame(
    {
        "Data": indexed_nav_a.index,
        "Carteira A": indexed_nav_a.values,
        "Carteira B": indexed_nav_b.reindex(indexed_nav_a.index).values,
    }
).melt(id_vars="Data", var_name="Carteira", value_name="Indice")

comparison_dd = pd.DataFrame(
    {
        "Data": dd_a.index,
        "Carteira A": dd_a.values,
        "Carteira B": dd_b.reindex(dd_a.index).values,
    }
).melt(id_vars="Data", var_name="Carteira", value_name="Drawdown")

impact_by_asset = pd.DataFrame(
    {
        "Ativo": selected_assets,
        "Classe": [ASSET_CATALOG.get(symbol, {}).get("classe", "Alternativos") for symbol in selected_assets],
        "Peso": weights,
        "Choque": stress_vector,
    }
)
impact_by_asset["Impacto"] = impact_by_asset["Peso"] * impact_by_asset["Choque"]
impact_by_asset["AtivoLabel"] = impact_by_asset["Ativo"].apply(_clean_title)

impact_by_class = impact_by_asset.groupby("Classe", as_index=False)["Impacto"].sum().sort_values("Impacto")

mean_corr = _mean_abs_correlation(returns_a)
insights = _build_insights(vol_a, max_dd_a, mean_corr, major_risk_symbol, major_risk_value, weights_by_asset)

profile = _portfolio_profile(vol_a, max_dd_a)
equity_weight = sum(
    weights_by_asset[s]
    for s in selected_assets
    if ASSET_CATALOG.get(s, {}).get("classe") in {"Ações Brasil", "Ações EUA"}
)

if equity_weight > 0.60:
    strategic_suggestion = "Carteira com exposição elevada a renda variável. Para perfis mais conservadores, recomenda-se aumentar a alocação em renda fixa."
elif major_risk_value > 0.35:
    strategic_suggestion = f"O risco está concentrado em {_clean_title(major_risk_symbol)}. Avaliar rebalanceamento para reduzir concentração de risco."
else:
    strategic_suggestion = "Estrutura de alocação equilibrada para o período. Manter monitoramento tático de curva de juros e variação cambial."

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown(
        f"<div class='chart-title'>1) Retorno acumulado comparado (base 100)</div>",
        unsafe_allow_html=True,
    )
    nav_chart = (
        alt.Chart(comparison_nav)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("Data:T", title=None),
            y=alt.Y("Indice:Q", title=None),
            color=alt.Color("Carteira:N", scale=alt.Scale(domain=["Carteira A", "Carteira B"], range=["#0b1f3a", "#7588a1"])),
            tooltip=[alt.Tooltip("Data:T"), alt.Tooltip("Carteira:N"), alt.Tooltip("Indice:Q", format=".2f")],
        )
        .properties(height=210)
    )
    st.altair_chart(nav_chart, width="stretch")

with chart_col2:
    st.markdown(
        f"<div class='chart-title'>2) Drawdown histórico comparado</div>",
        unsafe_allow_html=True,
    )
    dd_chart = (
        alt.Chart(comparison_dd)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Data:T", title=None),
            y=alt.Y("Drawdown:Q", axis=alt.Axis(format=".0%"), title=None),
            color=alt.Color("Carteira:N", scale=alt.Scale(domain=["Carteira A", "Carteira B"], range=["#991b1b", "#c8a28d"])),
            tooltip=[alt.Tooltip("Data:T"), alt.Tooltip("Carteira:N"), alt.Tooltip("Drawdown:Q", format=".2%")],
        )
        .properties(height=210)
    )
    st.altair_chart(dd_chart, width="stretch")

detail_col1, detail_col2 = st.columns(2)

with detail_col1:
    st.markdown("<div class='chart-title'>3) Contribuição de risco por ativo</div>", unsafe_allow_html=True)
    rc_chart = (
        alt.Chart(rc_df)
        .mark_bar()
        .encode(
            x=alt.X("Contribuição:Q", axis=alt.Axis(format=".0%", title=None)),
            y=alt.Y("AtivoLabel:N", sort="-x", title=None),
            color=alt.Color("Destaque:N", scale=alt.Scale(domain=["Maior risco", "Demais"], range=["#8b1e1e", "#0b1f3a"]), legend=None),
            tooltip=[
                alt.Tooltip("AtivoLabel:N", title="Ativo"),
                alt.Tooltip("Contribuição:Q", title="Contribuição (%)", format=".2%"),
            ],
        )
        .properties(height=240)
    )
    st.altair_chart(rc_chart, width="stretch")
    st.markdown(
        f"<div class='single-insight'>A estrutura de risco está concentrada em <strong>{_clean_title(major_risk_symbol)}</strong> ({major_risk_value:.2%}) apesar da alocação distribuída.</div>",
        unsafe_allow_html=True,
    )

with detail_col2:
    st.markdown("<div class='chart-title'>4) Teste de estresse por classe de ativos</div>", unsafe_allow_html=True)
    class_chart = (
        alt.Chart(impact_by_class)
        .mark_bar(color="#3e5c76")
        .encode(
            x=alt.X("Impacto:Q", axis=alt.Axis(format=".1%", title=None)),
            y=alt.Y("Classe:N", sort="x", title=None),
            tooltip=[alt.Tooltip("Classe:N"), alt.Tooltip("Impacto:Q", format=".2%")],
        )
        .properties(height=240)
    )
    st.altair_chart(class_chart, width="stretch")
    st.dataframe(
        impact_by_asset[["AtivoLabel", "Classe", "Impacto"]]
        .sort_values("Impacto")
        .rename(columns={"AtivoLabel": "Ativo", "Impacto": "Impacto no cenário"}),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("<div class='section-title'>Diagnóstico Automático</div>", unsafe_allow_html=True)
for message in insights:
    st.markdown(f"<div class='single-insight'>{message}</div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Parecer Estratégico</div>", unsafe_allow_html=True)
st.markdown(
    (
        f"<div class='single-insight'><strong>Classificação de risco:</strong> {profile}. "
        f"<strong>Fatores principais:</strong> concentração de risco em {_clean_title(major_risk_symbol)}, "
        f"drawdown máximo de {max_dd_a:.2%} e correlação média de {mean_corr:.2f}. "
        f"<strong>Recomendação:</strong> {strategic_suggestion}</div>"
    ),
    unsafe_allow_html=True,
)
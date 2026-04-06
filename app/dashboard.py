import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Garantir imports absolutos mesmo quando o Streamlit e executado de outro diretorio.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.market_data import get_price_dataframe
from core.portfolio import calculate_returns, portfolio_return, cumulative_return
from core.risk import volatility, drawdown, risk_contribution
from core.scenarios import stress_test


def load_local_css(css_file: Path):
    st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


st.set_page_config(page_title="Analise de Portfolio", layout="wide")
load_local_css(Path(__file__).with_name("styles.css"))

st.markdown(
    """
    <div class='hero'>
        <div class='hero-copy'>
            <div class='eyebrow'>Risk First / Portfolio Intelligence</div>
            <h1 class='app-title'>Analise de Portfolio</h1>
            <div class='app-subtitle'>Os filtros modificam os graficos instantaneamente.</div>
        </div>
        <div class='hero-chip'>Leitura objetiva, risco em primeiro plano</div>
    </div>
    """,
    unsafe_allow_html=True,
)

period_options = {
    "6 meses": "6mo",
    "1 ano": "1y",
    "2 anos": "2y",
    "5 anos": "5y",
}

default_weights = {
    "AAPL": 30,
    "MSFT": 30,
    "SPY": 40,
}

asset_universe = ["AAPL", "MSFT", "SPY", "QQQ", "TLT", "BND", "GLD"]

st.sidebar.markdown("<div class='sidebar-section-title'>Painel de Controle</div>", unsafe_allow_html=True)
st.sidebar.selectbox("Janela historica", list(period_options.keys()), index=2, key="period_label")
period_label = st.session_state["period_label"]
selected_assets = st.sidebar.multiselect(
    "Universo de ativos",
    options=asset_universe,
    default=["AAPL", "MSFT", "SPY"],
)

if not selected_assets:
    st.warning("Selecione ao menos um ativo na barra lateral.")
    st.stop()

st.sidebar.markdown("<div class='sidebar-section-title'>Alocacao</div>", unsafe_allow_html=True)
with st.sidebar.container():
    raw_weights = []
    for symbol in selected_assets:
        slider_default = default_weights.get(symbol, int(100 / len(selected_assets)))
        raw_weights.append(
            st.slider(
                f"{symbol}",
                min_value=0,
                max_value=100,
                value=slider_default,
                step=1,
                key=f"w_{symbol}",
            )
        )

raw_weights = np.array(raw_weights, dtype=float)
if raw_weights.sum() <= 0:
    st.warning("A soma dos pesos deve ser maior que 0%.")
    st.stop()

weights = raw_weights / raw_weights.sum()
st.sidebar.markdown(
    f"<div class='sidebar-note'>Soma normalizada dos pesos: <strong>{weights.sum() * 100:.0f}%</strong></div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div class='sidebar-section-title'>Cenario de Estresse</div>", unsafe_allow_html=True)
global_shock = st.sidebar.slider("Choque global (%)", -50, 20, -15)

shock_vector = np.full(len(selected_assets), global_shock / 100)


@st.cache_data(show_spinner=False)
def load_close_prices(symbols, period):
    frame = pd.DataFrame()
    for symbol in symbols:
        frame[symbol] = get_price_dataframe(symbol, period=period)["close"]

    frame = frame.dropna(how="all")
    return frame.dropna()


try:
    data = load_close_prices(tuple(selected_assets), period_options[period_label])
    if data.empty:
        raise ValueError("Sem dados apos alinhar as series dos ativos selecionados.")
except Exception as exc:
    st.error(f"Falha ao carregar dados de mercado: {exc}")
    st.info("Verifique conexao de rede e disponibilidade dos dados no Yahoo Finance.")
    st.stop()

returns = calculate_returns(data)
port_ret = portfolio_return(returns, weights)
cum_ret = cumulative_return(port_ret)
dd = drawdown(cum_ret)
rc = risk_contribution(returns, weights)
stress_impact = stress_test(weights, shock_vector)
last_return = (1 + port_ret.iloc[-1]) - 1 if not port_ret.empty else 0

best_asset = rc_df = None

assets_last_return = pd.Series(dtype=float)
for symbol in selected_assets:
    assets_last_return.loc[symbol] = data[symbol].pct_change().dropna().iloc[-1] if len(data[symbol].pct_change().dropna()) else 0

top_asset = assets_last_return.idxmax() if not assets_last_return.empty else "N/A"
top_asset_value = assets_last_return.max() if not assets_last_return.empty else 0
weak_asset = assets_last_return.idxmin() if not assets_last_return.empty else "N/A"
weak_asset_value = assets_last_return.min() if not assets_last_return.empty else 0

st.markdown("<div class='section-title'>Resumo executivo</div>", unsafe_allow_html=True)
summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
summary_col1.metric("Volatilidade anualizada", f"{volatility(port_ret):.2%}")
summary_col2.metric("Drawdown maximo", f"{dd.min():.2%}")
summary_col3.metric("Impacto estresse", f"{stress_impact:.2%}")
summary_col4.metric("Ultimo retorno", f"{last_return:.2%}")

st.markdown("<div class='section-title'>Leitura rapida</div>", unsafe_allow_html=True)
insight_col1, insight_col2 = st.columns(2)
with insight_col1:
    st.markdown(
        f"""
        <div class='insight-card'>
            <div class='insight-label'>Melhor ativo recente</div>
            <div class='insight-value'>{top_asset}</div>
            <div class='insight-meta'>{top_asset_value:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with insight_col2:
    st.markdown(
        f"""
        <div class='insight-card'>
            <div class='insight-label'>Mais fraco recente</div>
            <div class='insight-value'>{weak_asset}</div>
            <div class='insight-meta'>{weak_asset_value:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

rc_df = pd.DataFrame({"Ativo": selected_assets, "Contribuicao": rc})
rc_total = rc_df["Contribuicao"].sum()
if rc_total != 0:
    rc_df["Contribuicao"] = rc_df["Contribuicao"] / rc_total

st.markdown("<div class='section-title'>Painel de leitura</div>", unsafe_allow_html=True)
chart_col1, chart_col2, chart_col3 = st.columns(3)

with chart_col1:
    st.markdown("<div class='chart-card'><div class='chart-title'>Retorno acumulado</div></div>", unsafe_allow_html=True)
    st.line_chart(cum_ret, height=240, width="stretch")

with chart_col2:
    st.markdown("<div class='chart-card'><div class='chart-title'>Drawdown</div></div>", unsafe_allow_html=True)
    st.line_chart(dd, height=240, width="stretch")

with chart_col3:
    st.markdown("<div class='chart-card'><div class='chart-title'>Contribuicao de risco</div></div>", unsafe_allow_html=True)
    st.bar_chart(rc_df.set_index("Ativo"), height=240, width="stretch")
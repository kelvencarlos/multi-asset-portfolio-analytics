import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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
    <div class='topbar'>
        <div>
            <div class='eyebrow'>Plataforma de Monitoramento de Carteiras</div>
            <h1 class='app-title'>Painel de Risco e Performance</h1>
            <div class='app-subtitle'>Atualizacao imediata de indicadores conforme parametros de analise.</div>
        </div>
        <div class='topbar-tag'>Comite de Investimentos</div>
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

st.sidebar.markdown("<div class='sidebar-section-title'>Parametros</div>", unsafe_allow_html=True)
st.sidebar.selectbox("Horizonte de analise", list(period_options.keys()), index=2, key="period_label")
period_label = st.session_state["period_label"]
selected_assets = st.sidebar.multiselect(
    "Universo de ativos",
    options=asset_universe,
    default=["AAPL", "MSFT", "SPY"],
)

if not selected_assets:
    st.warning("Selecione ao menos um ativo para compor a carteira analisada.")
    st.stop()

st.sidebar.markdown("<div class='sidebar-section-title'>Alocacao Estrategica</div>", unsafe_allow_html=True)
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
    st.warning("A soma dos pesos deve ser superior a 0%.")
    st.stop()

weights = raw_weights / raw_weights.sum()
st.sidebar.markdown(
    f"<div class='sidebar-note'>Soma dos pesos (normalizada): <strong>{weights.sum() * 100:.0f}%</strong></div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div class='sidebar-section-title'>Stress Test</div>", unsafe_allow_html=True)
global_shock = st.sidebar.slider("Choque de mercado (%)", -50, 20, -15)

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
    st.error(f"Nao foi possivel carregar os dados de mercado: {exc}")
    st.info("Verifique conectividade e disponibilidade dos dados na fonte Yahoo Finance.")
    st.stop()

returns = calculate_returns(data)
port_ret = portfolio_return(returns, weights)
cum_ret = cumulative_return(port_ret)
dd = drawdown(cum_ret)
rc = risk_contribution(returns, weights)
stress_impact = stress_test(weights, shock_vector)
last_return = (1 + port_ret.iloc[-1]) - 1 if not port_ret.empty else 0

assets_last_return = pd.Series(dtype=float)
for symbol in selected_assets:
    assets_last_return.loc[symbol] = data[symbol].pct_change().dropna().iloc[-1] if len(data[symbol].pct_change().dropna()) else 0

top_asset = assets_last_return.idxmax() if not assets_last_return.empty else "N/A"
top_asset_value = assets_last_return.max() if not assets_last_return.empty else 0
weak_asset = assets_last_return.idxmin() if not assets_last_return.empty else "N/A"
weak_asset_value = assets_last_return.min() if not assets_last_return.empty else 0

max_dd = dd.min() if not dd.empty else 0
narrative_status = "com desempenho positivo recente" if last_return >= 0 else "com desempenho negativo recente"

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
summary_col1.metric("Volatilidade anualizada", f"{volatility(port_ret):.2%}")
summary_col2.metric("Drawdown maximo observado", f"{dd.min():.2%}")
summary_col3.metric("Impacto no stress test", f"{stress_impact:.2%}")
summary_col4.metric("Retorno do ultimo periodo", f"{last_return:.2%}")

st.markdown(
    f"<div class='single-insight'>Maior contribuicao positiva recente: <strong>{top_asset} ({top_asset_value:.2%})</strong> | Maior pressao negativa recente: <strong>{weak_asset} ({weak_asset_value:.2%})</strong></div>",
    unsafe_allow_html=True,
)

rc_df = pd.DataFrame({"Ativo": selected_assets, "Contribuicao": rc})
rc_total = rc_df["Contribuicao"].sum()
if rc_total != 0:
    rc_df["Contribuicao"] = rc_df["Contribuicao"] / rc_total

rc_df = rc_df.sort_values("Contribuicao", ascending=False)

indexed_nav = 100 * (cum_ret / cum_ret.iloc[0]) if not cum_ret.empty else cum_ret

chart_col1, chart_col2, chart_col3 = st.columns(3)

with chart_col1:
    st.markdown(
        f"<div class='chart-title'>1) Evolucao do patrimonio (base 100): nivel atual {indexed_nav.iloc[-1]:.1f}, carteira {narrative_status}</div>",
        unsafe_allow_html=True,
    )
    st.line_chart(indexed_nav, height=195, width="stretch")

with chart_col2:
    st.markdown(
        f"<div class='chart-title'>2) Profundidade de queda: drawdown maximo de {max_dd:.2%}</div>",
        unsafe_allow_html=True,
    )
    st.area_chart(dd, height=195, width="stretch")

with chart_col3:
    st.markdown("<div class='chart-title'>3) Distribuicao de risco por ativo (ordenada por relevancia)</div>", unsafe_allow_html=True)
    rc_chart = (
        alt.Chart(rc_df)
        .mark_bar(color="#0b1f3a")
        .encode(
            x=alt.X("Contribuicao:Q", axis=alt.Axis(format=".0%", title=None)),
            y=alt.Y("Ativo:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Ativo:N", title="Ativo"),
                alt.Tooltip("Contribuicao:Q", title="Contribuicao", format=".2%"),
            ],
        )
        .properties(height=195)
    )
    st.altair_chart(rc_chart, width="stretch")
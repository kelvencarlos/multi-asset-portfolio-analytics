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
    "<div class='app-title'>Analise de Portfolio</div>",
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

st.sidebar.header("Filtros")
period_label = st.sidebar.selectbox("Janela historica", list(period_options.keys()), index=2)
selected_assets = st.sidebar.multiselect(
    "Ativos",
    options=asset_universe,
    default=["AAPL", "MSFT", "SPY"],
)

if not selected_assets:
    st.warning("Selecione ao menos um ativo na barra lateral.")
    st.stop()

with st.sidebar.expander("Pesos (%)", expanded=True):
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
st.sidebar.caption(f"Soma normalizada dos pesos: {weights.sum() * 100:.0f}%")

with st.sidebar.expander("Cenario de Estresse", expanded=False):
    global_shock = st.slider("Choque global (%)", -50, 20, -15)

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

col1, col2, col3 = st.columns(3)
col1.metric("Volatilidade anualizada", f"{volatility(port_ret):.2%}")
col2.metric("Drawdown maximo", f"{dd.min():.2%}")
col3.metric("Impacto estresse", f"{stress_impact:.2%}")

rc_df = pd.DataFrame({"Ativo": selected_assets, "Contribuicao": rc})
rc_total = rc_df["Contribuicao"].sum()
if rc_total != 0:
    rc_df["Contribuicao"] = rc_df["Contribuicao"] / rc_total

tab_return, tab_drawdown, tab_risk = st.tabs(["Retorno", "Drawdown", "Risco"])

with tab_return:
    st.line_chart(cum_ret, height=300, use_container_width=True)

with tab_drawdown:
    st.line_chart(dd, height=300, use_container_width=True)

with tab_risk:
    st.bar_chart(rc_df.set_index("Ativo"), height=300, use_container_width=True)
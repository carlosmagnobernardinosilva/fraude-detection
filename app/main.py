"""
Fraud Detection BI — Página Inicial
"""
import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent

st.set_page_config(
    page_title="Fraud Detection BI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Remove default top padding */
    .block-container { padding-top: 2rem; }

    /* Navigation card */
    .nav-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2D7DD2 100%);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(45, 125, 210, 0.3);
        transition: transform 0.2s;
    }
    .nav-card:hover { transform: translateY(-4px); }
    .nav-card h2 { color: white; margin: 0.5rem 0; font-size: 1.4rem; }
    .nav-card p  { color: rgba(255,255,255,0.85); font-size: 0.9rem; margin: 0; }
    .nav-card .icon { font-size: 2.5rem; margin-bottom: 0.5rem; }

    /* KPI banner */
    .kpi-banner {
        background: #F0F4F9;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 2rem;
        border-left: 4px solid #2D7DD2;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    div[data-testid="metric-container"] label { color: #64748B; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:1rem; margin-bottom:2rem;">
    <span style="font-size:3rem;">🛡️</span>
    <div>
        <h1 style="margin:0; color:#1A1A2E; font-size:2rem;">Fraud Detection BI</h1>
        <p style="margin:0; color:#64748B; font-size:1rem;">
            Plataforma de análise e monitoramento de detecção de fraudes em transações
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Carregamento de estatísticas globais ──────────────────────────────────────
_SILVER_PATH = ROOT / "data" / "silver" / "train_silver.parquet"

_PIPELINE_INSTRUCTIONS = """
Execute o pipeline de preparação antes de usar o app:

```python
import pandas as pd
from src.context.pipeline_context import PipelineContext
from src.orchestrator.preparation_orchestrator import PreparationOrchestrator

ctx = PipelineContext(
    df_train    = pd.read_csv("data/raw/train.csv"),
    df_test     = pd.read_csv("data/raw/test.csv"),
    df_customer = pd.read_csv("data/raw/customer.csv"),
    df_terminal = pd.read_csv("data/raw/terminal.csv"),
)
PreparationOrchestrator().run(ctx)
```
"""


@st.cache_data
def load_summary():
    if not _SILVER_PATH.exists():
        return None
    return pd.read_parquet(
        _SILVER_PATH,
        columns=["TX_DATETIME", "TX_AMOUNT", "TX_FRAUD", "CUSTOMER_ID", "TERMINAL_ID"],
    )


with st.spinner("Carregando dados..."):
    df = load_summary()

if df is None:
    st.warning("Dados Silver não encontrados. O pipeline de preparação ainda não foi executado.")
    st.markdown(_PIPELINE_INSTRUCTIONS)
    st.stop()

n_tx      = len(df)
n_fraud   = df["TX_FRAUD"].sum()
n_legit   = n_tx - n_fraud
fraud_pct = n_fraud / n_tx
avg_amount = df["TX_AMOUNT"].mean()
periodo   = f"{df['TX_DATETIME'].min().strftime('%b %Y')} – {df['TX_DATETIME'].max().strftime('%b %Y')}"
n_clientes  = df["CUSTOMER_ID"].nunique()
n_terminais = df["TERMINAL_ID"].nunique()

# ── KPIs Globais ──────────────────────────────────────────────────────────────
st.markdown("### 📊 Visão Geral do Dataset")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Transações",       f"{n_tx:,}")
c2.metric("Fraudes",          f"{n_fraud:,}", f"{fraud_pct:.2%} do total")
c3.metric("Legítimas",        f"{n_legit:,}")
c4.metric("Ticket Médio",     f"R$ {avg_amount:.2f}")
c5.metric("Clientes Únicos",  f"{n_clientes:,}")
c6.metric("Terminais Únicos", f"{n_terminais:,}")

st.markdown(f"""
<div class="kpi-banner">
    📅 &nbsp;<strong>Período:</strong> {periodo} &nbsp;|&nbsp;
    ⚠️ &nbsp;<strong>Taxa de fraude:</strong> {fraud_pct:.3%} &nbsp;|&nbsp;
    💰 &nbsp;<strong>Montante total em fraudes:</strong>
    R$ {df.loc[df['TX_FRAUD']==1,'TX_AMOUNT'].sum():,.2f}
</div>
""", unsafe_allow_html=True)

# ── Navegação ─────────────────────────────────────────────────────────────────
st.markdown("### 🗂️ Painéis Disponíveis")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="nav-card">
        <div class="icon">🔍</div>
        <h2>Análise Exploratória (EDA)</h2>
        <p>
            Distribuição de transações, padrões temporais, perfil de valor,
            análise de clientes e terminais de alto risco.
        </p>
        <br/>
        <p><strong>→ Use o menu lateral para navegar</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="icon">💼</div>
        <h2>Análise de Negócio</h2>
        <p>
            Impacto financeiro do modelo, tradeoff entre fraudes detectadas e
            transações negadas, análise de sensibilidade e sumário executivo.
        </p>
        <br/>
        <p><strong>→ Use o menu lateral para navegar</strong></p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("🤖 Fraud Detection BI · Dados simulados · Modelo XGBoost + Optuna")

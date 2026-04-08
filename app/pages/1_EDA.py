"""
EDA — Análise Exploratória de Dados
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

st.set_page_config(
    page_title="EDA — Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="metric-container"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    div[data-testid="metric-container"] label { color: #64748B; font-size: 0.78rem; }
    .section-title {
        font-size: 1rem; font-weight: 600; color: #1E3A5F;
        border-left: 3px solid #2D7DD2; padding-left: 0.6rem;
        margin: 1.2rem 0 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Constantes visuais ────────────────────────────────────────────────────────
C_FRAUD  = "#E63946"
C_LEGIT  = "#2D7DD2"
C_WARN   = "#F4A261"
C_OK     = "#06D6A0"
TEMPLATE = "plotly_white"

DAY_LABELS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
MONTH_LABELS = {8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}


def style(fig, height=380):
    fig.update_layout(
        template=TEMPLATE, height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        font=dict(family="Arial, sans-serif", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#F0F0F0")
    return fig


# ── Dados ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = ROOT / "data" / "silver" / "train_silver.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["TX_HOUR"]       = df["TX_DATETIME"].dt.hour
    df["TX_DOW"]        = df["TX_DATETIME"].dt.dayofweek
    df["TX_DOW_LABEL"]  = df["TX_DOW"].map(dict(enumerate(DAY_LABELS)))
    df["TX_MONTH"]      = df["TX_DATETIME"].dt.month
    df["TX_MONTH_LABEL"]= df["TX_MONTH"].map(MONTH_LABELS)
    df["TX_DATE"]       = df["TX_DATETIME"].dt.date
    df["SAFRA"]         = df["TX_DATETIME"].dt.to_period("M").astype(str)
    df["FRAUD_LABEL"]   = df["TX_FRAUD"].map({0: "Legítima", 1: "Fraude"})
    return df


df_full = load_data()

if df_full is None:
    st.warning("Dados Silver não encontrados. Execute o pipeline de preparação antes de usar o app.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 EDA — Filtros")
    st.divider()

    safras_disp = sorted(df_full["SAFRA"].unique())
    safras_sel  = st.multiselect("📅 Safra (mês)", safras_disp, default=safras_disp,
                                  help="Filtra o mês de referência das transações")

    tipo_tx = st.radio("Tipo de transação", ["Todas", "Somente Fraudes", "Somente Legítimas"],
                        index=0)

    amount_min, amount_max = float(df_full["TX_AMOUNT"].min()), float(df_full["TX_AMOUNT"].max())
    amount_range = st.slider("💰 Faixa de valor (R$)", amount_min, amount_max,
                              (amount_min, amount_max), step=1.0)
    st.divider()
    st.caption(f"Dataset: {len(df_full):,} transações")

# ── Filtros aplicados ─────────────────────────────────────────────────────────
df = df_full[df_full["SAFRA"].isin(safras_sel)].copy()
df = df[(df["TX_AMOUNT"] >= amount_range[0]) & (df["TX_AMOUNT"] <= amount_range[1])]
if tipo_tx == "Somente Fraudes":
    df = df[df["TX_FRAUD"] == 1]
elif tipo_tx == "Somente Legítimas":
    df = df[df["TX_FRAUD"] == 0]

if df.empty:
    st.warning("Nenhum dado com os filtros selecionados.")
    st.stop()

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.markdown("# 🔍 Análise Exploratória de Dados")
st.caption(f"Transações filtradas: **{len(df):,}** · "
           f"Fraudes: **{df['TX_FRAUD'].sum():,}** ({df['TX_FRAUD'].mean():.3%})")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visão Geral",
    "⏱️ Padrões Temporais",
    "💰 Distribuição de Valores",
    "👥 Clientes & Terminais",
])

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 1 — VISÃO GERAL                                ║
# ╚══════════════════════════════════════════════════════╝
with tab1:
    n_tx    = len(df)
    n_fraud = int(df["TX_FRAUD"].sum())
    n_legit = n_tx - n_fraud

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total de Transações", f"{n_tx:,}")
    k2.metric("Fraudes",  f"{n_fraud:,}",  f"{n_fraud/n_tx:.3%}")
    k3.metric("Legítimas", f"{n_legit:,}", f"{n_legit/n_tx:.3%}")
    k4.metric("Ticket Médio (Fraude)",
              f"R$ {df.loc[df['TX_FRAUD']==1,'TX_AMOUNT'].mean():.2f}")
    k5.metric("Ticket Médio (Legítima)",
              f"R$ {df.loc[df['TX_FRAUD']==0,'TX_AMOUNT'].mean():.2f}")

    st.markdown('<div class="section-title">Composição do Dataset</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2])

    # Donut
    with col_a:
        fig_donut = go.Figure(go.Pie(
            labels=["Legítimas", "Fraudes"],
            values=[n_legit, n_fraud],
            hole=0.65,
            marker_colors=[C_LEGIT, C_FRAUD],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,}<extra></extra>",
        ))
        fig_donut.add_annotation(
            text=f"<b>{n_fraud/n_tx:.2%}</b><br>fraudes",
            x=0.5, y=0.5, font_size=16, showarrow=False,
        )
        fig_donut.update_layout(
            template=TEMPLATE, height=320,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True,
            legend=dict(orientation="h", x=0.15, y=-0.1),
            paper_bgcolor="rgba(0,0,0,0)",
            title=dict(text="Distribuição das Transações", font_size=13),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Boxplot amount por tipo
    with col_b:
        fig_box = px.box(
            df, x="FRAUD_LABEL", y="TX_AMOUNT",
            color="FRAUD_LABEL",
            color_discrete_map={"Fraude": C_FRAUD, "Legítima": C_LEGIT},
            points="outliers",
            labels={"FRAUD_LABEL": "", "TX_AMOUNT": "Valor (R$)"},
            title="Distribuição de Valor por Tipo de Transação",
        )
        st.plotly_chart(style(fig_box, 320), use_container_width=True)

    st.markdown('<div class="section-title">Volume por Safra</div>',
                unsafe_allow_html=True)

    # Barras empilhadas por safra
    safra_agg = (
        df.groupby(["SAFRA", "FRAUD_LABEL"])
        .size().reset_index(name="n")
    )
    fig_safra = px.bar(
        safra_agg, x="SAFRA", y="n", color="FRAUD_LABEL",
        color_discrete_map={"Fraude": C_FRAUD, "Legítima": C_LEGIT},
        labels={"SAFRA": "Safra", "n": "Transações", "FRAUD_LABEL": ""},
        title="Volume de Transações por Safra",
        barmode="stack",
    )
    st.plotly_chart(style(fig_safra, 320), use_container_width=True)


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 2 — PADRÕES TEMPORAIS                          ║
# ╚══════════════════════════════════════════════════════╝
with tab2:

    # ── Taxa de fraude: hora × dia da semana (heatmap) ──
    st.markdown('<div class="section-title">Taxa de Fraude por Hora e Dia da Semana</div>',
                unsafe_allow_html=True)

    heatmap_df = (
        df.groupby(["TX_DOW", "TX_HOUR"])["TX_FRAUD"]
        .mean()
        .reset_index()
        .pivot(index="TX_DOW", columns="TX_HOUR", values="TX_FRAUD")
    )
    heatmap_df.index = DAY_LABELS

    fig_heat = go.Figure(go.Heatmap(
        z=heatmap_df.values * 100,
        x=[f"{h:02d}h" for h in heatmap_df.columns],
        y=heatmap_df.index,
        colorscale=[[0, "#EBF4FF"], [0.5, C_WARN], [1, C_FRAUD]],
        hoverongaps=False,
        hovertemplate="Dia: %{y}<br>Hora: %{x}<br>Taxa fraude: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="% Fraude", ticksuffix="%"),
    ))
    fig_heat.update_layout(
        template=TEMPLATE, height=320,
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis_title="Hora do Dia",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Taxa de Fraude (%) — Hora × Dia", font_size=13),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    col_l, col_r = st.columns(2)

    # ── Volume por hora ──
    with col_l:
        st.markdown('<div class="section-title">Volume por Hora do Dia</div>',
                    unsafe_allow_html=True)
        hour_agg = (
            df.groupby(["TX_HOUR", "FRAUD_LABEL"])
            .size().reset_index(name="n")
        )
        fig_hour = px.bar(
            hour_agg, x="TX_HOUR", y="n", color="FRAUD_LABEL",
            color_discrete_map={"Fraude": C_FRAUD, "Legítima": C_LEGIT},
            labels={"TX_HOUR": "Hora", "n": "Transações", "FRAUD_LABEL": ""},
            barmode="group",
        )
        fig_hour.update_xaxes(dtick=2)
        st.plotly_chart(style(fig_hour, 320), use_container_width=True)

    # ── Volume por dia da semana ──
    with col_r:
        st.markdown('<div class="section-title">Volume por Dia da Semana</div>',
                    unsafe_allow_html=True)
        dow_agg = (
            df.groupby(["TX_DOW", "TX_DOW_LABEL", "FRAUD_LABEL"])
            .size().reset_index(name="n")
            .sort_values("TX_DOW")
        )
        fig_dow = px.bar(
            dow_agg, x="TX_DOW_LABEL", y="n", color="FRAUD_LABEL",
            color_discrete_map={"Fraude": C_FRAUD, "Legítima": C_LEGIT},
            labels={"TX_DOW_LABEL": "Dia", "n": "Transações", "FRAUD_LABEL": ""},
            barmode="group",
            category_orders={"TX_DOW_LABEL": DAY_LABELS},
        )
        st.plotly_chart(style(fig_dow, 320), use_container_width=True)

    # ── Série temporal diária ──
    st.markdown('<div class="section-title">Evolução Diária</div>',
                unsafe_allow_html=True)

    ts = (
        df.groupby(["TX_DATE", "TX_FRAUD"])
        .size().reset_index(name="n")
    )
    ts["TX_DATE"] = pd.to_datetime(ts["TX_DATE"])
    ts["Tipo"] = ts["TX_FRAUD"].map({0: "Legítimas", 1: "Fraudes"})

    fig_ts = px.line(
        ts, x="TX_DATE", y="n", color="Tipo",
        color_discrete_map={"Fraudes": C_FRAUD, "Legítimas": C_LEGIT},
        labels={"TX_DATE": "Data", "n": "Transações", "Tipo": ""},
    )
    fig_ts.update_traces(mode="lines")
    st.plotly_chart(style(fig_ts, 320), use_container_width=True)


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 3 — DISTRIBUIÇÃO DE VALORES                    ║
# ╚══════════════════════════════════════════════════════╝
with tab3:
    col_l, col_r = st.columns(2)

    # ── Histograma sobreposto ──
    with col_l:
        st.markdown('<div class="section-title">Distribuição de Valor por Tipo</div>',
                    unsafe_allow_html=True)
        fig_hist = px.histogram(
            df, x="TX_AMOUNT", color="FRAUD_LABEL",
            nbins=60, opacity=0.72, barmode="overlay",
            color_discrete_map={"Fraude": C_FRAUD, "Legítima": C_LEGIT},
            labels={"TX_AMOUNT": "Valor (R$)", "count": "Frequência", "FRAUD_LABEL": ""},
        )
        fig_hist.update_traces(marker_line_width=0)
        st.plotly_chart(style(fig_hist, 360), use_container_width=True)

    # ── Violin ──
    with col_r:
        st.markdown('<div class="section-title">Violin Plot — Forma da Distribuição</div>',
                    unsafe_allow_html=True)
        fig_violin = px.violin(
            df, x="FRAUD_LABEL", y="TX_AMOUNT",
            color="FRAUD_LABEL", box=True, points=False,
            color_discrete_map={"Fraude": C_FRAUD, "Legítima": C_LEGIT},
            labels={"FRAUD_LABEL": "", "TX_AMOUNT": "Valor (R$)"},
        )
        st.plotly_chart(style(fig_violin, 360), use_container_width=True)

    # ── Faixas de valor ──
    st.markdown('<div class="section-title">Taxa de Fraude por Faixa de Valor</div>',
                unsafe_allow_html=True)

    bins   = [0, 10, 25, 50, 75, 100, 150, 200, 300]
    labels = ["R$0-10", "R$10-25", "R$25-50", "R$50-75",
              "R$75-100", "R$100-150", "R$150-200", "R$200+"]
    df["FAIXA"] = pd.cut(df["TX_AMOUNT"], bins=bins, labels=labels, right=True)

    faixa_agg = (
        df.groupby("FAIXA", observed=True)
        .agg(total=("TX_FRAUD", "count"), fraudes=("TX_FRAUD", "sum"))
        .assign(taxa=lambda x: x["fraudes"] / x["total"] * 100)
        .reset_index()
    )

    fig_faixa = make_subplots(specs=[[{"secondary_y": True}]])
    fig_faixa.add_trace(
        go.Bar(x=faixa_agg["FAIXA"].astype(str), y=faixa_agg["total"],
               name="Total", marker_color="#BFDBFE", opacity=0.8),
        secondary_y=False,
    )
    fig_faixa.add_trace(
        go.Scatter(x=faixa_agg["FAIXA"].astype(str), y=faixa_agg["taxa"],
                   name="Taxa Fraude (%)", mode="lines+markers",
                   line=dict(color=C_FRAUD, width=2.5),
                   marker=dict(size=8)),
        secondary_y=True,
    )
    fig_faixa.update_layout(
        template=TEMPLATE, height=360,
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(orientation="h", y=1.08),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_faixa.update_yaxes(title_text="Nº de Transações", secondary_y=False, gridcolor="#F0F0F0")
    fig_faixa.update_yaxes(title_text="Taxa de Fraude (%)", secondary_y=True,
                            ticksuffix="%", gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_faixa, use_container_width=True)

    # ── Estatísticas descritivas ──
    st.markdown('<div class="section-title">Estatísticas Descritivas</div>',
                unsafe_allow_html=True)
    stats = (
        df.groupby("FRAUD_LABEL")["TX_AMOUNT"]
        .describe()
        .round(2)
        .T
        .rename_axis("Estatística")
        .reset_index()
    )
    stats.columns.name = None
    st.dataframe(stats, use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 4 — CLIENTES & TERMINAIS                       ║
# ╚══════════════════════════════════════════════════════╝
with tab4:
    col_l, col_r = st.columns(2)

    # ── Top terminais por risco ──
    with col_l:
        st.markdown('<div class="section-title">Top 20 Terminais por Taxa de Fraude</div>',
                    unsafe_allow_html=True)
        n_min = st.number_input("Mínimo de transações no terminal", 10, 500, 30, step=10)

        term_agg = (
            df.groupby("TERMINAL_ID")
            .agg(total=("TX_FRAUD", "count"), fraudes=("TX_FRAUD", "sum"))
            .query(f"total >= {n_min}")
            .assign(taxa=lambda x: x["fraudes"] / x["total"] * 100)
            .nlargest(20, "taxa")
            .reset_index()
        )

        fig_term = px.bar(
            term_agg.sort_values("taxa"),
            x="taxa", y="TERMINAL_ID",
            orientation="h",
            color="taxa",
            color_continuous_scale=[[0, "#EBF4FF"], [0.5, C_WARN], [1, C_FRAUD]],
            labels={"taxa": "Taxa Fraude (%)", "TERMINAL_ID": "Terminal ID"},
            hover_data={"total": True, "fraudes": True},
        )
        fig_term.update_coloraxes(showscale=False)
        fig_term.update_layout(yaxis=dict(type="category"))
        st.plotly_chart(style(fig_term, 460), use_container_width=True)

    # ── Perfil do cliente ──
    with col_r:
        st.markdown('<div class="section-title">Perfil de Clientes — Montante Médio vs Volume</div>',
                    unsafe_allow_html=True)

        cust_agg = (
            df.groupby("CUSTOMER_ID")
            .agg(
                total        =("TX_FRAUD", "count"),
                fraudes      =("TX_FRAUD", "sum"),
                mean_amount  =("TX_AMOUNT", "mean"),
                nb_tx_per_day=("mean_nb_tx_per_day", "first"),
            )
            .assign(taxa=lambda x: x["fraudes"] / x["total"])
            .query("total >= 5")
            .reset_index()
        )
        cust_agg = cust_agg.sample(min(2000, len(cust_agg)), random_state=42)

        fig_scat = px.scatter(
            cust_agg,
            x="mean_amount", y="total",
            color="taxa",
            color_continuous_scale=[[0, C_LEGIT], [0.5, C_WARN], [1, C_FRAUD]],
            opacity=0.6, size_max=8,
            labels={"mean_amount": "Ticket Médio (R$)", "total": "Nº de Transações",
                    "taxa": "Taxa Fraude"},
            hover_data={"CUSTOMER_ID": True, "fraudes": True},
        )
        fig_scat.update_coloraxes(colorbar_tickformat=".1%")
        st.plotly_chart(style(fig_scat, 460), use_container_width=True)

    # ── Mapa geográfico (coordenadas simuladas) ──
    st.markdown('<div class="section-title">Distribuição Geográfica dos Terminais</div>',
                unsafe_allow_html=True)

    geo_term = (
        df.groupby("TERMINAL_ID")
        .agg(
            x=("x_terminal_id", "first"),
            y=("y_terminal_id", "first"),
            total=("TX_FRAUD", "count"),
            fraudes=("TX_FRAUD", "sum"),
        )
        .assign(taxa=lambda d: d["fraudes"] / d["total"] * 100)
        .reset_index()
    )

    fig_geo = px.scatter(
        geo_term,
        x="x", y="y",
        color="taxa",
        size="total",
        size_max=18,
        opacity=0.75,
        color_continuous_scale=[[0, "#DBEAFE"], [0.5, C_WARN], [1, C_FRAUD]],
        labels={"x": "Coordenada X", "y": "Coordenada Y", "taxa": "Taxa Fraude (%)"},
        hover_data={"TERMINAL_ID": True, "total": True, "fraudes": True, "taxa": ":.2f"},
        title="Mapa de Risco dos Terminais (coordenadas simuladas)",
    )
    fig_geo.update_coloraxes(colorbar_ticksuffix="%")
    st.plotly_chart(style(fig_geo, 420), use_container_width=True)

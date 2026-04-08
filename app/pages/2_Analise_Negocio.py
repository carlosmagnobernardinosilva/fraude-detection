"""
Análise de Negócio — BI Financeiro
"""
import re
import glob
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent

st.set_page_config(
    page_title="Análise de Negócio — Fraud Detection",
    page_icon="💼",
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
        padding: 0.9rem 1.2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    div[data-testid="metric-container"] label { color: #64748B; font-size: 0.78rem; }
    .section-title {
        font-size: 1rem; font-weight: 600; color: #1E3A5F;
        border-left: 3px solid #2D7DD2; padding-left: 0.6rem;
        margin: 1.2rem 0 0.8rem 0;
    }
    .kpi-positive { color: #06D6A0; font-weight: 700; }
    .kpi-negative { color: #E63946; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
C_FRAUD    = "#E63946"
C_LEGIT    = "#2D7DD2"
C_POSITIVE = "#06D6A0"
C_WARN     = "#F4A261"
C_NEUTRAL  = "#8892B0"
TEMPLATE   = "plotly_white"
RANDOM_STATE = 42


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
    fig.update_yaxes(gridcolor="#F0F0F0")
    fig.update_xaxes(showgrid=False)
    return fig


# ── Carregamento de dados e modelo ────────────────────────────────────────────
@st.cache_resource
def load_model():
    pkls = sorted(glob.glob(str(ROOT / "data" / "models" / "xgboost_optuna*.pkl")))
    if not pkls:
        return None, None
    path = pkls[-1]
    m = re.search(r"thr([\d._]+)_[0-9a-f]{8}", Path(path).name)
    thr = float(m.group(1).rstrip("_").replace("_", ".")) if m else 0.30
    return joblib.load(path), thr


@st.cache_data
def load_val_data():
    x_path = ROOT / "data" / "gold" / "X_train.parquet"
    if not x_path.exists():
        return None, None, None, None
    X = pd.read_parquet(x_path)
    y = pd.read_parquet(ROOT / "data" / "gold" / "y_train.parquet").squeeze()
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    gold   = pd.read_parquet(ROOT / "data" / "gold" / "train_gold.parquet",
                              columns=["TX_AMOUNT"])
    silver = pd.read_parquet(ROOT / "data" / "silver" / "train_silver.parquet",
                              columns=["TX_DATETIME"])
    amount = gold.loc[X_val.index, "TX_AMOUNT"].values
    safra  = silver.loc[X_val.index, "TX_DATETIME"].dt.to_period("M").astype(str).values
    return X_val, y_val, amount, safra


@st.cache_data
def compute_scores(_model, X_val_index):
    X_val, _, _, _ = load_val_data()
    return _model.predict_proba(X_val)[:, 1]


pipeline, DEFAULT_THRESH = load_model()

if pipeline is None:
    st.error("Modelo não encontrado em `data/models/`. Execute o ExperimentLoggerAgent primeiro.")
    st.stop()

X_val, y_val, amount_val, safra_val = load_val_data()

if X_val is None:
    st.error("Dados Gold não encontrados em `data/gold/`. Execute o pipeline de preparação primeiro.")
    st.stop()
y_arr    = y_val.values
prob_val = compute_scores(pipeline, tuple(X_val.index.tolist()))

# ── Sidebar — controles interativos ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💼 Análise de Negócio")
    st.divider()

    st.markdown("#### 🎚️ Threshold do Modelo")
    threshold = st.slider(
        "Threshold de classificação",
        min_value=0.01, max_value=0.99,
        value=float(round(DEFAULT_THRESH, 2)),
        step=0.01,
        help="Ponto de corte para classificar uma transação como fraude",
    )

    st.divider()
    st.markdown("#### 💰 Premissas Financeiras")
    with st.expander("Configurar custos", expanded=False):
        custo_revisao = st.number_input(
            "Custo operacional por FP (R$)",
            min_value=0.0, max_value=200.0, value=5.0, step=1.0,
            help="Custo fixo de revisar/bloquear uma transação legítima"
        )
        taxa_churn = st.slider(
            "Taxa de churn por FP (%)",
            0.0, 20.0, 2.0, 0.5,
            help="% de clientes que cancelam após ter transação negada indevidamente"
        ) / 100
        ltv_cliente = st.number_input(
            "LTV do cliente (R$/ano)",
            min_value=0.0, max_value=5000.0, value=800.0, step=50.0,
        )

    custo_fp = custo_revisao + taxa_churn * ltv_cliente

    st.divider()
    st.caption(f"Modelo: `{Path(sorted(glob.glob(str(ROOT / 'data' / 'models' / 'xgboost_optuna*.pkl')))[-1]).name}`")
    st.caption(f"Threshold padrão (F2): **{DEFAULT_THRESH:.3f}**")

# ── Métricas reativas ao threshold ────────────────────────────────────────────
pred_val  = (prob_val >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_arr, pred_val).ravel()

tp_mask = (pred_val == 1) & (y_arr == 1)
fp_mask = (pred_val == 1) & (y_arr == 0)
fn_mask = (pred_val == 0) & (y_arr == 1)

salvo_tp  = amount_val[tp_mask].sum()
custo_fp_ = fp * custo_fp
resid_fn  = amount_val[fn_mask].sum()
perda_sem = amount_val[y_arr == 1].sum()
benef_liq = salvo_tp - custo_fp_
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
precisao  = tp / (tp + fp) if (tp + fp) > 0 else 0
fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0
auc_roc   = roc_auc_score(y_arr, prob_val)
avg_prec  = average_precision_score(y_arr, prob_val)

# ── Varredura de thresholds (cacheada, não depende do slider) ─────────────────
@st.cache_data
def compute_sweep(prob_bytes):
    prob = np.frombuffer(prob_bytes, dtype=np.float64)
    y    = y_arr
    rows = []
    for thr in np.linspace(0.01, 0.99, 300):
        p = (prob >= thr).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y, p).ravel()
        salvo_  = amount_val[(p == 1) & (y == 1)].sum()
        custo_  = fp_ * custo_fp
        rows.append({
            "threshold"   : thr,
            "tp": tp_, "fp": fp_, "fn": fn_, "tn": tn_,
            "valor_salvo" : salvo_,
            "custo_fp"    : custo_,
            "prejuizo_fn" : amount_val[(p == 0) & (y == 1)].sum(),
            "beneficio_liq": salvo_ - custo_,
            "recall"      : tp_ / (tp_ + fn_) if tp_ + fn_ > 0 else 0,
            "precisao"    : tp_ / (tp_ + fp_) if tp_ + fp_ > 0 else 0,
        })
    return pd.DataFrame(rows)


df_sweep = compute_sweep(prob_val.astype(np.float64).tobytes())
idx_fin    = df_sweep["beneficio_liq"].idxmax()
THRESH_FIN = df_sweep.loc[idx_fin, "threshold"]
BEN_FIN    = df_sweep.loc[idx_fin, "beneficio_liq"]

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.markdown("# 💼 Análise de Negócio — Detecção de Fraudes")
st.caption(f"Conjunto de validação · Threshold selecionado: **{threshold:.3f}** · "
           f"Modelo: XGBoost + Optuna")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Resumo Executivo",
    "📈 Impacto Financeiro",
    "⚖️ Tradeoff TP × FP",
    "📅 Por Safra",
    "🔬 Sensibilidade",
])

# ╔══════════════════════════════════════════════════════╗
# ║  TAB 1 — RESUMO EXECUTIVO                           ║
# ╚══════════════════════════════════════════════════════╝
with tab1:
    # Linha 1: KPIs financeiros
    st.markdown('<div class="section-title">Impacto Financeiro (Threshold: {:.3f})</div>'.format(threshold),
                unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("💰 Fraudes Bloqueadas (R$)",  f"R$ {salvo_tp:,.2f}",
              help="Valor total das fraudes detectadas e bloqueadas")
    f2.metric("⚠️ Custo das Negações (R$)",  f"R$ {custo_fp_:,.2f}",
              delta=f"-R$ {custo_fp_:,.0f}", delta_color="inverse",
              help="Custo de bloquear transações legítimas (FP)")
    f3.metric("🔴 Exposição Residual (R$)",  f"R$ {resid_fn:,.2f}",
              delta=f"-R$ {resid_fn:,.0f}", delta_color="inverse",
              help="Valor das fraudes que ainda passam (FN)")
    f4.metric("✅ Benefício Líquido (R$)", f"R$ {benef_liq:,.2f}",
              delta=f"{benef_liq/perda_sem:.1%} vs sem modelo")

    st.markdown('<div class="section-title">Desempenho do Modelo</div>',
                unsafe_allow_html=True)

    # Linha 2: KPIs de performance
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("TP — Fraudes detectadas", f"{tp:,}")
    m2.metric("FP — Legítimas negadas",  f"{fp:,}")
    m3.metric("FN — Fraudes perdidas",   f"{fn:,}")
    m4.metric("Recall",    f"{recall:.2%}")
    m5.metric("Precisão",  f"{precisao:.2%}")
    m6.metric("AUC-ROC",   f"{auc_roc:.4f}")

    st.markdown('<div class="section-title">Eficiência Operacional</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # Waterfall financeiro
    with col_a:
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Exposição sem Modelo", "Fraudes Bloqueadas (TP)",
               "Custo FP", "Benefício Líquido"],
            y=[-perda_sem, salvo_tp, -custo_fp_, 0],
            text=[f"R${-perda_sem:,.0f}", f"+R${salvo_tp:,.0f}",
                  f"-R${custo_fp_:,.0f}", f"R${benef_liq:,.0f}"],
            textposition="outside",
            connector={"line": {"color": "#CBD5E1"}},
            decreasing={"marker": {"color": C_FRAUD}},
            increasing={"marker": {"color": C_POSITIVE}},
            totals={"marker": {"color": C_LEGIT}},
        ))
        fig_wf.update_layout(
            template=TEMPLATE, height=380,
            title="Decomposição do Benefício Financeiro",
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(tickprefix="R$", gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    # Gauge de recall
    with col_b:
        razao_tp_fp = tp / max(fp, 1)
        fig_gauge = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        )
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=recall * 100,
            title={"text": "Recall de Fraude (%)"},
            delta={"reference": 80, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": C_POSITIVE},
                "steps": [
                    {"range": [0, 50],  "color": "#FEE2E2"},
                    {"range": [50, 75], "color": "#FEF3C7"},
                    {"range": [75, 100],"color": "#DCFCE7"},
                ],
                "threshold": {"line": {"color": C_FRAUD, "width": 3},
                              "thickness": 0.75, "value": 80},
            },
            number={"suffix": "%", "font": {"size": 32}},
        ), row=1, col=1)

        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number",
            value=round(razao_tp_fp, 2),
            title={"text": "Fraudes por Legítima Negada (TP/FP)"},
            gauge={
                "axis": {"range": [0, 5]},
                "bar": {"color": C_LEGIT},
                "steps": [
                    {"range": [0, 0.5], "color": "#FEE2E2"},
                    {"range": [0.5, 1], "color": "#FEF3C7"},
                    {"range": [1, 5],   "color": "#DCFCE7"},
                ],
                "threshold": {"line": {"color": "#94A3B8", "width": 2},
                              "thickness": 0.75, "value": 1},
            },
            number={"font": {"size": 32}},
        ), row=1, col=2)

        fig_gauge.update_layout(
            height=380, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 2 — IMPACTO FINANCEIRO POR THRESHOLD           ║
# ╚══════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="section-title">Benefício Líquido por Threshold</div>',
                unsafe_allow_html=True)
    st.caption(f"Threshold ótimo financeiro: **{THRESH_FIN:.3f}** (R$ {BEN_FIN:,.2f}) · "
               f"Selecionado: **{threshold:.3f}** (R$ {benef_liq:,.2f})")

    fig_sweep = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
    )
    # Painel superior: valores em R$
    fig_sweep.add_trace(go.Scatter(
        x=df_sweep["threshold"], y=df_sweep["beneficio_liq"],
        name="Benefício Líquido", line=dict(color=C_LEGIT, width=2.5),
        fill="tozeroy", fillcolor="rgba(45,125,210,0.08)",
        hovertemplate="thr=%{x:.3f}<br>Benef. Líq.: R$%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    fig_sweep.add_trace(go.Scatter(
        x=df_sweep["threshold"], y=df_sweep["valor_salvo"],
        name="Valor Salvo (TP)", line=dict(color=C_POSITIVE, width=1.5, dash="dot"),
    ), row=1, col=1)
    fig_sweep.add_trace(go.Scatter(
        x=df_sweep["threshold"], y=-df_sweep["custo_fp"],
        name="Custo FP", line=dict(color=C_WARN, width=1.5, dash="dot"),
    ), row=1, col=1)
    fig_sweep.add_trace(go.Scatter(
        x=df_sweep["threshold"], y=-df_sweep["prejuizo_fn"],
        name="Exposição FN", line=dict(color=C_FRAUD, width=1.5, dash="dot"),
    ), row=1, col=1)

    # Painel inferior: recall vs precisão
    fig_sweep.add_trace(go.Scatter(
        x=df_sweep["threshold"], y=df_sweep["recall"] * 100,
        name="Recall (%)", line=dict(color=C_POSITIVE, width=2),
    ), row=2, col=1)
    fig_sweep.add_trace(go.Scatter(
        x=df_sweep["threshold"], y=df_sweep["precisao"] * 100,
        name="Precisão (%)", line=dict(color=C_NEUTRAL, width=2),
    ), row=2, col=1)

    # Linhas verticais para threshold selecionado e ótimo financeiro
    for thr_v, color_v, label_v in [
        (threshold, "#64748B", f"Selecionado ({threshold:.3f})"),
        (THRESH_FIN, C_LEGIT,  f"Ótimo Fin. ({THRESH_FIN:.3f})"),
    ]:
        for r in [1, 2]:
            fig_sweep.add_vline(
                x=thr_v, line_dash="dash", line_color=color_v,
                line_width=1.5, row=r, col=1,
                annotation_text=label_v if r == 1 else "",
                annotation_position="top right",
                annotation_font_size=10,
            )

    fig_sweep.add_hline(y=0, line_color="#CBD5E1", line_width=1, row=1, col=1)

    fig_sweep.update_layout(
        template=TEMPLATE, height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.04),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig_sweep.update_yaxes(tickprefix="R$", gridcolor="#F0F0F0", row=1, col=1)
    fig_sweep.update_yaxes(ticksuffix="%", gridcolor="#F0F0F0", row=2, col=1)
    fig_sweep.update_xaxes(title_text="Threshold", row=2, col=1)
    st.plotly_chart(fig_sweep, use_container_width=True)


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 3 — TRADEOFF TP × FP                          ║
# ╚══════════════════════════════════════════════════════╝
with tab3:
    col_l, col_r = st.columns(2)

    # Curva TP vs FP
    with col_l:
        st.markdown('<div class="section-title">Fraudes Detectadas vs Legítimas Negadas</div>',
                    unsafe_allow_html=True)

        row_sel = df_sweep.iloc[(df_sweep["threshold"] - threshold).abs().argsort()[:1]]
        row_fin = df_sweep.iloc[(df_sweep["threshold"] - THRESH_FIN).abs().argsort()[:1]]

        fig_tradeoff = go.Figure()
        fig_tradeoff.add_trace(go.Scatter(
            x=df_sweep["fp"], y=df_sweep["tp"],
            mode="lines", name="Curva TP × FP",
            line=dict(color=C_LEGIT, width=2.5),
            hovertemplate="FP=%{x}<br>TP=%{y}<extra></extra>",
        ))
        fig_tradeoff.add_trace(go.Scatter(
            x=row_sel["fp"], y=row_sel["tp"], mode="markers",
            name=f"Selecionado ({threshold:.3f})",
            marker=dict(color="#64748B", size=14, symbol="diamond"),
        ))
        fig_tradeoff.add_trace(go.Scatter(
            x=row_fin["fp"], y=row_fin["tp"], mode="markers",
            name=f"Ótimo Fin. ({THRESH_FIN:.3f})",
            marker=dict(color=C_FRAUD, size=14, symbol="star"),
        ))
        fig_tradeoff.update_layout(
            xaxis_title="Transações Legítimas Negadas (FP)",
            yaxis_title="Fraudes Detectadas (TP)",
        )
        st.plotly_chart(style(fig_tradeoff, 380), use_container_width=True)

    # Razão TP/FP por threshold
    with col_r:
        st.markdown('<div class="section-title">Eficiência: Fraudes por Legítima Negada (TP/FP)</div>',
                    unsafe_allow_html=True)

        ratio = df_sweep["tp"] / df_sweep["fp"].replace(0, np.nan)
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(
            x=df_sweep["threshold"], y=ratio,
            mode="lines", line=dict(color=C_POSITIVE, width=2.5),
            fill="tozeroy", fillcolor="rgba(6,214,160,0.08)",
            name="TP/FP",
        ))
        fig_ratio.add_hline(y=1, line_dash="dot", line_color=C_FRAUD,
                            annotation_text="Breakeven (1:1)",
                            annotation_position="bottom right",
                            annotation_font_size=10)
        for thr_v, color_v in [(threshold, "#64748B"), (THRESH_FIN, C_LEGIT)]:
            fig_ratio.add_vline(x=thr_v, line_dash="dash",
                                line_color=color_v, line_width=1.5)
        fig_ratio.update_layout(
            xaxis_title="Threshold",
            yaxis_title="Fraudes detectadas por legítima negada",
            yaxis=dict(rangemode="tozero"),
        )
        st.plotly_chart(style(fig_ratio, 380), use_container_width=True)

    # Tabela comparativa de cenários
    st.markdown('<div class="section-title">Comparação de Cenários</div>',
                unsafe_allow_html=True)

    def metrics_for(thr):
        p = (prob_val >= thr).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_arr, p).ravel()
        salvo_ = amount_val[(p == 1) & (y_arr == 1)].sum()
        custo_ = fp_ * custo_fp
        return {
            "Threshold": f"{thr:.3f}",
            "TP": tp_, "FP": fp_, "FN": fn_,
            "Recall": f"{tp_/(tp_+fn_):.2%}" if tp_+fn_ > 0 else "—",
            "Precisão": f"{tp_/(tp_+fp_):.2%}" if tp_+fp_ > 0 else "—",
            "Valor Salvo (R$)": f"R$ {salvo_:,.2f}",
            "Custo FP (R$)": f"R$ {custo_:,.2f}",
            "Benefício Líq. (R$)": f"R$ {salvo_-custo_:,.2f}",
        }

    cen_df = pd.DataFrame([
        {"Cenário": "Sem modelo",            **metrics_for(1.0)},
        {"Cenário": f"Selecionado ({threshold:.3f})", **metrics_for(threshold)},
        {"Cenário": f"Ótimo Fin. ({THRESH_FIN:.3f})", **metrics_for(THRESH_FIN)},
    ])
    cen_df.loc[0, ["TP","FP","FN","Recall","Precisão","Valor Salvo (R$)","Custo FP (R$)"]] = \
        [0, 0, int(y_arr.sum()), "0.00%", "—", "R$ 0.00", "R$ 0.00"]
    cen_df.loc[0, "Benefício Líq. (R$)"] = f"R$ {-perda_sem:,.2f}"

    st.dataframe(cen_df, use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 4 — POR SAFRA                                  ║
# ╚══════════════════════════════════════════════════════╝
with tab4:
    df_safra_base = pd.DataFrame({
        "safra" : safra_val,
        "target": y_arr,
        "score" : prob_val,
        "pred"  : pred_val,
        "amount": amount_val,
    })

    safras_ord = sorted(df_safra_base["safra"].unique())
    rows = []
    for s in safras_ord:
        sub = df_safra_base[df_safra_base["safra"] == s]
        tn_, fp_, fn_, tp_ = confusion_matrix(sub["target"], sub["pred"]).ravel()
        salvo_ = sub.loc[(sub["pred"] == 1) & (sub["target"] == 1), "amount"].sum()
        resid_ = sub.loc[(sub["pred"] == 0) & (sub["target"] == 1), "amount"].sum()
        custo_ = fp_ * custo_fp
        rows.append({
            "Safra": s, "Total": len(sub),
            "Fraudes": int(sub["target"].sum()),
            "taxa_fraude": sub["target"].mean(),
            "TP": tp_, "FP": fp_, "FN": fn_,
            "recall": tp_ / (tp_ + fn_) if tp_ + fn_ > 0 else 0,
            "valor_salvo": salvo_, "custo_fp": custo_,
            "exposicao_fn": resid_,
            "beneficio_liq": salvo_ - custo_,
        })

    df_s = pd.DataFrame(rows)

    # Gráfico principal: barras empilhadas + linha de benefício
    st.markdown('<div class="section-title">Impacto Financeiro por Safra</div>',
                unsafe_allow_html=True)

    fig_safra = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.06,
    )
    x = df_s["Safra"]
    fig_safra.add_trace(go.Bar(
        x=x, y=df_s["valor_salvo"], name="Valor Salvo (TP)",
        marker_color=C_POSITIVE, opacity=0.85,
    ), row=1, col=1)
    fig_safra.add_trace(go.Bar(
        x=x, y=-df_s["custo_fp"], name="Custo FP",
        marker_color=C_WARN, opacity=0.85,
    ), row=1, col=1)
    fig_safra.add_trace(go.Bar(
        x=x, y=-df_s["exposicao_fn"], name="Exposição FN",
        marker_color=C_FRAUD, opacity=0.60,
    ), row=1, col=1)
    fig_safra.add_trace(go.Scatter(
        x=x, y=df_s["beneficio_liq"], name="Benefício Líq.",
        mode="lines+markers", line=dict(color=C_LEGIT, width=2.5),
        marker=dict(size=9), zorder=5,
    ), row=1, col=1)
    fig_safra.add_trace(go.Scatter(
        x=x, y=df_s["recall"] * 100, name="Recall (%)",
        mode="lines+markers", line=dict(color=C_POSITIVE, width=2),
        marker=dict(size=8),
    ), row=2, col=1)

    fig_safra.add_hline(y=0, line_color="#CBD5E1", line_width=1, row=1, col=1)
    fig_safra.update_layout(
        template=TEMPLATE, height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        barmode="relative",
        legend=dict(orientation="h", y=1.04),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig_safra.update_yaxes(tickprefix="R$", gridcolor="#F0F0F0", row=1, col=1)
    fig_safra.update_yaxes(ticksuffix="%", gridcolor="#F0F0F0", row=2, col=1)
    st.plotly_chart(fig_safra, use_container_width=True)

    # Tabela resumo
    st.markdown('<div class="section-title">Detalhe por Safra</div>', unsafe_allow_html=True)
    disp_df = df_s[["Safra","Total","Fraudes","TP","FP","FN"]].copy()
    disp_df["Taxa Fraude"]      = df_s["taxa_fraude"].map("{:.2%}".format)
    disp_df["Recall"]           = df_s["recall"].map("{:.2%}".format)
    disp_df["Valor Salvo"]      = df_s["valor_salvo"].map("R$ {:,.2f}".format)
    disp_df["Custo FP"]         = df_s["custo_fp"].map("R$ {:,.2f}".format)
    disp_df["Benefício Líq."]   = df_s["beneficio_liq"].map("R$ {:,.2f}".format)
    st.dataframe(disp_df, use_container_width=True, hide_index=True)


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 5 — SENSIBILIDADE                              ║
# ╚══════════════════════════════════════════════════════╝
with tab5:
    st.markdown('<div class="section-title">Sensibilidade do Benefício Líquido às Premissas de Custo</div>',
                unsafe_allow_html=True)
    st.caption("Cada célula mostra o benefício líquido (R$) com o threshold selecionado no sidebar.")

    custos_rev = [0.5, 1, 2, 5, 10, 20, 50]
    taxas_ch   = [0.00, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

    grid_data = np.zeros((len(custos_rev), len(taxas_ch)))
    for i, cr in enumerate(custos_rev):
        for j, tc in enumerate(taxas_ch):
            cf = cr + tc * ltv_cliente
            grid_data[i, j] = salvo_tp - fp * cf

    fig_heat = go.Figure(go.Heatmap(
        z=grid_data,
        x=[f"{t:.1%}" for t in taxas_ch],
        y=[f"R${c}" for c in custos_rev],
        colorscale=[[0, C_FRAUD], [0.4, "#FEF3C7"], [0.6, "#D1FAE5"], [1, C_POSITIVE]],
        hoverongaps=False,
        hovertemplate=(
            "Revisão: %{y}<br>Churn: %{x}<br>"
            "Benefício Líq.: R$%{z:,.0f}<extra></extra>"
        ),
        colorbar=dict(title="Benefício Líq. (R$)", tickprefix="R$"),
        text=[[f"R${v:,.0f}" for v in row] for row in grid_data],
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig_heat.update_layout(
        template=TEMPLATE, height=440,
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis_title="Taxa de Churn por FP",
        yaxis_title="Custo Operacional por FP",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=f"Heatmap de Sensibilidade — thr={threshold:.3f} · "
                 f"TP={tp} · FP={fp} · LTV=R${ltv_cliente:.0f}",
            font_size=12,
        ),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Cenário base destacado
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Cenário atual (sidebar)", f"R$ {benef_liq:,.2f}")
    col_b.metric("Melhor caso (R$0.50 rev, 0% churn)",
                 f"R$ {salvo_tp - fp*(0.5 + 0.00*ltv_cliente):,.2f}")
    col_c.metric("Pior caso (R$50 rev, 20% churn)",
                 f"R$ {salvo_tp - fp*(50 + 0.20*ltv_cliente):,.2f}")

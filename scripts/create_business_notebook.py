"""Script para gerar o notebook de análise de negócio."""
import json

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

# ── 0. Título ──────────────────────────────────────────────────────────────
cells.append(md(
    "# Análise de Negócio — Detecção de Fraudes\n\n"
    "**Objetivo:** Avaliar se o modelo de detecção de fraudes compensa financeiramente, "
    "quantificando o tradeoff entre fraudes capturadas e transações legítimas negadas.\n\n"
    "**Perguntas respondidas:**\n"
    "1. Qual o impacto financeiro do modelo vs não ter modelo?\n"
    "2. O tradeoff (fraudes pegas × falsos positivos) é favorável?\n"
    "3. Qual threshold maximiza o retorno financeiro líquido?\n"
    "4. Qual a exposição residual (fraudes que ainda passam)?\n"
    "5. O impacto é estável ao longo do tempo (por safra)?\n"
    "6. Quão sensível é o resultado a diferentes premissas de custo?"
))

# ── 1. Imports ─────────────────────────────────────────────────────────────
cells.append(code(
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n\n"
    "import glob\n"
    "import os\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.ticker as mticker\n"
    "import seaborn as sns\n"
    "import joblib\n\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.metrics import confusion_matrix\n\n"
    "plt.rcParams.update({'figure.dpi': 130, 'axes.spines.top': False, 'axes.spines.right': False})\n"
    "sns.set_palette('husl')\n\n"
    "RANDOM_STATE = 42"
))

# ── 2. Parâmetros de Negócio ───────────────────────────────────────────────
cells.append(md("---\n## 1. Parâmetros de Negócio\n\nDefina as premissas financeiras abaixo. Todos os valores em R$."))

cells.append(code(
    "# ── Premissas financeiras (ajuste conforme a realidade do negócio) ──────\n\n"
    "# Custo de uma transação legítima negada (Falso Positivo)\n"
    "CUSTO_REVISAO_FP = 5.00    # R$ — custo fixo de revisar/bloquear uma transação legítima\n"
    "TAXA_CHURN_FP    = 0.02    # 2% dos clientes com FP cancelam o cartão\n"
    "LTV_CLIENTE      = 800.00  # R$ — valor médio anual de um cliente retido\n\n"
    "# Custo total de um FP = custo operacional + perda esperada de churn\n"
    "CUSTO_FP = CUSTO_REVISAO_FP + (TAXA_CHURN_FP * LTV_CLIENTE)\n\n"
    "# Custo operacional de manter o modelo — por período de análise\n"
    "CUSTO_OPERACIONAL_MODELO = 0.00  # R$ — inclua se quiser no ROI\n\n"
    "print('Premissas de Custo:')\n"
    "print(f'  Custo por FP (transação legítima negada) : R$ {CUSTO_FP:.2f}')\n"
    "print(f'    └─ Revisão operacional                 : R$ {CUSTO_REVISAO_FP:.2f}')\n"
    "print(f'    └─ Perda esperada por churn            : R$ {TAXA_CHURN_FP*LTV_CLIENTE:.2f}  ({TAXA_CHURN_FP:.0%} x R${LTV_CLIENTE:.0f})')\n"
    "print(f'  Custo por FN (fraude não detectada)      : R$ TX_AMOUNT (variável por transação)')"
))

# ── 3. Carregamento ────────────────────────────────────────────────────────
cells.append(md("---\n## 2. Carregamento do Modelo e Dados"))

cells.append(code(
    "# ── Modelo mais recente (XGBoost Optuna) ────────────────────────────────\n"
    "model_files = sorted(glob.glob('../data/models/xgboost_optuna*.pkl'))\n"
    "assert model_files, 'Nenhum modelo encontrado em data/models/. Execute o ExperimentLoggerAgent primeiro.'\n"
    "MODEL_PATH = model_files[-1]\n"
    "print(f'Modelo carregado: {os.path.basename(MODEL_PATH)}')\n\n"
    "pipeline = joblib.load(MODEL_PATH)\n\n"
    "# Extrair threshold do nome do arquivo\n"
    "_fname = os.path.basename(MODEL_PATH)\n"
    "try:\n"
    "    _thr_str = _fname.split('thr')[1].split('.pkl')[0]\n"
    "    # Normaliza formato: 0_300 → 0.300 ou 0.300 permanece\n"
    "    THRESH_OPT = float(_thr_str.replace('_', '.')) if '_' in _thr_str else float(_thr_str)\n"
    "except Exception:\n"
    "    THRESH_OPT = 0.30\n"
    "print(f'Threshold de classificação: {THRESH_OPT:.3f}')"
))

cells.append(code(
    "# ── Features e split (idêntico ao notebook de modelagem) ─────────────────\n"
    "X = pd.read_parquet('../data/gold/X_train.parquet')\n"
    "y = pd.read_parquet('../data/gold/y_train.parquet').squeeze()\n\n"
    "X_train, X_val, y_train, y_val = train_test_split(\n"
    "    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y\n"
    ")\n\n"
    "# TX_AMOUNT do gold — não entra no modelo, mas é essencial para o custo financeiro\n"
    "gold = pd.read_parquet('../data/gold/train_gold.parquet', columns=['TX_AMOUNT'])\n"
    "_, gold_val, _, _ = train_test_split(\n"
    "    gold, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y\n"
    ")\n"
    "amount_val = gold_val['TX_AMOUNT'].values\n\n"
    "# Safra (mês de referência) a partir do silver\n"
    "silver = pd.read_parquet('../data/silver/train_silver.parquet', columns=['TX_DATETIME'])\n"
    "_, silver_val, _, _ = train_test_split(\n"
    "    silver, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y\n"
    ")\n"
    "safra_val = silver_val['TX_DATETIME'].dt.to_period('M').astype(str).values\n\n"
    "print(f'Validação : {X_val.shape[0]:,} transações | fraude: {y_val.mean():.4%}')\n"
    "print(f'Montante total de fraudes (val): R$ {amount_val[y_val.values == 1].sum():,.2f}')"
))

# ── 4. Predições ───────────────────────────────────────────────────────────
cells.append(md("---\n## 3. Predições no Conjunto de Validação"))

cells.append(code(
    "prob_val = pipeline.predict_proba(X_val)[:, 1]\n"
    "pred_val = (prob_val >= THRESH_OPT).astype(int)\n"
    "y_arr    = y_val.values\n\n"
    "tn, fp, fn, tp = confusion_matrix(y_arr, pred_val).ravel()\n\n"
    "print(f'Threshold : {THRESH_OPT:.3f}')\n"
    "print()\n"
    "print(f'  TP (fraudes detectadas)       : {tp:>6,}')\n"
    "print(f'  FP (legítimas negadas)        : {fp:>6,}')\n"
    "print(f'  TN (legítimas aprovadas)      : {tn:>6,}')\n"
    "print(f'  FN (fraudes não detectadas)   : {fn:>6,}')\n"
    "print()\n"
    "print(f'  Recall de fraude (TPR)        : {tp/(tp+fn):.2%}')\n"
    "print(f'  Taxa de falsos positivos (FPR): {fp/(fp+tn):.4%}')\n"
    "print(f'  Precisão                      : {tp/(tp+fp):.2%}')"
))

# ── 5. Matriz de Custo-Benefício ───────────────────────────────────────────
cells.append(md("---\n## 4. Matriz de Custo-Benefício"))

cells.append(code(
    "tp_mask = (pred_val == 1) & (y_arr == 1)\n"
    "fp_mask = (pred_val == 1) & (y_arr == 0)\n"
    "fn_mask = (pred_val == 0) & (y_arr == 1)\n\n"
    "valor_salvo_tp   = amount_val[tp_mask].sum()    # fraudes bloqueadas (R$ salvo)\n"
    "custo_fp_total   = fp * CUSTO_FP                # custo das negações indevidas\n"
    "prejuizo_fn      = amount_val[fn_mask].sum()    # fraudes que passaram\n"
    "perda_sem_modelo = amount_val[y_arr == 1].sum() # baseline: 100% das fraudes passam\n"
    "beneficio_liquido = valor_salvo_tp - custo_fp_total - CUSTO_OPERACIONAL_MODELO\n\n"
    "print('=' * 56)\n"
    "print(' MATRIZ FINANCEIRA — CONJUNTO DE VALIDAÇÃO')\n"
    "print('=' * 56)\n"
    "print(f'  {\"Fraudes bloqueadas (TP)\":<35}: R$ {valor_salvo_tp:>10,.2f}')\n"
    "print(f'  {\"Custo transações negadas (FP)\":<35}: R$ {custo_fp_total:>10,.2f}')\n"
    "print(f'  {\"Fraudes não detectadas (FN)\":<35}: R$ {prejuizo_fn:>10,.2f}')\n"
    "print('-' * 56)\n"
    "print(f'  {\"Benefício líquido c/ modelo\":<35}: R$ {beneficio_liquido:>10,.2f}')\n"
    "print(f'  {\"Exposição sem modelo\":<35}: R$ {perda_sem_modelo:>10,.2f}')\n"
    "print(f'  {\"Redução de exposição\":<35}: {valor_salvo_tp/perda_sem_modelo:>10.2%}')\n"
    "print('=' * 56)\n"
    "print()\n"
    "print(f'  Razão TP/FP : {tp}/{fp} → {tp/max(fp,1):.1f} fraudes bloqueadas por cada legítima negada')\n"
    "print(f'  Custo médio FP vs valor médio de fraude : R${CUSTO_FP:.2f} vs R${amount_val[y_arr==1].mean():.2f}')"
))

# ── 6. Sweep de Threshold ──────────────────────────────────────────────────
cells.append(md(
    "---\n## 5. Impacto Financeiro por Threshold\n\n"
    "Varremos todos os thresholds para identificar o ponto de máximo benefício líquido. "
    "O threshold ótimo financeiro pode diferir do threshold ótimo por F2."
))

cells.append(code(
    "thresholds = np.linspace(0.01, 0.99, 300)\n"
    "resultados = []\n\n"
    "for thr in thresholds:\n"
    "    p = (prob_val >= thr).astype(int)\n"
    "    tn_, fp_, fn_, tp_ = confusion_matrix(y_arr, p).ravel()\n"
    "    salvo_  = amount_val[(p == 1) & (y_arr == 1)].sum()\n"
    "    custo_  = fp_ * CUSTO_FP\n"
    "    resid_  = amount_val[(p == 0) & (y_arr == 1)].sum()\n"
    "    resultados.append({\n"
    "        'threshold'   : thr,\n"
    "        'tp'          : tp_,\n"
    "        'fp'          : fp_,\n"
    "        'fn'          : fn_,\n"
    "        'valor_salvo' : salvo_,\n"
    "        'custo_fp'    : custo_,\n"
    "        'prejuizo_fn' : resid_,\n"
    "        'beneficio_liq': salvo_ - custo_,\n"
    "        'recall'      : tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0,\n"
    "        'precisao'    : tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0,\n"
    "    })\n\n"
    "df_thr = pd.DataFrame(resultados)\n\n"
    "idx_fin    = df_thr['beneficio_liq'].idxmax()\n"
    "THRESH_FIN = df_thr.loc[idx_fin, 'threshold']\n"
    "BEN_FIN    = df_thr.loc[idx_fin, 'beneficio_liq']\n\n"
    "idx_f2  = (df_thr['threshold'] - THRESH_OPT).abs().idxmin()\n"
    "BEN_F2  = df_thr.loc[idx_f2, 'beneficio_liq']\n\n"
    "print(f'Threshold otimizado (F2)     : {THRESH_OPT:.3f}  → benefício líq. R$ {BEN_F2:,.2f}')\n"
    "print(f'Threshold ótimo financeiro   : {THRESH_FIN:.3f}  → benefício líq. R$ {BEN_FIN:,.2f}')\n"
    "print(f'Ganho adicional potencial    : R$ {BEN_FIN - BEN_F2:,.2f}')"
))

cells.append(code(
    "fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,\n"
    "                         gridspec_kw={'height_ratios': [2, 1]})\n\n"
    "# Painel 1: impacto financeiro\n"
    "ax1 = axes[0]\n"
    "ax1.plot(df_thr['threshold'], df_thr['beneficio_liq'],  color='#2196F3', lw=2.5, label='Benefício Líquido (Salvo - Custo FP)')\n"
    "ax1.plot(df_thr['threshold'], df_thr['valor_salvo'],    color='#4CAF50', lw=1.5, ls='--', label='Fraudes Bloqueadas (R$)')\n"
    "ax1.plot(df_thr['threshold'], -df_thr['custo_fp'],      color='#F44336', lw=1.5, ls='--', label='Custo FP (negativo)')\n"
    "ax1.plot(df_thr['threshold'], -df_thr['prejuizo_fn'],   color='#FF9800', lw=1.5, ls=':',  label='Exposição Residual FN (negativo)')\n"
    "ax1.axvline(THRESH_OPT, color='gray',    ls='--', lw=1.2, label=f'Threshold F2 ({THRESH_OPT:.3f})')\n"
    "ax1.axvline(THRESH_FIN, color='#1565C0', ls='-',  lw=1.8, label=f'Threshold Financeiro ({THRESH_FIN:.3f})')\n"
    "ax1.axhline(0, color='black', lw=0.8)\n"
    "ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x:,.0f}'))\n"
    "ax1.set_ylabel('R$')\n"
    "ax1.set_title('Impacto Financeiro por Threshold', fontweight='bold')\n"
    "ax1.legend(fontsize=8, ncol=2)\n\n"
    "# Painel 2: recall vs precisão\n"
    "ax2 = axes[1]\n"
    "ax2.plot(df_thr['threshold'], df_thr['recall'],   color='#4CAF50', lw=2, label='Recall')\n"
    "ax2.plot(df_thr['threshold'], df_thr['precisao'], color='#9C27B0', lw=2, label='Precisão')\n"
    "ax2.axvline(THRESH_OPT, color='gray',    ls='--', lw=1.2)\n"
    "ax2.axvline(THRESH_FIN, color='#1565C0', ls='-',  lw=1.8)\n"
    "ax2.set_xlabel('Threshold')\n"
    "ax2.set_ylabel('Taxa')\n"
    "ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0%}'))\n"
    "ax2.legend(fontsize=9)\n\n"
    "fig.suptitle('Tradeoff Financeiro vs Threshold de Classificação', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ── 7. Tradeoff TP x FP ────────────────────────────────────────────────────
cells.append(md("---\n## 6. Tradeoff: Fraudes Detectadas × Transações Legítimas Negadas"))

cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n\n"
    "# Painel A: curva TP vs FP (volume absoluto)\n"
    "ax = axes[0]\n"
    "ax.plot(df_thr['fp'], df_thr['tp'], color='#2196F3', lw=2.5)\n\n"
    "row_f2  = df_thr.iloc[(df_thr['threshold'] - THRESH_OPT).abs().argsort()[:1]]\n"
    "row_fin = df_thr.iloc[(df_thr['threshold'] - THRESH_FIN).abs().argsort()[:1]]\n"
    "ax.scatter(row_f2['fp'],  row_f2['tp'],  s=120, color='gray',    zorder=5, label=f'Threshold F2 ({THRESH_OPT:.3f})')\n"
    "ax.scatter(row_fin['fp'], row_fin['tp'], s=120, color='#F44336', zorder=5, label=f'Threshold Financeiro ({THRESH_FIN:.3f})')\n"
    "ax.set_xlabel('Transações Legítimas Negadas (FP)')\n"
    "ax.set_ylabel('Fraudes Detectadas (TP)')\n"
    "ax.set_title('Volume: Fraudes Detectadas vs Legítimas Negadas', fontweight='bold')\n"
    "ax.legend(fontsize=9)\n\n"
    "# Painel B: razão TP/FP ao longo dos thresholds\n"
    "ax2 = axes[1]\n"
    "ratio = df_thr['tp'] / df_thr['fp'].replace(0, np.nan)\n"
    "ax2.plot(df_thr['threshold'], ratio, color='#4CAF50', lw=2.5)\n"
    "ax2.axvline(THRESH_OPT, color='gray',    ls='--', lw=1.2, label=f'Threshold F2 ({THRESH_OPT:.3f})')\n"
    "ax2.axvline(THRESH_FIN, color='#1565C0', ls='-',  lw=1.8, label=f'Threshold Financeiro ({THRESH_FIN:.3f})')\n"
    "ax2.axhline(1, color='#F44336', ls=':', lw=1.2, label='Breakeven (1 fraude : 1 legítima)')\n"
    "ax2.set_xlabel('Threshold')\n"
    "ax2.set_ylabel('Fraudes detectadas por transação negada (TP/FP)')\n"
    "ax2.set_title('Eficiência: Razão TP/FP por Threshold', fontweight='bold')\n"
    "ax2.legend(fontsize=9)\n"
    "ax2.set_ylim(bottom=0)\n\n"
    "plt.suptitle('Tradeoff Operacional', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "v_f2  = row_f2.iloc[0]\n"
    "v_fin = row_fin.iloc[0]\n"
    "print(f'Threshold F2   → {v_f2[\"tp\"]:.0f} fraudes detectadas, {v_f2[\"fp\"]:.0f} legítimas negadas  (razão {v_f2[\"tp\"]/max(v_f2[\"fp\"],1):.1f}:1)')\n"
    "print(f'Threshold Fin. → {v_fin[\"tp\"]:.0f} fraudes detectadas, {v_fin[\"fp\"]:.0f} legítimas negadas (razão {v_fin[\"tp\"]/max(v_fin[\"fp\"],1):.1f}:1)')"
))

# ── 8. Com vs Sem Modelo ───────────────────────────────────────────────────
cells.append(md(
    "---\n## 7. Comparação: Com Modelo vs Sem Modelo\n\n"
    "Baseline **sem modelo**: todas as transações são aprovadas → 100% das fraudes passam."
))

cells.append(code(
    "total_fraudes = int(y_arr.sum())\n"
    "total_legit   = int((y_arr == 0).sum())\n"
    "total_tx      = len(y_arr)\n\n"
    "# Com modelo — threshold F2\n"
    "p_f2 = (prob_val >= THRESH_OPT).astype(int)\n"
    "tn_f2, fp_f2, fn_f2, tp_f2 = confusion_matrix(y_arr, p_f2).ravel()\n"
    "salvo_f2  = amount_val[(p_f2==1) & (y_arr==1)].sum()\n"
    "custo_f2  = fp_f2 * CUSTO_FP\n"
    "resid_f2  = amount_val[(p_f2==0) & (y_arr==1)].sum()\n"
    "benef_f2  = salvo_f2 - custo_f2\n\n"
    "# Com modelo — threshold financeiro\n"
    "p_fin = (prob_val >= THRESH_FIN).astype(int)\n"
    "tn_fin, fp_fin, fn_fin, tp_fin = confusion_matrix(y_arr, p_fin).ravel()\n"
    "salvo_fin = amount_val[(p_fin==1) & (y_arr==1)].sum()\n"
    "custo_fin = fp_fin * CUSTO_FP\n"
    "resid_fin = amount_val[(p_fin==0) & (y_arr==1)].sum()\n"
    "benef_fin = salvo_fin - custo_fin\n\n"
    "cenarios = {\n"
    "    'Sem Modelo': {\n"
    "        'Fraudes Detectadas': 0, 'FP': 0,\n"
    "        'Valor Salvo (R$)': 0, 'Custo FP (R$)': 0,\n"
    "        'Exposição Residual (R$)': perda_sem_modelo,\n"
    "        'Benefício Líquido (R$)': -perda_sem_modelo,\n"
    "    },\n"
    "    f'Modelo (thr={THRESH_OPT:.3f})': {\n"
    "        'Fraudes Detectadas': tp_f2, 'FP': fp_f2,\n"
    "        'Valor Salvo (R$)': salvo_f2, 'Custo FP (R$)': custo_f2,\n"
    "        'Exposição Residual (R$)': resid_f2,\n"
    "        'Benefício Líquido (R$)': benef_f2,\n"
    "    },\n"
    "    f'Modelo (thr={THRESH_FIN:.3f}) [Financeiro]': {\n"
    "        'Fraudes Detectadas': tp_fin, 'FP': fp_fin,\n"
    "        'Valor Salvo (R$)': salvo_fin, 'Custo FP (R$)': custo_fin,\n"
    "        'Exposição Residual (R$)': resid_fin,\n"
    "        'Benefício Líquido (R$)': benef_fin,\n"
    "    },\n"
    "}\n\n"
    "df_cen = pd.DataFrame(cenarios).T\n"
    "display(\n"
    "    df_cen.style\n"
    "    .format({\n"
    "        'Fraudes Detectadas': '{:.0f}', 'FP': '{:.0f}',\n"
    "        'Valor Salvo (R$)': 'R$ {:,.2f}', 'Custo FP (R$)': 'R$ {:,.2f}',\n"
    "        'Exposição Residual (R$)': 'R$ {:,.2f}', 'Benefício Líquido (R$)': 'R$ {:,.2f}',\n"
    "    })\n"
    "    .background_gradient(subset=['Benefício Líquido (R$)'], cmap='RdYlGn')\n"
    "    .set_caption('Comparação de Cenários — Conjunto de Validação')\n"
    ")\n\n"
    "print(f'\\nO modelo (thr={THRESH_OPT:.3f}) evita R$ {benef_f2:,.2f} de perda líquida')\n"
    "print(f'Redução de {benef_f2/perda_sem_modelo:.2%} da exposição total vs sem modelo')"
))

# ── 9. Análise por Safra ────────────────────────────────────────────────────
cells.append(md("---\n## 8. Impacto Financeiro por Safra (Temporal)"))

cells.append(code(
    "df_safra = pd.DataFrame({\n"
    "    'safra' : safra_val,\n"
    "    'target': y_arr,\n"
    "    'score' : prob_val,\n"
    "    'pred'  : p_f2,\n"
    "    'amount': amount_val,\n"
    "})\n\n"
    "safras_ord = sorted(df_safra['safra'].unique())\n"
    "rows = []\n"
    "for s in safras_ord:\n"
    "    sub = df_safra[df_safra['safra'] == s]\n"
    "    tn_, fp_, fn_, tp_ = confusion_matrix(sub['target'], sub['pred']).ravel()\n"
    "    salvo_ = sub.loc[(sub['pred']==1) & (sub['target']==1), 'amount'].sum()\n"
    "    resid_ = sub.loc[(sub['pred']==0) & (sub['target']==1), 'amount'].sum()\n"
    "    custo_ = fp_ * CUSTO_FP\n"
    "    rows.append({\n"
    "        'Safra': s, 'Total TX': len(sub), 'Fraudes': int(sub['target'].sum()),\n"
    "        'Taxa Fraude': sub['target'].mean(),\n"
    "        'TP': tp_, 'FP': fp_, 'FN': fn_,\n"
    "        'Recall': tp_ / (tp_ + fn_) if tp_ + fn_ > 0 else 0,\n"
    "        'Valor Salvo': salvo_, 'Custo FP': custo_, 'Exposição FN': resid_,\n"
    "        'Benefício Líq.': salvo_ - custo_,\n"
    "    })\n\n"
    "df_s = pd.DataFrame(rows)\n"
    "display(\n"
    "    df_s.style\n"
    "    .format({\n"
    "        'Taxa Fraude': '{:.2%}', 'Recall': '{:.2%}',\n"
    "        'Valor Salvo': 'R$ {:,.2f}', 'Custo FP': 'R$ {:,.2f}',\n"
    "        'Exposição FN': 'R$ {:,.2f}', 'Benefício Líq.': 'R$ {:,.2f}',\n"
    "    })\n"
    "    .background_gradient(subset=['Benefício Líq.'], cmap='RdYlGn')\n"
    "    .set_caption(f'Impacto Financeiro por Safra — threshold {THRESH_OPT:.3f}')\n"
    ")"
))

cells.append(code(
    "fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,\n"
    "                         gridspec_kw={'height_ratios': [2, 1]})\n"
    "x = np.arange(len(safras_ord))\n\n"
    "ax1 = axes[0]\n"
    "ax1.bar(x, df_s['Valor Salvo'],   color='#4CAF50', alpha=0.85, label='Valor Salvo (TP)')\n"
    "ax1.bar(x, -df_s['Custo FP'],     color='#F44336', alpha=0.75, label='Custo FP')\n"
    "ax1.bar(x, -df_s['Exposição FN'], color='#FF9800', alpha=0.60, label='Exposição Residual (FN)')\n"
    "ax1.plot(x, df_s['Benefício Líq.'], color='#1565C0', marker='o', lw=2.5, ms=8, label='Benefício Líq.', zorder=5)\n"
    "ax1.axhline(0, color='black', lw=0.8)\n"
    "ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'R${v:,.0f}'))\n"
    "ax1.set_ylabel('R$')\n"
    "ax1.set_title('Impacto Financeiro por Safra', fontweight='bold')\n"
    "ax1.legend(fontsize=9, ncol=2)\n\n"
    "ax2 = axes[1]\n"
    "ax2.plot(x, df_s['Recall'], color='#4CAF50', marker='o', lw=2, ms=7)\n"
    "ax2.fill_between(x, df_s['Recall'], alpha=0.15, color='#4CAF50')\n"
    "ax2.axhline(df_s['Recall'].mean(), color='gray', ls='--', lw=1.2, label=f\"Média {df_s['Recall'].mean():.2%}\")\n"
    "ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))\n"
    "ax2.set_xticks(x)\n"
    "ax2.set_xticklabels(safras_ord, rotation=20, ha='right', fontsize=10)\n"
    "ax2.set_ylabel('Recall')\n"
    "ax2.set_title('Recall de Fraude por Safra', fontweight='bold')\n"
    "ax2.legend(fontsize=9)\n\n"
    "plt.suptitle('Estabilidade Financeira Temporal', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ── 10. Exposição Residual ─────────────────────────────────────────────────
cells.append(md(
    "---\n## 9. Exposição Residual — Fraudes que Ainda Passam (FN)\n\n"
    "Análise das fraudes não detectadas: quanto valem, qual a concentração de valor."
))

cells.append(code(
    "fn_mask_f2 = (p_f2 == 0) & (y_arr == 1)\n"
    "amount_fn  = amount_val[fn_mask_f2]\n\n"
    "print(f'Fraudes não detectadas (FN) : {fn_f2:,}')\n"
    "print(f'Valor total FN              : R$ {amount_fn.sum():,.2f}')\n"
    "print(f'Valor médio por FN          : R$ {amount_fn.mean():.2f}')\n"
    "print(f'FN de alto valor (> R$100)  : {(amount_fn > 100).sum()} transações  ->  R$ {amount_fn[amount_fn > 100].sum():,.2f}')\n"
    "print()\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n\n"
    "# Histograma de valor dos FN\n"
    "ax1 = axes[0]\n"
    "ax1.hist(amount_fn, bins=40, color='#FF7043', edgecolor='white', alpha=0.85)\n"
    "ax1.axvline(amount_fn.mean(), color='black', ls='--', lw=1.5, label=f'Média R${amount_fn.mean():.2f}')\n"
    "ax1.set_xlabel('Valor da Transação (R$)')\n"
    "ax1.set_ylabel('Frequência')\n"
    "ax1.set_title('Distribuição de Valor — Fraudes Não Detectadas (FN)', fontweight='bold')\n"
    "ax1.legend(fontsize=9)\n\n"
    "# Curva de Pareto (concentração de valor)\n"
    "sorted_fn  = np.sort(amount_fn)[::-1]\n"
    "cumsum_fn  = np.cumsum(sorted_fn) / sorted_fn.sum()\n"
    "pct_casos  = np.arange(1, len(sorted_fn) + 1) / len(sorted_fn)\n\n"
    "ax2 = axes[1]\n"
    "ax2.plot(pct_casos * 100, cumsum_fn * 100, color='#FF7043', lw=2.5)\n"
    "ax2.axhline(80, color='gray', ls=':', lw=1, label='80% do valor total')\n"
    "idx_80 = np.searchsorted(cumsum_fn, 0.80)\n"
    "if idx_80 < len(pct_casos):\n"
    "    ax2.axvline(pct_casos[idx_80] * 100, color='gray', ls=':', lw=1)\n"
    "    ax2.annotate(f'{pct_casos[idx_80]:.1%} dos FN',\n"
    "                 xy=(pct_casos[idx_80]*100, 80),\n"
    "                 xytext=(pct_casos[idx_80]*100 + 5, 60),\n"
    "                 fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))\n"
    "ax2.set_xlabel('% das Fraudes Não Detectadas (ordenadas por valor desc.)')\n"
    "ax2.set_ylabel('% do Valor Total Acumulado')\n"
    "ax2.set_title('Concentração de Valor — Fraudes Não Detectadas (Pareto)', fontweight='bold')\n"
    "ax2.legend(fontsize=9)\n\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ── 11. Sensibilidade ──────────────────────────────────────────────────────
cells.append(md(
    "---\n## 10. Análise de Sensibilidade\n\n"
    "Como o benefício líquido se comporta sob diferentes premissas de custo? "
    "Isso testa a robustez da decisão de implantar o modelo."
))

cells.append(code(
    "custos_revisao = [1, 2, 5, 10, 20, 50]\n"
    "taxas_churn    = [0.00, 0.01, 0.02, 0.05, 0.10]\n\n"
    "grid = pd.DataFrame(\n"
    "    index  =[f'R${c:.0f}/FP revisao' for c in custos_revisao],\n"
    "    columns=[f'{t:.0%} churn' for t in taxas_churn],\n"
    "    dtype=float\n"
    ")\n\n"
    "for cr in custos_revisao:\n"
    "    for tc in taxas_churn:\n"
    "        custo_fp_i = cr + tc * LTV_CLIENTE\n"
    "        benef_i    = salvo_f2 - fp_f2 * custo_fp_i\n"
    "        grid.loc[f'R${cr:.0f}/FP revisao', f'{tc:.0%} churn'] = benef_i\n\n"
    "fig, ax = plt.subplots(figsize=(11, 5))\n"
    "sns.heatmap(\n"
    "    grid.astype(float), annot=True, fmt=',.0f', cmap='RdYlGn',\n"
    "    linewidths=0.5, ax=ax,\n"
    "    cbar_kws={'label': 'Benefício Líquido (R$)'},\n"
    ")\n"
    "ax.set_xlabel('Taxa de Churn por FP')\n"
    "ax.set_ylabel('Custo Operacional de Revisão')\n"
    "ax.set_title(\n"
    "    f'Sensibilidade do Benefício Líquido (thr={THRESH_OPT:.3f})\\n'\n"
    "    f'LTV fixo = R${LTV_CLIENTE:.0f} | TP = {tp_f2} fraudes bloqueadas | FP = {fp_f2} legítimas negadas',\n"
    "    fontweight='bold'\n"
    ")\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "print(f'Pior cenário  (R$50/FP, 10% churn) : R$ {grid.iloc[-1, -1]:,.2f}')\n"
    "print(f'Melhor cenário (R$1/FP,  0% churn) : R$ {grid.iloc[ 0,  0]:,.2f}')\n"
    "print(f'Cenário base   (R${CUSTO_REVISAO_FP:.0f}/FP, {TAXA_CHURN_FP:.0%} churn) : R$ {benef_f2:,.2f}')"
))

# ── 12. Sumário Executivo ──────────────────────────────────────────────────
cells.append(md("---\n## 11. Sumário Executivo"))

cells.append(code(
    "roi_vs_nada = benef_f2 / perda_sem_modelo\n\n"
    "print('=' * 62)\n"
    "print(' SUMÁRIO EXECUTIVO — DETECCAO DE FRAUDE')\n"
    "print('=' * 62)\n"
    "print()\n"
    "print(f'  Período analisado     : {safras_ord[0]} a {safras_ord[-1]}')\n"
    "print(f'  Transações avaliadas  : {total_tx:,}')\n"
    "print(f'  Fraudes reais         : {total_fraudes:,}  ({total_fraudes/total_tx:.2%} do volume)')\n"
    "print()\n"
    "print('  DESEMPENHO DO MODELO')\n"
    "print(f'    Threshold aplicado  : {THRESH_OPT:.3f}')\n"
    "print(f'    Fraudes bloqueadas  : {tp_f2:,} de {total_fraudes:,}  ({tp_f2/total_fraudes:.2%} recall)')\n"
    "print(f'    Legítimas negadas   : {fp_f2:,} de {total_legit:,}  ({fp_f2/total_legit:.4%} do volume legítimo)')\n"
    "print(f'    Razão TP:FP         : {tp_f2/max(fp_f2,1):.1f} fraudes bloqueadas por cada legítima negada')\n"
    "print()\n"
    "print('  IMPACTO FINANCEIRO (conjunto de validação)')\n"
    "print(f'    Fraudes evitadas (R$)       : R$ {salvo_f2:>10,.2f}')\n"
    "print(f'    Custo das negações (R$)     : R$ {custo_f2:>10,.2f}')\n"
    "print(f'    Exposição residual (FN)     : R$ {resid_f2:>10,.2f}')\n"
    "print(f'    Benefício líquido           : R$ {benef_f2:>10,.2f}')\n"
    "print()\n"
    "print(f'    Reducao da exposição total  : {roi_vs_nada:.2%}  (vs cenário sem modelo)')\n"
    "print()\n"
    "print('  THRESHOLD RECOMENDADO')\n"
    "if abs(THRESH_FIN - THRESH_OPT) > 0.01:\n"
    "    diff = BEN_FIN - BEN_F2\n"
    "    print(f'    Threshold F2 (atual)        : {THRESH_OPT:.3f}')\n"
    "    print(f'    Threshold financeiro ótimo  : {THRESH_FIN:.3f}')\n"
    "    print(f'    Ganho adicional potencial   : R$ {diff:,.2f}')\n"
    "    print(f'    Recomendacao: considerar ajuste para {THRESH_FIN:.3f} para maximizar ROI')\n"
    "else:\n"
    "    print(f'    O threshold F2 ({THRESH_OPT:.3f}) coincide com o otimo financeiro.')\n"
    "print('=' * 62)"
))

# ── Montar e salvar ────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open("notebooks/analise_negocio.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook criado: notebooks/analise_negocio.ipynb ({len(cells)} células)")

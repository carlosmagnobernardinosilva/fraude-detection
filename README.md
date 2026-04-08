# Fraud Detection — End-to-End ML Pipeline

Sistema end-to-end de detecção de fraudes em cartões de crédito em terminais físicos (POS), desenvolvido como projeto de portfólio.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-lightgreen?logo=xgboost)
![MLflow](https://img.shields.io/badge/MLflow-tracking-orange?logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)

---

## Visão Geral

Pipeline semi-automatizado que cobre desde a ingestão dos dados brutos até o registro de experimentos no MLflow. A modelagem é realizada manualmente em notebook Jupyter, usando os dados preparados pelo pipeline.

**Stack:** Python · XGBoost · LightGBM · Scikit-learn · MLflow · SHAP · Streamlit · Pandas · Groq API

**Dataset:**
| Arquivo | Descrição | Volume |
|---|---|---|
| `train.csv` | Transações de treino (ago–dez/2021) | 291.231 linhas |
| `test.csv` | Transações de teste (jan/2022+) | 226.731 linhas |
| `customer.csv` | Perfil de clientes (coordenadas, comportamento) | 998 clientes |
| `terminal.csv` | Localização dos terminais POS | 1.994 terminais |

Taxa de fraude global: **~2,26%** — classes fortemente desbalanceadas.

---

## Arquitetura

O pipeline segue o padrão **Agent → Skill → Hook** com um objeto `PipelineContext` compartilhado:

- **Agent** — orquestra uma fase; nunca contém lógica de negócio diretamente
- **Skill** — função pura e stateless; única responsabilidade (ex: `cast_types`, `join_datasets`)
- **Hook** — validação e auditoria nas fronteiras do pipeline; nunca transforma dados
- **PipelineContext** — dataclass que carrega o estado entre todos os agentes

```
PipelineContext (dados brutos carregados)
        │
        ▼
[pre_data_hook]              ← valida schema, taxa de fraude, volume
        │
        ▼
DataPreparationAgent
    ├── skill: type_casting  ← corrige tipos (TX_DATETIME, TX_FRAUD, etc.)
    ├── skill: joiner        ← LEFT JOIN train + customer + terminal
    └── skill: filter        ← remove duplicatas, TX_AMOUNT <= 0, nulos críticos
        │
        ▼
[post_data_hook]             ← loga tempo, % removido, nulos restantes
        │
        ▼
[pre_feature_hook]           ← valida tipo TX_DATETIME
        │
        ▼
FeatureEngineeringAgent
    ├── skill: feature_creator   ← features de negócio (baseadas na EDA)
    ├── skill: feature_explorer  ← LLM (Groq) sugere e testa novas hipóteses
    └── skill: feature_selector  ← IV + RandomForest para seleção
        │
        ▼
[post_feature_hook]          ← detecta leakage, valida X/y, loga taxa fraude
        │
        ▼
        [Humano modela aqui — Jupyter Notebook]
        │
        ▼
[pre_logging_hook]           ← bloqueia se métricas abaixo do mínimo
        │
        ▼
ExperimentLoggerAgent
    └── skill: mlflow_logger     ← log_params, log_metrics, log_model, log_artifacts
        │
        ▼
[post_logging_hook]          ← confirma run_id, loga resumo
```

---

## Estrutura de Arquivos

```
fraud-detection/
│
├── src/
│   ├── context/
│   │   └── pipeline_context.py            ← dataclass de estado do pipeline
│   ├── orchestrator/
│   │   └── preparation_orchestrator.py    ← orquestra agentes + hooks
│   ├── agents/
│   │   ├── DataPreparationAgent.py        ← type_casting + joiner + filter
│   │   ├── FeatureEngineeringAgent.py     ← creator + explorer + selector
│   │   └── ExperimentLoggerAgent.py       ← mlflow_logger
│   ├── skills/
│   │   ├── type_casting.py
│   │   ├── joiner.py
│   │   ├── filter.py
│   │   ├── feature_creator.py
│   │   ├── feature_explorer.py            ← integração com Groq API (LLM)
│   │   ├── feature_selector.py
│   │   ├── mlflow_logger.py
│   │   └── persistence.py                 ← save/load Silver e Gold
│   └── hooks/
│       ├── data_hooks.py
│       ├── feature_hooks.py
│       └── logging_hooks.py
│
├── data/
│   ├── raw/                               ← CSVs originais (não versionados)
│   ├── silver/                            ← Parquet limpo e tipado (não versionado)
│   ├── gold/                              ← Parquet com features (não versionado)
│   └── models/                            ← modelos serializados (não versionados)
│
├── notebooks/
│   ├── entendimento_dados.ipynb           ← EDA
│   ├── analise_negocio.ipynb              ← análise de negócio
│   ├── modelagem_baseline.ipynb           ← modelos baseline
│   └── modelagem_xgboost_optuna.ipynb     ← otimização com Optuna
│
├── app/                                   ← Streamlit BI
├── reports/                               ← catálogo de features, relatórios
└── requirements.txt
```

---

## Medallion Architecture

Os dados passam por três camadas, todas em Parquet:

```
data/raw/    → CSVs originais, nunca modificados
data/silver/ → limpos, tipados e com JOIN (DataPreparationAgent)
data/gold/   → Silver + features engineered + selecionadas (FeatureEngineeringAgent)
```

Agentes pulam o reprocessamento se a camada já existir. Para forçar: `DataPreparationAgent(force_rerun=True)`.

---

## Features Criadas

<details>
<summary>Temporais</summary>

| Feature | Descrição |
|---|---|
| TX_HOUR | Hora da transação |
| TX_DAY_OF_WEEK | Dia da semana |
| TX_DAY_OF_MONTH | Dia do mês |
| TX_MONTH / TX_YEAR | Mês e ano |
| TX_DURING_WEEKEND | Flag fim de semana |
| TX_NIGHT_FLAG | Flag período noturno |
| PERIODO_DIA / PERIODO_DIA_NUM | Manhã / Tarde / Noite |

</details>

<details>
<summary>Valor da transação</summary>

| Feature | Descrição |
|---|---|
| TX_AMOUNT_LOG | log1p do valor |
| TX_AMOUNT_ROUNDED | Flag valor sem centavos |
| TX_ABOVE_MEAN_FLAG | Acima da média do cliente |
| TX_AMOUNT_ZSCORE | Desvio vs. histórico do cliente |

</details>

<details>
<summary>Risco do terminal (delay 7 dias para evitar leakage)</summary>

| Feature | Descrição |
|---|---|
| TERM_NB_TX_1D/7D/30D | Volume de transações no terminal |
| TERM_RISK_1D/7D/30D | Taxa de fraude do terminal por janela |

</details>

<details>
<summary>Janelas temporais do cliente (sem leakage — closed='left')</summary>

Para W ∈ {1H, 2H, 4H, 8H, 12H, 24H, 48H, 72H, 7D, 14D, 21D, 30D, 45D}:

`TX_CUST_NB_TX_W` · `TX_CUST_SUM_AMT_W` · `TX_CUST_MEAN_AMT_W` · `TX_CUST_MEDIAN_AMT_W` · `TX_CUST_MIN_AMT_W` · `TX_CUST_MAX_AMT_W`

</details>

<details>
<summary>Comportamentais e geográficas</summary>

| Feature | Descrição |
|---|---|
| TX_TIME_SINCE_LAST_TX | Minutos desde a última transação do cliente |
| TX_FLAG_SAME_TERMINAL | Flag mesmo terminal da transação anterior |
| TX_CUST_NIGHT_RATIO | Proporção de transações noturnas (expanding) |
| TX_CUST_DISTINCT_TERMINALS | Terminais distintos usados (expanding) |
| DIST_CUSTOMER_TERMINAL | Distância euclidiana cliente ↔ terminal |

</details>

<details>
<summary>Razões e velocidade</summary>

`RATIO_AMT_VS_CUST_24H` · `RATIO_AMT_VS_CUST_7D` · `RATIO_TERM_RISK_1D_7D` · `RATIO_AMT_VS_GLOBAL_MEAN` · `RATIO_NB_TX_1H_24H` · `RATIO_MAX_VS_MEAN_24H` · `TX_AMOUNT_X_DIST` · `TX_VELOCITY_24H`

</details>

---

## Achados da EDA

A análise exploratória revelou padrões claros que guiaram a construção das features:

- **Concentração de risco por terminal:** o terminal 565 apresentou taxa de fraude **9,68× acima da média** — evidenciando que o histórico do terminal é um sinal forte
- **Exposição ampla de clientes:** **75,15%** dos clientes foram envolvidos em ao menos 1 fraude no período, indicando que o comportamento individual também importa
- **Período noturno como fator de risco:** transações entre 00h e 04h concentraram a maior taxa de fraude (**2,43%**), quase o dobro da média geral
- **Valor da transação é discriminativo:** o teste Mann-Whitney confirmou diferença significativa nos montantes entre fraudes e transações legítimas (p = 0,003)
- **Total fraudado no período:** **R$ 376.210,13** — representando impacto financeiro real e mensurável

---

## O que o Modelo Resolveu

Com base nos padrões identificados na EDA, o pipeline produziu um modelo XGBoost otimizado com Optuna capaz de:

- **Detectar fraudes com alta sensibilidade** — recall superior a 80% no conjunto de validação, recuperando a maior parte do valor em risco
- **Controlar falsos positivos** — threshold otimizado pelo F2-score para equilibrar a detecção máxima de fraudes com o mínimo de transações legítimas bloqueadas
- **Quantificar o impacto financeiro** — o app Streamlit calcula em tempo real o benefício líquido do modelo (valor salvo em fraudes bloqueadas menos o custo operacional de cada falso positivo), permitindo ajuste interativo do threshold conforme as premissas de negócio
- **Rastrear experimentos de forma reproduzível** — todos os modelos treinados são registrados no MLflow com parâmetros, métricas, threshold e lista de features, garantindo auditabilidade completa

---

## Principais Decisões de Design

| Decisão | Motivo |
|---|---|
| Hooks explícitos (não decorators) | Mais legível e fácil de debugar |
| LEFT JOIN no joiner | Nunca perde transações silenciosamente |
| Delay de 7 dias no risco do terminal | Evita data leakage |
| `closed='left'` nas janelas do cliente | Garante ausência de leakage nas rolling windows |
| `pre_logging_hook` com threshold mínimo | Bloqueia registro se AUC-ROC < 0.80 ou AP < 0.40 |
| Modelagem manual em notebook | Fase que exige julgamento humano; pipeline apenas prepara os dados |

---

## Setup

```bash
# Clone o repositório
git clone https://github.com/carlosmagnobernardinosilva/fraude-detection.git
cd fraude-detection

# Instale as dependências
pip install -r requirements.txt

# Configure a variável de ambiente para o feature_explorer (opcional)
export GROQ_API_KEY=sua_chave_aqui
```

Coloque os CSVs originais em `data/raw/` e execute o pipeline:

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
ctx = PreparationOrchestrator().run(ctx)
# ctx.X_train e ctx.y_train prontos para modelagem
```

Após treinar o modelo, registre o experimento:

```python
from src.agents.ExperimentLoggerAgent import ExperimentLoggerAgent
from src.hooks.logging_hooks import pre_logging_hook, post_logging_hook
import time

ctx.model      = model
ctx.model_name = "xgboost_v1"
ctx.threshold  = 0.48
ctx.metrics    = {"auc_roc": 0.97, "average_precision": 0.82}

pre_logging_hook(ctx)
if not ctx.has_errors():
    start = time.time()
    ctx = ExperimentLoggerAgent().run(ctx)
    post_logging_hook(ctx, start)
```

Para visualizar os resultados:

```bash
streamlit run app/main.py
```


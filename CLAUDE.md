# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest
pytest tests/test_specific.py::test_name   # single test

# Run the automated preparation pipeline (from project root)
python -c "
from src.orchestrator.preparation_orchestrator import PreparationOrchestrator
from src.context.pipeline_context import PipelineContext
import pandas as pd
ctx = PipelineContext(
    df_train=pd.read_csv('data/raw/train.csv'),
    df_test=pd.read_csv('data/raw/test.csv'),
    df_customer=pd.read_csv('data/raw/customer.csv'),
    df_terminal=pd.read_csv('data/raw/terminal.csv'),
)
ctx = PreparationOrchestrator().run(ctx)
"

# Streamlit app (not yet implemented)
streamlit run app/main.py

# Jupyter for EDA / modeling
jupyter notebook notebooks/
```

## Architecture

The pipeline uses an **Agent → Skill → Hook** pattern with a shared `PipelineContext` object:

```
PipelineContext (shared state carrier)
       │
       ▼
PreparationOrchestrator          ← src/orchestrator/preparation_orchestrator.py
  ├─ DataPreparationAgent        ← src/agents/DataPreparationAgent.py
  │    calls: type_casting → joiner → filter → persistence (save Silver)
  │
  └─ FeatureEngineeringAgent     ← src/agents/FeatureEngineeringAgent.py
       calls: feature_creator → feature_explorer → feature_selector → persistence (save Gold)

[Human models in notebook using Gold data]

ExperimentLoggerAgent            ← src/agents/ExperimentLoggerAgent.py
  calls: mlflow_logger           ← src/skills/mlflow_logger.py
```

**Roles:**
- **Agents** orchestrate a phase; they call skills, never contain business logic themselves.
- **Skills** (`src/skills/`) are stateless functions — each has a single responsibility (e.g., `cast_types`, `join_datasets`, `create_features`).
- **Hooks** (`src/hooks/`) validate and audit at pipeline boundaries; they never transform data. Called explicitly by the orchestrator, not automatically.
- **PipelineContext** (`src/context/pipeline_context.py`) is the shared dataclass passed through every stage. It carries raw DataFrames, prepared DataFrames, feature lists, model reference, warnings, and errors.

**What is and isn't implemented:**

| Area | Status |
|---|---|
| `src/orchestrator/`, `src/agents/`, `src/skills/`, `src/hooks/` | Fully implemented |
| `src/models/`, `src/pipeline/`, `src/utils/` | Empty stubs |
| `app/` (Streamlit) | Not started |
| `tests/` | Not started |

## Data — Medallion Architecture

Raw CSVs live in `data/raw/` (not version-controlled). The pipeline writes two Parquet layers:

- **Silver** (`data/silver/train_silver.parquet`, `test_silver.parquet`) — cleaned, typed, joined (train + customer + terminal). Written by `DataPreparationAgent`.
- **Gold** (`data/gold/train_gold.parquet`) — Silver + all engineered features. Written by `FeatureEngineeringAgent`.

Both layers are handled by `src/skills/persistence.py`. Agents skip reprocessing if the layer already exists (use `force_rerun=True` to override).

Serialized models go in `data/models/`.

## Key Design Notes

- **Semi-automated pipeline:** `PreparationOrchestrator` is fully automated (data prep + feature engineering). Modeling is intentional done by the human in a Jupyter notebook using the Gold Parquet. `ExperimentLoggerAgent` is then called manually to log to MLflow.

- **LLM-assisted feature engineering:** `src/skills/feature_explorer.py` calls the **Groq API** (`llama-3.3-70b-versatile`) to suggest additional feature hypotheses. These are validated by IV (Information Value) before acceptance. Requires `GROQ_API_KEY` in the environment.

- **File naming convention:** Agent files use PascalCase (`DataPreparationAgent.py`) and imports in the orchestrator also use PascalCase (`from src.agents.DataPreparationAgent import ...`).

- **Class imbalance:** Fraud is ~2% of transactions. Use `imbalanced-learn` samplers at modeling time. Evaluate using AUC-PR, not accuracy. `logging_hooks.py` enforces minimum AUC/AP thresholds before MLflow logging proceeds.

- **Features created by `feature_creator`:**
  - *Temporal:* TX_HOUR, TX_DAY_OF_WEEK, TX_DAY_OF_MONTH, TX_MONTH, TX_YEAR, TX_DURING_WEEKEND, TX_NIGHT_FLAG, PERIODO_DIA/NUM
  - *Sequential:* TX_TIME_SINCE_LAST_TX (minutes since last tx per customer), TX_FLAG_SAME_TERMINAL
  - *Amount:* TX_AMOUNT_LOG, TX_AMOUNT_ROUNDED, TX_ABOVE_MEAN_FLAG
  - *Terminal risk (7-day delay):* TERM_NB_TX_{1,7,30}D, TERM_RISK_{1,7,30}D
  - *Geographic:* DIST_CUSTOMER_TERMINAL (euclidean)
  - *Z-score:* TX_AMOUNT_ZSCORE
  - *Customer rolling windows* (closed='left', no leakage) for W ∈ {1H,2H,4H,8H,12H,24H,48H,72H,7D,14D,21D,30D,45D}: TX_CUST_NB_TX_W, TX_CUST_SUM_AMT_W, TX_CUST_MEAN_AMT_W, TX_CUST_MEDIAN_AMT_W, TX_CUST_MIN_AMT_W, TX_CUST_MAX_AMT_W
  - *Behavioral:* TX_CUST_NIGHT_RATIO (expanding), TX_CUST_DISTINCT_TERMINALS (expanding)
  - *Ratios:* RATIO_AMT_VS_CUST_24H, RATIO_AMT_VS_CUST_7D, RATIO_TERM_RISK_1D_7D, RATIO_AMT_VS_GLOBAL_MEAN, RATIO_NB_TX_1H_24H, RATIO_MAX_VS_MEAN_24H, TX_AMOUNT_X_DIST, TX_VELOCITY_24H

"""
Skill: filter
-------------
Remove registros inválidos ou inconsistentes.
Cada filtro aplicado é logado com o volume removido.

Retorna o DataFrame filtrado e a lista de avisos gerados.
Quem decide o que fazer com os avisos é o agente — a skill
não acessa nem modifica o PipelineContext.
"""

import pandas as pd
from loguru import logger


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    initial  = len(df)
    warnings = []

    # ── 1. Duplicatas ──────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["TRANSACTION_ID"])
    removed = before - len(df)
    if removed:
        msg = f"{removed:,} transações duplicadas removidas"
        logger.warning(f"[filter] {msg}")
        warnings.append(msg)

    # ── 2. Valores negativos ou zerados em TX_AMOUNT ───────────────────────
    before = len(df)
    df = df[df["TX_AMOUNT"] > 0]
    removed = before - len(df)
    if removed:
        msg = f"{removed:,} transações com TX_AMOUNT <= 0 removidas"
        logger.warning(f"[filter] {msg}")
        warnings.append(msg)

    # ── 3. Nulos em colunas críticas ───────────────────────────────────────
    critical_cols = ["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"]
    critical_cols = [c for c in critical_cols if c in df.columns]

    before = len(df)
    df = df.dropna(subset=critical_cols)
    removed = before - len(df)
    if removed:
        msg = f"{removed:,} linhas com nulos em colunas críticas removidas"
        logger.warning(f"[filter] {msg}")
        warnings.append(msg)

    total_removed = initial - len(df)
    logger.info(
        f"[filter] Concluído — {len(df):,} linhas mantidas "
        f"({total_removed:,} removidas no total)"
    )

    return df, warnings
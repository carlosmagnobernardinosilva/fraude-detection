"""
Skill: type_casting
-------------------
Corrige os tipos das variáveis brutas.
Não remove nem transforma dados — só garante os tipos corretos.
"""

import pandas as pd
from loguru import logger


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Datetime
    if "TX_DATETIME" in df.columns:
        df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

    # Inteiros
    for col in ["TRANSACTION_ID", "CUSTOMER_ID", "TERMINAL_ID", "TX_FRAUD"]:
        if col in df.columns:
            df[col] = df[col].astype("int64")

    # Float
    for col in ["TX_AMOUNT"]:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    logger.info(f"[type_casting] Tipos ajustados — {df.shape[1]} colunas")
    return df
"""
Skill: joiner
-------------
Faz o merge entre as 3 bases usando as chaves corretas.
Sempre LEFT JOIN — garante que nenhuma transação é perdida.
"""

import pandas as pd
from loguru import logger


def join_datasets(
    df_train: pd.DataFrame,
    df_customer: pd.DataFrame,
    df_terminal: pd.DataFrame,
) -> pd.DataFrame:

    before = len(df_train)

    df = df_train.merge(df_customer, on="CUSTOMER_ID", how="left")
    df = df.merge(df_terminal, on="TERMINAL_ID", how="left")

    after = len(df)

    if before != after:
        logger.warning(
            f"[joiner] Volume alterado após merge: {before:,} → {after:,} linhas. "
            "Verifique duplicatas nas bases auxiliares."
        )
    else:
        logger.info(f"[joiner] Merge OK — {after:,} linhas | {df.shape[1]} colunas")

    return df
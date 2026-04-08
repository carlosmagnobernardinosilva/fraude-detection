"""
Skill: persistence
------------------
Salva e carrega cada camada da Medallion Architecture em Parquet.

Raw    → dados brutos (não tocamos aqui)
Silver → dados limpos e unidos (DataPreparationAgent)
Gold   → dados com features prontos para o modelo (FeatureEngineeringAgent)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# ── Caminhos padrão ───────────────────────────────────────────────────────

SILVER_PATH = Path("data/silver")
GOLD_PATH   = Path("data/gold")


# ── SILVER ────────────────────────────────────────────────────────────────

def save_silver(df_train: pd.DataFrame, df_test: pd.DataFrame = None) -> None:
    """Salva os datasets limpos e unidos na camada silver."""
    SILVER_PATH.mkdir(parents=True, exist_ok=True)

    df_train.to_parquet(SILVER_PATH / "train_silver.parquet", index=False)
    logger.info(f"[persistence] Silver salvo: train_silver.parquet ({len(df_train):,} linhas)")

    if df_test is not None:
        df_test.to_parquet(SILVER_PATH / "test_silver.parquet", index=False)
        logger.info(f"[persistence] Silver salvo: test_silver.parquet ({len(df_test):,} linhas)")


def load_silver() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Carrega os datasets da camada silver."""
    train_path = SILVER_PATH / "train_silver.parquet"
    test_path  = SILVER_PATH / "test_silver.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Silver não encontrado: {train_path}")

    df_train = pd.read_parquet(train_path)
    df_test  = pd.read_parquet(test_path) if test_path.exists() else None

    logger.info(f"[persistence] Silver carregado: {len(df_train):,} linhas")
    return df_train, df_test


def silver_exists() -> bool:
    return (SILVER_PATH / "train_silver.parquet").exists()


# ── GOLD ──────────────────────────────────────────────────────────────────

def save_gold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_columns: list,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    df_train: pd.DataFrame = None,
    df_test: pd.DataFrame = None,
) -> None:
    """
    Salva a camada Gold em dois formatos:

    - train_gold.parquet / test_gold.parquet : DataFrame completo com todas as
      features (Silver + engineered), exceto identificadores. Ideal para EDA e
      exploração no notebook.
    - X_train.parquet / X_test.parquet       : Matrizes com apenas as features
      selecionadas (passaram no IV), prontas para o modelo.
    - y_train.parquet / y_test.parquet       : Target.
    - feature_columns.txt                    : Lista das features selecionadas.
    """
    GOLD_PATH.mkdir(parents=True, exist_ok=True)

    # ── DataFrame completo (Silver + engineered) ──────────────────────────
    if df_train is not None:
        df_train.to_parquet(GOLD_PATH / "train_gold.parquet", index=False)
        logger.info(
            f"[persistence] Gold salvo: train_gold.parquet "
            f"{df_train.shape[0]:,} linhas × {df_train.shape[1]} colunas"
        )
    if df_test is not None:
        df_test.to_parquet(GOLD_PATH / "test_gold.parquet", index=False)
        logger.info(
            f"[persistence] Gold salvo: test_gold.parquet "
            f"{df_test.shape[0]:,} linhas × {df_test.shape[1]} colunas"
        )

    # ── Matrizes prontas para o modelo ────────────────────────────────────
    pd.DataFrame(X_train, columns=feature_columns).to_parquet(
        GOLD_PATH / "X_train.parquet", index=False
    )
    pd.DataFrame({"TX_FRAUD": y_train}).to_parquet(
        GOLD_PATH / "y_train.parquet", index=False
    )

    if X_test is not None:
        pd.DataFrame(X_test, columns=feature_columns).to_parquet(
            GOLD_PATH / "X_test.parquet", index=False
        )
    if y_test is not None:
        pd.DataFrame({"TX_FRAUD": y_test}).to_parquet(
            GOLD_PATH / "y_test.parquet", index=False
        )

    # Lista de features como txt — fácil de auditar
    (GOLD_PATH / "feature_columns.txt").write_text("\n".join(feature_columns))

    logger.info(
        f"[persistence] Gold salvo: X_train {X_train.shape} | "
        f"{len(feature_columns)} features selecionadas"
    )


def load_gold() -> tuple[np.ndarray, np.ndarray, list, np.ndarray | None, np.ndarray | None]:
    """
    Carrega X, y e feature_columns da camada gold.

    Retorna:
        X_train, y_train, feature_columns, X_test, y_test
    """
    for f in ["X_train.parquet", "y_train.parquet", "feature_columns.txt"]:
        if not (GOLD_PATH / f).exists():
            raise FileNotFoundError(f"Gold não encontrado: {GOLD_PATH / f}")

    feature_columns = (GOLD_PATH / "feature_columns.txt").read_text().splitlines()

    X_train = pd.read_parquet(GOLD_PATH / "X_train.parquet").values
    y_train = pd.read_parquet(GOLD_PATH / "y_train.parquet")["TX_FRAUD"].values

    X_test = (
        pd.read_parquet(GOLD_PATH / "X_test.parquet").values
        if (GOLD_PATH / "X_test.parquet").exists() else None
    )
    y_test = (
        pd.read_parquet(GOLD_PATH / "y_test.parquet")["TX_FRAUD"].values
        if (GOLD_PATH / "y_test.parquet").exists() else None
    )

    logger.info(f"[persistence] Gold carregado: X_train {X_train.shape} | {len(feature_columns)} features")
    return X_train, y_train, feature_columns, X_test, y_test


def gold_exists() -> bool:
    return (GOLD_PATH / "X_train.parquet").exists()
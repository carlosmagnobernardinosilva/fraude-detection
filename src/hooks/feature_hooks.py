"""
Hooks: feature_hooks
--------------------
Funções chamadas explicitamente pelo orquestrador
antes e depois do FeatureEngineeringAgent.

Não criam features — validam o que foi criado e detectam leakage.
"""

import time
import pandas as pd
from loguru import logger

from src.context import PipelineContext


# Colunas que indicam leakage se presentes no X_train
LEAKAGE_COLS = [
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
    "TRANSACTION_ID",
    "TX_DATETIME",
]


# ── PRE ───────────────────────────────────────────────────────────────────

def pre_feature_hook(ctx: PipelineContext) -> None:
    """
    Executado antes do FeatureEngineeringAgent.
    Garante que o DataFrame preparado está pronto para receber features.
    """
    logger.info("[feature_hooks] pre_feature_hook — validando df_train_prepared...")

    if ctx.df_train_prepared is None:
        ctx.add_error("pre_feature_hook", "df_train_prepared ausente.")
        return

    # TX_DATETIME precisa ser datetime para criar features temporais
    if "TX_DATETIME" in ctx.df_train_prepared.columns:
        dtype = ctx.df_train_prepared["TX_DATETIME"].dtype
        if not pd.api.types.is_datetime64_any_dtype(dtype):
            ctx.add_error(
                "pre_feature_hook",
                f"TX_DATETIME não é datetime — tipo encontrado: {dtype}. "
                "Verifique o type_casting."
            )
            logger.error("[feature_hooks] TX_DATETIME com tipo incorreto.")
            return

    logger.info("[feature_hooks] pre_feature_hook — OK")


# ── POST ──────────────────────────────────────────────────────────────────

def post_feature_hook(ctx: PipelineContext, start_time: float) -> None:
    """
    Executado depois do FeatureEngineeringAgent.
    Valida leakage, volume de features e consistência do X_train.
    """
    elapsed = time.time() - start_time
    logger.info(f"[feature_hooks] post_feature_hook — FeatureEngineeringAgent concluído em {elapsed:.2f}s")

    if ctx.X_train is None or ctx.y_train is None:
        ctx.add_error("post_feature_hook", "X_train ou y_train não foram gerados.")
        return

    # Verifica leakage nas features selecionadas
    leaking = [c for c in ctx.feature_columns if c in LEAKAGE_COLS]
    if leaking:
        ctx.add_error(
            "post_feature_hook",
            f"LEAKAGE DETECTADO nas features selecionadas: {leaking}"
        )
        logger.error(f"[feature_hooks] LEAKAGE: {leaking}")
        return

    # Verifica consistência entre X e y
    if ctx.X_train.shape[0] != ctx.y_train.shape[0]:
        ctx.add_error(
            "post_feature_hook",
            f"X_train ({ctx.X_train.shape[0]}) e y_train ({ctx.y_train.shape[0]}) "
            "com volumes diferentes."
        )
        return

    # Verifica se há features suficientes
    if len(ctx.feature_columns) < 3:
        ctx.add_warning(
            "post_feature_hook",
            f"Apenas {len(ctx.feature_columns)} features selecionadas — verifique o IV threshold."
        )
        logger.warning(f"[feature_hooks] Poucas features: {ctx.feature_columns}")

    # Loga taxa de fraude no y_train
    fraud_rate = ctx.y_train.mean()
    logger.info(
        f"[feature_hooks] X_train: {ctx.X_train.shape} | "
        f"Features: {len(ctx.feature_columns)} | "
        f"Taxa fraude y_train: {fraud_rate:.2%}"
    )
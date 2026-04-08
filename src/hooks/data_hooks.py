"""
Hooks: data_hooks
-----------------
Funções chamadas explicitamente pelo orquestrador
antes e depois do DataPreparationAgent.

Não transformam dados — apenas validam, auditam e logam.
"""

import time
import pandas as pd
from loguru import logger

from src.context import PipelineContext


# ── PRE ───────────────────────────────────────────────────────────────────

def pre_data_hook(ctx: PipelineContext) -> None:
    """
    Executado antes do DataPreparationAgent.
    Valida que os dados brutos foram carregados corretamente.
    """
    logger.info("[data_hooks] pre_data_hook — validando dados brutos...")

    # Verifica volumes mínimos
    if ctx.df_train is not None and len(ctx.df_train) < 1000:
        ctx.add_warning(
            "pre_data_hook",
            f"df_train com volume baixo: {len(ctx.df_train)} linhas."
        )

    # Verifica colunas obrigatórias no treino
    required = ["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TX_FRAUD"]
    if ctx.df_train is not None:
        missing = [c for c in required if c not in ctx.df_train.columns]
        if missing:
            ctx.add_error("pre_data_hook", f"Colunas ausentes em df_train: {missing}")
            logger.error(f"[data_hooks] Colunas ausentes: {missing}")
            return

    # Verifica taxa de fraude — alerta se estiver muito fora do esperado
    if ctx.df_train is not None and "TX_FRAUD" in ctx.df_train.columns:
        fraud_rate = ctx.df_train["TX_FRAUD"].mean()
        if fraud_rate < 0.01 or fraud_rate > 0.10:
            ctx.add_warning(
                "pre_data_hook",
                f"Taxa de fraude fora do intervalo esperado: {fraud_rate:.2%}"
            )
            logger.warning(f"[data_hooks] Taxa de fraude inesperada: {fraud_rate:.2%}")

    logger.info("[data_hooks] pre_data_hook — OK")


# ── POST ──────────────────────────────────────────────────────────────────

def post_data_hook(ctx: PipelineContext, start_time: float) -> None:
    """
    Executado depois do DataPreparationAgent.
    Valida o resultado e loga estatísticas da preparação.
    """
    elapsed = time.time() - start_time
    logger.info(f"[data_hooks] post_data_hook — DataPreparationAgent concluído em {elapsed:.2f}s")

    if ctx.df_train_prepared is None:
        ctx.add_error("post_data_hook", "df_train_prepared não foi gerado.")
        return

    original  = len(ctx.df_train)
    prepared  = len(ctx.df_train_prepared)
    removed   = original - prepared
    pct       = removed / original * 100 if original > 0 else 0

    logger.info(
        f"[data_hooks] Linhas originais: {original:,} | "
        f"Após preparação: {prepared:,} | "
        f"Removidas: {removed:,} ({pct:.2f}%)"
    )

    # Alerta se mais de 5% das linhas foram removidas
    if pct > 5:
        ctx.add_warning(
            "post_data_hook",
            f"{pct:.2f}% das linhas removidas na preparação — verifique os filtros."
        )
        logger.warning(f"[data_hooks] Alto volume de linhas removidas: {pct:.2f}%")

    # Verifica nulos no resultado
    nulls = ctx.df_train_prepared.isnull().sum().sum()
    if nulls > 0:
        ctx.add_warning("post_data_hook", f"{nulls} valores nulos encontrados após preparação.")
        logger.warning(f"[data_hooks] {nulls} nulos após preparação.")
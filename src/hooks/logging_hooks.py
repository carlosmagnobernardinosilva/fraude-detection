"""
Hooks: logging_hooks
--------------------
Funções chamadas explicitamente pelo orquestrador
antes e depois do ExperimentLoggerAgent.

Garante que só experimentos com qualidade mínima
são registrados no MLflow.
"""

import time
from loguru import logger

from src.context import PipelineContext


# Thresholds mínimos para aceitar o registro
MIN_AUC_ROC           = 0.80
MIN_AVERAGE_PRECISION = 0.40


# ── PRE ───────────────────────────────────────────────────────────────────

def pre_logging_hook(ctx: PipelineContext) -> None:
    """
    Executado antes do ExperimentLoggerAgent.
    Valida que o modelo e as métricas atendem ao mínimo esperado
    antes de registrar o experimento.
    """
    logger.info("[logging_hooks] pre_logging_hook — validando antes de registrar...")

    # Modelo presente
    if ctx.model is None:
        ctx.add_error("pre_logging_hook", "Nenhum modelo encontrado no contexto.")
        return

    # Métricas presentes
    if not ctx.metrics:
        ctx.add_error("pre_logging_hook", "Métricas ausentes — injete ctx.metrics antes de registrar.")
        return

    # Thresholds mínimos
    auc = ctx.metrics.get("auc_roc", 0)
    ap  = ctx.metrics.get("average_precision", 0)

    if auc < MIN_AUC_ROC:
        ctx.add_error(
            "pre_logging_hook",
            f"AUC-ROC ({auc:.4f}) abaixo do mínimo ({MIN_AUC_ROC}). "
            "Modelo não será registrado."
        )
        logger.error(f"[logging_hooks] AUC-ROC insuficiente: {auc:.4f}")
        return

    if ap < MIN_AVERAGE_PRECISION:
        ctx.add_error(
            "pre_logging_hook",
            f"Average Precision ({ap:.4f}) abaixo do mínimo ({MIN_AVERAGE_PRECISION}). "
            "Modelo não será registrado."
        )
        logger.error(f"[logging_hooks] Average Precision insuficiente: {ap:.4f}")
        return

    logger.info(
        f"[logging_hooks] pre_logging_hook — métricas OK "
        f"(AUC={auc:.4f} | AP={ap:.4f})"
    )


# ── POST ──────────────────────────────────────────────────────────────────

def post_logging_hook(ctx: PipelineContext, start_time: float) -> None:
    """
    Executado depois do ExperimentLoggerAgent.
    Confirma que o run_id foi gerado e loga o resumo final.
    """
    elapsed = time.time() - start_time
    logger.info(f"[logging_hooks] post_logging_hook — ExperimentLoggerAgent concluído em {elapsed:.2f}s")

    if ctx.run_id is None:
        ctx.add_error("post_logging_hook", "run_id não foi gerado — verifique o MLflow.")
        logger.error("[logging_hooks] run_id ausente após logging.")
        return

    logger.info(
        f"[logging_hooks] Experimento registrado com sucesso!\n"
        f"  run_id     : {ctx.run_id}\n"
        f"  model      : {ctx.model_name}\n"
        f"  threshold  : {ctx.threshold}\n"
        f"  métricas   : {ctx.metrics}\n"
        f"  features   : {len(ctx.feature_columns)}"
    )
"""
PreparationOrchestrator
-----------------------
Orquestra a sequência de agentes da fase de preparação e feature engineering.

Responsabilidades:
  - Validar o contexto antes de cada etapa
  - Decidir se avança, para ou emite alerta
  - Nunca executar lógica de negócio diretamente (isso é papel dos agentes)

Fluxo:
  PipelineContext
       │
       ▼
  DataPreparationAgent      ← limpa, tipifica e une as bases
       │
       ▼
  FeatureEngineeringAgent   ← cria e seleciona features
       │
       ▼
  PipelineContext pronto para o humano modelar

O ExperimentLoggerAgent é chamado separadamente pelo humano
após treinar e avaliar o modelo no notebook.
"""

import time

from loguru import logger

from src.context import PipelineContext
from src.agents.DataPreparationAgent import DataPreparationAgent
from src.agents.FeatureEngineeringAgent import FeatureEngineeringAgent
from src.hooks.data_hooks import pre_data_hook, post_data_hook
from src.hooks.feature_hooks import pre_feature_hook, post_feature_hook


class PreparationOrchestrator:

    def __init__(self, use_explorer: bool = True, n_suggestions: int = 8):
        """
        Parâmetros:
            use_explorer   : ativa exploração de features via LLM
            n_suggestions  : número de hipóteses pedidas ao LLM
        """
        self.data_agent    = DataPreparationAgent()
        self.feature_agent = FeatureEngineeringAgent(
            use_explorer=use_explorer,
            n_suggestions=n_suggestions,
        )

    def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info("=" * 55)
        logger.info("  PreparationOrchestrator — iniciando pipeline")
        logger.info("=" * 55)

        # ── Etapa 1: Preparação dos dados ─────────────────────────────
        pre_data_hook(ctx)
        if ctx.has_errors():
            logger.error("[PreparationOrchestrator] pre_data_hook falhou — pipeline interrompido.")
            return ctx

        t0 = time.time()
        ctx = self.data_agent.run(ctx)
        post_data_hook(ctx, t0)

        if ctx.has_errors():
            logger.error(
                "[PreparationOrchestrator] Erros na preparação dos dados — "
                "pipeline interrompido."
            )
            logger.error(f"[PreparationOrchestrator] Erros: {ctx.errors}")
            return ctx

        if not ctx.is_ready_for_feature_engineering():
            logger.error(
                "[PreparationOrchestrator] df_train_prepared não foi gerado — "
                "pipeline interrompido."
            )
            return ctx

        # ── Etapa 2: Feature engineering ──────────────────────────────
        pre_feature_hook(ctx)
        if ctx.has_errors():
            logger.error("[PreparationOrchestrator] pre_feature_hook falhou — pipeline interrompido.")
            return ctx

        t1 = time.time()
        ctx = self.feature_agent.run(ctx)
        post_feature_hook(ctx, t1)

        if ctx.has_errors():
            logger.error(
                "[PreparationOrchestrator] Erros no feature engineering — "
                "verifique o contexto antes de modelar."
            )
            logger.error(f"[PreparationOrchestrator] Erros: {ctx.errors}")
            return ctx

        # ── Avisos acumulados ─────────────────────────────────────────
        if ctx.warnings:
            logger.warning(
                f"[PreparationOrchestrator] {len(ctx.warnings)} aviso(s) — "
                f"revise antes de modelar: {ctx.warnings}"
            )

        # ── Resumo final ──────────────────────────────────────────────
        logger.info("=" * 55)
        logger.info("  PreparationOrchestrator — pipeline concluído")
        logger.info("=" * 55)
        logger.info(f"\n{ctx.summary()}")

        return ctx
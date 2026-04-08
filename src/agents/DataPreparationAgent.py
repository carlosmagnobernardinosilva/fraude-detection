"""
DataPreparationAgent
--------------------
Responsável por entregar os dados limpos, tipados e unidos
prontos para o FeatureEngineeringAgent.

Salva o resultado na camada Silver (data/silver/).

Orquestra 3 skills em sequência:
  1. type_casting  → corrige tipos de variáveis
  2. joiner        → merge train + customer + terminal
  3. filter        → remove registros inválidos
  4. persistence   → salva silver/train_silver.parquet
"""

from loguru import logger

from src.context import PipelineContext
from src.skills.type_casting import cast_types
from src.skills.joiner import join_datasets
from src.skills.filter import apply_filters
from src.skills.persistence import save_silver, load_silver, silver_exists


class DataPreparationAgent:

    def __init__(self, force_rerun: bool = False):
        """
        Parâmetros:
            force_rerun : se False e silver já existir, carrega do disco
                          sem reprocessar. Use True para forçar reprocessamento.
        """
        self.force_rerun = force_rerun

    def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info("[DataPreparationAgent] Iniciando...")

        # ── Atalho: carrega silver se já existir ──────────────────────
        if not self.force_rerun and silver_exists():
            logger.info("[DataPreparationAgent] Silver encontrado — carregando do disco.")
            ctx.df_train_prepared, ctx.df_test_prepared = load_silver()
            return ctx

        if not ctx.is_ready_for_preparation():
            msg = "Dados brutos não carregados. Preencha df_train, df_customer e df_terminal."
            ctx.add_error("DataPreparationAgent", msg)
            logger.error(f"[DataPreparationAgent] {msg}")
            return ctx

        try:
            # ── 1. Corrigir tipos ──────────────────────────────────────
            logger.info("[DataPreparationAgent] Etapa 1/4 — type_casting")
            df_train = cast_types(ctx.df_train)
            df_test  = cast_types(ctx.df_test) if ctx.df_test is not None else None

            # ── 2. Join com customer e terminal ───────────────────────
            logger.info("[DataPreparationAgent] Etapa 2/4 — join")
            df_train = join_datasets(df_train, ctx.df_customer, ctx.df_terminal)
            if df_test is not None:
                df_test = join_datasets(df_test, ctx.df_customer, ctx.df_terminal)

            # ── 3. Filtros ────────────────────────────────────────────
            logger.info("[DataPreparationAgent] Etapa 3/4 — filters")
            df_train, train_warnings = apply_filters(df_train)
            for w in train_warnings:
                ctx.add_warning("filter[train]", w)

            if df_test is not None:
                df_test, test_warnings = apply_filters(df_test)
                for w in test_warnings:
                    ctx.add_warning("filter[test]", w)

            # ── 4. Persiste na camada Silver ──────────────────────────
            logger.info("[DataPreparationAgent] Etapa 4/4 — salvando Silver")
            save_silver(df_train, df_test)

            ctx.df_train_prepared = df_train
            ctx.df_test_prepared  = df_test

            logger.info(
                f"[DataPreparationAgent] Concluído — "
                f"{df_train.shape[0]:,} linhas x {df_train.shape[1]} colunas → Silver"
            )

        except Exception as e:
            ctx.add_error("DataPreparationAgent", str(e))
            logger.exception(f"[DataPreparationAgent] Erro inesperado: {e}")

        return ctx
"""
FeatureEngineeringAgent
-----------------------
Responsável por criar e selecionar as features que entrarão no modelo.
Recebe df_train_prepared (Silver) e salva X_train, y_train na camada Gold.

Orquestra 3 skills em sequência:
  1. feature_creator   → features baseadas na EDA
  2. feature_explorer  → LLM sugere e testa novas hipóteses
  3. feature_selector  → IV + RandomForest para seleção
  4. persistence       → salva gold/X_train.parquet, y_train.parquet
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.context import PipelineContext
from src.skills.feature_creator import create_features
from src.skills.feature_explorer import explore_features, print_exploration_report
from src.skills.feature_selector import select_features
from src.skills.persistence import save_gold, load_gold, gold_exists


TARGET = "TX_FRAUD"

EXCLUDE_COLS = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_FRAUD",
    "PERIODO_DIA",
    "month_reference",
    # Coordenadas brutas: redundantes com DIST_CUSTOMER_TERMINAL e
    # funcionam como pseudo-identificadores de localização (risco de overfitting)
    "x_customer_id",
    "y_customer_id",
    "x_terminal_id",
    "y_terminal_id",
]


class FeatureEngineeringAgent:

    def __init__(self, use_explorer: bool = True, n_suggestions: int = 8, force_rerun: bool = False):
        """
        Parâmetros:
            use_explorer  : ativa exploração de features via LLM
            n_suggestions : número de hipóteses pedidas ao LLM
            force_rerun   : se False e gold já existir, carrega do disco
        """
        self.use_explorer  = use_explorer
        self.n_suggestions = n_suggestions
        self.force_rerun   = force_rerun

    def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info("[FeatureEngineeringAgent] Iniciando...")

        # ── Atalho: carrega gold se já existir ────────────────────────
        if not self.force_rerun and gold_exists():
            logger.info("[FeatureEngineeringAgent] Gold encontrado — carregando do disco.")
            ctx.X_train, ctx.y_train, ctx.feature_columns, ctx.X_test, ctx.y_test = load_gold()
            return ctx

        if not ctx.is_ready_for_feature_engineering():
            msg = "df_train_prepared não encontrado. Rode o DataPreparationAgent antes."
            ctx.add_error("FeatureEngineeringAgent", msg)
            logger.error(f"[FeatureEngineeringAgent] {msg}")
            return ctx

        try:
            # ── 1. Features baseadas no negócio ───────────────────────
            logger.info("[FeatureEngineeringAgent] Etapa 1/4 — feature_creator")

            # Concatena train+test antes de criar features para que as janelas
            # temporais do teste enxerguem o histórico do treino.
            # Uma coluna marcadora preserva a identidade de cada split após o
            # sort interno do create_features.
            has_test = ctx.df_test_prepared is not None
            if has_test:
                df_all = pd.concat(
                    [
                        ctx.df_train_prepared.assign(_IS_TRAIN=True),
                        ctx.df_test_prepared.assign(_IS_TRAIN=False),
                    ],
                    ignore_index=True,
                )
                df_all = create_features(df_all)
                df_train = df_all[df_all["_IS_TRAIN"]].drop(columns="_IS_TRAIN").reset_index(drop=True)
                df_test  = df_all[~df_all["_IS_TRAIN"]].drop(columns="_IS_TRAIN").reset_index(drop=True)
            else:
                df_train = create_features(ctx.df_train_prepared)
                df_test  = None

            known_features = [
                c for c in df_train.columns
                if c not in EXCLUDE_COLS and c not in ctx.df_train_prepared.columns
            ]

            # ── 2. Exploração via LLM ─────────────────────────────────
            if self.use_explorer:
                logger.info("[FeatureEngineeringAgent] Etapa 2/4 — feature_explorer (LLM)")
                df_train, report = explore_features(
                    df=df_train,
                    target=TARGET,
                    existing_features=known_features,
                    n_suggestions=self.n_suggestions,
                )
                print_exploration_report(report)

                if df_test is not None:
                    for item in [r for r in report if r["approved"]]:
                        try:
                            exec(item["code"], {"__builtins__": {}, "df": df_test, "np": np, "pd": pd})
                        except Exception as e:
                            logger.warning(f"[FeatureEngineeringAgent] '{item['name']}' no test: {e}")
            else:
                logger.info("[FeatureEngineeringAgent] Etapa 2/4 — feature_explorer desativado")

            # ── 3. Seleção de features ────────────────────────────────
            logger.info("[FeatureEngineeringAgent] Etapa 3/4 — feature_selector")
            candidate_cols = [c for c in df_train.columns if c not in EXCLUDE_COLS]

            selected_cols, iv_dict, importance_dict, correlation_dict = select_features(
                df=df_train,
                feature_cols=candidate_cols,
                target=TARGET,
            )

            # ── 4. Salva na camada Gold ───────────────────────────────
            logger.info("[FeatureEngineeringAgent] Etapa 4/4 — salvando Gold")

            # Colunas a manter no parquet analítico (remove só identificadores)
            _ID_COLS = {"TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "month_reference"}
            gold_cols_train = [c for c in df_train.columns if c not in _ID_COLS]
            gold_cols_test  = [c for c in df_test.columns  if c not in _ID_COLS] if df_test is not None else None

            X_train = df_train[selected_cols].values
            y_train = df_train[TARGET].values
            X_test  = df_test[selected_cols].values if df_test is not None else None
            y_test  = df_test[TARGET].values if df_test is not None and TARGET in df_test.columns else None

            save_gold(
                X_train, y_train, selected_cols, X_test, y_test,
                df_train=df_train[gold_cols_train],
                df_test=df_test[gold_cols_test] if df_test is not None else None,
            )

            # ── Persiste no contexto ──────────────────────────────────
            ctx.df_train_features  = df_train
            ctx.df_test_features   = df_test
            ctx.X_train            = X_train
            ctx.y_train            = y_train
            ctx.X_test             = X_test
            ctx.y_test             = y_test
            ctx.feature_columns    = selected_cols
            ctx.feature_iv         = iv_dict
            ctx.feature_importance = importance_dict
            ctx.feature_correlation = correlation_dict

            logger.info(
                f"[FeatureEngineeringAgent] Concluído — "
                f"{len(selected_cols)} features | X_train: {X_train.shape} → Gold"
            )

        except Exception as e:
            ctx.add_error("FeatureEngineeringAgent", str(e))
            logger.exception(f"[FeatureEngineeringAgent] Erro inesperado: {e}")

        return ctx
"""
ExperimentLoggerAgent
---------------------
Orquestra o registro do experimento no MLflow após o humano
treinar e avaliar o modelo manualmente.

Não treina, não avalia, não decide — só valida o contexto
e delega cada etapa do log para a skill mlflow_logger.

Uso correto (com hooks de validação de qualidade):
    import time
    from src.hooks.logging_hooks import pre_logging_hook, post_logging_hook

    ctx.model       = modelo_treinado
    ctx.model_name  = "xgboost_v1"
    ctx.threshold   = 0.48
    ctx.metrics     = {
        "auc_roc": 0.97,
        "average_precision": 0.82,
        "f1_fraud": 0.80,
    }

    pre_logging_hook(ctx)           # bloqueia se AUC < 0.80 ou AP < 0.40
    if not ctx.has_errors():
        t0  = time.time()
        ctx = ExperimentLoggerAgent().run(ctx)
        post_logging_hook(ctx, t0)  # confirma run_id e loga resumo final

    print(ctx.run_id)
"""

import mlflow
from loguru import logger

from src.context import PipelineContext
from src.skills.mlflow_logger import (
    log_params,
    log_metrics,
    log_features,
    log_threshold,
    log_model,
    log_feature_artifact,
)


class ExperimentLoggerAgent:

    def __init__(
        self,
        experiment_name: str = "fraud-detection",
        tracking_uri: str = "mlruns",
        models_path: str = "data/models",
    ):
        self.experiment_name = experiment_name
        self.tracking_uri    = tracking_uri
        self.models_path     = models_path

    def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info("[ExperimentLoggerAgent] Iniciando registro do experimento...")

        if not ctx.is_ready_for_logging():
            msg = (
                "Contexto incompleto. "
                "Verifique: model, model_name, metrics e feature_columns."
            )
            ctx.add_error("ExperimentLoggerAgent", msg)
            logger.error(f"[ExperimentLoggerAgent] {msg}")
            return ctx

        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run(run_name=ctx.model_name) as run:
                run_id = run.info.run_id

                log_params(ctx.model)
                log_metrics(ctx.metrics)
                log_features(ctx.feature_columns, ctx.feature_iv)
                log_threshold(ctx.threshold)
                log_model(ctx.model, ctx.model_name, run_id, self.models_path)
                log_feature_artifact(ctx.feature_columns, run_id, self.models_path)

                ctx.run_id = run_id

            logger.info(f"[ExperimentLoggerAgent] Concluído — run_id: {ctx.run_id}")

        except Exception as e:
            ctx.add_error("ExperimentLoggerAgent", str(e))
            logger.exception(f"[ExperimentLoggerAgent] Erro inesperado: {e}")

        return ctx
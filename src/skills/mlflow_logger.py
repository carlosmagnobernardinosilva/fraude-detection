"""
Skill: mlflow_logger
--------------------
Funções de registro no MLflow.
Cada função é responsável por uma parte do log — independentes entre si.

O agente decide o que e quando chamar.
"""

import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn
from loguru import logger


# ── Parâmetros do modelo ───────────────────────────────────────────────────

RELEVANT_PARAMS = [
    "n_estimators", "max_depth", "learning_rate",
    "subsample", "colsample_bytree", "scale_pos_weight",
    "random_state", "num_leaves", "min_child_samples",
]


def log_params(model) -> None:
    """Loga os hiperparâmetros relevantes do modelo."""
    try:
        params = model.get_params()
        filtered = {k: v for k, v in params.items() if k in RELEVANT_PARAMS}
        if filtered:
            mlflow.log_params(filtered)
            logger.info(f"[mlflow_logger] Parâmetros registrados: {filtered}")
    except AttributeError:
        logger.warning("[mlflow_logger] Modelo não possui get_params() — params não registrados.")


# ── Métricas ───────────────────────────────────────────────────────────────

def log_metrics(metrics: dict) -> None:
    """Loga as métricas de avaliação do modelo."""
    mlflow.log_metrics(metrics)
    logger.info(f"[mlflow_logger] Métricas registradas: {metrics}")


# ── Features ───────────────────────────────────────────────────────────────

def log_features(feature_columns: list, feature_iv: dict = None) -> None:
    """Loga a lista de features e os IVs das top 20."""
    mlflow.log_param("n_features", len(feature_columns))
    mlflow.log_param("features", ", ".join(feature_columns))

    if feature_iv:
        iv_metrics = {
            f"iv_{col}": round(iv, 4)
            for col, iv in list(feature_iv.items())[:20]
        }
        mlflow.log_metrics(iv_metrics)

    logger.info(f"[mlflow_logger] {len(feature_columns)} features registradas.")


# ── Threshold ──────────────────────────────────────────────────────────────

def log_threshold(threshold: float) -> None:
    """Loga o threshold de classificação."""
    mlflow.log_param("threshold", threshold)
    logger.info(f"[mlflow_logger] Threshold registrado: {threshold}")


# ── Modelo ─────────────────────────────────────────────────────────────────

def log_model(model, model_name: str, run_id: str, models_path: str) -> Path:
    """
    Salva o modelo em disco (.pkl) e registra no MLflow.
    Retorna o caminho do arquivo salvo.
    """
    path = Path(models_path)
    path.mkdir(parents=True, exist_ok=True)

    model_file = path / f"{model_name}_{run_id[:8]}.pkl"
    joblib.dump(model, model_file)
    mlflow.sklearn.log_model(model, artifact_path=model_name)

    logger.info(f"[mlflow_logger] Modelo salvo: {model_file}")
    return model_file


# ── Lista de features como artefato ───────────────────────────────────────

def log_feature_artifact(feature_columns: list, run_id: str, models_path: str) -> None:
    """Salva a lista de features como arquivo .txt e registra como artefato."""
    path = Path(models_path)
    path.mkdir(parents=True, exist_ok=True)

    features_file = path / f"features_{run_id[:8]}.txt"
    features_file.write_text("\n".join(feature_columns))
    mlflow.log_artifact(str(features_file))

    logger.info(f"[mlflow_logger] Feature list salva: {features_file}")
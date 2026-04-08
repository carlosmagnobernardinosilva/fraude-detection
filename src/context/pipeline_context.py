"""
PipelineContext
---------------
Shared state carrier passado por todos os agentes, skills e hooks do pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PipelineContext:
    # ── Dados brutos (entrada) ────────────────────────────────────────
    df_train: pd.DataFrame = field(default=None)
    df_test: Optional[pd.DataFrame] = field(default=None)
    df_customer: Optional[pd.DataFrame] = field(default=None)
    df_terminal: Optional[pd.DataFrame] = field(default=None)

    # ── Dados preparados — camada Silver ─────────────────────────────
    df_train_prepared: Optional[pd.DataFrame] = field(default=None)
    df_test_prepared: Optional[pd.DataFrame] = field(default=None)

    # ── Dados com features — camada Gold (DataFrames) ─────────────────
    df_train_features: Optional[pd.DataFrame] = field(default=None)
    df_test_features: Optional[pd.DataFrame] = field(default=None)

    # ── Matrizes de treino/teste (numpy) ─────────────────────────────
    X_train: Optional[np.ndarray] = field(default=None)
    y_train: Optional[np.ndarray] = field(default=None)
    X_test: Optional[np.ndarray] = field(default=None)
    y_test: Optional[np.ndarray] = field(default=None)

    # ── Informações de features ───────────────────────────────────────
    feature_columns: list = field(default_factory=list)
    feature_iv: dict = field(default_factory=dict)
    feature_importance: dict = field(default_factory=dict)
    feature_correlation: dict = field(default_factory=dict)

    # ── Modelo e experimento ──────────────────────────────────────────
    model: object = field(default=None)
    model_name: Optional[str] = field(default=None)
    threshold: float = field(default=0.5)
    metrics: dict = field(default_factory=dict)
    run_id: Optional[str] = field(default=None)

    # ── Diagnóstico ───────────────────────────────────────────────────
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    # ── Métodos auxiliares ────────────────────────────────────────────

    def add_error(self, source: str, message: str) -> None:
        self.errors.append(f"[{source}] {message}")

    def add_warning(self, source: str, message: str) -> None:
        self.warnings.append(f"[{source}] {message}")

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def is_ready_for_preparation(self) -> bool:
        return self.df_train is not None and self.df_customer is not None and self.df_terminal is not None

    def is_ready_for_feature_engineering(self) -> bool:
        return self.df_train_prepared is not None

    def is_ready_for_logging(self) -> bool:
        return (
            self.model is not None
            and len(self.metrics) > 0
            and len(self.feature_columns) > 0
        )

    def summary(self) -> str:
        lines = ["── PipelineContext Summary ──"]

        def _shape(df):
            if df is None:
                return "None"
            return f"{df.shape[0]:,} linhas × {df.shape[1]} colunas"

        def _arr(arr):
            if arr is None:
                return "None"
            return str(arr.shape)

        lines.append(f"  df_train              : {_shape(self.df_train)}")
        lines.append(f"  df_test               : {_shape(self.df_test)}")
        lines.append(f"  df_customer           : {_shape(self.df_customer)}")
        lines.append(f"  df_terminal           : {_shape(self.df_terminal)}")
        lines.append(f"  df_train_prepared     : {_shape(self.df_train_prepared)}")
        lines.append(f"  df_test_prepared      : {_shape(self.df_test_prepared)}")
        lines.append(f"  df_train_features     : {_shape(self.df_train_features)}")
        lines.append(f"  X_train               : {_arr(self.X_train)}")
        lines.append(f"  y_train               : {_arr(self.y_train)}")
        lines.append(f"  X_test                : {_arr(self.X_test)}")
        lines.append(f"  feature_columns       : {len(self.feature_columns)} features")
        lines.append(f"  model                 : {self.model_name or 'não injetado'}")
        lines.append(f"  metrics               : {self.metrics or 'pendente'}")
        lines.append(f"  errors                : {len(self.errors)}")
        lines.append(f"  warnings              : {len(self.warnings)}")
        return "\n".join(lines)

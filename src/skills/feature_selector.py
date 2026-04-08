"""
Skill: feature_selector
-----------------------
Seleciona as features mais relevantes em três passos:

  1. Filtro adaptativo por esparsidade
     - Feature esparsa  (zeros > sparsity_threshold, padrão 50%):
         usa correlação ponto-bisserial |r| >= corr_threshold (padrão 0.05)
         Motivo: IV é penalizado por bins dominados por zeros, subestimando
         features como taxas de risco calculadas com delay.
     - Feature densa (zeros <= sparsity_threshold):
         usa IV >= iv_threshold (padrão 0.02)
         Referência: IV < 0.02 → inútil | 0.02–0.1 → fraco | 0.1–0.3 → médio | > 0.3 → forte

  2. Remoção de redundância por correlação entre features
     - Calcula matriz de correlação de Pearson entre as features aprovadas no passo 1
     - Para cada par com |r| > redundancy_threshold (padrão 0.85), descarta a de menor IV
     - Garante que features altamente colineares não entrem juntas no modelo

  3. Importância via modelo simples (RandomForest)
     - Treina um RF leve sobre as features selecionadas nos passos 1+2
     - Retorna ranking de importância para referência do humano
     - Não descarta features automaticamente nessa etapa (decisão do humano)
"""

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pointbiserialr
from sklearn.ensemble import RandomForestClassifier


# ── IV ────────────────────────────────────────────────────────────────────

def _compute_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 10) -> float:
    """Calcula o Information Value de uma feature contínua ou discreta."""
    try:
        if df[feature].nunique() > bins:
            df = df.copy()
            df[feature] = pd.qcut(df[feature], q=bins, duplicates="drop")

        grouped = df.groupby(feature)[target].agg(["sum", "count"])
        grouped.columns = ["events", "total"]
        grouped["non_events"] = grouped["total"] - grouped["events"]

        total_events     = grouped["events"].sum()
        total_non_events = grouped["non_events"].sum()

        if total_events == 0 or total_non_events == 0:
            return 0.0

        grouped["dist_events"]     = grouped["events"]     / total_events
        grouped["dist_non_events"] = grouped["non_events"] / total_non_events

        # Evita log(0)
        grouped = grouped[
            (grouped["dist_events"] > 0) & (grouped["dist_non_events"] > 0)
        ]

        grouped["woe"] = np.log(grouped["dist_events"] / grouped["dist_non_events"])
        grouped["iv"]  = (grouped["dist_events"] - grouped["dist_non_events"]) * grouped["woe"]

        return grouped["iv"].sum()

    except Exception:
        return 0.0


# ── Seleção ───────────────────────────────────────────────────────────────

def _compute_correlation(df: pd.DataFrame, feature: str, target: str) -> float:
    """Correlação ponto-bisserial entre feature numérica e target binário."""
    try:
        x = df[feature].fillna(0).astype(float)
        y = df[target].astype(float)
        corr, _ = pointbiserialr(y, x)
        return round(float(corr), 6)
    except Exception:
        return 0.0


def _remove_redundant_features(
    cols: list,
    df: pd.DataFrame,
    iv_dict: dict,
    redundancy_threshold: float,
) -> tuple[list, list]:
    """
    Remove features redundantes por correlação de Pearson entre si.

    Para cada par (i, j) com |r| > redundancy_threshold, descarta a feature
    de menor IV. Retorna (features_mantidas, features_descartadas).
    """
    if len(cols) < 2:
        return cols, []

    corr_matrix = df[cols].fillna(0).corr(method="pearson").abs()

    # Ordena por IV decrescente para sempre manter a mais informativa
    sorted_cols = sorted(cols, key=lambda c: iv_dict.get(c, 0.0), reverse=True)

    kept      = []
    dropped   = set()

    for col in sorted_cols:
        if col in dropped:
            continue
        kept.append(col)
        # Marca como descartadas todas as features altamente correlacionadas
        # com a atual que ainda não foram processadas
        for other in sorted_cols:
            if other == col or other in dropped or other in kept:
                continue
            if corr_matrix.loc[col, other] > redundancy_threshold:
                dropped.add(other)

    return kept, list(dropped)


def select_features(
    df: pd.DataFrame,
    feature_cols: list,
    target: str,
    iv_threshold: float = 0.02,
    corr_threshold: float = 0.05,
    sparsity_threshold: float = 0.50,
    redundancy_threshold: float = 0.85,
    rf_n_estimators: int = 100,
    rf_max_depth: int = 5,
    random_state: int = 42,
) -> tuple[list, dict, dict, dict]:
    """
    Retorna:
      selected_cols     : lista de features selecionadas
      iv_dict           : IV de cada feature candidata
      importance_dict   : importância (RF) das features selecionadas
      correlation_dict  : correlação ponto-bisserial de cada feature candidata com o target
    """

    # ── Passo 1: Correlação e IV para todas as candidatas ─────────────────
    logger.info(
        f"[feature_selector] Calculando correlação e IV para {len(feature_cols)} features..."
    )

    correlation_dict = {}
    iv_dict = {}
    for col in feature_cols:
        correlation_dict[col] = _compute_correlation(df, col, target)
        iv_dict[col] = _compute_iv(df, col, target)

    correlation_dict = dict(sorted(
        correlation_dict.items(), key=lambda x: abs(x[1]), reverse=True
    ))
    iv_dict = dict(sorted(iv_dict.items(), key=lambda x: x[1], reverse=True))

    # ── Passo 2: Filtro adaptativo por esparsidade ────────────────────────
    sparse_cols = []
    dense_cols  = []
    for col in feature_cols:
        zero_ratio = (df[col].fillna(0) == 0).mean()
        if zero_ratio > sparsity_threshold:
            sparse_cols.append(col)
        else:
            dense_cols.append(col)

    logger.info(
        f"[feature_selector] Esparsidade — {len(sparse_cols)} esparsas "
        f"(zeros > {sparsity_threshold:.0%}) | {len(dense_cols)} densas"
    )

    passed_sparse = [
        col for col in sparse_cols
        if abs(correlation_dict[col]) >= corr_threshold
    ]
    passed_dense = [
        col for col in dense_cols
        if iv_dict[col] >= iv_threshold
    ]

    pre_redundancy = passed_sparse + passed_dense

    logger.info(
        f"[feature_selector] Esparsas aprovadas por correlação |r| >= {corr_threshold}: "
        f"{len(passed_sparse)}/{len(sparse_cols)} — {passed_sparse}"
    )
    logger.info(
        f"[feature_selector] Densas aprovadas por IV >= {iv_threshold}: "
        f"{len(passed_dense)}/{len(dense_cols)} — {passed_dense}"
    )
    logger.info(
        f"[feature_selector] Candidatas após filtro target: {len(pre_redundancy)}"
    )

    if not pre_redundancy:
        logger.warning("[feature_selector] Nenhuma feature selecionada.")
        return [], iv_dict, {}, correlation_dict

    # ── Passo 2: Remoção de redundância inter-features ────────────────────
    logger.info(
        f"[feature_selector] Removendo redundância entre features "
        f"(|r_pearson| > {redundancy_threshold})..."
    )
    selected_cols, dropped_redundant = _remove_redundant_features(
        cols=pre_redundancy,
        df=df,
        iv_dict=iv_dict,
        redundancy_threshold=redundancy_threshold,
    )

    logger.info(
        f"[feature_selector] Redundantes removidas ({len(dropped_redundant)}): "
        f"{dropped_redundant}"
    )
    logger.info(
        f"[feature_selector] Total selecionadas: {len(selected_cols)} | "
        f"removidas (total): {len(feature_cols) - len(selected_cols)}"
    )

    if not selected_cols:
        logger.warning("[feature_selector] Nenhuma feature restou após remoção de redundância.")
        return [], iv_dict, {}, correlation_dict

    # ── Passo 3: RandomForest leve para ranking de importância ────────────
    logger.info("[feature_selector] Treinando RF para ranking de importância...")

    X = df[selected_cols].fillna(0).values
    y = df[target].values

    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X, y)

    importance_dict = dict(sorted(
        zip(selected_cols, rf.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    ))

    logger.info(
        f"[feature_selector] Top 5 por importância RF: "
        f"{list(importance_dict.keys())[:5]}"
    )

    return selected_cols, iv_dict, importance_dict, correlation_dict
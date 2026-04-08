"""
Skill: feature_creator
----------------------
Cria todas as features derivadas a partir do DataFrame preparado.
Não remove nem filtra linhas — só adiciona colunas.

Features criadas:
  Temporais       : TX_HOUR, TX_DAY_OF_WEEK, TX_DAY_OF_MONTH, TX_MONTH, TX_YEAR,
                    TX_DURING_WEEKEND, TX_NIGHT_FLAG, PERIODO_DIA, PERIODO_DIA_NUM
  Sequenciais     : TX_TIME_SINCE_LAST_TX, TX_FLAG_SAME_TERMINAL
  Valor           : TX_AMOUNT_LOG, TX_AMOUNT_ROUNDED, TX_ABOVE_MEAN_FLAG
  Risco terminal  : TERM_NB_TX_{1,7,30}D, TERM_RISK_{1,7,30}D (delay=7d)
  Geográfica      : DIST_CUSTOMER_TERMINAL (distância euclidiana)
  Z-score         : TX_AMOUNT_ZSCORE
  Rolling cliente : TX_CUST_{NB_TX,SUM_AMT,MEAN_AMT,MEDIAN_AMT,MIN_AMT,MAX_AMT}_{W}
                    para W ∈ {1H,2H,4H,8H,12H,24H,48H,72H,7D,14D,21D,30D,45D}
                    closed='left' — exclui a transação corrente (sem leakage)
  Comportamentais : TX_CUST_NIGHT_RATIO, TX_CUST_DISTINCT_TERMINALS
  Razões          : RATIO_AMT_VS_CUST_24H, RATIO_AMT_VS_CUST_7D,
                    RATIO_TERM_RISK_1D_7D, RATIO_AMT_VS_GLOBAL_MEAN,
                    RATIO_NB_TX_1H_24H, RATIO_MAX_VS_MEAN_24H,
                    TX_AMOUNT_X_DIST, TX_VELOCITY_24H

Nota: mean_amount, std_amount e mean_nb_tx_per_day já vêm do join com customer.csv.
"""

import numpy as np
import pandas as pd
from loguru import logger


# Janelas de tempo para features rolling por cliente
ROLLING_WINDOWS = [
    ("1H",  "1h"),
    ("2H",  "2h"),
    ("4H",  "4h"),
    ("8H",  "8h"),
    ("12H", "12h"),
    ("24H", "24h"),
    ("48H", "48h"),
    ("72H", "72h"),
    ("7D",  "7D"),
    ("14D", "14D"),
    ("21D", "21D"),
    ("30D", "30D"),
    ("45D", "45D"),
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("TX_DATETIME").reset_index(drop=True)

    df = _time_features(df)
    df = _amount_features(df)
    df = _terminal_risk_features(df)
    df = _geo_features(df)
    df = _amount_zscore(df)
    df = _customer_rolling_features(df)
    df = _behavioral_features(df)
    df = _ratio_features(df)

    logger.info(f"[feature_creator] {df.shape[1]} colunas após criação de features")
    return df


# ── Temporais ──────────────────────────────────────────────────────────────

def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["TX_HOUR"]           = df["TX_DATETIME"].dt.hour
    df["TX_DAY_OF_WEEK"]    = df["TX_DATETIME"].dt.dayofweek   # 0=seg, 6=dom
    df["TX_DAY_OF_MONTH"]   = df["TX_DATETIME"].dt.day
    df["TX_MONTH"]          = df["TX_DATETIME"].dt.month
    df["TX_YEAR"]           = df["TX_DATETIME"].dt.year
    df["TX_DURING_WEEKEND"] = (df["TX_DAY_OF_WEEK"] >= 5).astype(int)

    # Período do dia
    def _periodo(hour: int) -> str:
        if 5 <= hour < 12:
            return "manha"
        elif 12 <= hour < 18:
            return "tarde"
        else:
            return "noite"

    df["PERIODO_DIA"]     = df["TX_HOUR"].apply(_periodo)
    periodo_map           = {"manha": 0, "tarde": 1, "noite": 2}
    df["PERIODO_DIA_NUM"] = df["PERIODO_DIA"].map(periodo_map)

    # Flag noite: 18h–04h59 (alto risco por EDA)
    df["TX_NIGHT_FLAG"] = (~df["TX_HOUR"].between(5, 17)).astype(int)

    logger.info("[feature_creator] Features temporais criadas")
    return df


# ── Valor ──────────────────────────────────────────────────────────────────

def _amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df["TX_AMOUNT_LOG"]     = np.log1p(df["TX_AMOUNT"])
    df["TX_AMOUNT_ROUNDED"] = (df["TX_AMOUNT"] % 1 == 0).astype(int)

    if "mean_amount" in df.columns:
        df["TX_ABOVE_MEAN_FLAG"] = (df["TX_AMOUNT"] > df["mean_amount"]).astype(int)

    logger.info("[feature_creator] Features de valor criadas")
    return df


# ── Risco do terminal ──────────────────────────────────────────────────────

def _terminal_risk_features(
    df: pd.DataFrame,
    windows: list = [1, 7, 30],
    delay: int = 7,
) -> pd.DataFrame:
    """
    Volume e taxa de fraude por terminal, calculados com delay para evitar leakage.
    O delay garante que não usamos transações futuras ao momento da predição.

    Estratégia vetorizada (O(n log n)):
      rolling(delay+w, closed='left') - rolling(delay, closed='left')
      = contagem em [t-(delay+w), t) - contagem em [t-delay, t)
      = contagem em [t-(delay+w), t-delay)

    Isso reproduz exatamente a janela com delay sem usar iterrows.
    """
    has_fraud = "TX_FRAUD" in df.columns

    # Ordena por terminal + tempo para o rolling funcionar corretamente
    original_index = df.index
    df_work = df[
        ["TRANSACTION_ID", "TERMINAL_ID", "TX_DATETIME", "TX_AMOUNT"]
        + (["TX_FRAUD"] if has_fraud else [])
    ].sort_values(["TERMINAL_ID", "TX_DATETIME"]).set_index("TX_DATETIME")

    for w in windows:
        col_nb   = f"TERM_NB_TX_{w}D"
        col_risk = f"TERM_RISK_{w}D"

        outer = f"{delay + w}D"
        inner = f"{delay}D"

        grp_amt = df_work.groupby("TERMINAL_ID", group_keys=False)["TX_AMOUNT"]

        nb_outer = grp_amt.transform(
            lambda x, o=outer: x.rolling(o, closed="left", min_periods=0).count()
        )
        nb_inner = grp_amt.transform(
            lambda x, i=inner: x.rolling(i, closed="left", min_periods=0).count()
        )
        df_work[col_nb] = (nb_outer - nb_inner).clip(lower=0).astype(int)

        if has_fraud:
            grp_fraud = df_work.groupby("TERMINAL_ID", group_keys=False)["TX_FRAUD"]
            fraud_outer = grp_fraud.transform(
                lambda x, o=outer: x.rolling(o, closed="left", min_periods=0).sum()
            )
            fraud_inner = grp_fraud.transform(
                lambda x, i=inner: x.rolling(i, closed="left", min_periods=0).sum()
            )
            fraud_window = (fraud_outer - fraud_inner).clip(lower=0)
            df_work[col_risk] = (fraud_window / df_work[col_nb].replace(0, 1)).fillna(0)
        else:
            df_work[col_risk] = 0.0

        logger.info(f"[feature_creator] Risco terminal {w}d criado (delay={delay}d)")

    # Junta de volta ao df original preservando a ordem
    result_cols = (
        [f"TERM_NB_TX_{w}D" for w in windows]
        + [f"TERM_RISK_{w}D" for w in windows]
    )
    df_work = df_work.reset_index()[["TRANSACTION_ID"] + result_cols]
    df = df.merge(df_work, on="TRANSACTION_ID", how="left")

    return df


# ── Geográfica ─────────────────────────────────────────────────────────────

def _geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distância euclidiana entre a localização habitual do cliente e o terminal."""
    geo_cols = ["x_customer_id", "y_customer_id", "x_terminal_id", "y_terminal_id"]

    if all(c in df.columns for c in geo_cols):
        df["DIST_CUSTOMER_TERMINAL"] = np.sqrt(
            (df["x_customer_id"] - df["x_terminal_id"]) ** 2 +
            (df["y_customer_id"] - df["y_terminal_id"]) ** 2
        )
        logger.info("[feature_creator] Distância cliente-terminal criada")
    else:
        logger.warning("[feature_creator] Colunas geográficas ausentes — distância não criada")

    return df


# ── Z-score de valor ───────────────────────────────────────────────────────

def _amount_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Desvio do valor relativo ao histórico do cliente (mean/std do customer.csv)."""
    if all(c in df.columns for c in ["mean_amount", "std_amount"]):
        df["TX_AMOUNT_ZSCORE"] = (
            (df["TX_AMOUNT"] - df["mean_amount"]) /
            df["std_amount"].replace(0, 1)
        )
        logger.info("[feature_creator] Z-score de TX_AMOUNT criado")
    else:
        logger.warning("[feature_creator] mean_amount/std_amount ausentes — z-score não criado")

    return df


# ── Rolling por cliente ────────────────────────────────────────────────────

def _customer_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada janela temporal, calcula por cliente:
      - Contagem de transações (NB_TX)
      - Soma, média, mediana, mínimo e máximo de TX_AMOUNT

    Usa closed='left' no rolling: a janela é [t-w, t), excluindo a transação corrente.
    Isso evita leakage — cada feature reflete apenas o histórico anterior.

    Internamente ordena por [CUSTOMER_ID, TX_DATETIME] para o rolling por grupo,
    depois restaura a ordem original (global por TX_DATETIME).
    """
    # Ordem original: o índice 0..n-1 reflete sort global por TX_DATETIME
    df_sorted = df.sort_values(["CUSTOMER_ID", "TX_DATETIME"]).copy()
    df_indexed = df_sorted.set_index("TX_DATETIME")

    for w_name, w_str in ROLLING_WINDOWS:
        grp = df_indexed.groupby("CUSTOMER_ID", group_keys=False)["TX_AMOUNT"]

        df_sorted[f"TX_CUST_NB_TX_{w_name}"] = (
            grp.transform(lambda x, w=w_str: x.rolling(w, closed="left", min_periods=0).count())
            .values
        )
        df_sorted[f"TX_CUST_SUM_AMT_{w_name}"] = (
            grp.transform(lambda x, w=w_str: x.rolling(w, closed="left", min_periods=0).sum().fillna(0))
            .values
        )
        df_sorted[f"TX_CUST_MEAN_AMT_{w_name}"] = (
            grp.transform(lambda x, w=w_str: x.rolling(w, closed="left", min_periods=0).mean().fillna(0))
            .values
        )
        df_sorted[f"TX_CUST_MIN_AMT_{w_name}"] = (
            grp.transform(lambda x, w=w_str: x.rolling(w, closed="left", min_periods=0).min().fillna(0))
            .values
        )
        df_sorted[f"TX_CUST_MAX_AMT_{w_name}"] = (
            grp.transform(lambda x, w=w_str: x.rolling(w, closed="left", min_periods=0).max().fillna(0))
            .values
        )

        try:
            df_sorted[f"TX_CUST_MEDIAN_AMT_{w_name}"] = (
                grp.transform(lambda x, w=w_str: x.rolling(w, closed="left", min_periods=0).median().fillna(0))
                .values
            )
        except Exception:
            # Fallback se a versão do pandas não suportar median em rolling temporal
            df_sorted[f"TX_CUST_MEDIAN_AMT_{w_name}"] = df_sorted[f"TX_CUST_MEAN_AMT_{w_name}"]

        logger.info(f"[feature_creator] Rolling cliente {w_name} criado")

    # Restaura ordem original pelo índice inteiro (global sort por TX_DATETIME)
    return df_sorted.sort_index()


# ── Comportamentais ────────────────────────────────────────────────────────

def _count_distinct_expanding(series: pd.Series) -> pd.Series:
    """
    Para cada posição i, conta quantos valores distintos existem em series[0:i].
    Equivale a expanding().nunique() com shift(1), mas em O(n) via set.
    """
    seen: set = set()
    counts = []
    for val in series:
        counts.append(len(seen))   # conta ANTES de registrar a transação corrente
        seen.add(val)
    return pd.Series(counts, index=series.index)


def _behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features comportamentais sequenciais do cliente.
    Todas calculadas sem incluir a transação corrente (sem leakage).
    """
    df_sorted = df.sort_values(["CUSTOMER_ID", "TX_DATETIME"]).copy()

    # 1. Tempo desde a última transação do mesmo cliente (em minutos)
    df_sorted["TX_TIME_SINCE_LAST_TX"] = (
        df_sorted.groupby("CUSTOMER_ID")["TX_DATETIME"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .fillna(0)
    )

    # 2. Flag: transação consecutiva no mesmo terminal do cliente
    df_sorted["TX_FLAG_SAME_TERMINAL"] = (
        df_sorted.groupby("CUSTOMER_ID")["TERMINAL_ID"]
        .transform(lambda x: (x == x.shift(1)).astype(int))
        .fillna(0)
        .astype(int)
    )

    # 3. Proporção histórica de transações noturnas do cliente (expanding, sem leakage)
    if "TX_NIGHT_FLAG" in df_sorted.columns:
        df_sorted["TX_CUST_NIGHT_RATIO"] = (
            df_sorted.groupby("CUSTOMER_ID")["TX_NIGHT_FLAG"]
            .transform(lambda x: x.expanding().mean().shift(1))
            .fillna(0)
        )

    # 4. Número de terminais distintos já utilizados pelo cliente (expanding, sem leakage)
    df_sorted["TX_CUST_DISTINCT_TERMINALS"] = (
        df_sorted.groupby("CUSTOMER_ID")["TERMINAL_ID"]
        .transform(_count_distinct_expanding)
        .astype(int)
    )

    logger.info("[feature_creator] Features comportamentais criadas")
    return df_sorted.sort_index()


# ── Razões entre features ──────────────────────────────────────────────────

def _ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Razões entre features para capturar desvios relativos.
    Todas as divisões usam +eps para evitar divisão por zero.
    """
    eps = 1e-6

    # Valor da transação vs média do cliente nas últimas 24h
    if "TX_CUST_MEAN_AMT_24H" in df.columns:
        df["RATIO_AMT_VS_CUST_24H"] = df["TX_AMOUNT"] / (df["TX_CUST_MEAN_AMT_24H"] + eps)

    # Valor da transação vs média do cliente nos últimos 7d
    if "TX_CUST_MEAN_AMT_7D" in df.columns:
        df["RATIO_AMT_VS_CUST_7D"] = df["TX_AMOUNT"] / (df["TX_CUST_MEAN_AMT_7D"] + eps)

    # Risco do terminal: curto prazo vs longo prazo (spike súbito de risco)
    if all(c in df.columns for c in ["TERM_RISK_1D", "TERM_RISK_7D"]):
        df["RATIO_TERM_RISK_1D_7D"] = df["TERM_RISK_1D"] / (df["TERM_RISK_7D"] + eps)

    # Valor vs média histórica global do cliente (do customer.csv)
    if "mean_amount" in df.columns:
        df["RATIO_AMT_VS_GLOBAL_MEAN"] = df["TX_AMOUNT"] / (df["mean_amount"] + eps)

    # Intensidade recente: transações na última 1h vs últimas 24h
    if all(c in df.columns for c in ["TX_CUST_NB_TX_1H", "TX_CUST_NB_TX_24H"]):
        df["RATIO_NB_TX_1H_24H"] = df["TX_CUST_NB_TX_1H"] / (df["TX_CUST_NB_TX_24H"] + eps)

    # Pico de valor: máximo vs média na última 24h
    if all(c in df.columns for c in ["TX_CUST_MAX_AMT_24H", "TX_CUST_MEAN_AMT_24H"]):
        df["RATIO_MAX_VS_MEAN_24H"] = df["TX_CUST_MAX_AMT_24H"] / (df["TX_CUST_MEAN_AMT_24H"] + eps)

    # Valor ponderado pela distância (transação cara em terminal distante = suspeito)
    if "DIST_CUSTOMER_TERMINAL" in df.columns:
        df["TX_AMOUNT_X_DIST"] = df["TX_AMOUNT"] * np.log1p(df["DIST_CUSTOMER_TERMINAL"])

    # Velocidade de transações nas últimas 24h (transações/hora)
    if all(c in df.columns for c in ["TX_CUST_NB_TX_24H", "TX_TIME_SINCE_LAST_TX"]):
        # TX_TIME_SINCE_LAST_TX em minutos → converte para horas
        df["TX_VELOCITY_24H"] = df["TX_CUST_NB_TX_24H"] / (
            df["TX_TIME_SINCE_LAST_TX"].clip(lower=1).div(60)
        )

    logger.info("[feature_creator] Features de razão criadas")
    return df

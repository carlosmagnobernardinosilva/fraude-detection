"""
Skill: feature_explorer
-----------------------
Usa um LLM (Groq — llama-3.3-70b-versatile) para sugerir hipóteses de novas
features com base no schema e contexto de negócio do dataset.

Requer a variável de ambiente GROQ_API_KEY.

Fluxo:
  1. Monta um prompt com schema do DataFrame + contexto de negócio completo
  2. O LLM sugere N hipóteses de features com justificativa e código Python
  3. Para cada hipótese, tenta criar a feature nos dados reais (exec com sandbox)
  4. Calcula o IV de cada feature criada
  5. Retorna apenas as que passaram no threshold de IV
  6. Humano revisa o relatório antes de aprovar para o modelo

A decisão final de usar ou não cada feature é sempre do humano.
"""

import json
import os
import re

import numpy as np
import pandas as pd
import requests
from loguru import logger


# ── Contexto de negócio injetado no prompt ─────────────────────────────────

BUSINESS_CONTEXT = """
Você é um especialista sênior em detecção de fraudes em cartões de crédito.
Estamos trabalhando com transações realizadas em terminais físicos (POS).

Schema do DataFrame (após join de train + customer + terminal + feature_creator):
{schema}

Informações relevantes do negócio:
- Taxa de fraude global: ~2,26% (classes muito desbalanceadas)
- Período: agosto a dezembro de 2021
- 998 clientes, 1994 terminais, ~291k transações
- Fraudes são mais frequentes à noite e em terminais comprometidos
- mean_amount e std_amount representam o comportamento histórico do cliente (de customer.csv)
- x_customer_id / y_customer_id e x_terminal_id / y_terminal_id são coordenadas geográficas
- Padrões conhecidos de fraude neste domínio:
  * Velocity attacks: muitas transações em curto intervalo
  * Card testing: transações pequenas antes de uma grande
  * Terminal comprometido: muitas fraudes no mesmo terminal
  * Account takeover: mudança súbita no padrão comportamental do cliente
  * Geographical anomaly: uso do cartão em local incomum para o cliente

Features já criadas pelo feature_creator (NÃO sugira estas):
{existing_features}

Sugira exatamente {n_suggestions} hipóteses de novas features que:
- Não sejam trivialmente deriváveis das features acima
- Capturem comportamentos comportamentais, de negócio ou anomalias relevantes
- Incluam razões/proporções entre features existentes que ainda não estejam listadas
- Explorem interações entre tempo, valor, localização e risco do terminal

Para cada hipótese:
1. Dê um nome em UPPER_SNAKE_CASE (ex: RATIO_AMOUNT_TERM_MEAN_7D)
2. Explique em 1 frase por que essa feature pode indicar fraude
3. Forneça o código Python exato para criar a feature em um DataFrame chamado `df`
   - Use apenas pandas (pd) e numpy (np)
   - A feature deve resultar em uma coluna numérica
   - Não use loops — prefira operações vetorizadas
   - `df` já contém TODAS as colunas do schema acima
   - Proteja divisões com + 1e-6 para evitar divisão por zero

Responda SOMENTE em JSON válido, sem texto fora do JSON, neste formato:
[
  {{
    "name": "NOME_DA_FEATURE",
    "justification": "Por que essa feature indica fraude",
    "code": "df['NOME_DA_FEATURE'] = <expressão pandas/numpy>"
  }}
]
"""


# ── Chamada ao LLM ─────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """Chama a API da Groq (llama-3.3-70b-versatile) e retorna o texto da resposta."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY não encontrada. "
            "Defina a variável de ambiente antes de usar feature_explorer."
        )

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        },
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(
            f"Groq API error {response.status_code}: {response.text}"
        )
    return response.json()["choices"][0]["message"]["content"]


# ── Extrai JSON da resposta ────────────────────────────────────────────────

def _parse_suggestions(raw: str) -> list[dict]:
    """Extrai e valida o JSON retornado pelo LLM."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError("LLM não retornou JSON válido.")
    return json.loads(match.group())


# ── Tenta criar a feature nos dados reais ─────────────────────────────────

def _try_create_feature(df: pd.DataFrame, suggestion: dict) -> pd.Series | None:
    """
    Executa o código sugerido pelo LLM em ambiente controlado.
    Retorna a Series criada ou None se falhar.
    """
    name = suggestion["name"]
    code = suggestion["code"]

    try:
        local_df = df.copy()
        exec(code, {"__builtins__": {}, "df": local_df, "np": np, "pd": pd})
        if name not in local_df.columns:
            logger.warning(f"[feature_explorer] '{name}' não foi criada pelo código gerado.")
            return None
        series = local_df[name]
        if not pd.api.types.is_numeric_dtype(series):
            logger.warning(f"[feature_explorer] '{name}' não é numérica — ignorada.")
            return None
        return series
    except Exception as e:
        logger.warning(f"[feature_explorer] Erro ao criar '{name}': {e}")
        return None


# ── IV simples ────────────────────────────────────────────────────────────

def _quick_iv(series: pd.Series, target: pd.Series, bins: int = 10) -> float:
    """Calcula IV rápido para avaliar o poder preditivo da feature gerada."""
    try:
        df_tmp = pd.DataFrame({"feat": series, "target": target}).dropna()
        if df_tmp["feat"].nunique() > bins:
            df_tmp["feat"] = pd.qcut(df_tmp["feat"], q=bins, duplicates="drop")

        grouped = df_tmp.groupby("feat")["target"].agg(["sum", "count"])
        grouped.columns = ["events", "total"]
        grouped["non_events"] = grouped["total"] - grouped["events"]

        te = grouped["events"].sum()
        tn = grouped["non_events"].sum()
        if te == 0 or tn == 0:
            return 0.0

        grouped["de"] = grouped["events"] / te
        grouped["dn"] = grouped["non_events"] / tn
        grouped = grouped[(grouped["de"] > 0) & (grouped["dn"] > 0)]
        grouped["woe"] = np.log(grouped["de"] / grouped["dn"])
        grouped["iv"]  = (grouped["de"] - grouped["dn"]) * grouped["woe"]
        return grouped["iv"].sum()
    except Exception:
        return 0.0


# ── Função principal ──────────────────────────────────────────────────────

def explore_features(
    df: pd.DataFrame,
    target: str = "TX_FRAUD",
    existing_features: list = None,
    n_suggestions: int = 12,
    iv_threshold: float = 0.02,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Parâmetros:
        df                : DataFrame preparado (após feature_creator)
        target            : nome da coluna alvo
        existing_features : features já criadas pelo feature_creator
        n_suggestions     : quantas hipóteses pedir ao LLM
        iv_threshold      : IV mínimo para aprovar a feature

    Retorna:
        df                : DataFrame com as novas features aprovadas adicionadas
        report            : lista com detalhes de cada hipótese (para revisão humana)
    """
    existing_features = existing_features or []

    # ── 1. Monta schema para o prompt ─────────────────────────────────────
    schema_lines = [f"  - {col}: {str(dtype)}" for col, dtype in df.dtypes.items()]
    schema_str   = "\n".join(schema_lines)

    prompt = BUSINESS_CONTEXT.format(
        schema=schema_str,
        existing_features=", ".join(existing_features) if existing_features else "nenhuma ainda",
        n_suggestions=n_suggestions,
    )

    # ── 2. Chama o LLM ────────────────────────────────────────────────────
    logger.info(f"[feature_explorer] Solicitando {n_suggestions} hipóteses ao LLM...")
    raw_response = _call_llm(prompt)
    suggestions  = _parse_suggestions(raw_response)
    logger.info(f"[feature_explorer] {len(suggestions)} hipóteses recebidas")

    # ── 3. Testa cada hipótese ────────────────────────────────────────────
    report = []
    df_out = df.copy()

    for suggestion in suggestions:
        name          = suggestion["name"]
        justification = suggestion["justification"]
        code          = suggestion["code"]

        series = _try_create_feature(df_out, suggestion)

        if series is None:
            report.append({
                "name": name,
                "justification": justification,
                "code": code,
                "iv": None,
                "status": "ERRO — feature não pôde ser criada",
                "approved": False,
            })
            continue

        iv = _quick_iv(series, df_out[target])

        approved = iv >= iv_threshold
        df_out[name] = series

        status = f"APROVADA (IV={iv:.4f})" if approved else f"REPROVADA (IV={iv:.4f} < {iv_threshold})"
        logger.info(f"[feature_explorer] {name}: {status}")

        report.append({
            "name": name,
            "justification": justification,
            "code": code,
            "iv": round(iv, 4),
            "status": status,
            "approved": approved,
        })

    # Remove features reprovadas do DataFrame de saída
    reprovadas = [r["name"] for r in report if not r["approved"] and r["name"] in df_out.columns]
    df_out = df_out.drop(columns=reprovadas, errors="ignore")

    approved_count = sum(1 for r in report if r["approved"])
    logger.info(
        f"[feature_explorer] Concluído — "
        f"{approved_count}/{len(suggestions)} features aprovadas pelo IV"
    )

    return df_out, report


# ── Utilitário: imprime o relatório para revisão humana ───────────────────

def print_exploration_report(report: list[dict]) -> None:
    sep  = "=" * 60
    line = "-" * 60
    print(f"\n{sep}")
    print("  RELATORIO DE EXPLORACAO DE FEATURES")
    print(sep)
    for item in report:
        print(f"\n{line}")
        print(f"  Feature    : {item['name']}")
        print(f"  Hipotese   : {item['justification']}")
        print(f"  IV         : {item['iv']}")
        print(f"  Status     : {item['status']}")
        print(f"  Codigo     : {item['code']}")
    print(f"\n{sep}")
    print("  Revise as features aprovadas antes de avancar para o modelo.")
    print(f"{sep}\n")

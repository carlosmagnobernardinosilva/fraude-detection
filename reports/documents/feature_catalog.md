# Catálogo de Features — Fraud Detection

Gerado em: 2026-04-01
Dataset: 291.222 transações | 998 clientes | 1.994 terminais | ~2,26% fraude
Pipeline: `feature_creator` (negócio) + `feature_explorer` (LLM/Groq)
Seleção: `feature_selector` com filtro adaptativo por esparsidade → IV ou correlação ponto-bisserial

**Critérios de seleção:**
- Feature **esparsa** (> 50% zeros): aprovada se `|corr_pbs| ≥ 0.05`
- Feature **densa** (≤ 50% zeros): aprovada se `IV ≥ 0.02`
- Ranking final por RandomForest (não descarta — é referência para o humano)

**Legenda IV:**  `< 0.02` inútil | `0.02–0.1` fraco | `0.1–0.3` médio | `> 0.3` forte

---

## 1. Features Temporais

| Feature | Descrição | IV | Corr PBS | Status |
|---|---|---|---|---|
| `TX_HOUR` | Hora do dia da transação (0–23) | baixo | baixo | REJEITADA (densa, IV < 0.02) |
| `TX_DAY_OF_WEEK` | Dia da semana (0=seg, 6=dom) | baixo | baixo | REJEITADA (densa, IV < 0.02) |
| `TX_DAY_OF_MONTH` | Dia do mês (1–31) | baixo | baixo | REJEITADA (densa, IV < 0.02) |
| `TX_MONTH` | Mês da transação (1–12) | baixo | baixo | REJEITADA (densa, IV < 0.02) |
| `TX_YEAR` | Ano da transação | baixo | baixo | REJEITADA (densa, IV < 0.02) |
| `TX_DURING_WEEKEND` | Flag: transação ocorreu no fim de semana (1/0) | baixo | baixo | REJEITADA (densa, IV < 0.02) |
| `TX_NIGHT_FLAG` | Flag: transação ocorreu entre 22h e 6h (1/0) | baixo | baixo | REJEITADA (densa, IV < 0.02) |
| `PERIODO_DIA` | Categoria textual do período: madrugada/manhã/tarde/noite | — | — | Excluída do modelo (categórica/identificador) |
| `PERIODO_DIA_NUM` | Período do dia codificado numericamente (0–3) | baixo | baixo | REJEITADA (densa, IV < 0.02) |

> **Por que temporais foram rejeitadas?** Fraude ocorre em todos os horários e dias. O padrão relevante não é o horário absoluto, mas o desvio do comportamento habitual do cliente — capturado pelas features de rolling window e ratios.

---

## 2. Features de Valor (TX_AMOUNT)

| Feature | Descrição | IV | Status |
|---|---|---|---|
| `TX_AMOUNT` | Valor bruto da transação em EUR | baixo | REJEITADA (densa, IV < 0.02) |
| `TX_AMOUNT_LOG` | `log1p(TX_AMOUNT)` — normaliza a distribuição assimétrica do valor | baixo | REJEITADA (densa, IV < 0.02) |
| `TX_AMOUNT_ROUNDED` | Flag: valor é múltiplo de 10 (possível card testing) | baixo | REJEITADA (densa, IV < 0.02) |
| `TX_AMOUNT_ZSCORE` | Z-score do valor do cliente vs. sua média histórica (`mean_amount`) | baixo | REJEITADA (densa, IV < 0.02) |
| `TX_ABOVE_MEAN_FLAG` | Flag: valor acima da média histórica do cliente (1/0) | baixo | REJEITADA (densa, IV < 0.02) |

> **Por que valor bruto foi rejeitado?** O valor absoluto tem pouco poder preditivo isolado. O que importa é o valor *relativo* ao comportamento do cliente — capturado pelos ratios e rolling windows.

---

## 3. Features de Risco do Terminal

> Calculadas com **delay de 7 dias** para evitar data leakage (fraude confirmada só após investigação).

| Feature | Descrição | IV | Status |
|---|---|---|---|
| `TERM_NB_TX_1D` | Nº de transações no terminal nos últimos 1 dia | esparsa | REJEITADA (|corr| < 0.05) |
| `TERM_NB_TX_7D` | Nº de transações no terminal nos últimos 7 dias | esparsa | REJEITADA (|corr| < 0.05) |
| `TERM_NB_TX_30D` | Nº de transações no terminal nos últimos 30 dias | esparsa | **SELECIONADA** |
| `TERM_RISK_1D` | Taxa de fraude no terminal nos últimos 1 dia (com delay 7d) | esparsa | **SELECIONADA** |
| `TERM_RISK_7D` | Taxa de fraude no terminal nos últimos 7 dias (com delay 7d) | esparsa | **SELECIONADA** |
| `TERM_RISK_30D` | Taxa de fraude no terminal nos últimos 30 dias (com delay 7d) | esparsa | **SELECIONADA** |

> **Por que `TERM_NB_TX_1D/7D` foram rejeitadas?** São esparsas (muitos terminais com zero transações na janela curta) e a correlação ponto-bisserial ficou abaixo de 0.05. O volume de transações em si não indica fraude — o que indica é a *taxa* de fraude.

---

## 4. Features Sequenciais

| Feature | Descrição | IV | Status |
|---|---|---|---|
| `TX_TIME_SINCE_LAST_TX` | Minutos desde a última transação do mesmo cliente | esparsa | **SELECIONADA** |
| `TX_FLAG_SAME_TERMINAL` | Flag: mesmo terminal da transação anterior do cliente (1/0) | baixo | REJEITADA (densa, IV < 0.02) |

---

## 5. Features Geográficas

| Feature | Descrição | IV | Status |
|---|---|---|---|
| `DIST_CUSTOMER_TERMINAL` | Distância euclidiana entre coordenadas do cliente e do terminal | baixo | REJEITADA (densa, IV < 0.02) |
| `TX_AMOUNT_X_DIST` | Interação: `TX_AMOUNT × DIST_CUSTOMER_TERMINAL` | baixo | REJEITADA (densa, IV < 0.02) |

> **Observação:** A distância euclidiana entre coordenadas sintéticas tem poder preditivo limitado neste dataset. Em dados reais de GPS poderia ser muito mais relevante.

---

## 6. Features de Rolling Window por Cliente

> Janelas calculadas com `closed='left'` (sem look-ahead). Para cada janela W ∈ {1H, 2H, 4H, 8H, 12H, 24H, 48H, 72H, 7D, 14D, 21D, 30D, 45D}.

### 6.1 Contagem de transações (`TX_CUST_NB_TX_W`)

| Feature | Selecionada? | Motivo de rejeição |
|---|---|---|
| `TX_CUST_NB_TX_1H` | **Sim** | — |
| `TX_CUST_NB_TX_2H` | **Sim** | — |
| `TX_CUST_NB_TX_4H` | **Sim** | — |
| `TX_CUST_NB_TX_8H` | **Sim** | — |
| `TX_CUST_NB_TX_12H` | **Sim** | — |
| `TX_CUST_NB_TX_24H` | **Sim** | — |
| `TX_CUST_NB_TX_48H` | **Sim** | — |
| `TX_CUST_NB_TX_72H` | **Sim** | — |
| `TX_CUST_NB_TX_7D` | **Sim** | — |
| `TX_CUST_NB_TX_14D` | **Sim** | — |
| `TX_CUST_NB_TX_21D` | **Sim** | — |
| `TX_CUST_NB_TX_30D` | **Sim** | — |
| `TX_CUST_NB_TX_45D` | Não | REJEITADA (densa, IV < 0.02) |

### 6.2 Soma do valor (`TX_CUST_SUM_AMT_W`)

| Feature | Selecionada? | Motivo de rejeição |
|---|---|---|
| `TX_CUST_SUM_AMT_1H` | **Sim** | — |
| `TX_CUST_SUM_AMT_2H` | **Sim** | — |
| `TX_CUST_SUM_AMT_4H` | **Sim** | — |
| `TX_CUST_SUM_AMT_8H` | **Sim** | — |
| `TX_CUST_SUM_AMT_12H` | **Sim** | — |
| `TX_CUST_SUM_AMT_24H` | **Sim** | — |
| `TX_CUST_SUM_AMT_48H` | **Sim** | — |
| `TX_CUST_SUM_AMT_72H` | **Sim** | — |
| `TX_CUST_SUM_AMT_7D` | **Sim** | — |
| `TX_CUST_SUM_AMT_14D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_SUM_AMT_21D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_SUM_AMT_30D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_SUM_AMT_45D` | Não | REJEITADA (densa, IV < 0.02) |

### 6.3 Média do valor (`TX_CUST_MEAN_AMT_W`)

| Feature | Selecionada? | Motivo de rejeição |
|---|---|---|
| `TX_CUST_MEAN_AMT_1H` | **Sim** | — |
| `TX_CUST_MEAN_AMT_2H` | **Sim** | — |
| `TX_CUST_MEAN_AMT_4H` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEAN_AMT_8H` | **Sim** | — |
| `TX_CUST_MEAN_AMT_12H` | **Sim** | — |
| `TX_CUST_MEAN_AMT_24H` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEAN_AMT_48H` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEAN_AMT_72H` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEAN_AMT_7D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEAN_AMT_14D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEAN_AMT_21D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEAN_AMT_30D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEAN_AMT_45D` | Não | REJEITADA (densa, IV < 0.02) |

### 6.4 Máximo do valor (`TX_CUST_MAX_AMT_W`)

| Feature | Selecionada? | Motivo de rejeição |
|---|---|---|
| `TX_CUST_MAX_AMT_1H` | **Sim** | — |
| `TX_CUST_MAX_AMT_2H` | **Sim** | — |
| `TX_CUST_MAX_AMT_4H` | **Sim** | — |
| `TX_CUST_MAX_AMT_8H` | **Sim** | — |
| `TX_CUST_MAX_AMT_12H` | **Sim** | — |
| `TX_CUST_MAX_AMT_24H` | **Sim** | — |
| `TX_CUST_MAX_AMT_48H` | **Sim** | — |
| `TX_CUST_MAX_AMT_72H` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MAX_AMT_7D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MAX_AMT_14D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MAX_AMT_21D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MAX_AMT_30D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MAX_AMT_45D` | Não | REJEITADA (densa, IV < 0.02) |

### 6.5 Mediana do valor (`TX_CUST_MEDIAN_AMT_W`)

| Feature | Selecionada? | Motivo de rejeição |
|---|---|---|
| `TX_CUST_MEDIAN_AMT_1H` | **Sim** | — |
| `TX_CUST_MEDIAN_AMT_2H` | **Sim** | — |
| `TX_CUST_MEDIAN_AMT_4H` | **Sim** | — |
| `TX_CUST_MEDIAN_AMT_8H` | **Sim** | — |
| `TX_CUST_MEDIAN_AMT_12H` | **Sim** | — |
| `TX_CUST_MEDIAN_AMT_24H` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEDIAN_AMT_48H` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEDIAN_AMT_72H` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEDIAN_AMT_7D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEDIAN_AMT_14D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEDIAN_AMT_21D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEDIAN_AMT_30D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MEDIAN_AMT_45D` | Não | REJEITADA (densa, IV < 0.02) |

### 6.6 Mínimo do valor (`TX_CUST_MIN_AMT_W`)

| Feature | Selecionada? | Motivo de rejeição |
|---|---|---|
| `TX_CUST_MIN_AMT_1H` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MIN_AMT_2H` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MIN_AMT_4H` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MIN_AMT_8H` | **Sim** | — |
| `TX_CUST_MIN_AMT_12H` | **Sim** | — |
| `TX_CUST_MIN_AMT_24H` | **Sim** | — |
| `TX_CUST_MIN_AMT_48H` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MIN_AMT_72H` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MIN_AMT_7D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MIN_AMT_14D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MIN_AMT_21D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MIN_AMT_30D` | Não | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_MIN_AMT_45D` | Não | REJEITADA (densa, IV < 0.02) |

> **Padrão geral das rolling windows:** Janelas curtas (1H–48H) têm mais poder preditivo do que janelas longas (14D+). Fraude tende a ser detectada por *aceleração* recente no comportamento, não por médias de longo prazo. Janelas longas têm IV baixo porque normalizam o comportamento fraudulento com períodos normais.

---

## 7. Features Comportamentais (Expanding)

| Feature | Descrição | IV | Status |
|---|---|---|---|
| `TX_CUST_NIGHT_RATIO` | Proporção de transações noturnas do cliente (expanding, sem leakage) | baixo | REJEITADA (densa, IV < 0.02) |
| `TX_CUST_DISTINCT_TERMINALS` | Número de terminais distintos usados pelo cliente (expanding) | baixo | REJEITADA (densa, IV < 0.02) |

---

## 8. Features de Ratio e Interação

| Feature | Descrição | IV | Status |
|---|---|---|---|
| `RATIO_AMT_VS_CUST_24H` | `TX_AMOUNT / TX_CUST_MEAN_AMT_24H` — valor vs. média do cliente 24h | baixo | REJEITADA (densa, IV < 0.02) |
| `RATIO_AMT_VS_CUST_7D` | `TX_AMOUNT / TX_CUST_MEAN_AMT_7D` — valor vs. média do cliente 7d | baixo | REJEITADA (densa, IV < 0.02) |
| `RATIO_TERM_RISK_1D_7D` | `TERM_RISK_1D / TERM_RISK_7D` — aceleração de risco no terminal | esparsa | **SELECIONADA** |
| `RATIO_AMT_VS_GLOBAL_MEAN` | `TX_AMOUNT / média global` — valor relativo ao mercado | baixo | REJEITADA (densa, IV < 0.02) |
| `RATIO_NB_TX_1H_24H` | `TX_CUST_NB_TX_1H / TX_CUST_NB_TX_24H` — concentração de transações na última hora | baixo | REJEITADA (densa, IV < 0.02) |
| `RATIO_MAX_VS_MEAN_24H` | `TX_CUST_MAX_AMT_24H / TX_CUST_MEAN_AMT_24H` — pico vs. média 24h | densa | **SELECIONADA** |
| `TX_AMOUNT_X_DIST` | `TX_AMOUNT × DIST_CUSTOMER_TERMINAL` | baixo | REJEITADA (densa, IV < 0.02) |
| `TX_VELOCITY_24H` | `TX_CUST_NB_TX_24H / TX_CUST_SUM_AMT_24H` — frequência por valor gasto | densa | **SELECIONADA** |

---

## 9. Features do LLM (feature_explorer via Groq llama-3.3-70b)

| Feature | Descrição | IV | Status |
|---|---|---|---|
| `RATIO_TERM_RISK_30D_VS_7D` | `TERM_RISK_30D / TERM_RISK_7D` — compara risco de longo prazo com curto prazo no terminal. Alto valor indica que o terminal tem histórico crônico de fraude, não apenas um episódio recente | **0.6059** | **SELECIONADA** |
| `RATIO_TX_AMOUNT_VS_TERM_RISK_1D` | `TX_AMOUNT / (TERM_RISK_1D + ε)` — valor da transação ponderado pelo risco imediato do terminal. Captura transações de alto valor em terminais recentemente comprometidos | **0.3585** | **SELECIONADA** |
| `RATIO_TX_TIME_SINCE_LAST_TX_VS_CUST_MEAN` | `TX_TIME_SINCE_LAST_TX / (TX_CUST_MEAN_AMT_1H + ε)` — tempo desde última transação relativizado pelo comportamento recente do cliente | **0.1936** | **SELECIONADA** |
| `RATIO_TX_AMOUNT_VS_CUST_MAX_24H` | `TX_AMOUNT / (TX_CUST_MAX_AMT_24H + ε)` — valor atual vs. máximo do cliente nas últimas 24h. Detecta quando a transação corrente supera o pico recente | **0.0363** | **SELECIONADA** |
| `RATIO_TX_AMOUNT_VS_CUST_MEAN_30D` | `TX_AMOUNT / TX_CUST_MEAN_AMT_30D` — valor vs. média de longo prazo do cliente | 0.0046 | REJEITADA (IV < 0.02) |
| `RATIO_TX_CUST_NIGHT_RATIO_VS_GLOBAL` | Proporção de transações noturnas do cliente vs. média global | 0.0047 | REJEITADA (IV < 0.02) |
| `RATIO_TERM_NB_TX_1D_VS_7D` | `TERM_NB_TX_1D / TERM_NB_TX_7D` — concentração de volume no terminal no curto prazo | 0.0033 | REJEITADA (IV < 0.02) |
| `RATIO_TX_CUST_DISTINCT_TERMINALS_VS_GLOBAL` | Terminais distintos do cliente / total de terminais | 0.0068 | REJEITADA (IV < 0.02) |

---

## 10. Colunas de Identificação / Raw (excluídas do modelo)

| Coluna | Descrição |
|---|---|
| `TRANSACTION_ID` | Identificador único da transação |
| `TX_DATETIME` | Timestamp completo |
| `CUSTOMER_ID` | ID do cliente |
| `TERMINAL_ID` | ID do terminal |
| `TX_FRAUD` | **Target** — 1 se fraudulenta, 0 caso contrário |
| `mean_amount` | Média histórica do valor de compra do cliente (de customer.csv) |
| `std_amount` | Desvio padrão histórico do valor de compra do cliente |
| `mean_nb_tx_per_day` | Média de transações por dia do cliente |
| `x_customer_id` / `y_customer_id` | Coordenadas geográficas do cliente |
| `x_terminal_id` / `y_terminal_id` | Coordenadas geográficas do terminal |
| `month_reference` | Mês de referência (usado no join) |

---

## Resumo

| Categoria | Geradas | Selecionadas |
|---|---|---|
| Temporais | 9 | 0 |
| Valor | 5 | 0 |
| Risco terminal | 6 | 4 |
| Sequenciais | 2 | 1 |
| Geográficas | 2 | 0 |
| Rolling cliente (NB_TX) | 13 | 12 |
| Rolling cliente (SUM_AMT) | 13 | 8 |
| Rolling cliente (MEAN_AMT) | 13 | 3 |
| Rolling cliente (MAX_AMT) | 13 | 7 |
| Rolling cliente (MEDIAN_AMT) | 13 | 5 |
| Rolling cliente (MIN_AMT) | 13 | 3 |
| Comportamentais | 2 | 0 |
| Ratio / Interação | 8 | 3 |
| LLM (feature_explorer) | 8 | 4 |
| **Total** | **114** | **51** |

> CSV completo com IV e correlação de cada feature: `reports/documents/feature_catalog.csv`

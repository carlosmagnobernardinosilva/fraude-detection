# 🛡️ Fraud Detection — Detecção de Fraudes em Transações

Sistema end-to-end de detecção de fraudes com pipeline modular (Agentes, Skills, Hooks) e app interativo em Streamlit.

---

## 🗂️ Estrutura do Projeto

```
fraud-detection/
│
├── 📂 app/                    # Aplicação Streamlit
│   ├── components/            # Componentes reutilizáveis de UI
│   └── pages/                 # Páginas do app (predição, dashboard, modelo)
│
├── 📂 data/
│   ├── raw/                   # Dados brutos (não versionados)
│   ├── processed/             # Dados após feature engineering
│   └── models/                # Modelos treinados serializados
│
├── 📂 notebooks/              # Análise Exploratória (EDA)
│
├── 📂 src/
│   ├── agents/                # Agentes: DataAgent, FeatureAgent, PredictionAgent
│   ├── skills/                # Skills: preprocessing, feature_engineering, evaluation
│   ├── hooks/                 # Hooks: data_hooks, model_hooks, prediction_hooks
│   ├── models/                # Wrappers dos modelos (XGBoost, LightGBM)
│   ├── pipeline/              # Orquestração do pipeline de treino
│   └── utils/                 # Logger, config, helpers
│
├── 📂 reports/
│   └── figures/               # Gráficos e visualizações exportadas
│
├── 📂 tests/                  # Testes unitários
│
├── 📂 .github/workflows/      # CI/CD com GitHub Actions
│
├── config.yaml                # Configurações centralizadas
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Stack

`Python` · `XGBoost` · `LightGBM` · `MLflow` · `SHAP` · `Streamlit` · `Scikit-learn`

---

## 👤 Autor

**Carlos Magno Bernardino Silva** — Data Scientist | Credit Risk & Fraud Detection  
🔗 [LinkedIn](https://linkedin.com/in/carlosmagno) · 🐙 [GitHub](https://github.com/carlosmagnobernardinosilva)

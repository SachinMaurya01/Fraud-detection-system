# Project PRD: End-to-End Fraud Detection ML System

## 1. Overview

**Goal:** Build a production-style, end-to-end ML system for transaction fraud detection —
covering feature engineering, model training, experiment tracking, model registry, serving,
and monitoring — plus a GenAI explanation layer on top, to demonstrate the hybrid
ML-systems + GenAI skill set relevant to Forward Deployed Engineer / GenAI application roles.

**Why this project:** Most ML portfolio projects stop at "trained a model in a notebook."
This one demonstrates the full lifecycle a real ML system needs in production, and adds an
LLM-powered layer that makes the system usable by a non-technical stakeholder — the part
that differentiates an FDE candidate from a pure modeling candidate.

**Hardware constraint:** RTX 3050 (limited VRAM). Design choice: use CPU-friendly gradient
boosting (XGBoost/LightGBM) instead of deep learning — this is *also* the industry-standard
choice for tabular fraud data, so it's not a compromise, it's the correct engineering decision.

**Target roles:** GenAI Application Engineer, Forward Deployed Engineer, ML-adjacent
software engineering roles.

---

## 2. Dataset

| Option | Size | Notes |
|---|---|---|
| [Credit Card Fraud Detection (Kaggle, ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | ~150MB, 285K rows | Best starting point — already PCA-anonymized, highly imbalanced (~0.17% fraud), fast to iterate on |
| [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/competitions/ieee-fraud-detection) | ~1.2GB | More realistic raw features, harder feature engineering, better portfolio story if you want a bigger challenge |

**Recommendation:** Start with the ULB dataset for the first full pass through the pipeline,
then optionally swap in IEEE-CIS once the system works end-to-end.

---

## 3. Repo / Folder Structure

```
fraud-detection-system/
├── README.md                     # Project overview, architecture diagram, setup instructions
├── PRD.md                        # This document
├── pyproject.toml / requirements.txt
├── .env.example                  # MLFLOW_TRACKING_URI, API keys, etc.
├── docker-compose.yml            # MLflow server + FastAPI service + monitoring stack
│
├── data/
│   ├── raw/                      # Original downloaded dataset (gitignored)
│   ├── processed/                # Cleaned/engineered features (gitignored)
│   └── data_dictionary.md        # Column definitions, source, notes
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
│
├── src/
│   ├── features/
│   │   ├── build_features.py     # Feature engineering pipeline (importable, testable)
│   │   └── preprocessing.py      # Scaling, encoding, imbalance handling (SMOTE/class weights)
│   ├── training/
│   │   ├── train.py              # Trains model, logs to MLflow
│   │   ├── config.yaml           # Hyperparameters, paths
│   │   └── evaluate.py           # Precision/recall/PR-AUC (NOT accuracy — imbalanced data)
│   ├── registry/
│   │   └── promote_model.py      # Registers + promotes best run to "Production" stage
│   ├── serving/
│   │   ├── main.py               # FastAPI app
│   │   ├── schemas.py            # Pydantic request/response models
│   │   └── model_loader.py       # Loads model from MLflow registry
│   ├── explain/
│   │   ├── shap_explainer.py     # Computes SHAP values per prediction
│   │   └── llm_explainer.py      # Turns SHAP output into plain-English explanation via LLM
│   └── monitoring/
│       ├── drift_report.py       # Evidently AI data/prediction drift reports
│       └── scheduled_check.py    # Cron-style script to run drift checks periodically
│
├── tests/
│   ├── test_features.py
│   ├── test_api.py               # FastAPI TestClient tests for /predict, /health
│   └── test_drift.py
│
├── monitoring/
│   ├── grafana/                  # Dashboard configs (optional stretch)
│   └── reports/                  # Generated Evidently HTML reports
│
├── mlruns/                       # MLflow local tracking store (gitignored, or point to remote)
└── deployment/
    ├── Dockerfile
    └── render.yaml / fly.toml    # Free-tier deploy config
```

---

## 4. Architecture

```
Raw Data → Feature Pipeline → Training (XGBoost/LightGBM)
                                     │
                                     ▼
                         MLflow Tracking (log runs)
                                     │
                                     ▼
                       MLflow Model Registry (Staging → Production)
                                     │
                                     ▼
                    FastAPI /predict endpoint (loads Production model)
                          │                        │
                          ▼                        ▼
                  SHAP explanation         Evidently drift monitoring
                          │
                          ▼
              LLM explanation layer (natural-language "why flagged")
```

---

## 5. Tools & Resources

| Layer | Tool | Docs / Resource |
|---|---|---|
| Data manipulation | Pandas or Polars | [pandas docs](https://pandas.pydata.org/docs/), [Polars docs](https://docs.pola.rs/) |
| Modeling | XGBoost or LightGBM | [XGBoost docs](https://xgboost.readthedocs.io/), [LightGBM docs](https://lightgbm.readthedocs.io/) |
| Imbalance handling | imbalanced-learn (SMOTE) | [imblearn docs](https://imbalanced-learn.org/stable/) |
| Experiment tracking | MLflow | [MLflow Tracking Quickstart](https://mlflow.org/docs/latest/getting-started/) |
| Model registry | MLflow Model Registry | [MLflow Model Registry docs](https://mlflow.org/docs/latest/model-registry.html) |
| Serving | FastAPI + Uvicorn | [FastAPI docs](https://fastapi.tiangolo.com/) |
| Explainability | SHAP | [SHAP docs](https://shap.readthedocs.io/) |
| GenAI layer | Anthropic API / OpenAI API | [Anthropic API docs](https://docs.claude.com) |
| Monitoring / drift | Evidently AI | [Evidently docs](https://docs.evidentlyai.com/) |
| Containerization | Docker + docker-compose | [Docker docs](https://docs.docker.com/) |
| Free deployment | Render or Fly.io | [Render docs](https://render.com/docs), [Fly.io docs](https://fly.io/docs/) |
| Testing | pytest + FastAPI TestClient | [FastAPI testing docs](https://fastapi.tiangolo.com/tutorial/testing/) |

---

## 6. Timeline (Weekend-Project Pacing)

Assumes ~4–6 hours per weekend, plus small weekday tweaks. ~7 weekends total.

| Weekend | Focus | Deliverable |
|---|---|---|
| **1** | Setup + EDA | Repo scaffolded, dataset downloaded, EDA notebook complete — understand class imbalance, feature distributions |
| **2** | Feature engineering | `build_features.py` pipeline, reproducible train/test split, imbalance strategy decided (SMOTE vs class weights) |
| **3** | Model training + MLflow | Baseline model logged to MLflow; 3–5 experiment runs comparing hyperparameters/imbalance strategies; pick best by PR-AUC |
| **4** | Model registry + FastAPI | Best model registered and promoted to "Production"; `/predict` endpoint live locally, returns prediction + probability |
| **5** | Explainability + GenAI layer | SHAP integrated into API response; LLM explainer turns SHAP values into a plain-English "why flagged" summary |
| **6** | Monitoring | Evidently drift report comparing live/incoming data vs. training distribution; a scheduled script that flags drift |
| **7** | Polish + deploy | Dockerized, deployed to Render/Fly.io free tier, README with architecture diagram, demo video/GIF for resume |

**Stretch goals (if time allows):** Grafana dashboard for live monitoring, A/B comparing two
registered model versions, a small Streamlit/React front-end for the fraud analyst chat interface.

---

## 7. Success Criteria

- [ ] Full pipeline runs end-to-end from raw data to a live `/predict` API call
- [ ] At least 3 MLflow experiment runs logged and comparable
- [ ] Model correctly registered and versioned (not just saved as a pickle file)
- [ ] API returns a prediction **and** a natural-language explanation for any input transaction
- [ ] A drift report can be generated on demand showing at least one real or simulated drift scenario
- [ ] Entire system runs via `docker-compose up` with no manual steps
- [ ] README clearly explains the architecture — this is what a recruiter/interviewer will actually read first

---

## 8. Interview Talking Points This Project Should Give You

- "I chose gradient boosting over deep learning because it's both hardware-appropriate and
  the correct tool for tabular fraud data — not because of a compute limitation."
- "I used PR-AUC instead of accuracy because with 0.17% positive class, a model that predicts
  'not fraud' every time is 99.8% accurate and useless."
- "I added an LLM explanation layer on top of SHAP so a non-technical fraud analyst could
  understand *why* a transaction was flagged, not just that it was."
- "I built drift monitoring in from the start because a fraud model's biggest risk isn't a bad
  initial model — it's silent degradation as fraud patterns evolve."
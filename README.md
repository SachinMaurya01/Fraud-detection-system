# Fraud Detection System — End-to-End Credit Card Fraud Classification

A comprehensive experimental study on highly imbalanced tabular fraud detection using classical ML and shallow neural networks. Evaluated on the [Kaggle Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (284,807 transactions, 0.172% fraud).

This README documents the exact pipeline, algorithms, hyperparameters, and benchmark results reproduced from `fraud_detection.ipynb:1`.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Algorithms Evaluated](#algorithms-evaluated)
5. [Benchmarks](#benchmarks)
6. [Comparative Analysis](#comparative-analysis)
7. [Training Dynamics](#training-dynamics)
8. [How to Reproduce](#how-to-reproduce)
9. [Project Structure](#project-structure)
10. [Technology Stack](#technology-stack)
11. [Limitations & Next Steps](#limitations--next-steps)
12. [Future Scope & Improvements](#future-scope--improvements)
13. [References](#references)

---

## Overview

The project addresses binary classification of fraudulent vs. legitimate credit card transactions under extreme class imbalance. Rather than optimizing for accuracy (a naive all-negative classifier achieves 99.83% accuracy), the study optimizes for **fraud-class precision, recall, and F1-score** — the metrics that matter operationally.

Two experimental regimes are compared:

| Regime | Training Set | Class Distribution | Purpose |
|--------|-------------|-------------------|---------|
| **A — Imbalanced** | First 240,000 rows in temporal order | 239,563 : 437 (99.82% : 0.18%) | Realistic production distribution |
| **B — Balanced** | Undersampled + shuffled 984 rows (700 train) | 353 : 347 train / 72 : 70 val | Controlled ablation to isolate model capacity under balanced conditions |

> **Key finding:** The shallow neural network (2 hidden units, 73 parameters) achieved the best fraud-class F1 on the realistic imbalanced validation set (F1 = 0.71), while logistic regression and LinearSVC trained on balanced data achieved F1 = 0.96 on the balanced validation set — a result that does not transfer to production without recalibration.

---

## Dataset

**Source:** `data/raw/creditcard.csv` — ULB Machine Learning Group, via Kaggle.

| Property | Value |
|----------|-------|
| Rows | 284,807 |
| Columns | 31 (30 features + `Class`) |
| Features | `Time` (seconds since first transaction), `V1`–`V28` (PCA-anonymized), `Amount` |
| Target | `Class` — `0` = legitimate (284,315), `1` = fraud (492) |
| Fraud rate | 0.1727% (1 in 579) |
| Time span | 0 – 172,792 seconds (48 hours / 2 days) |
| Missing values | None |
| `Amount` | mean 88.35, std 250.12, range 0 – 25,691.16 |
| `V1`–`V28` | Zero-centered (PCA), std ~0.3–1.96 |

Exploratory analysis (`fraud_detection.ipynb:3`) includes per-feature histograms (5x6 grid, 30 bins) and `df.describe()` confirming heavy tails on `V1`, `V2`, `V15`, and `Amount`.

---

## Methodology

### Preprocessing — `fraud_detection.ipynb:5`

```python
from sklearn.preprocessing import RobustScaler

new_df = df.copy()
new_df['Amount'] = RobustScaler().fit_transform(new_df['Amount'].values.reshape(-1,1))
new_df['Time']   = (time - time.min()) / (time.max() - time.min())  # Min-Max to [0, 1]
# V1–V28 left unchanged (already PCA-standardized)
```

| Feature | Transformation | Rationale |
|---------|---------------|-----------|
| `Amount` | `RobustScaler` (median/IQR) | Robust to extreme outliers (max $25k, 75th percentile $77); prevents skew from dominating PCA features |
| `Time` | Min-Max normalization to [0, 1] | Preserves temporal ordering; compresses 172k-second range to unit interval for gradient-based optimizers |
| `V1`–`V28` | None | Already zero-mean, unit-scale from PCA; additional scaling would destroy anonymized variance structure |
| `Class` | Target, excluded from features | — |

### Data Splitting

**Regime A — Temporal (imbalanced) — `fraud_detection.ipynb:4–6`**

```python
train, test, val = new_df[:240000], new_df[240000:262000], new_df[262000:]
# train: (240000, 31) — 239563:437  | test: (22000, 31) — 21963:37 | val: (22807, 31) — 22789:18
x_train, y_train = train_np[:,:-1], train_np[:,-1]  # (240000, 30) / (240000,)
```

Sequential slicing preserves temporal order — no shuffling, no stratification, no leakage. The validation set simulates "future" transactions.

**Regime B — Balanced undersampling — `fraud_detection.ipynb:16–20`**

```python
not_frauds = new_df.query('Class == 0')  # 284,315
frauds     = new_df.query('Class == 1')  # 492
balanced_df = pd.concat([frauds, not_frauds.sample(len(frauds), random_state=1)])  # 984 rows, 50:50
balanced_df = balanced_df.sample(frac=1, random_state=1)  # shuffle

x_train_b, y_train_b = balanced_df_np[:700, :-1], ...   # 700 train — 353:347
x_test_b,  y_test_b  = balanced_df_np[700:842, :-1], ... # 142 test  — 73:69
x_val_b,   y_val_b   = balanced_df_np[842:, :-1], ...    # 142 val   — 72:70
```

Balanced regime uses random undersampling of the majority class (no SMOTE, no synthetic generation). This is a clean ablation but produces a validation set that is **not representative of production traffic**.

---

## Algorithms Evaluated

### 1. Logistic Regression — `fraud_detection.ipynb:7–8`

| Hyperparameter | Value |
|---------------|-------|
| Solver | `lbfgs` (sklearn default) |
| Penalty | L2 (default) |
| Class weight | None |
| Max iter | 100 (default) |

Linear baseline. Trained on both imbalanced and balanced splits.

### 2. Shallow Neural Network (Imbalanced) — `fraud_detection.ipynb:9–13`

```python
shallow_nn = Sequential([
    InputLayer((30,)),
    Dense(2, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])
# compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit(validation_data=(x_val, y_val), epochs=50, callbacks=ModelCheckpoint(monitor='val_loss'))
```

| Property | Value |
|----------|-------|
| Architecture | 30 → 2 (ReLU) → BatchNorm → 1 (Sigmoid) |
| Total params | 73 (69 trainable, 4 non-trainable) |
| Optimizer | Adam (default lr 0.001) |
| Loss | Binary cross-entropy |
| Epochs | 50 |
| Batch size | 32 (Keras default, 7500 steps/epoch) |
| Checkpoint | `val_loss`, `save_best_only=True` |

Representative minimal-capacity network testing whether fraud is linearly separable with small non-linearity.

### 3. Random Forest — `fraud_detection.ipynb:14`

```python
RandomForestClassifier(max_depth=2, n_jobs=-1)
```

| Hyperparameter | Value |
|---------------|-------|
| `max_depth` | 2 (heavily regularized, 4 leaf nodes max) |
| `n_estimators` | 100 (default) |
| `n_jobs` | -1 (all cores) |

Intentionally shallow to prevent overfitting on the minority class.

### 4. Gradient Boosting Classifier — `fraud_detection.ipynb:15`

```python
GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
```

| Hyperparameter | Value |
|---------------|-------|
| `n_estimators` | 50 |
| `learning_rate` | 1.0 (aggressive, no shrinkage) |
| `max_depth` | 1 (decision stumps) |
| `random_state` | 0 |

Boosted stumps — classic bias-variance tradeoff for imbalanced tabular data. Balanced variant (`fraud_detection.ipynb:29`) uses `max_depth=2`.

### 5. Linear Support Vector Classifier — `fraud_detection.ipynb:16`

```python
LinearSVC(class_weight='balanced')
```

| Hyperparameter | Value |
|---------------|-------|
| `class_weight` | `balanced` (inverse-frequency weighting) |
| Loss | Squared hinge (default) |
| Dual | auto |

Only model in Regime A that explicitly corrects for imbalance via class weights. Threshold is at 0 (not calibrated probability).

### 6. Balanced-Regime Variants — `fraud_detection.ipynb:21–34`

| Model ID | Description | Training Data |
|----------|-------------|---------------|
| `logistic_model_b` | Logistic Regression (default) | `x_train_b` (700, 50:50) |
| `shallow_nn_b` | Dense(2)-ReLU-BN-Sigmoid, 40 epochs | `x_train_b` |
| `shallow_nn_b1` | Dense(1)-ReLU-BN-Sigmoid, 40 epochs | `x_train_b` |
| `gbc_b` | GBC stump ensemble (`max_depth=2`) | `x_train_b` |
| `svc_b` | LinearSVC `class_weight='balanced'` | `x_train_b` |

All balanced models evaluated on `x_val_b` (142 rows, 72:70 split).

---

## Benchmarks

All metrics are from `sklearn.metrics.classification_report` on the **held-out validation set** (`fraud_detection.ipynb:8,13–16,21,28,29,32–34`). Accuracy is reported but should not be used for model selection on imbalanced data.

### Regime A — Imbalanced Validation Set (`x_val`, n = 22,807, fraud = 18, prevalence 0.079%)

| # | Model | Fraud Precision | Fraud Recall | Fraud F1 | Accuracy | Macro Avg F1 | Weighted F1 | Support (Fraud) |
|---|-------|----------------|-------------|----------|----------|-------------|-------------|-----------------|
| A1 | **Logistic Regression** | **0.83** | 0.28 | 0.42 | 1.00 | 0.71 | 1.00 | 18 |
| A2 | **Shallow NN (2 units, 50e)** | 0.75 | **0.67** | **0.71** | 1.00 | **0.85** | 1.00 | 18 |
| A3 | **Random Forest (d=2)** | **0.89** | 0.44 | 0.59 | 1.00 | 0.80 | 1.00 | 18 |
| A4 | **Gradient Boosting (stumps)** | 0.71 | 0.56 | 0.62 | 1.00 | 0.81 | 1.00 | 18 |
| A5 | **LinearSVC (balanced)** | 0.05 | **0.72** | 0.09 | 0.99 | 0.54 | 0.99 | 18 |

Additional training signal: Logistic Regression train accuracy = 0.9991 (`fraud_detection.ipynb:7`).

### Regime B — Balanced Validation Set (`x_val_b`, n = 142, fraud = 70, prevalence 49.3%)

| # | Model | Not-Fraud P/R/F1 | Fraud P/R/F1 | Accuracy | Macro F1 | Notes |
|---|-------|-------------------|-------------|----------|----------|-------|
| B1 | **Logistic Regression (balanced)** | 0.92 / 1.00 / 0.96 | 1.00 / 0.91 / **0.96** | **0.96** | **0.96** | `fraud_detection.ipynb:21` |
| B2 | **Shallow NN (2 units, 40e)** | 0.94 / 1.00 / 0.97 | 1.00 / 0.93 / **0.96** | **0.96** | **0.96** | `fraud_detection.ipynb:24,28` |
| B3 | **Shallow NN (1 unit, 40e)** | 0.94 / 0.96 / 0.94 | 0.95 / 0.91 / 0.93 | 0.94 | 0.94 | `fraud_detection.ipynb:34` (held-out `x_test_b`) |
| B4 | **Gradient Boosting (d=2)** | 0.73 / 1.00 / 0.84 | 1.00 / 0.61 / 0.76 | 0.81 | 0.80 | `fraud_detection.ipynb:29` — see note below |
| B5 | **LinearSVC (balanced)** | 0.92 / 1.00 / 0.96 | 1.00 / 0.91 / **0.96** | **0.96** | **0.96** | `fraud_detection.ipynb:32–33` — see note below |

> **Reproducibility note on B4/B5:** Cells `fraud_detection.ipynb:29,32,33` instantiate `gbc_b`/`svc_b` on balanced data but call `.predict()` on the stale imbalanced-trained objects `gbc`/`svc`. The reported numbers therefore reflect the imbalanced-trained models evaluated on the balanced validation set, not the newly fitted balanced models. Treat B4/B5 as indicative; re-running with `gbc_b.predict` / `svc_b.predict` is recommended before citing.

### Raw Summary Table (Fraud-Class F1 as Primary Comparator)

```
Model                          Regime        Fraud F1    Fraud Recall    Fraud Precision
─────────────────────────────    ──────────    ────────    ────────────    ───────────────
Logistic Regression              Imbalanced    0.42        0.28            0.83
Shallow NN (2 units)             Imbalanced    0.71        0.67            0.75  ← best imbalanced
Random Forest (d=2)              Imbalanced    0.59        0.44            0.89
Gradient Boosting (stumps)       Imbalanced    0.62        0.56            0.71
LinearSVC (balanced weights)     Imbalanced    0.09        0.72            0.05
Logistic Regression              Balanced      0.96        0.91            1.00
Shallow NN (2 units)             Balanced      0.96        0.93            1.00  ← best balanced
Shallow NN (1 unit)              Balanced      0.93        0.91            0.95
```

---

## Comparative Analysis

### 1. The accuracy trap

Every model in Regime A reports accuracy ≥ 0.99. A dummy classifier predicting `Not Fraud` universally would score 0.9992 on `x_val`. Accuracy is therefore **not informative** for model selection. All comparisons below use fraud-class F1, precision, and recall.

### 2. Regime A — which model handles real-world imbalance best?

**Winner: Shallow NN (F1 = 0.71, Recall = 0.67, Precision = 0.75)**

| Criterion | Best Model | Value | Why it matters |
|-----------|-----------|-------|----------------|
| **F1 (harmonic mean)** | Shallow NN | 0.71 | Best balance of catching fraud vs. avoiding false alarms |
| **Recall (catch rate)** | LinearSVC / Shallow NN | 0.72 / 0.67 | LinearSVC catches slightly more fraud but at catastrophic precision cost |
| **Precision (false-alarm control)** | Random Forest | 0.89 | Fewest false positives per true fraud caught |
| **Failure mode** | LinearSVC | F1 0.09 | `class_weight='balanced'` over-corrects; 95% of fraud predictions are false positives |

**Practical interpretation for a 22,807-transaction validation window (18 frauds):**

- **Shallow NN:** Catches ~12/18 frauds, fires ~16 fraud alerts (4 false positives).
- **Logistic Regression:** Catches ~5/18 frauds, fires ~6 alerts (1 false positive) — precise but misses 72% of fraud.
- **Random Forest:** Catches ~8/18 frauds, fires ~9 alerts (1 false positive) — strong precision, moderate recall.
- **LinearSVC:** Catches ~13/18 frauds but fires ~260 alerts (~247 false positives) — operationally unusable.

### 3. Regime B — does balancing the training set help?

Fraud F1 jumps from 0.42–0.71 (Regime A) to 0.93–0.96 (Regime B) for comparable architectures. This is expected: the balanced validation set has ~625x higher fraud prevalence, so the task is fundamentally easier.

**Critical caveat:** Regime B validation performance **does not estimate production performance**. A model with F1 = 0.96 on 50:50 data will degrade sharply when deployed against 0.17% prevalence unless:
- Predicted probabilities are recalibrated (e.g., Platt scaling, isotonic regression), or
- The decision threshold is retuned on an imbalanced holdout (optimize PR-AUC / F-beta), or
- The model is evaluated via cross-validated PR-AUC on the original distribution.

Undersampling also discards 99.8% of legitimate transactions, losing information about the majority-class manifold.

### 4. Model capacity vs. data regime

| Model | Imbalanced F1 | Balanced F1 | Delta |
|-------|--------------|-------------|-------|
| Logistic Regression | 0.42 | 0.96 | +0.54 |
| Shallow NN (2 units) | 0.71 | 0.96 | +0.25 |
| Gradient Boosting | 0.62 | 0.76 | +0.14 |

The shallow NN shows the **smallest delta**, indicating it is least sensitive to class imbalance — it already extracts much of the separable signal under the realistic distribution.

### 5. Architecture sensitivity (balanced regime)

- **2-unit vs. 1-unit hidden layer:** F1 drops from 0.96 → 0.93, accuracy 0.96 → 0.94. The 2-unit model is worth the 30 extra parameters; both are tiny (73 vs ~40 params) and train in <2 seconds on CPU.
- **Random Forest `max_depth=2`:** Underperforms all other balanced models (F1 0.59 imbalanced). Depth-2 trees are too shallow for this feature space without boosting.

### 6. Cost of `class_weight='balanced'` without threshold tuning

LinearSVC on imbalanced data achieves the highest recall (0.72) but the lowest precision (0.05) — a 14:1 false-positive-to-true-positive ratio. For fraud operations, where each false positive triggers manual review or customer friction, this tradeoff is unacceptable despite the high recall. Threshold tuning or probability calibration would be required to make this viable.

---

## Training Dynamics

### Shallow NN — Imbalanced (50 epochs) — `fraud_detection.ipynb:11`

```
Epoch  1: acc 0.9939  loss 0.0475  val_acc 0.9996  val_loss 0.0028
Epoch  2: acc 0.9993  loss 0.0034  val_acc 0.9996  val_loss 0.0025  ← best val_loss
Epoch 14: acc 0.9993  loss 0.0032  val_acc 0.9996  val_loss 0.0025
Epoch 50: acc 0.9993  loss 0.0032  val_acc 0.9996  val_loss 0.0027
```

Convergence by epoch 2; remaining 48 epochs are flat (loss 0.0031–0.0033, val_loss 0.0025–0.0038). Early stopping at epoch 2–3 would save ~94% of training time with no loss in quality.

### Shallow NN — Balanced, 2 units (40 epochs) — `fraud_detection.ipynb:23`

```
Epoch  1: acc 0.7186  loss 0.5785  val_acc 0.8592  val_loss 0.4744
Epoch 10: acc 0.8971  loss 0.3922  val_acc 0.9296  val_loss 0.4250
Epoch 20: acc 0.9314  loss 0.2915  val_acc 0.9507  val_loss 0.3153
Epoch 30: acc 0.9357  loss 0.2394  val_acc 0.9648  val_loss 0.2427
Epoch 40: acc 0.9443  loss 0.1996  val_acc 0.9648  val_loss 0.2001
```

Smooth monotonic improvement; no overfitting. Larger hidden layer converges to lower loss than the 1-unit variant (final val_loss 0.20 vs 0.285).

### Shallow NN — Balanced, 1 unit (40 epochs) — `fraud_detection.ipynb:26–27`

```
Epoch  1: acc 0.4843  loss 0.7323  val_acc 0.4648  val_loss 0.6629
Epoch 20: acc 0.8429  loss 0.4342  val_acc 0.8662  val_loss 0.4341
Epoch 40: acc 0.9114  loss 0.2630  val_acc 0.9155  val_loss 0.2857
```

Starts near chance (single ReLU unit has limited capacity), converges slower and to a higher final loss.

---

## How to Reproduce

### Prerequisites

- Python ≥ 3.13, `uv` or `pip`
- Dataset: `data/raw/creditcard.csv` (download from [Kaggle ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place at `data/raw/creditcard.csv`)

### Installation

```bash
uv sync                          # or: pip install -e .
# or manually:
pip install jupyter pandas numpy scikit-learn tensorflow matplotlib seaborn
```

### Run the notebook

```bash
jupyter notebook fraud_detection.ipynb
# Execute cells top-to-bottom. Runtime: ~3–5 minutes on CPU (no GPU required).
```

### Expected artifacts

- `shallow_nn.keras` / `shallow_nn_b.keras` — saved checkpoints (best `val_loss`)
- All classification reports printed to cell outputs (no external metrics file; parse from notebook JSON if automating)

### Reproducibility notes

- `random_state=1` for undersampling and shuffling; `random_state=0` for GBC.
- No global `np.random.seed` or `tf.random.set_seed` set — minor run-to-run variance expected for NN weight initialization.
- Temporal split is deterministic (row-order slicing).

---

## Project Structure

```
fraud-detection-system/
├── fraud_detection.ipynb   # Complete experimental pipeline (this README documents it)
├── data/
│   └── raw/
│       └── creditcard.csv  # Raw ULB dataset (gitignored, ~150 MB)
├── pyproject.toml          # Dependencies (pandas, sklearn, tensorflow, jupyter, etc.)
├── README.md               # This file
└── main.py                 # Placeholder entry point
```

> The target production architecture includes MLflow tracking and registry, FastAPI serving, SHAP and LLM explanations, Evidently drift monitoring, and Docker deployment. The notebook implements the modeling core that precedes that system layer.

---

## Technology Stack

| Layer | Library | Version |
|-------|---------|---------|
| Data | `pandas`, `numpy` | ≥ 3.0.5, ≥ 2.5.2 |
| Preprocessing | `scikit-learn` (`RobustScaler`) | ≥ 1.9.0 |
| Classical ML | `scikit-learn` (LogisticRegression, RandomForest, GradientBoosting, LinearSVC) | ≥ 1.9.0 |
| Deep Learning | `tensorflow` / `keras` (Sequential, Dense, BatchNormalization) | ≥ 2.21.0 |
| Evaluation | `scikit-learn` (`classification_report`) | — |
| Visualization | `matplotlib` | ≥ 3.11.1 |
| Notebook | `jupyter` | ≥ 1.1.1 |

---

## Limitations & Next Steps

### Known limitations in the current notebook

1. **Evaluation bug — `fraud_detection.ipynb:29,32,33`:** `gbc_b`/`svc_b` are fitted but predictions use stale `gbc`/`svc` objects. Re-evaluate with `gbc_b.predict(x_val_b)` and `svc_b.predict(x_val_b)` before reporting balanced GBC/SVC numbers.
2. **Checkpoint overwrite — `fraud_detection.ipynb:22,26`:** Both `shallow_nn_b` and `shallow_nn_b1` write to `shallow_nn_b.keras`; the second training run silently overwrites the first checkpoint.
3. **Redundant re-training — `fraud_detection.ipynb:22`:** `shallow_nn.fit(...)` is called with `shallow_nn_b`'s checkpoint object, re-training the original imbalanced model a second time (40 extra epochs, no new evaluation).
4. **No probability calibration** for any model — critical for imbalanced deployment.
5. **No PR-AUC / ROC-AUC** reported — PR-AUC is the recommended primary metric for imbalanced data; adding `average_precision_score` / `roc_auc_score` would strengthen comparisons.
6. **Validation set is tiny for fraud (n=18)** — confidence intervals on Regime A fraud metrics are wide (±~0.20 on precision/recall). K-fold or bootstrapped evaluation recommended.

### Recommended next steps

- [ ] Fix evaluation bugs and re-run balanced GBC/SVC evaluation
- [ ] Add `average_precision_score` (PR-AUC) and `roc_auc_score` for all models
- [ ] Threshold tuning on `x_test` (F-beta optimization, e.g., F2 for recall-weighted fraud detection)
- [ ] Probability calibration (Platt / isotonic) + expected calibration error
- [ ] Stratified K-fold or time-series cross-validation for tighter confidence intervals
- [ ] Replace undersampling with SMOTE / class-weighted loss for information-preserving imbalance handling
- [ ] Integrate MLflow tracking (`src/training/train.py`) and promote best model to registry
- [ ] SHAP explainability + LLM natural-language layer (`src/explain/`)
- [ ] Evidently drift monitoring (`src/monitoring/`)

---

## Future Scope & Improvements

This section outlines a structured roadmap to evolve the current notebook-based study (`fraud_detection.ipynb:1`) into a production-grade, research-competitive fraud detection system. Items are prioritized by impact.

### 1. Modeling — From Baselines to State-of-the-Art for Tabular Fraud

| Area | Current State | Proposed Improvement | Expected Impact |
|------|--------------|---------------------|-----------------|
| **Gradient boosting** | `sklearn` GBC with stumps (`n_estimators=50, max_depth=1`) | **XGBoost / LightGBM / CatBoost** with native handling of imbalance (`scale_pos_weight`, `is_unbalance`), leaf-wise growth, and GPU acceleration | PR-AUC +5–12% on ULB; industry standard for tabular fraud |
| **Ensembles & stacking** | Single models evaluated in isolation | Stacking / voting ensemble (e.g., Shallow NN + LightGBM + Logistic Regression) with calibrated meta-learner; bagging with balanced subsamples (EasyEnsemble, BalancedRandomForest) | Reduced variance; tighter confidence intervals on n=18 fraud val set |
| **Cost-sensitive & focal learning** | `LinearSVC(class_weight='balanced')` only | Focal loss (`tensorflow_addons`), class-weighted cross-entropy, and threshold-moving via `cost_matrix` (false-negative cost >> false-positive cost) | Direct optimization for business cost, not just F1 |
| **Anomaly / unsupervised detection** | Supervised only | Autoencoder / Variational Autoencoder trained on legitimate transactions; reconstruction error as fraud score. Isolation Forest / One-Class SVM baselines | Detects novel fraud patterns with zero labeled examples; complements supervised models |
| **Sequential & deep tabular models** | 2-unit shallow NN (73 params) | TabNet, FT-Transformer, TabPFN for tabular attention; LSTM/Temporal-CNN on transaction sequences per cardholder (requires IEEE-CIS dataset with card IDs) | Captures temporal dependencies that PCA features obscure |
| **Hyperparameter optimization** | Manual defaults | Optuna / Ray Tune with PR-AUC objective, stratified time-series cross-validation, and early stopping | Systematic gain without manual trial-and-error |

### 2. Data & Feature Engineering

**Current limitation:** `V1`–`V28` are PCA-anonymized (`fraud_detection.ipynb:1`), so domain features cannot be engineered. `Amount` and `Time` are the only interpretable signals.

| Improvement | Description | Effort |
|-------------|-------------|--------|
| **Temporal features** | Hour-of-day, day-of-week, time-since-last-transaction, transaction velocity (count/amount in last N hours) | Low — derived from `Time` |
| **Behavioral aggregates** | Rolling mean/std of `Amount` per time window, distance from cardholder centroid (if card ID available) | Medium — requires grouping key |
| **Graph features** | Merchant–cardholder bipartite graph; PageRank / community detection to flag fraud rings | High — needs merchant/card identifiers |
| **Dataset upgrade** | Migrate to **IEEE-CIS Fraud Detection** (~1.2 GB, raw categorical + transaction + identity tables) | Medium — unlocks realistic feature engineering and joins |
| **Synthetic augmentation** | Replace naive undersampling with **SMOTE, ADASYN, Borderline-SMOTE** (`imbalanced-learn`), or generative models (CTGAN, TabDDPM) for minority oversampling | Low–Medium — preserves majority information while balancing |
| **Feature selection & importance** | Permutation importance, SHAP-based selection, and correlation pruning on `V1`–`V28` to reduce noise | Low |

### 3. Evaluation & Validation Maturity

| Gap Today | Improvement | Why It Matters |
|-----------|-------------|----------------|
| Accuracy reported; no PR-AUC / ROC-AUC (`fraud_detection.ipynb:29` noted) | Add `average_precision_score` (PR-AUC), `roc_auc_score`, and **Precision-Recall curves** per model; use PR-AUC as primary selector | Only discriminative metric under 0.17% prevalence |
| Single temporal split; fraud val n=18 → wide CIs | **Time-series cross-validation** (expanding window), stratified K-fold, and bootstrapped CIs for precision/recall | Reliable model comparison; current 0.71 vs 0.62 F1 gap may not be significant |
| Fixed 0.5 threshold | **Threshold tuning** via F-beta (F2 for recall-weighted fraud), Youden's J, or cost-optimal threshold on `x_test` | Moves operating point to business optimum (e.g., 5:1 cost of FN:FP) |
| No calibration | **Platt scaling / isotonic regression** + reliability diagrams, Expected Calibration Error (ECE), Brier score | Required for probability-based downstream actions (manual review queues, risk scoring) |
| No statistical testing | McNemar / paired bootstrap tests for model-vs-model fraud-F1 differences | Prevents overclaiming small F1 deltas |

### 4. MLOps & Productionization

```
Current:  notebook → manual .keras checkpoint
Target:   Raw Data → Feature Pipeline → Training (tracked) → Registry → Serving → Monitoring
```

| Component | Tooling | Next Step |
|-----------|-----------------------------|-----------|
| **Experiment tracking** | MLflow Tracking | Wrap training in `src/training/train.py` with `mlflow.log_params / log_metrics`; log PR-AUC, confusion matrices, and artifacts per run |
| **Model registry** | MLflow Model Registry | `src/registry/promote_model.py` to version and stage (`Staging` → `Production`); enforce validation gate (PR-AUC > threshold on holdout) |
| **Feature pipeline** | `src/features/build_features.py` | Extract preprocessing (`RobustScaler`, time normalization) from notebook into importable, tested module with `fit` on train only (prevent leakage) |
| **Serving** | FastAPI + Uvicorn (`src/serving/main.py`) | `/predict`, `/health`, `/explain` endpoints; Pydantic schemas (`schemas.py`); load model from registry via `model_loader.py` |
| **Containerization** | Docker + docker-compose | `deployment/Dockerfile` + `docker-compose.yml` (MLflow + FastAPI + monitoring) for `docker-compose up` reproducibility |
| **CI/CD** | GitHub Actions | Lint, `pytest`, notebook execution test, and MLflow model validation on PR |
| **Deployment** | Render / Fly.io free tier | Live demo endpoint for portfolio |

### 5. Explainability, Trust & Human-in-the-Loop

| Capability | Implementation | Stakeholder Value |
|------------|----------------|-------------------|
| **SHAP values** | `shap.TreeExplainer` / `DeepExplainer` per prediction (`src/explain/shap_explainer.py`) | Feature-level attribution for every flagged transaction |
| **LLM explanation layer** | `src/explain/llm_explainer.py` via Anthropic/OpenAI API — converts SHAP output to plain English: *"Flagged due to unusually high amount ($2,400 vs. $22 median) and anomalous V14/V17 pattern"* | Non-technical fraud analyst can act without ML expertise |
| **Counterfactuals** | DiCE / Alibi — "What minimal change would flip this prediction to legitimate?" | Actionable recourse and false-positive triage |
| **Global interpretability** | SHAP summary / dependence plots, partial dependence on `Amount`/`Time` | Model audit and regulator compliance |

### 6. Monitoring, Drift & Retraining

| Signal | Tool | Action |
|--------|------|--------|
| **Data drift** | Evidently AI (`src/monitoring/drift_report.py`) — compare live `V1`–`V28`/`Amount` distributions vs. training | Alert when PSI / KS-test exceeds threshold |
| **Prediction drift** | Track fraud-score distribution shift | Detect silent concept drift as fraudsters adapt |
| **Performance drift** | Delayed labels — monitor precision@k and review-queue conversion rate | Trigger retraining when PR-AUC drops > X% |
| **Scheduled checks** | `src/monitoring/scheduled_check.py` cron job + Grafana dashboard (`monitoring/grafana/`) | Automated daily drift report (HTML via Evidently) |
| **Retraining strategy** | Incremental / online learning vs. full retrain; champion–challenger A/B on registry | Prevents catastrophic forgetting while adapting to new fraud patterns |

### 7. Real-Time & Scale

- **Streaming inference:** Kafka / Kinesis consumer → feature store (Feast) → FastAPI low-latency endpoint (p50 < 50 ms).
- **Feature store:** Centralized `Amount`/`Time` transformations and velocity features with point-in-time correctness.
- **Scalability:** Horizontal scaling of FastAPI (Uvicorn workers), model caching, and batch prediction for offline scoring of historical ledgers.
- **Latency budget:** Quantize / distill shallow NN or LightGBM for edge deployment if sub-10 ms required.

### 8. Security, Privacy & Compliance

- **PII & anonymization:** ULB dataset is already PCA-anonymized; IEEE-CIS migration requires tokenization and vaulting of card identifiers.
- **Federated learning:** Train across banks without sharing raw transactions (Flower / FedML).
- **Differential privacy:** Add DP-SGD for NN training when sharing model weights externally.
- **Auditability:** Immutable MLflow run lineage, model cards, and explainability logs for regulatory review (PCI-DSS, GDPR).

### 9. Research & Stretch Goals

- **Cost-sensitive active learning:** Prioritize uncertain / high-risk transactions for human labeling to grow the 492-fraud set efficiently.
- **Graph neural networks** for fraud-ring detection on transaction graphs.
- **Causal inference:** Distinguish correlation (e.g., high `Amount` correlates with fraud) from causal drivers to avoid spurious blocks.
- **Multi-modal fusion:** Combine tabular transaction data with text (merchant descriptions) and behavioral biometrics.

### Suggested Roadmap

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1 — Correctness** | 1 weekend | Fix evaluation bugs (`fraud_detection.ipynb:29,32`), add PR-AUC/threshold tuning/calibration, time-series CV |
| **Phase 2 — Modeling uplift** | 2 weekends | XGBoost/LightGBM + SMOTE + Optuna; ensemble vs. shallow NN bake-off on PR-AUC |
| **Phase 3 — MLOps foundation** | 2 weekends | `src/features` + `src/training` + MLflow tracking + Model Registry |
| **Phase 4 — Serving & explainability** | 1 weekend | FastAPI + SHAP + LLM explanation layer |
| **Phase 5 — Monitoring & deploy** | 1 weekend | Evidently drift + Grafana + Docker + Render/Fly.io deploy |
| **Phase 6 — Scale & research** | Ongoing | Streaming, feature store, GNN / autoencoder experiments, IEEE-CIS migration |

---

## References

- Kaggle — Credit Card Fraud Detection (ULB): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Dal Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced Classification," IEEE SSCI 2015.
- scikit-learn — Imbalanced classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure
- Chollet, *Deep Learning with Python* — minimal-capacity network design for tabular data.

---

*Generated from `fraud_detection.ipynb` (3,052 lines, 34 code cells). All benchmark numbers are verbatim from notebook cell outputs; hyperparameters and architecture details are sourced to `fraud_detection.ipynb:<line>` where applicable.*

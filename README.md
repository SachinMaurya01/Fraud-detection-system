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

This section outlines general, practical improvements to make the current notebook-based study (`fraud_detection.ipynb:1`) more accurate and scalable in real-world conditions.

### A. Improving Accuracy

Accuracy in fraud detection is not about overall accuracy — it is about catching more fraud with fewer false alarms under extreme imbalance (0.17% fraud).

#### 1. Better Data and Feature Engineering

- **Richer features from existing signals:** Derive hour-of-day, time-since-last-transaction, and transaction velocity (count and total amount in the last N hours) from `Time`; rolling statistics on `Amount` to capture behavioral deviation. Current `V1`–`V28` are PCA-anonymized (`fraud_detection.ipynb:1`), which limits domain feature design — moving to a dataset with raw attributes (e.g., merchant, device, card history) would unlock far stronger signals.
- **Preserve information when balancing:** Replace random undersampling (which discards 99.8% of legitimate transactions) with SMOTE, ADASYN, or Borderline-SMOTE for synthetic minority oversampling, or use class-weighted losses that keep all data.
- **Feature selection:** Apply permutation importance and correlation pruning on `V1`–`V28` to remove noisy components and reduce overfitting.

#### 2. Stronger Modeling

- **Modern gradient boosting:** Replace the baseline `sklearn` GBC stumps (`fraud_detection.ipynb:15`) with XGBoost, LightGBM, or CatBoost. They provide native imbalance handling (`scale_pos_weight`), leaf-wise growth, regularization, and GPU acceleration, typically improving PR-AUC by 5–12% on this dataset.
- **Ensembles and stacking:** Combine diverse models (e.g., shallow NN + gradient boosting + logistic regression) via voting or stacking with a calibrated meta-learner to reduce variance. On a validation set with only 18 fraud cases, this stabilizes performance estimates.
- **Cost-sensitive learning:** Move beyond `class_weight='balanced'` to focal loss and class-weighted cross-entropy where a false negative (missed fraud) costs significantly more than a false positive (manual review).
- **Anomaly detection as complement:** Train autoencoders or variational autoencoders on legitimate transactions only and use reconstruction error as a fraud score. This catches novel fraud patterns with no labeled examples and pairs well with supervised models.
- **Systematic tuning:** Use Optuna or Ray Tune with PR-AUC as the objective, time-series cross-validation, and early stopping instead of manual defaults.

#### 3. More Rigorous Evaluation

- **Use the right metrics:** Add PR-AUC (average precision), ROC-AUC, and full precision-recall curves. PR-AUC is the primary discriminator when prevalence is 0.17%; accuracy is not informative (`Comparative Analysis: The accuracy trap`).
- **Tune the decision threshold:** The default 0.5 threshold is rarely optimal. Tune for F2 (recall-weighted), business cost (e.g., 5:1 cost of missed fraud vs. false alarm), or precision@k on a held-out test set.
- **Calibrate probabilities:** Apply Platt scaling or isotonic regression and measure calibration via reliability diagrams, Expected Calibration Error (ECE), and Brier score. Calibrated scores are required for risk-ranked review queues.
- **Reduce variance:** Replace the single temporal split with expanding-window time-series cross-validation or stratified K-fold with bootstrapped confidence intervals. Add statistical tests (McNemar, paired bootstrap) before claiming one model beats another.

### B. Improving Scalability

Scalability covers data volume, training time, inference latency, and operational overhead — not just model size.

#### 1. Scalable Data Processing

- **Decouple preprocessing from the notebook:** Extract `RobustScaler` and time normalization (`fraud_detection.ipynb:5`) into a reusable, tested module that fits only on training data to prevent leakage and can be reused in batch and online pipelines.
- **Handle larger datasets:** For datasets beyond the 284K-row ULB set, use columnar formats (Parquet), chunked processing with Dask or Spark, and a feature store (e.g., Feast) for point-in-time correct, reusable transformations.

#### 2. Scalable Training

- **Distributed and accelerated training:** LightGBM/XGBoost with histogram-based training and GPU support scales to millions of rows without code changes. For neural networks, use mixed-precision and data-parallel training.
- **Experiment tracking and registry:** Track every run (parameters, PR-AUC, confusion matrices, artifacts) with MLflow or Weights & Biases, version models in a registry, and gate promotion on holdout PR-AUC. This avoids manual `.keras` checkpoint handling (`fraud_detection.ipynb:11`) and enables reproducibility.
- **Automated pipelines:** Orchestrate preprocessing → training → evaluation → registration with a pipeline tool (e.g., Prefect, Airflow) and CI checks (lint, unit tests, notebook execution) on every change.

#### 3. Scalable and Reliable Inference

- **API serving:** Serve the best model behind a FastAPI (or gRPC) service with `/predict` and `/health` endpoints, Pydantic validation, and model loading from the registry. This replaces ad-hoc `model.predict()` calls in the notebook.
- **Containerization and orchestration:** Package the service with Docker and deploy on Kubernetes or a managed container platform with horizontal auto-scaling, load balancing, and rolling updates. Cache models in memory and support batch prediction for offline scoring of historical ledgers.
- **Latency optimization:** For real-time authorization (p50 < 50 ms), consider model quantization, distillation, or using LightGBM which is typically faster than neural networks on tabular data. Use async I/O and worker scaling (Uvicorn/Gunicorn) for throughput.
- **Streaming architecture (when needed):** For high-throughput environments, add a Kafka/Kinesis consumer that enriches transactions from the feature store and calls the inference service; keep a synchronous fast path for low-latency decisions and an async path for monitoring and logging.

#### 4. Monitoring, Maintenance, and Operations

- **Drift detection:** Monitor data drift (feature distributions for `Amount`, `V1`–`V28`), prediction drift (fraud-score shift), and performance drift (precision@k, PR-AUC on delayed labels) with tools like Evidently AI. Alert on PSI or KS-test thresholds.
- **Retraining strategy:** Define when to retrain — incremental updates vs. full retrain, and champion–challenger A/B testing before promoting a new version. This handles concept drift as fraud patterns evolve.
- **Observability:** Add structured logging, metrics (Prometheus), and dashboards (Grafana) for request rate, latency, error rate, and fraud-rate trends.
- **Security and compliance:** Keep the current PCA anonymization for privacy; for richer datasets, add tokenization of card identifiers, audit trails, and model cards for regulatory review. Consider federated learning if training across institutions without sharing raw data, and differential privacy when sharing model artifacts.

#### 5. Practical Roadmap (Accuracy + Scalability)

| Priority | Focus | Example Deliverable | Impact |
|----------|-------|---------------------|--------|
| **High** | Evaluation correctness | Fix evaluation bugs (`fraud_detection.ipynb:29,32`), add PR-AUC, threshold tuning, calibration, and time-series CV | Trustworthy model comparison; avoids shipping a misleading 0.96 F1 that collapses in production |
| **High** | Modeling uplift | LightGBM/XGBoost + SMOTE or class-weighted loss + Optuna tuning; ensemble bake-off | Direct lift in fraud recall and precision |
| **Medium** | Pipeline modularization | Reusable preprocessing and training modules, tracked experiments, model registry | Reproducibility, scalability, and safe promotion to production |
| **Medium** | Serving and scalability | Containerized FastAPI service with auto-scaling, batch + real-time paths | Handles growth from thousands to millions of transactions per day |
| **Medium** | Monitoring | Drift reports and performance dashboards with automated retraining triggers | Prevents silent degradation as fraud tactics change |
| **Lower** | Advanced research | Autoencoder anomaly detection, graph-based fraud-ring features, richer datasets | Long-term differentiation for novel or coordinated fraud |

---

## References

- Kaggle — Credit Card Fraud Detection (ULB): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Dal Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced Classification," IEEE SSCI 2015.
- scikit-learn — Imbalanced classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure
- Chollet, *Deep Learning with Python* — minimal-capacity network design for tabular data.

---

*Generated from `fraud_detection.ipynb` (3,052 lines, 34 code cells). All benchmark numbers are verbatim from notebook cell outputs; hyperparameters and architecture details are sourced to `fraud_detection.ipynb:<line>` where applicable.*

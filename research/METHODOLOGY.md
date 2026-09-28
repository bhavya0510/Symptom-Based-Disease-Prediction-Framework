# Research Methodology & Evaluation Framework

## Core Research Contribution
This work contributes an integrated evaluation framework that combines probability calibration, selective prediction through abstention, sequential symptom acquisition, and component-level explainability, while explicitly auditing the limitations of small symptom-disease datasets.

---

## Literature Review & Research Gap Matrix

| Work | Calibration | Selective Abstention | Sequential Questioning | Explainability (XAI) | Dataset Auditing |
|:---|:---:|:---:|:---:|:---:|:---:|
| Previous study 1 | No | No | No | Yes | No |
| Previous study 2 | No | No | Yes | No | No |
| Previous study 3 | Yes | No | No | Yes | No |
| **Proposed Framework** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

---

## 1. Data and Deduplication Protocol

`stage1_data_splitting.py` loads the raw symptom CSV files (4,962 total records across 132 binary symptoms and 41 disease classes). 

**Pattern Deduplication Protocol:** The original dataset contained 4,962 records, but only 305 unique symptom vectors were present. Because repeated identical symptom vectors do not provide additional information about the mapping from symptoms to disease, pattern-level deduplication was performed before splitting. Consequently, all subsequent experiments treat the 305 unique patterns as the effective dataset.

The 305 unique rows are split with stratification into:
- 70% Train (213 samples)
- 15% Validation (46 samples)
- 15% Test (46 samples)

---

## 2. Model Performance Benchmarking & Dataset Limitation

**Test Performance:** 46/46 test cases were correctly classified ($1.0000$ accuracy).

**⚠️ Dataset Limitation:** However, the test partition contains only 46 unique patterns across 41 disease classes, limiting conclusions about generalization. The evaluation should therefore be interpreted as a proof-of-concept assessment of the framework rather than an estimate of population-level diagnostic performance.

---

## 3. Probability Calibration Methodology

**Validation-Only Method Selection:** Calibration method selection was performed exclusively on the validation set. The test set was not used for selecting isotonic regression. After selection, the calibrated model was evaluated once on the held-out test set.

**Metrics Improvement:**
- Brier Score: $0.0019 \rightarrow 0.0001$
- Expected Calibration Error (ECE): $0.2341 \rightarrow 0.0141$
- Log Loss: $0.2875 \rightarrow 0.0148$

---

## 4. Sequential Symptom Acquisition & Paired Statistical Significance

In `stage7_sequential_questioning.py`, Mutual Information (MI) questioning is evaluated against a Random-selection baseline using **strict paired experimental control** (identical test cases, same initial symptoms, same starting model, same max questions, same stopping threshold).

**Paired Statistical Significance ($N=46$ test cases with 1 initial symptom):**
- **Confidence Gain:** MI ($+0.3355$) vs Random ($+0.1171$), Mean Diff $= +0.2184$, 95% CI: $[0.1491, 0.2877]$, $p_{ttest} = 1.70 \times 10^{-7}$, $p_{wilcoxon} = 2.53 \times 10^{-6}$.
- **Questions Needed:** MI ($1.50$) vs Random ($3.26$), Mean Diff $= -1.76$, 95% CI: $[-2.38, -1.14]$, $p = 1.0 \times 10^{-6}$.

---

## 5. Component Ablation Study & Metric Note

`stage10_ablation_study.py` evaluates 5 system configurations: Baseline $\rightarrow$ Uncalibrated Ensemble $\rightarrow$ Calibrated Ensemble $\rightarrow$ Abstention $\rightarrow$ Full System.

**Ablation Metric Note:** Because point accuracy is saturated on this dataset ($1.0000$), calibration and selective-prediction metrics (Brier Score, ECE, Log Loss, Coverage) provide more informative evidence for component-level comparison.

---

## 6. XGBoost Component-Level SHAP Explainability

Stage 8 fits a SHAP `TreeExplainer` on the XGBoost component model.
*Note:* This provides XGBoost component-level SHAP explanations, not explanations for the entire RF + XGBoost calibrated ensemble.

---

## 7. Exact Implementation Details

To ensure exact scientific reproducibility:
- **Environment:** Python 3.9+ / 3.14, `scikit-learn==1.9.1`, `xgboost==3.4.1`, `shap==0.52.0`, `scipy==1.18.1`.
- **Data Partitioning:** Stratified 70/15/15 split (`random_state=42`).
- **Random Forest Parameters:** `n_estimators=100`, `max_depth=None`, `random_state=42`.
- **XGBoost Parameters:** `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `random_state=42`, `eval_metric='mlogloss'`.
- **MLP Architecture:** `hidden_layer_sizes=(100, 50)`, `activation='relu'`, `max_iter=500`, `random_state=42`.
- **SVM Parameters:** `kernel='rbf'`, `C=1.0`, `probability=True`, `random_state=42`.
- **RandomizedSearchCV:** `n_iter=20`, `cv=5`, `random_state=42`.
- **Calibration Parameters:** Isotonic regression fitted on validation probabilities (`cv='prefit'`).

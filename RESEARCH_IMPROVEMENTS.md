# Integrated Evaluation Framework: Research & Methodology Improvements

## Executive Summary
This document summarizes the comprehensive enhancements made to the Symptom-Based Disease Prediction Framework. Rather than positioning the project as a "new prediction algorithm", this work contributes an **uncertainty-aware evaluation framework** combining probability calibration, selective prediction through abstention, paired sequential symptom acquisition, component-level explainability, and rigorous auditing of small symptom-disease datasets.

---

## 12 Core Research Improvements

### 1. Reframed Novelty & Core Contribution
- **Previous Position:** Claimed a novel disease prediction model.
- **Updated Position:** Contributes an integrated evaluation framework combining probability calibration, selective abstention, paired sequential questioning, and XGBoost component-level explainability while auditing small dataset limits.
- **Added Text:** *"This work contributes an integrated evaluation framework that combines probability calibration, selective prediction through abstention, sequential symptom acquisition, and component-level explainability, while explicitly auditing the limitations of small symptom-disease datasets."*

### 2. Dataset Deduplication Protocol (4,962 → 305 Unique Patterns)
- **Clarification:** The raw dataset contains 4,962 records, but only 305 unique symptom vectors exist across 41 disease classes.
- **Protocol:** Because repeated identical symptom vectors do not provide additional information about the mapping from symptoms to disease, pattern-level deduplication was performed before splitting. All experiments treat the 305 unique patterns as the effective dataset.

### 3. Test Set Accuracy Reporting (46/46 Test Cases)
- **Reporting:** 46/46 test cases correctly classified ($1.0000$ accuracy).
- **Caveat:** Immediately emphasized in Abstract, Results, and Conclusion: *"However, the test partition contains only 46 unique patterns across 41 disease classes, limiting conclusions about generalization."*

### 4. Strengthened Dataset Limitation Statement
- **Added Sentence:** *"The evaluation should therefore be interpreted as a proof-of-concept assessment of the framework rather than an estimate of population-level diagnostic performance."*

### 5. Rigorous Paired Experimental Control for MI vs Random
- **Experimental Control:** Evaluated Mutual Information (MI) vs Random questioning on the exact same test cases (same initial symptoms, same starting model, same max questions, same stopping threshold).
- **Result:** On 1 initial symptom, MI required an average of $1.50$ follow-up questions vs Random $3.26$ questions.

### 6. Paired Statistical Significance Testing
- **Statistical Methods:** Paired t-test (`scipy.stats.ttest_rel`) and Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) with 95% Confidence Intervals.
- **Key Results (1 Initial Symptom, $N=46$ identical cases):**
  - **Confidence Gain:** MI ($+0.3355$) vs Random ($+0.1171$), Mean Diff $= +0.2184$, 95% CI: $[0.1491, 0.2877]$, $p_{ttest} = 1.70 \times 10^{-7}$, $p_{wilcoxon} = 2.53 \times 10^{-6}$.
  - **Questions Needed:** MI ($1.50$) vs Random ($3.26$), Mean Diff $= -1.76$, 95% CI: $[-2.38, -1.14]$, $p = 1.0 \times 10^{-6}$.

### 7. Literature Review & Specific Research Gap Matrix
Added explicit comparison matrix establishing the research gap:

| Work | Calibration | Selective Abstention | Sequential Questioning | Explainability (XAI) | Dataset Auditing |
|:---|:---:|:---:|:---:|:---:|:---:|
| Previous study 1 | No | No | No | Yes | No |
| Previous study 2 | No | No | Yes | No | No |
| Previous study 3 | Yes | No | No | Yes | No |
| **Proposed Framework** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

### 8. Exact Implementation & Hyperparameter Details
Added complete reproducible setup details:
- **Environment:** Python 3.9+ / 3.14, `scikit-learn==1.9.1`, `xgboost==3.4.1`, `shap==0.52.0`, `scipy==1.18.1`.
- **Partitioning:** Stratified 70% Train (213), 15% Validation (46), 15% Test (46) with `random_state=42`.
- **Random Forest:** `n_estimators=100`, `max_depth=None`, `random_state=42`.
- **XGBoost:** `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `random_state=42`, `eval_metric='mlogloss'`.
- **MLP Architecture:** `hidden_layer_sizes=(100, 50)`, `activation='relu'`, `max_iter=500`, `random_state=42`.
- **SVM Parameters:** `kernel='rbf'`, `C=1.0`, `probability=True`, `random_state=42`.
- **RandomizedSearchCV:** `n_iter=20`, `cv=5`, `random_state=42`.
- **Calibration Parameters:** Isotonic regression fitted on validation probabilities (`cv='prefit'`).

### 9. Prominent Calibration Methodology Clarification
- **Explicit Statement:** *"Calibration method selection was performed exclusively on the validation set. The test set was not used for selecting isotonic regression. After selection, the calibrated model was evaluated once on the held-out test set."*

### 10. Explicit SHAP Labeling & Scope
- **Updated Caption/Heading:** *"XGBoost component-level SHAP explanation"*
- **Clarification:** Honest description noting that SHAP explains the XGBoost component feature attributions rather than the full RF + XGBoost ensemble.

### 11. Component Ablation Metrics Adjustment
- **Reasoning:** Accuracy is saturated at $1.0000$ on the 46-example test set.
- **Added Note:** *"Because point accuracy is saturated on this dataset, calibration and selective-prediction metrics provide more informative evidence for component-level comparison."*
- **Ablation Comparison:** Showed calibration reducing Brier Score from $0.0019 \rightarrow 0.0001$, ECE from $0.2341 \rightarrow 0.0141$, and Log Loss from $0.2875 \rightarrow 0.0148$.

### 12. Non-Clinical Framing & Terminology
- **Language Alignment:** Consistently replaced diagnostic claims with decision support, candidate condition, disease prediction, and research prototype.

---

## Artifacts Generated

### CSV Files
- `results/sequential_questioning_statistical_test.csv` (Paired t-test, Wilcoxon, 95% CI, p-values)
- `results/sequential_questioning_evaluation.csv` (MI vs Random paired evaluation)
- `results/ablation_study.csv` (Component metrics: Brier, ECE, Log Loss, Coverage)
- `results/final_calibrated_metrics.csv` (Test set calibration metrics)
- `results/threshold_analysis.csv` (Abstention sweep)
- `results/generalization_table.csv` (Representative disease classes)

### Figures
- `results/system_architecture.png` (Labeled with XGBoost Component-Level SHAP)
- `results/sequential_questioning_comparison.png` (Paired MI vs Random comparison plots)
- `results/ablation_study_visualization.png` (Brier Score, ECE, Log Loss comparison)
- `results/confidence_abstention_curve.png` (Coverage & error rate vs threshold)
- `results/reliability_diagram_comparison.png` (Calibration reliability curves)
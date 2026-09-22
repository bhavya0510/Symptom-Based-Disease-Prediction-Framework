# Methodology

## Data and split protocol

`stage1_data_splitting.py` loads the two raw symptom CSVs (Training.csv and Testing.csv), removes empty-header and `Unnamed:*` columns before concatenation, then removes exact duplicate rows at the pattern level to prevent data leakage. The dataset contains 4,962 total rows but only 305 unique symptom patterns. Class support ranges from 5 to 10 unique patterns per disease, so all estimates have substantial uncertainty.

The unique rows are split once, with stratification by diagnosis, into 70% train (213 samples), 15% validation (46 samples), and 15% test (46 samples). The script asserts zero symptom-pattern overlap between every pair of splits (Train-Test overlap: 0, Val-Test overlap: 0). It saves stratified 5-fold CV splits to `data/processed/cv_splits.pkl` for hyperparameter tuning.

`results/per_class_split_counts.csv` documents each diagnosis's total unique patterns, train, validation, and test counts. The total count ranges from 5 to 10; six classes have only 5 unique patterns (AIDS, Acne, Allergy, Gastroenteritis, Heart attack, Urinary tract infection), making held-out results especially unstable.

**Generalization reporting scope**: The project reports representative class support counts and the overall dataset limitation, but it does not claim a full per-class performance table because the test split contains only 46 total samples and most classes contribute 1–2 examples. The class-support summary is therefore used to document pattern scarcity and generalization risk rather than to infer stable per-disease accuracy estimates.

**Enhanced Analysis (Stage 9)**: `stage9_duplicate_leakage_analysis.py` provides comprehensive analysis of dataset limitations, including pattern frequency distribution, symptom sparsity analysis, and detailed duplicate analysis. This stage quantifies the impact of limited unique patterns on model performance.

**Reproducibility and evidence**: The authoritative numerical claims in this repository are the CSV and PNG artifacts saved under `research/results/` by the stage scripts. The documentation reports only metrics directly produced by those scripts; it does not claim that every final figure was regenerated in a separate pass beyond the recorded artifacts.

## Evaluation: 5-fold CV + Held-Out Test

The primary model comparison uses stratified 5-fold cross-validation on the 305 unique rows. `results/model_comparison_cv.csv` reports CV F1-Macro as mean ± standard deviation. The held-out test set (46 samples) provides deployment metrics in `results/model_comparison_test.csv`.

**⚠️ DATASET LIMITATION**: This dataset has only 305 unique patterns for 4,962 total rows. Some disease classes have only 5 unique patterns, leading to high variance in estimates. XGBoost shows realistic performance (CV F1-Macro: 0.9081 ± 0.0479, Test F1-Macro: 0.9837, Test Accuracy: 0.9783) while other models achieve perfect scores (1.0000) due to the small dataset size.

## Models and ensemble

Five baseline models were trained with hyperparameter tuning via RandomizedSearchCV on the CV folds:
- Logistic Regression (solver='lbfgs', C=[0.1, 1, 10])
- SVM (RBF kernel, C=[0.1, 1, 10], gamma=['scale', 'auto'])
- Random Forest (n_estimators=[50, 100], max_depth=[5, 10, None])
- XGBoost (max_depth=[3, 6], learning_rate=[0.1, 0.3], n_estimators=[50, 100])
- MLP (hidden_layer_sizes=[(50,), (100,)], alpha=[0.001, 0.01])

Two ensemble approaches were evaluated:
- Soft-voting ensemble (RF + XGBoost)
- Stacking ensemble (RF + XGBoost → Logistic Regression)

Both achieved 1.0000 test F1-Macro. Soft-voting was selected as the production ensemble.

## Probability calibration

Stage 4 applied isotonic regression calibration to the ensemble using the validation set (46 samples). **Enhanced calibration metrics** now include:
- Brier Score (multiclass)
- Expected Calibration Error (ECE)
- Log Loss
- Reliability diagrams (before/after calibration)

Brier scores: sigmoid=0.0032, isotonic=0.0000. Isotonic calibration was selected. The calibrated ensemble achieved 1.0000 test accuracy and 1.0000 test F1-Macro. Calibration improvements are quantified and visualized in `results/calibration_metrics.csv` and reliability diagrams.

## Top-K predictions

Stage 5 computed Top-1, Top-3, and Top-5 accuracy on the test set:
- Calibrated Ensemble: Top-1=1.0000, Top-3=1.0000, Top-5=1.0000
- XGBoost: Top-1=0.9783, Top-3=0.9783, Top-5=1.0000
- Other baselines: 1.0000 across all Top-K metrics

## Uncertainty and abstention

Stage 6 swept confidence thresholds from 0.10 to 0.95 on the validation set. **Enhanced abstention analysis** now includes:
- Coverage metrics (proportion of samples covered)
- Error rate analysis (error on confident predictions)
- Comprehensive threshold sweep table
- Visualization of accuracy vs. abstention rate vs. coverage

Optimal threshold: 0.10 (minimum due to high model confidence). Test set: 0% abstention rate (model very confident, average confidence=0.9793). The abstention mechanism is implemented but rarely triggers due to dataset characteristics. Full analysis available in `results/threshold_analysis.csv`.

## Sequential symptom questioning

Stage 7 computed mutual information for 132 symptoms × 41 diseases using training data. A greedy information-gain selector picks the most informative next symptom when confidence is below threshold. The system is stateful and supports up to 5 follow-up questions per session.

**Enhanced evaluation (Stage 7)**: Now includes:
- Average questions needed for different initial symptom counts
- Confidence improvement metrics from sequential questioning
- Accuracy impact analysis
- Comprehensive evaluation on test set samples

Evaluation results are saved in `results/sequential_questioning_evaluation.csv`.

## SHAP explainability

Stage 8 fitted a SHAP TreeExplainer on the XGBoost component of the ensemble. Global summary plots show the most important symptoms (phlegm, acute_liver_failure, irritability, blurred_and_distorted_vision, swelling_of_stomach). Per-prediction explanations return top contributing symptoms with direction (positive/negative impact).

## Ablation study (Stage 10 - NEW)

`stage10_ablation_study.py` systematically evaluates the contribution of each system component:
1. Baseline (best single model)
2. Ensemble without calibration
3. Ensemble with calibration
4. Ensemble with calibration + abstention
5. Full system (all components)

This quantifies the incremental improvement from each enhancement and helps understand which components provide the most value. Results are saved in `results/ablation_study.csv` with visualizations in `results/ablation_study_visualization.png`.

## Production deployment

The Flask backend (`HealthGuardAI/backend/app.py`) serves:
- `POST /predict`: Returns top-K predictions, probabilities, SHAP explanations, and abstention flag
- `POST /followup`: Sequential questioning with mutual information-based symptom selection
- `GET /health`: System health check

All artifacts are in `HealthGuardAI/backend/artifacts/`:
- `model.pkl`: Calibrated ensemble (RF + XGBoost)
- `encoder.pkl`: Label encoder (41 disease classes)
- `metadata.pkl`: Symptom names, model type, calibration method, abstention threshold
- `mutual_information.pkl`: MI matrix for sequential questioning
- `shap_system.pkl`: SHAP explainer and XGBoost model
- `abstention_config.pkl`: Optimal confidence threshold (0.10)
- **NEW**: Enhanced results files with calibration metrics, threshold analysis, sequential questioning evaluation, duplicate analysis, and ablation study

## Limitations

**Critical dataset limitation**: This dataset has only 305 unique symptom patterns for 4,962 total rows. Six disease classes have only 5 unique patterns each, making held-out evaluation results highly unstable. Perfect scores for most models reflect dataset characteristics rather than generalizable performance. XGBoost shows more realistic performance (CV F1-Macro: 0.9081 ± 0.0479), suggesting it's less prone to overfitting this small dataset.

This is a small, synthetic symptom-pattern dataset and is not clinical validation. No medical decision should be made from this model.

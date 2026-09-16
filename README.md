# HealthGuard AI: Adaptive Symptom-Based Diagnostic Engine

> **Explainable, Uncertainty-Aware & Sequential Disease Prediction Framework**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-brightgreen.svg)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An advanced machine learning framework and interactive web application for symptom-based disease prediction. Built with calibrated ensemble learning, local/global SHAP model explainability, confidence-based abstention, and stateful sequential symptom acquisition.

---

## 🌟 Key Features & Research Contributions

1. **Upgraded Dataset & Strict Deduplication Protocol**
   - Trained on the *Disease Prediction Using Machine Learning* dataset (**4,962 total records, 132 binary symptom features, 41 disease classes**).
   - Applied pattern-level deduplication (**305 unique symptom vectors**) to eliminate exact duplicate records across train/validation/test splits, preventing severe data leakage.

2. **Rigorous Model Benchmarking**
   - Evaluates 5 baseline models (**Logistic Regression, SVM, Random Forest, XGBoost, Neural Network / MLP**) against the proposed ensemble.
   - Evaluated across **Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), Macro-F1, and Weighted-F1** using stratified 5-fold cross-validation and held-out test set.

3. **Proposed Calibrated Ensemble**
   - **Soft-Voting Ensemble** combining **Random Forest** and **XGBoost**.
   - Calibrated using **Isotonic Regression** to convert raw model scores into true probability distributions.

4. **Explainable AI (SHAP Interpretability)**
   - Integrated SHAP (`TreeExplainer`) to compute local feature attribution for every individual prediction, highlighting positive (aggravating) and negative (mitigating) symptom impacts.

5. **Top-K Differential Diagnosis**
   - Provides ranked candidate diseases (Top-1, Top-3, Top-5) alongside calibrated confidence scores.

6. **Uncertainty Quantification & Abstention**
   - Automated confidence thresholding (`0.10`). If maximum prediction confidence is below threshold, the system abstains from making a forced diagnosis and prompts:
     > *"Insufficient information — please provide additional symptoms"*

7. **Sequential Symptom Acquisition**
   - Stateful follow-up questioning driven by a **Mutual Information matrix** ($132 \text{ symptoms} \times 41 \text{ diseases}$). Dynamically selects unasked symptoms with maximum expected information gain when predictions are uncertain.

---

## 📸 Web Application Screenshots

### 1. Hero Dashboard & System Overview
![Hero Dashboard](docs/screenshots/hero_dashboard.png)

### 2. AI Disease Prediction & SHAP Explanation Factors
![Disease Prediction Results](docs/screenshots/disease_prediction_results.png)

### 3. Dark Mode UI Preview
![Dark Mode Preview](docs/screenshots/dark_mode_preview.png)

---

## 📊 Empirical Model Performance Comparison

| Model | CV F1-Macro | Test Accuracy | Test Precision | Test Recall | Test F1-Macro |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **SVM** | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Random Forest** | 0.9940 ± 0.0120 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **XGBoost** | 0.9081 ± 0.0479 | 0.9783 | 0.9878 | 0.9878 | 0.9837 |
| **MLP** | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Calibrated Ensemble (RF + XGB)** | **1.0000 ± 0.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

*Note: High baseline metric scores reflect dataset characteristics (305 unique symptom patterns for 41 classes). XGBoost demonstrates realistic variance under stratified 5-fold cross-validation.*

---

## 🔄 System Architecture & Data Flow

```
          Symptom Dataset (4,962 records / 132 symptoms / 41 diseases)
                                      ↓
                     Data Cleaning & Pattern Deduplication (305 unique)
                                      ↓
                        Stratified Train / Val / Test (70/15/15)
                                      ↓
                       ┌──────────────┼──────────────┐
                       ↓              ↓              ↓
                      SVM             RF          XGBoost
                       ↓              ↓              ↓
                       └──────────────┼──────────────┘
                                      ↓
                          Proposed Soft-Voting Ensemble
                                      ↓
                       Isotonic Probability Calibration
                                      ↓
                           Uncertainty Detection
                                      ↓
                      ┌───────────────┴───────────────┐
                      ↓                               ↓
               Confident (≥ threshold)         Uncertain (< threshold)
                      ↓                               ↓
               Top-K Diseases                  Ask Useful Symptoms (MI)
                      ↓                               ↓
               SHAP Explanation                Update Prediction Loop
                                      ↓
                             Flask Web Application
```

---

## ⚡ Quick Start & Installation

### Prerequisites
- Python 3.9+
- pip & virtualenv

### 1. Clone & Set Up Research Pipeline
```bash
# Clone repository
git clone https://github.com/bhavya0510/Symptom-Based-Disease-Prediction-Framework.git
cd Symptom-Based-Disease-Prediction-Framework/research

# Create virtual environment & install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run complete training pipeline (Stages 1 to 8)
python train_pipeline.py
```

### 2. Launch Flask Web Server
```bash
# Start Flask app
cd ../HealthGuardAI/backend
../../research/venv/bin/python app.py
```
Open **`http://127.0.0.1:5050`** in your browser.

---

## 🔌 API Reference

| Endpoint | Method | Description |
|:---|:---:|:---|
| `/predict` | `POST` | Accepts user symptom array; returns Top-K predictions, SHAP explanations, and abstention status. |
| `/followup` | `POST` | Sequential questioning endpoint driven by Mutual Information. |
| `/symptoms` | `GET` | Returns full list of 132 available binary symptom features. |
| `/model-comparison` | `GET` | Returns model evaluation benchmarks across CV and Test sets. |
| `/health` | `GET` | System health check and model artifact status. |

---

## 📁 Repository Structure

```
.
├── HealthGuardAI/               # Production Web Application
│   ├── backend/                 # Flask API Server & Artifacts
│   │   ├── app.py               # Flask backend application
│   │   ├── shap_utils.py        # Local SHAP explanation helper
│   │   └── artifacts/           # Calibrated model, metadata, & SHAP explainer
│   └── frontend/                # Web Dashboard
│       ├── index.html           # Main HTML structure
│       ├── script.js            # Frontend logic & dynamic API calls
│       └── style.css            # Modern CSS design system & dark mode
├── research/                    # Machine Learning Research Framework
│   ├── train_pipeline.py        # Master pipeline executor (Stages 1-8)
│   ├── stage1_data_splitting.py # Data cleaning & stratified deduplication
│   ├── stage2_baseline_comparison.py # Baseline model benchmarking
│   ├── stage3_ensemble_model.py # Soft-voting & stacking ensemble build
│   ├── stage4_probability_calibration.py # Isotonic probability calibration
│   ├── stage5_topk_metrics.py   # Top-1/3/5 accuracy metrics
│   ├── stage6_uncertainty_abstention.py # Confidence threshold sweep
│   ├── stage7_sequential_questioning.py # Mutual Information calculator
│   ├── stage8_shap_explainability.py    # SHAP global & local explainers
│   ├── data/                    # Cleaned & processed datasets
│   └── results/                 # Metrics CSVs, confusion matrices & plots
├── docs/screenshots/            # Application Screenshots
├── New Dataset/                 # Raw Kaggle Training & Testing CSVs
└── README.md                    # Main Project Documentation
```

---

## ⚠️ Medical Disclaimer
This project is a decision-support prototype created strictly for academic research and demonstration purposes. It does not provide medical advice or diagnosis. Always consult a qualified healthcare professional for medical evaluation.

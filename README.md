# Health Guard AI - Research-Grade Disease Prediction Framework

## Overview
**Explainable and Uncertainty-Aware Symptom-Based Disease Prediction Framework with Sequential Symptom Acquisition**

An advanced ML framework for disease prediction featuring:
- **Ensemble Learning**: Calibrated Random Forest + XGBoost ensemble
- **Probability Calibration**: Isotonic regression for reliable confidence estimates
- **Uncertainty Quantification**: Confidence-based abstention mechanism
- **Sequential Questioning**: Information gain-based follow-up symptom selection
- **Explainable AI**: SHAP-based model explanations
- **Top-K Predictions**: Differential diagnosis with multiple candidate diseases

**Dataset**: Kaggle Disease Prediction Using Machine Learning (4,962 samples, 132 symptoms, 41 diseases)

Model: Calibrated Ensemble (RF + XGBoost)  
Backend: Enhanced Flask API with research endpoints  
Frontend: Responsive HTML/CSS/JS with SHAP explanations  
ML Stack: scikit-learn, XGBoost, SHAP, pandas, joblib  

## 📸 Application Screenshots

### 1. Hero Dashboard & Overview
![HealthGuard AI Dashboard Overview](docs/screenshots/hero_dashboard.png)

### 2. Symptom Analysis & Disease Prediction Results
![Symptom Analysis & Disease Prediction Results](docs/screenshots/disease_prediction_results.png)

### 3. Dark Mode UI Preview
![Dark Mode UI Preview](docs/screenshots/dark_mode_preview.png)

## Dataset Details
| Feature | Type | Values |
|---------|------|---------|
| Fever | Binary | Yes/No |
| Cough | Binary | Yes/No |
| Fatigue | Binary | Yes/No |
| Difficulty Breathing | Binary | Yes/No |
| Age | Numeric | 19-90 |
| Gender | Categorical | Male/Female |
| Blood Pressure | Categorical | Low/Normal/High |
| Cholesterol Level | Categorical | Low/Normal/High |
| Target | Multi-class (116 diseases) | Influenza, Asthma, Pneumonia... |

## Quick Start

### Research Framework Setup
```bash
# Navigate to research directory
cd research

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run training pipeline (Stages 1-4)
python train_pipeline.py

# This will:
# - Clean and split data
# - Train baseline models
# - Build ensemble model
# - Apply probability calibration
# - Copy artifacts to backend
```

### Run Enhanced Application
```bash
# Start Flask backend with research features
cd HealthGuardAI/backend
../../research/venv/bin/python app.py
```
URL: http://127.0.0.1:5050

### Legacy Quick Start (Original Dataset)
```bash
cd HealthGuardAI
pip install -r requirements.txt
cd backend
python train_model.py  # Uses legacy dataset
python app.py
```
URL: http://127.0.0.1:5000

## API Endpoints

### Enhanced Research Endpoints (Port 5050)

**POST /predict** - Enhanced prediction with research features
```json
{
  "symptoms": ["itching", "skin_rash", "high_fever"],
  "top_k": 3
}
```
**Response**:
```json
{
  "predictions": [
    {"disease": "Fungal infection", "confidence": 0.92},
    {"disease": "Allergy", "confidence": 0.05},
    {"disease": "Drug Reaction", "confidence": 0.02}
  ],
  "top_prediction": "Fungal infection",
  "max_confidence": 0.92,
  "should_abstain": false,
  "abstention_threshold": 0.10,
  "shap_explanation": [
    {"symptom": "itching", "shap_value": 0.45, "impact": "positive"},
    {"symptom": "skin_rash", "shap_value": 0.32, "impact": "positive"}
  ],
  "disclaimer": "This is an AI decision-support prototype, not medical advice."
}
```

**POST /followup** - Sequential symptom questioning
```json
{
  "session_id": "user123",
  "symptoms": ["itching"],
  "max_questions": 5
}
```
**Response** (if uncertain):
```json
{
  "status": "question",
  "next_question": "Do you have skin rash?",
  "symptom_name": "skin_rash",
  "current_confidence": 0.35,
  "questions_asked": ["skin_rash"],
  "questions_remaining": 4
}
```
**Response** (if confident):
```json
{
  "status": "final",
  "prediction": "Fungal infection",
  "confidence": 0.92,
  "questions_asked": ["skin_rash", "nodal_skin_eruptions"],
  "disclaimer": "This is an AI decision-support prototype, not medical advice."
}
```

**GET /health** - System health check
```json
{
  "status": "healthy",
  "model_type": "Calibrated Ensemble (RF + XGBoost)",
  "n_classes": 41,
  "n_features": 132,
  "shap_available": true,
  "abstention_available": true
}
```

### Legacy Endpoint (Port 5000 - Original Dataset)
**POST /predict** - Original simple prediction
```json
{
  "symptoms": ["Fever", "Cough"],
  "age": 25,
  "gender": "Male"
}
```

## Frontend Features
- **Enhanced Results**: Top-K disease predictions with confidence bars
- **SHAP Explanations**: Key contributing factors for predictions
- **Abstention Handling**: "Insufficient information" messages when uncertain
- **Sequential Questioning**: Chat-style follow-up interface (UI ready)
- **Responsive Design**: Mobile/desktop compatibility
- **Dark Mode**: Theme switching
- **Loading States**: Progress indicators
- **Error Handling**: Graceful degradation

## Model Performance

### Research Framework Results
- **Dataset**: 4,962 samples, 41 diseases, 132 symptoms
- **Calibrated Ensemble**: 99.6% accuracy, 99.6% F1-macro
- **Top-K Accuracy**: Top-1 = 99.6%, Top-3 = 100%, Top-5 = 100%
- **Calibration**: Brier score improved from 0.0030 to 0.0000
- **Uncertainty**: Optimal threshold = 0.10, abstention rate = 0%

### Baseline Comparison
All models achieved near-perfect performance due to dataset limitations:
- Logistic Regression: 100% accuracy
- SVM: 100% accuracy
- Random Forest: 100% accuracy
- XGBoost: 100% accuracy
- MLP: 100% accuracy

### Legacy Model Results (Archived)
- **Original Dataset**: 349 samples, 116 diseases, 4 symptoms + patient profile
- **Location**: `research/data/legacy/Disease_symptom_and_patient_profile_dataset.csv`
- **Random Forest**: Trained on full dataset (no split for small classes)
- **Features**: 4 binary symptoms + 4 profile variables
- **Note**: Replaced by larger Kaggle dataset with 132 symptoms, 41 diseases

## Tech Stack
```
Research Framework:
- Backend: Python 3.14, Flask, scikit-learn 1.9, XGBoost 3.4, SHAP 0.52
- Frontend: HTML5, CSS3, Vanilla JS (enhanced)
- ML: Calibrated Ensemble (RF + XGBoost), Isotonic Calibration, SHAP TreeExplainer
- Research: Mutual Information, Sequential Questioning, Uncertainty Quantification

Legacy System:
- Backend: Python 3.12, Flask 3.0, scikit-learn 1.5
- Frontend: HTML5, CSS3, Vanilla JS
- ML: RandomForestClassifier, ColumnTransformer, LabelEncoder
```

## Project Structure
```
Symptom-Based-Disease-Prediction-main/
├── research/                          # Research framework
│   ├── stage0_setup.py               # Dataset verification
│   ├── stage1_data_splitting.py      # Data cleaning & splitting
│   ├── stage2_baseline_comparison.py # Baseline model training
│   ├── stage3_ensemble_model.py      # Ensemble development
│   ├── stage4_probability_calibration.py # Probability calibration
│   ├── stage5_topk_metrics.py        # Top-K accuracy computation
│   ├── stage6_uncertainty_abstention.py # Uncertainty quantification
│   ├── stage7_sequential_questioning.py # Sequential questioning
│   ├── stage8_shap_explainability.py # SHAP integration
│   ├── train_pipeline.py             # End-to-end training pipeline
│   ├── requirements.txt              # Research dependencies
│   ├── METHODOLOGY.md                # Comprehensive methodology
│   ├── RESULTS_SUMMARY.md            # Results summary
│   ├── data/
│   │   ├── raw/                     # Original datasets
│   │   ├── processed/               # Cleaned & split data
│   │   └── legacy/                  # Legacy dataset
│   └── results/                     # Research artifacts
│       ├── model_comparison.csv      # Model performance table
│       ├── shap_summary_*.png        # SHAP visualizations
│       ├── reliability_diagram.png   # Calibration visualization
│       └── *.pkl                     # Trained models & systems
├── HealthGuardAI/
│   ├── backend/
│   │   ├── app.py                   # Enhanced Flask API
│   │   ├── app_old_backup.py        # Original Flask API (backup)
│   │   ├── train_model.py           # Legacy training script
│   │   ├── create_metadata.py       # Metadata generation
│   │   ├── artifacts/               # Production model artifacts
│   │   │   ├── model.pkl           # Calibrated ensemble
│   │   │   ├── encoder.pkl         # Label encoder
│   │   │   ├── metadata.pkl         # System metadata
│   │   │   ├── mutual_information.pkl # MI matrix
│   │   │   ├── shap_system.pkl      # SHAP explainer
│   │   │   └── abstention_config.pkl # Abstention config
│   │   ├── model.pkl                # Legacy model (if exists)
│   │   ├── encoder.pkl              # Legacy encoder (if exists)
│   │   └── metadata.pkl             # Legacy metadata (if exists)
│   ├── frontend/
│   │   ├── index.html
│   │   ├── style.css
│   │   └── script.js                # Enhanced with research features
│   ├── dataset/
│   │   └── Disease_symptom...csv    # Legacy dataset
│   └── requirements.txt             # Legacy dependencies
├── New Dataset/                      # Original enhanced dataset
│   ├── Training.csv
│   └── Testing.csv
├── Disease_symptom...csv            # Legacy dataset (root level)
└── README.md                        # This file
```

## Limitations & Disclaimer

### Research Framework Limitations
- **Dataset Limitations**: Only 305 unique patterns for 4,962 samples limits generalization assessment
- **Pattern Overlap**: Significant overlap between train/test splits affects traditional metrics
- **SHAP Scope**: Explanations limited to XGBoost component (ensemble limitation)
- **Abstention Rate**: Low abstention due to high model confidence (dataset characteristic)

### General Limitations
- **Educational Prototype**: Not for clinical diagnosis without validation
- **Not Medical Advice**: Always consult healthcare professionals
- **Dataset Specific**: Performance may not generalize to real clinical data
- **Symptom Coverage**: Limited to 132 symptoms in training dataset

### Disclaimer
This is an AI decision-support prototype for research purposes. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Troubleshooting

### Research Framework
```bash
# Training pipeline fails
cd research
source venv/bin/activate
pip install -r requirements.txt  # Ensure all dependencies installed

# Backend import errors
cd HealthGuardAI/backend
../../research/venv/bin/python app.py  # Use research venv

# SHAP errors
# SHAP requires specific dependencies - ensure research venv is activated
```

### Legacy System
```bash
Flask not found → python app.py
Model not found → rerun train_model.py
CORS error → flask-cors installed
```

## Research Documentation

- **Methodology**: `research/METHODOLOGY.md` - Comprehensive research methodology
- **Results**: `research/RESULTS_SUMMARY.md` - Detailed results summary
- **Artifacts**: `research/results/` - All generated plots, models, and data files

## Citation

If you use this research framework, please cite:

```
Explainable and Uncertainty-Aware Symptom-Based Disease Prediction Framework
with Sequential Symptom Acquisition
```

## Future Work

1. **Dataset Expansion**: Acquire larger, diverse clinical datasets
2. **External Validation**: Test on independent datasets
3. **Clinical Integration**: Pilot testing with healthcare professionals
4. **User Studies**: Evaluate explainability and uncertainty communication
5. **Model Enhancement**: Explore more sophisticated architectures
6. **Real-world Deployment**: Clinical validation studies

Ready for production demo! All specs implemented.

# HealthGuard AI: Adaptive Symptom-Based Disease Prediction Prototype

An interactive web application demonstrating the **Explainable, Uncertainty-Aware Sequential Evaluation Framework for Symptom-Based Disease Prediction**.

---

## Screenshots & Visual Walkthrough

### 1. Hero Dashboard & Clinical Overview
![Hero Dashboard](../docs/screenshots/hero_dashboard.png)

### 2. Symptom Duration & Live Clinical Triage
![Symptom Duration Tracker](../docs/screenshots/symptom_duration_tracker.png)

### 3. AI Health Assessment & Prolonged Fever Clinical Advisory
![Disease Prediction Results](../docs/screenshots/disease_prediction_results.png)

### 4. Recommended Diagnostic Laboratory Tests (CBC, Dengue NS1, Typhoid Widal)
![Recommended Medical Diagnostic Tests](../docs/screenshots/clinical_diagnostic_tests.png)

### 5. Printable Doctor Consultation Summary Sheet
![Doctor Consultation Sheet](../docs/screenshots/doctor_consultation_sheet.png)

### 6. Modern Dark Mode Theme
![Dark Mode Preview](../docs/screenshots/dark_mode_preview.png)

---

## 🚀 Running the Web Application

```bash
# 1. Navigate to backend directory
cd backend

# 2. Run Flask server
../../venv/bin/python app.py
```

Access the UI at: **`http://127.0.0.1:5050`**

---

## 🔌 Core API Endpoints

- **`POST /predict`**: Returns Top-K disease candidates, calibrated probabilities, XGBoost component-level SHAP explanation factors, and abstention status.
- **`POST /followup`**: Interactive follow-up questioning based on Mutual Information calculation.
- **`GET /symptoms`**: Returns full list of 132 binary symptom features.
- **`GET /model-comparison`**: Returns research benchmark evaluation metrics across 5 baseline models and the calibrated ensemble.
- **`GET /health`**: Server health check.

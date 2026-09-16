# HealthGuard AI: Adaptive Symptom-Based Diagnostic Engine

An interactive web application demonstrating the **Explainable, Uncertainty-Aware & Sequential Disease Prediction Framework**.

---

## 📸 Screenshots

### 1. Hero Dashboard & Overview
![Hero Dashboard](../docs/screenshots/hero_dashboard.png)

### 2. Disease Prediction & SHAP Key Factors
![Disease Prediction Results](../docs/screenshots/disease_prediction_results.png)

### 3. Dark Mode UI Preview
![Dark Mode Preview](../docs/screenshots/dark_mode_preview.png)

---

## 🚀 Running the Web Application

```bash
# 1. Navigate to backend directory
cd backend

# 2. Run Flask server
../../research/venv/bin/python app.py
```

Access the UI at: **`http://127.0.0.1:5050`**

---

## 🔌 Core API Endpoints

- **`POST /predict`**: Returns Top-K disease candidates, calibrated probabilities, local SHAP explanation factors, and abstention status.
- **`POST /followup`**: Interactive follow-up questioning based on Mutual Information calculation.
- **`GET /symptoms`**: Returns full list of 132 binary symptom features.
- **`GET /model-comparison`**: Returns research benchmark evaluation metrics across 5 baseline models and the ensemble.
- **`GET /health`**: Server health check.

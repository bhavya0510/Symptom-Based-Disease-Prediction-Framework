"""Importable SHAP helpers; no functions from a ``__main__`` training script are pickled."""
import numpy as np

def explain_prediction(explainer, xgb_model, input_frame, symptom_names, top_k=5):
    values = explainer.shap_values(input_frame)
    predicted = int(xgb_model.predict(input_frame)[0])
    if isinstance(values, list):
        values = values[predicted]
    if values.ndim == 3:
        values = values[0]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    indices = np.argsort(np.abs(values[0]))[-top_k:][::-1]
    return [{"symptom": symptom_names[int(i)], "shap_value": float(values[0][i]),
             "impact": "positive" if values[0][i] > 0 else "negative"} for i in indices]

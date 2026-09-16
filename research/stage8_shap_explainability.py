"""
STAGE 8: Explainable AI (SHAP)
Integrate SHAP for model explainability:
- Fit TreeExplainer on XGBoost component of ensemble
- Generate global SHAP summary plots
- Create per-prediction SHAP explanations
- Implement production function for real-time explanations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STAGE 8: SHAP Explainable AI Integration")
print("=" * 80)

# Load training data for SHAP background
print("\nLoading training data...")
train_df = pd.read_csv('data/processed/train.csv')
X_train = train_df.drop(columns=['prognosis'])
y_train = train_df['prognosis']

print(f"Training data: {X_train.shape}")

# Load label encoder
le = joblib.load('results/label_encoder.pkl')
print(f"Classes: {len(le.classes_)}")

# Extract XGBoost model from ensemble
print("\nExtracting XGBoost model from ensemble...")
best_models = joblib.load('results/baseline_models.pkl')
xgb_model = best_models['XGBoost']
print("✅ Extracted XGBoost model")

# Fit SHAP TreeExplainer
print("\nFitting SHAP TreeExplainer...")
explainer = shap.TreeExplainer(xgb_model)
print("✅ SHAP explainer fitted")

# Compute SHAP values for a sample of training data (for global plots)
print("\nComputing SHAP values for global analysis...")
sample_size = min(100, len(X_train))  # Use subset for efficiency
X_sample = X_train.sample(n=sample_size, random_state=42)
shap_values = explainer.shap_values(X_sample)

print(f"✅ Computed SHAP values for {sample_size} samples")

# Generate global SHAP summary plot
print("\nGenerating global SHAP summary plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title('Global SHAP Summary - Feature Importance')
plt.tight_layout()
plt.savefig('results/shap_summary_bar.png', dpi=300, bbox_inches='tight')
print("✅ Saved SHAP summary bar plot to results/shap_summary_bar.png")

# Generate detailed summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title('Global SHAP Summary - Feature Impact')
plt.tight_layout()
plt.savefig('results/shap_summary_detailed.png', dpi=300, bbox_inches='tight')
print("✅ Saved SHAP summary detailed plot to results/shap_summary_detailed.png")

# Function to get SHAP explanation for a single prediction
def get_shap_explanation(input_vector, explainer, symptom_names, top_k=10):
    """
    Get SHAP explanation for a single prediction
    
    Args:
        input_vector: pandas DataFrame with symptom values
        explainer: fitted SHAP explainer
        symptom_names: list of symptom names
        top_k: number of top features to return
    
    Returns:
        explanation: dict with top contributing symptoms and their SHAP values
    """
    # Compute SHAP values for this prediction
    shap_values_single = explainer.shap_values(input_vector)
    
    # Get the predicted class
    prediction = xgb_model.predict(input_vector)[0]
    
    # Get SHAP values for the predicted class
    if isinstance(shap_values_single, list):
        # Multi-class case - SHAP returns list of arrays
        shap_values_class = shap_values_single[prediction]
    else:
        # Binary case
        shap_values_class = shap_values_single
    
    # Handle different SHAP output formats
    if len(shap_values_class.shape) == 3:
        shap_values_class = shap_values_class[0]  # Remove sample dimension
    elif len(shap_values_class.shape) == 1:
        shap_values_class = shap_values_class.reshape(1, -1)
    
    # Get absolute SHAP values and sort
    abs_shap = np.abs(shap_values_class[0])
    top_indices = np.argsort(abs_shap)[-top_k:][::-1]
    
    # Create explanation
    explanation = {
        'predicted_class': le.classes_[prediction],
        'top_features': []
    }
    
    for idx in top_indices:
        symptom = symptom_names[int(idx)]
        shap_value = shap_values_class[0][int(idx)]
        explanation['top_features'].append({
            'symptom': symptom,
            'shap_value': float(shap_value),
            'impact': 'positive' if shap_value > 0 else 'negative'
        })
    
    return explanation

# Test SHAP explanation function
print("\nTesting SHAP explanation function...")

# Create a test case (symptoms for liver disease)
test_symptoms = {symptom: 0 for symptom in X_train.columns}
test_symptoms['yellowing_of_eyes'] = 1
test_symptoms['nausea'] = 1
test_symptoms['loss_of_appetite'] = 1
test_symptoms['abdominal_pain'] = 1

test_input = pd.DataFrame([test_symptoms])
explanation = get_shap_explanation(test_input, explainer, X_train.columns.tolist())

print(f"Test case: Liver disease symptoms")
print(f"Predicted class: {explanation['predicted_class']}")
print(f"Top contributing features:")
for feature in explanation['top_features'][:5]:
    print(f"  {feature['symptom']}: {feature['shap_value']:.4f} ({feature['impact']})")

# Generate per-prediction SHAP plots for paper examples
print("\nGenerating example per-prediction SHAP explanations...")

# Example 1: Fungal infection
test_symptoms_1 = {symptom: 0 for symptom in X_train.columns}
test_symptoms_1['itching'] = 1
test_symptoms_1['skin_rash'] = 1
test_symptoms_1['nodal_skin_eruptions'] = 1
test_input_1 = pd.DataFrame([test_symptoms_1])

explanation_1 = get_shap_explanation(test_input_1, explainer, X_train.columns.tolist())
print(f"Example 1 - Fungal infection: {explanation_1['predicted_class']}")
for feature in explanation_1['top_features'][:5]:
    print(f"  {feature['symptom']}: {feature['shap_value']:.4f} ({feature['impact']})")

# Example 2: Malaria
test_symptoms_2 = {symptom: 0 for symptom in X_train.columns}
test_symptoms_2['high_fever'] = 1
test_symptoms_2['chills'] = 1
test_symptoms_2['joint_pain'] = 1
test_symptoms_2['headache'] = 1
test_input_2 = pd.DataFrame([test_symptoms_2])

explanation_2 = get_shap_explanation(test_input_2, explainer, X_train.columns.tolist())
print(f"Example 2 - Malaria: {explanation_2['predicted_class']}")
for feature in explanation_2['top_features'][:5]:
    print(f"  {feature['symptom']}: {feature['shap_value']:.4f} ({feature['impact']})")

# Example 3: Typhoid
test_symptoms_3 = {symptom: 0 for symptom in X_train.columns}
test_symptoms_3['high_fever'] = 1
test_symptoms_3['headache'] = 1
test_symptoms_3['nausea'] = 1
test_symptoms_3['abdominal_pain'] = 1
test_symptoms_3['diarrhoea'] = 1
test_input_3 = pd.DataFrame([test_symptoms_3])

explanation_3 = get_shap_explanation(test_input_3, explainer, X_train.columns.tolist())
print(f"Example 3 - Typhoid: {explanation_3['predicted_class']}")
for feature in explanation_3['top_features'][:5]:
    print(f"  {feature['symptom']}: {feature['shap_value']:.4f} ({feature['impact']})")

# Save example explanations
example_explanations = {
    'fungal_infection': explanation_1,
    'malaria': explanation_2,
    'typhoid': explanation_3
}
joblib.dump(example_explanations, 'results/shap_example_explanations.pkl')
print("✅ Saved example explanations")

# Save data/model state only.  Do not pickle this script's __main__ functions:
# the backend imports its explanation logic from backend/shap_utils.py.
shap_system = {
    'explainer': explainer,
    'xgb_model': xgb_model,
    'symptom_names': X_train.columns.tolist()
}

joblib.dump(shap_system, 'results/shap_system.pkl')
print("✅ Saved SHAP system to results/shap_system.pkl")

# Print feature importance summary
print("\nTop 10 most important symptoms globally:")
# Handle different SHAP value structures
if isinstance(shap_values, list):
    # Multi-class case
    mean_abs_shap = np.mean([np.abs(np.mean(class_shap, axis=0)) for class_shap in shap_values], axis=0)
else:
    # Single output case
    mean_abs_shap = np.abs(np.mean(shap_values, axis=0))

if len(mean_abs_shap.shape) > 1:
    mean_abs_shap = mean_abs_shap.flatten()

# Ensure we have the right number of features
if len(mean_abs_shap) != len(X_train.columns):
    print(f"Warning: SHAP values shape {mean_abs_shap.shape} doesn't match features {len(X_train.columns)}")
    # Use feature importance from model instead
    importances = xgb_model.feature_importances_
    top_features_idx = np.argsort(importances)[-10:][::-1]
    for i, idx in enumerate(top_features_idx):
        symptom = X_train.columns[idx]
        importance = importances[idx]
        print(f"  {i+1}. {symptom}: {importance:.4f}")
else:
    top_features_idx = np.argsort(mean_abs_shap)[-10:][::-1]
    for i, idx in enumerate(top_features_idx):
        symptom = X_train.columns[idx]
        importance = mean_abs_shap[idx]
        print(f"  {i+1}. {symptom}: {importance:.4f}")

print(f"\n" + "="*80)
print("STAGE 8 SUMMARY")
print("="*80)
print(f"SHAP explainability system implemented:")
print(f"  - TreeExplainer fitted on XGBoost model")
print(f"  - Global summary plots generated")
print(f"  - Per-prediction explanation function implemented")
print(f"  - Example explanations for 3 disease cases")
print(f"  - Production-ready SHAP system saved")
print(f"  - Top 10 most important symptoms identified")
print("="*80)
print("✅ STAGE 8 COMPLETE!")
print("="*80)

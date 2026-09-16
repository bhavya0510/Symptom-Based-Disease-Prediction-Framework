"""
Enhanced Flask Backend for Symptom-Based Disease Prediction
Integrates research components: calibrated ensemble, SHAP, uncertainty, sequential questioning
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from shap_utils import explain_prediction

frontend_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_folder, static_url_path='')
CORS(app)

# Load model and artifacts
print("Loading model and artifacts...")
backend_dir = os.path.dirname(__file__)
artifacts_dir = os.path.join(backend_dir, 'artifacts')

required_artifacts = ('model.pkl', 'encoder.pkl', 'metadata.pkl')
missing_artifacts = [name for name in required_artifacts if not os.path.isfile(os.path.join(artifacts_dir, name))]
if missing_artifacts:
    raise RuntimeError(f"Required research artifacts are missing from {artifacts_dir}: {missing_artifacts}")
print("Using research artifacts only...")
model = joblib.load(os.path.join(artifacts_dir, 'model.pkl'))
label_encoder = joblib.load(os.path.join(artifacts_dir, 'encoder.pkl'))
metadata = joblib.load(os.path.join(artifacts_dir, 'metadata.pkl'))
    
shap_system = None
mutual_info = None
abstention_config = None
    
try:
    shap_system = joblib.load(os.path.join(artifacts_dir, 'shap_system.pkl'))
    print("✅ SHAP system loaded")
except Exception as error:
    print(f"⚠️  SHAP system not available: {error}")
    
try:
    mutual_info = joblib.load(os.path.join(artifacts_dir, 'mutual_information.pkl'))
    print("✅ Mutual information loaded")
except Exception as error:
    print(f"⚠️  Mutual information not available: {error}")
    
try:
    abstention_config = joblib.load(os.path.join(artifacts_dir, 'abstention_config.pkl'))
    print("✅ Abstention config loaded")
except Exception as error:
    print(f"⚠️  Abstention config not available: {error}")
    
symptom_names = metadata['symptom_names']
n_classes = metadata['n_classes']
confidence_threshold = abstention_config['optimal_threshold'] if abstention_config else 0.5

print(f"Loaded {n_classes} disease classes")
print(f"Symptom features: {len(symptom_names)}")
print("API ready!")

# In-memory session storage for sequential questioning
sessions = defaultdict(dict)

def get_shap_explanation(input_vector, explainer, xgb_model, symptom_names, le, top_k=5):
    """
    Get SHAP explanation for a single prediction (local function to avoid pickling issues)
    """
    # Compute SHAP values for this prediction
    shap_values_single = explainer.shap_values(input_vector)
    
    # Get the predicted class
    prediction = xgb_model.predict(input_vector)[0]
    
    # Get SHAP values for the predicted class
    if isinstance(shap_values_single, list):
        shap_values_class = shap_values_single[prediction]
    else:
        shap_values_class = shap_values_single
    
    # Handle different SHAP output formats
    if len(shap_values_class.shape) == 3:
        shap_values_class = shap_values_class[0]
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

def create_symptom_vector(symptoms_list):
    """Create symptom vector from list of symptom names"""
    symptom_vector = {symptom: 0 for symptom in symptom_names}
    for symptom in symptoms_list:
        if symptom in symptom_vector:
            symptom_vector[symptom] = 1
    return symptom_vector

@app.route('/predict', methods=['POST'])
def predict():
    """
    Enhanced prediction endpoint
    Returns top-K predictions with probabilities, SHAP explanations, and abstention flag
    """
    try:
        data = request.json
        
        # Extract symptoms
        symptoms_input = data.get('symptoms', [])
        top_k = data.get('top_k', 3)
        custom_threshold = data.get('abstention_threshold')
        
        # Use custom threshold if provided, otherwise use default
        current_threshold = custom_threshold if custom_threshold is not None else confidence_threshold
        
        # Create symptom vector
        symptom_vector = create_symptom_vector(symptoms_input)
        input_df = pd.DataFrame([symptom_vector])
        
        # Get predictions
        prediction_proba = model.predict_proba(input_df)[0]
        prediction = model.predict(input_df)[0]
        
        # Get top-K predictions
        top_indices = np.argsort(prediction_proba)[-top_k:][::-1]
        predictions = []
        for idx in top_indices:
            disease = label_encoder.inverse_transform([idx])[0]
            confidence = float(prediction_proba[idx])
            predictions.append({
                'disease': disease,
                'confidence': confidence
            })
        
        # Check for abstention
        max_confidence = float(np.max(prediction_proba))
        should_abstain = max_confidence < current_threshold
        
        # Get SHAP explanation if available
        shap_explanation = None
        if shap_system and not should_abstain:
            try:
                explainer = shap_system['explainer']
                xgb_model = shap_system['xgb_model']
                shap_explanation = get_shap_explanation(input_df, explainer, xgb_model, symptom_names, label_encoder)
                shap_explanation = shap_explanation['top_features']
            except Exception as e:
                print(f"SHAP explanation error: {e}")
        
        response = {
            'predictions': predictions,
            'top_prediction': predictions[0]['disease'] if predictions else None,
            'max_confidence': max_confidence,
            'should_abstain': should_abstain,
            'message': 'Insufficient information — please provide additional symptoms' if should_abstain else None,
            'abstention_threshold': current_threshold,
            'shap_explanation': shap_explanation,
            'disclaimer': 'This is an AI decision-support prototype, not medical advice. Consult a healthcare professional.'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/followup', methods=['POST'])
def followup():
    """
    Sequential questioning endpoint
    Returns next question or final prediction based on current state
    """
    try:
        data = request.json
        
        # Get session ID (or create one)
        session_id = data.get('session_id', 'default')
        
        # Get current symptom state
        current_symptoms = data.get('symptoms', [])
        max_questions = data.get('max_questions', 5)
        custom_threshold = data.get('abstention_threshold')
        
        # Use custom threshold if provided, otherwise use default
        current_threshold = custom_threshold if custom_threshold is not None else confidence_threshold
        
        # Update session state
        sessions[session_id]['symptoms'] = current_symptoms
        sessions[session_id]['questions_asked'] = sessions[session_id].get('questions_asked', [])
        
        # Create symptom vector
        symptom_vector = create_symptom_vector(current_symptoms)
        input_df = pd.DataFrame([symptom_vector])
        
        # Get current prediction
        prediction_proba = model.predict_proba(input_df)[0]
        max_confidence = float(np.max(prediction_proba))
        
        # Check if confident enough
        if max_confidence >= current_threshold or len(sessions[session_id]['questions_asked']) >= max_questions:
            # Return final prediction
            prediction = model.predict(input_df)[0]
            disease = label_encoder.inverse_transform([prediction])[0]
            
            return jsonify({
                'status': 'final',
                'prediction': disease,
                'confidence': max_confidence,
                'questions_asked': sessions[session_id]['questions_asked'],
                'disclaimer': 'This is an AI decision-support prototype, not medical advice. Consult a healthcare professional.'
            })
        
        # Get next question using mutual information
        next_symptom = None
        if mutual_info:
            try:
                # Simple implementation: select symptom with highest MI not yet asked
                available_symptoms = [s for s in symptom_names if s not in current_symptoms]
                if available_symptoms:
                    # Get top disease candidates
                    top_disease_indices = np.argsort(prediction_proba)[-5:][::-1]
                    
                    # Calculate expected information gain
                    best_mi = -1
                    for symptom in available_symptoms:
                        symptom_idx = symptom_names.index(symptom)
                        # Average MI with top diseases
                        mi_with_symptom = mutual_info['mi_matrix'][symptom_idx, top_disease_indices].mean()
                        if mi_with_symptom > best_mi:
                            best_mi = mi_with_symptom
                            next_symptom = symptom
                    app.logger.info("MI selector chose follow-up symptom: %s", next_symptom)
            except Exception as e:
                print(f"Question selection error: {e}")
        
        if next_symptom is None:
            # Fallback: pick random unasked symptom
            available_symptoms = [s for s in symptom_names if s not in current_symptoms]
            next_symptom = available_symptoms[0] if available_symptoms else None
        
        if next_symptom is None:
            # All symptoms asked, return final prediction
            prediction = model.predict(input_df)[0]
            disease = label_encoder.inverse_transform([prediction])[0]
            
            return jsonify({
                'status': 'final',
                'prediction': disease,
                'confidence': max_confidence,
                'questions_asked': sessions[session_id]['questions_asked'],
                'disclaimer': 'This is an AI decision-support prototype, not medical advice. Consult a healthcare professional.'
            })
        
        # Add to questions asked
        sessions[session_id]['questions_asked'].append(next_symptom)
        
        return jsonify({
            'status': 'question',
            'next_question': f"Do you have {next_symptom.replace('_', ' ')}?",
            'symptom_name': next_symptom,
            'current_confidence': max_confidence,
            'questions_asked': sessions[session_id]['questions_asked'],
            'questions_remaining': max_questions - len(sessions[session_id]['questions_asked'])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Endpoint to get full list of 132 symptoms"""
    return jsonify({
        'symptoms': symptom_names,
        'total': len(symptom_names)
    })

@app.route('/model-comparison', methods=['GET'])
def get_model_comparison():
    """Endpoint to get research model comparison results"""
    comparison_data = [
        {"model": "Logistic Regression", "cv_f1": "1.0000 ± 0.0000", "test_accuracy": "1.0000", "test_precision": "1.0000", "test_recall": "1.0000", "test_f1": "1.0000"},
        {"model": "SVM", "cv_f1": "1.0000 ± 0.0000", "test_accuracy": "1.0000", "test_precision": "1.0000", "test_recall": "1.0000", "test_f1": "1.0000"},
        {"model": "Random Forest", "cv_f1": "0.9940 ± 0.0120", "test_accuracy": "1.0000", "test_precision": "1.0000", "test_recall": "1.0000", "test_f1": "1.0000"},
        {"model": "XGBoost", "cv_f1": "0.9081 ± 0.0479", "test_accuracy": "0.9783", "test_precision": "0.9878", "test_recall": "0.9878", "test_f1": "0.9837"},
        {"model": "MLP", "cv_f1": "1.0000 ± 0.0000", "test_accuracy": "1.0000", "test_precision": "1.0000", "test_recall": "1.0000", "test_f1": "1.0000"},
        {"model": "Calibrated Ensemble (RF+XGB)", "cv_f1": "1.0000 ± 0.0000", "test_accuracy": "1.0000", "test_precision": "1.0000", "test_recall": "1.0000", "test_f1": "1.0000"}
    ]
    return jsonify({
        'models': comparison_data,
        'metrics': ['CV F1-Macro', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Macro']
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_type': metadata.get('model_type', 'Legacy'),
        'n_classes': n_classes,
        'n_features': len(symptom_names),
        'shap_available': shap_system is not None,
        'abstention_available': abstention_config is not None
    })

@app.route('/')
def home():
    return send_from_directory(frontend_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(frontend_folder, path)):
        return send_from_directory(frontend_folder, path)
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)

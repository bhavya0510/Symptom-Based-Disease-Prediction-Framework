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

frontend_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_folder, static_url_path='')
CORS(app)

# Load model and artifacts
print("Loading model and artifacts...")
backend_dir = os.path.dirname(__file__)
artifacts_dir = os.path.join(backend_dir, 'artifacts')

# Check if artifacts exist, if not use old models for backward compatibility
if os.path.exists(artifacts_dir):
    print("Using new research artifacts...")
    model = joblib.load(os.path.join(artifacts_dir, 'model.pkl'))
    label_encoder = joblib.load(os.path.join(artifacts_dir, 'encoder.pkl'))
    metadata = joblib.load(os.path.join(artifacts_dir, 'metadata.pkl'))
    
    # Load optional research components
    shap_system = None
    mutual_info = None
    abstention_config = None
    
    try:
        shap_system = joblib.load(os.path.join(artifacts_dir, 'shap_system.pkl'))
        print("✅ SHAP system loaded")
    except:
        print("⚠️  SHAP system not available")
    
    try:
        mutual_info = joblib.load(os.path.join(artifacts_dir, 'mutual_information.pkl'))
        print("✅ Mutual information loaded")
    except:
        print("⚠️  Mutual information not available")
    
    try:
        abstention_config = joblib.load(os.path.join(artifacts_dir, 'abstention_config.pkl'))
        print("✅ Abstention config loaded")
    except:
        print("⚠️  Abstention config not available")
    
    symptom_names = metadata['symptom_names']
    n_classes = metadata['n_classes']
    confidence_threshold = abstention_config['optimal_threshold'] if abstention_config else 0.5
    
else:
    print("Using legacy models...")
    model = joblib.load(os.path.join(backend_dir, 'model.pkl'))
    label_encoder = joblib.load(os.path.join(backend_dir, 'encoder.pkl'))
    metadata = joblib.load(os.path.join(backend_dir, 'metadata.pkl'))
    symptom_names = metadata['feature_cols']
    n_classes = len(label_encoder.classes_)
    shap_system = None
    mutual_info = None
    abstention_config = None
    confidence_threshold = 0.5

print(f"Loaded {n_classes} disease classes")
print(f"Symptom features: {len(symptom_names)}")
print("API ready!")

# In-memory session storage for sequential questioning
sessions = defaultdict(dict)

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
        should_abstain = max_confidence < confidence_threshold
        
        # Get SHAP explanation if available
        shap_explanation = None
        if shap_system and not should_abstain:
            try:
                explainer = shap_system['explainer']
                xgb_model = shap_system['xgb_model']
                shap_values_single = explainer.shap_values(input_df)
                
                # Get SHAP values for predicted class
                if isinstance(shap_values_single, list):
                    shap_values_class = shap_values_single[prediction]
                else:
                    shap_values_class = shap_values_single
                
                # Handle format
                if len(shap_values_class.shape) == 3:
                    shap_values_class = shap_values_class[0]
                elif len(shap_values_class.shape) == 1:
                    shap_values_class = shap_values_class.reshape(1, -1)
                
                # Get top features
                abs_shap = np.abs(shap_values_class[0])
                top_feature_indices = np.argsort(abs_shap)[-5:][::-1]
                
                shap_explanation = []
                for idx in top_feature_indices:
                    symptom = symptom_names[idx]
                    shap_value = float(shap_values_class[0][idx])
                    shap_explanation.append({
                        'symptom': symptom,
                        'shap_value': shap_value,
                        'impact': 'positive' if shap_value > 0 else 'negative'
                    })
            except Exception as e:
                print(f"SHAP explanation error: {e}")
        
        response = {
            'predictions': predictions,
            'top_prediction': predictions[0]['disease'] if predictions else None,
            'max_confidence': max_confidence,
            'should_abstain': should_abstain,
            'abstention_threshold': confidence_threshold,
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
        if max_confidence >= confidence_threshold or len(sessions[session_id]['questions_asked']) >= max_questions:
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

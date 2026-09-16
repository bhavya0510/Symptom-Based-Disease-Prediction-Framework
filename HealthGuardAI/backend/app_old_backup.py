from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

frontend_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_folder, static_url_path='')
CORS(app)  # Enable CORS for frontend

# Load model, encoder, metadata
print("Loading model and metadata...")
backend_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(backend_dir, 'model.pkl'))
label_encoder = joblib.load(os.path.join(backend_dir, 'encoder.pkl'))
metadata = joblib.load(os.path.join(backend_dir, 'metadata.pkl'))
feature_cols = metadata['feature_cols']
cat_cols = metadata['cat_cols']
num_cols = metadata['num_cols']

print("Loaded features:", feature_cols)
print("API ready!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract input
        symptoms_input = data.get('symptoms', [])  # ["Fever", "Cough"]
        age = data.get('age')
        gender = data.get('gender')
        blood_pressure = data.get('bloodPressure') or data.get('blood_pressure')
        cholesterol = data.get('cholesterolLevel') or data.get('cholesterol_level')
        
        # Create input DataFrame with same columns as training
        input_data = {}
        for col in feature_cols:
            input_data[col] = 'No'  # Default No for symptoms
        
        # Set symptoms to Yes (input is list of symptom names)
        for symptom in symptoms_input:
            symptom = symptom.strip().title()  # Normalize: "fever" -> "Fever"
            if symptom in feature_cols:
                input_data[symptom] = 'Yes'
        
        # Set patient data if provided
        if age is not None and age != '':
            input_data['Age'] = int(age)
        if gender:
            input_data['Gender'] = gender.title()
        if blood_pressure:
            input_data['Blood Pressure'] = blood_pressure.title()
        if cholesterol:
            input_data['Cholesterol Level'] = cholesterol.title()
        
        # Fill missing with proper defaults
        for col in feature_cols:
            if col not in input_data:
                if col == 'Age':
                    input_data[col] = 30
                elif col == 'Gender':
                    input_data[col] = 'Male'
                elif col == 'Blood Pressure':
                    input_data[col] = 'Normal'
                elif col == 'Cholesterol Level':
                    input_data[col] = 'Normal'
                else:
                    input_data[col] = 'No'
        
        df_input = pd.DataFrame([input_data])
        print("Input data:", df_input[feature_cols].to_dict('records'))
        
        # Predict
        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0]
        
        # Top 5 predictions for richer insights
        top_indices = np.argsort(probabilities)[::-1][:5]
        predictions = []
        for idx in top_indices:
            disease = label_encoder.inverse_transform([idx])[0]
            conf = probabilities[idx]
            predictions.append({'disease': disease, 'confidence': float(conf)})
        
        return jsonify({
            'predictions': predictions,
            'top_prediction': predictions[0]['disease'],
            'disclaimer': 'This is an AI decision-support prototype, not medical advice. Consult a healthcare professional.'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

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

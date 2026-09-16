"""
Create metadata file for backend
"""
import joblib
import os
import pandas as pd

backend_dir = os.path.dirname(__file__)
artifacts_dir = os.path.join(backend_dir, 'artifacts')

# Load label encoder
label_encoder = joblib.load(os.path.join(artifacts_dir, 'encoder.pkl'))
abstention_config = joblib.load(os.path.join(artifacts_dir, 'abstention_config.pkl'))

# Load training data to get symptom names
train_df = pd.read_csv(os.path.join(artifacts_dir, 'disease_data_clean.csv'))
symptom_names = [col for col in train_df.columns if col != 'prognosis']

# Create metadata
metadata = {
    'symptom_names': symptom_names,
    'disease_classes': label_encoder.classes_.tolist(),
    'n_classes': len(label_encoder.classes_),
    'n_features': len(symptom_names),
    'model_type': 'Calibrated Ensemble (RF + XGBoost)',
    'calibration_method': 'isotonic',
    'abstention_threshold': float(abstention_config['optimal_threshold'])
}

joblib.dump(metadata, os.path.join(artifacts_dir, 'metadata.pkl'))
print("✅ Created metadata.pkl")
print(f"Symptoms: {len(symptom_names)}, Diseases: {len(label_encoder.classes_)}")

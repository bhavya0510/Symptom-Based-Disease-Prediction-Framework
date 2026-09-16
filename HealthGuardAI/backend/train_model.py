"""DEPRECATED: legacy patient-profile trainer. Not used by the application or research pipeline."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

print("DEPRECATED: use research/train_pipeline.py; this legacy dataset is not a production input.")
df = pd.read_csv('../dataset/Disease_symptom_and_patient_profile_dataset.csv')

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

target_col = df.columns[0]
print(f"Target column: {target_col}")

exclude_cols = [target_col]
if 'Outcome Variable' in df.columns:
    exclude_cols.append('Outcome Variable')
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"Features: {feature_cols}")

X = df[feature_cols]
y = df[target_col]

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()

print("Categorical:", cat_cols)
print("Numeric:", num_cols)

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Classes: {len(le.classes_)}")

preprocessor = ColumnTransformer(
    transformers=[
('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore', dtype=np.float64), cat_cols),
        ('num', 'passthrough', num_cols)
    ])

print("Training...")
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1))
])
# X_clean = X.fillna('No')\nX_clean = X.copy()
pipeline.fit(X, y_enc)

print("✅ Training complete!")
print("Saved model.pkl")

joblib.dump(pipeline, 'model.pkl')
joblib.dump(le, 'encoder.pkl')
joblib.dump({'feature_cols': feature_cols, 'cat_cols': cat_cols, 'num_cols': num_cols}, 'metadata.pkl')

print("Files saved. Run: python -m flask --app app run")

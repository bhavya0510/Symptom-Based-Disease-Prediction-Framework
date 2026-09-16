"""
Investigate why XGBoost performs differently from other models
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib

print("Investigating XGBoost vs other models performance...")

# Load data
train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')

X_train = train_df.drop(columns=['prognosis'])
y_train = train_df['prognosis']
X_test = test_df.drop(columns=['prognosis'])
y_test = test_df['prognosis']

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Check if any test patterns exist in training (data leakage check)
train_patterns = X_train.drop_duplicates()
test_patterns = X_test.drop_duplicates()

overlap = pd.concat([train_patterns, test_patterns]).duplicated().sum()
print(f"Pattern overlap between train and test: {overlap}")

# Load the trained XGBoost model
best_models = joblib.load('results/baseline_models.pkl')
xgb_model = best_models['XGBoost']

# Get predictions
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)

# Find misclassified samples
misclassified = X_test[y_pred != y_test_enc]
print(f"\nMisclassified samples: {len(misclassified)} out of {len(X_test)}")

# Check if misclassified samples have unique patterns
misclassified_patterns = misclassified.drop_duplicates()
print(f"Unique patterns among misclassified: {len(misclassified_patterns)}")

# Check diseases with most errors
error_mask = y_pred != y_test_enc
error_diseases = y_test[error_mask].value_counts()
print(f"\nDiseases with most errors:")
print(error_diseases.head(10))

# Check if these diseases have few training samples
print(f"\nTraining samples for error diseases:")
for disease in error_diseases.head(5).index:
    train_count = (y_train == disease).sum()
    test_count = (y_test == disease).sum()
    print(f"{disease}: Train={train_count}, Test={test_count}, Errors={error_diseases[disease]}")

# Try to understand why other models get 100%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

lr_model = best_models['Logistic Regression']
svm_model = best_models['SVM']

lr_pred_enc = lr_model.predict(X_test)
svm_pred_enc = svm_model.predict(X_test)

lr_errors = (lr_pred_enc != y_test_enc).sum()
svm_errors = (svm_pred_enc != y_test_enc).sum()
xgb_errors = (y_pred != y_test_enc).sum()

print(f"\nLogistic Regression errors: {lr_errors}")
print(f"SVM errors: {svm_errors}")
print(f"XGBoost errors: {xgb_errors}")

# Check if LR and SVM are memorizing
print(f"\nLR training accuracy: {accuracy_score(y_train_enc, lr_model.predict(X_train)):.4f}")
print(f"SVM training accuracy: {accuracy_score(y_train_enc, svm_model.predict(X_train)):.4f}")
print(f"XGBoost training accuracy: {accuracy_score(y_train_enc, xgb_model.predict(X_train)):.4f}")

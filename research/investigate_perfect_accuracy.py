"""
Investigate why all models achieve 100% accuracy
Check for data leakage and dataset characteristics
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("Investigating perfect accuracy issue...")

# Load the splits
train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Check if test set patterns exist in training set
X_train = train_df.drop(columns=['prognosis'])
y_train = train_df['prognosis']
X_test = test_df.drop(columns=['prognosis'])
y_test = test_df['prognosis']

# Check for overlapping symptom patterns between train and test
train_patterns = X_train.drop_duplicates()
test_patterns = X_test.drop_duplicates()

print(f"\nUnique symptom patterns in train: {len(train_patterns)}")
print(f"Unique symptom patterns in test: {len(test_patterns)}")

# Find overlap
combined = pd.concat([train_patterns, test_patterns])
duplicates = combined[combined.duplicated()]
print(f"Overlapping patterns between train and test: {len(duplicates)}")

if len(duplicates) > 0:
    print("⚠️  WARNING: Data leakage detected - test patterns exist in training!")
    print("Sample overlapping patterns:")
    print(duplicates.head())
else:
    print("✅ No overlapping patterns between train and test")

# Check if same disease always has same symptom pattern
print("\nChecking disease-symptom pattern uniqueness...")
for disease in y_train.unique()[:5]:  # Check first 5 diseases
    disease_data = train_df[train_df['prognosis'] == disease]
    unique_patterns = disease_data.drop_duplicates(subset=[col for col in disease_data.columns if col != 'prognosis'])
    print(f"{disease}: {len(disease_data)} samples, {len(unique_patterns)} unique patterns")

# Try a simple decision tree to see depth needed
print("\nTraining simple Decision Tree to check complexity...")
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train)
train_acc = accuracy_score(y_train, dt.predict(X_train))
test_acc = accuracy_score(y_test, dt.predict(X_test))
print(f"Decision Tree (depth=5) - Train: {train_acc:.4f}, Test: {test_acc:.4f}")

# Try unlimited depth
dt_unlimited = DecisionTreeClassifier(random_state=42)
dt_unlimited.fit(X_train, y_train)
train_acc_unlim = accuracy_score(y_train, dt_unlimited.predict(X_train))
test_acc_unlim = accuracy_score(y_test, dt_unlimited.predict(X_test))
print(f"Decision Tree (unlimited) - Train: {train_acc_unlim:.4f}, Test: {test_acc_unlim:.4f}")
print(f"Tree depth: {dt_unlimited.get_depth()}")

# Check feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': dt_unlimited.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

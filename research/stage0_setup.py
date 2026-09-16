"""
STAGE 0: Dataset verification
Verify the new dataset loads correctly and check dimensions
"""
import pandas as pd
import numpy as np

print("=" * 60)
print("STAGE 0: Dataset Verification")
print("=" * 60)

# Load training data
train_df = pd.read_csv('data/raw/Training.csv')
print(f"\nTraining data shape: {train_df.shape}")
print(f"Training data columns: {len(train_df.columns)}")
print(f"First few columns: {train_df.columns[:5].tolist()}")

# Load testing data
test_df = pd.read_csv('data/raw/Testing.csv')
print(f"\nTesting data shape: {test_df.shape}")
print(f"Testing data columns: {len(test_df.columns)}")

# Check for prognosis column
print(f"\nTarget column in training: {'prognosis' in train_df.columns}")
print(f"Target column in testing: {'prognosis' in test_df.columns}")

# Check unique classes
if 'prognosis' in train_df.columns:
    unique_classes = train_df['prognosis'].nunique()
    print(f"\nUnique disease classes: {unique_classes}")
    print(f"Classes: {sorted(train_df['prognosis'].unique())}")

# Check for unnamed columns
unnamed_cols_train = [col for col in train_df.columns if 'Unnamed' in col]
unnamed_cols_test = [col for col in test_df.columns if 'Unnamed' in col]
print(f"\nUnnamed columns in training: {unnamed_cols_train}")
print(f"Unnamed columns in testing: {unnamed_cols_test}")

# Verify symptom columns count (should be 132)
symptom_cols_train = [col for col in train_df.columns if col != 'prognosis']
print(f"\nSymptom columns in training: {len(symptom_cols_train)}")

print("\n" + "=" * 60)
print("✅ Dataset verification complete!")
print("=" * 60)

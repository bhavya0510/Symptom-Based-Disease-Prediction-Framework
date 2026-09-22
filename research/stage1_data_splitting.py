"""
STAGE 1: Data Cleaning & Splitting (Leakage-Free)
- Load and concatenate Training.csv and Testing.csv
- Remove exact duplicate rows at pattern level to prevent data leakage
- Drop any fully-empty/unnamed columns
- Verify: 132 symptom columns, 1 `prognosis` column, 41 unique classes
- Save a cleaned combined file to `research/data/processed/disease_data_clean.csv`
- Build a stratified split: 70% train / 15% validation / 15% test at PATTERN LEVEL
- Save indices/files separately so every model uses the identical split
- Implement stratified 5-fold cross-validation on train+val portion
- Print and save class distribution (bar chart) and per-class split counts
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
from pathlib import Path
import joblib

RESEARCH_DIR = Path(__file__).resolve().parent
os.chdir(RESEARCH_DIR)

print("=" * 80)
print("STAGE 1: Data Cleaning & Splitting (Leakage-Free)")
print("=" * 80)

# Create directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load datasets
print("\nLoading datasets...")
train_df = pd.read_csv('data/raw/Training.csv')
test_df = pd.read_csv('data/raw/Testing.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")

# Concatenate datasets
print("\nConcatenating datasets...")
combined_df = pd.concat([train_df, test_df], ignore_index=True)
print(f"Combined shape: {combined_df.shape}")

# Remove exact duplicate rows at pattern level to prevent leakage
print("\nRemoving exact duplicate rows at pattern level...")
initial_rows = len(combined_df)
combined_df = combined_df.drop_duplicates()
final_rows = len(combined_df)
print(f"Removed {initial_rows - final_rows} duplicate rows")
print(f"Shape after deduplication: {combined_df.shape}")
print(f"⚠️  NOTE: Using unique patterns only to prevent data leakage")

# Drop unnamed columns
print("\nDropping unnamed columns...")
unnamed_cols = [col for col in combined_df.columns if 'Unnamed' in str(col)]
if unnamed_cols:
    print(f"Found unnamed columns: {unnamed_cols}")
    combined_df = combined_df.drop(columns=unnamed_cols)
    print(f"Shape after dropping unnamed: {combined_df.shape}")
else:
    print("No unnamed columns found")

# Verify structure
print("\nVerifying dataset structure...")
print(f"Total columns: {len(combined_df.columns)}")
print(f"Prognosis column present: {'prognosis' in combined_df.columns}")

symptom_cols = [col for col in combined_df.columns if col != 'prognosis']
print(f"Symptom columns: {len(symptom_cols)}")
print(f"Expected: 132 symptom columns")

if len(symptom_cols) != 132:
    print(f"⚠️  WARNING: Expected 132 symptom columns, found {len(symptom_cols)}")
else:
    print("✅ Symptom column count correct")

# Check unique classes
unique_classes = combined_df['prognosis'].nunique()
print(f"\nUnique disease classes: {unique_classes}")
print(f"Expected: 41 classes")

if unique_classes != 41:
    print(f"⚠️  WARNING: Expected 41 classes, found {unique_classes}")
else:
    print("✅ Class count correct")

# Display class distribution
print("\nClass distribution:")
class_counts = combined_df['prognosis'].value_counts().sort_index()
print(class_counts)

# Display per-class unique pattern counts
print("\nUnique rows per disease:")
unique_counts = combined_df.drop_duplicates().groupby('prognosis').size().sort_index()
print(unique_counts)

# Check for minimum patterns per class
min_patterns = unique_counts.min()
max_patterns = unique_counts.max()
print(f"\nMin unique patterns per class: {min_patterns}")
print(f"Max unique patterns per class: {max_patterns}")

if min_patterns < 5:
    print(f"⚠️  WARNING: Some classes have fewer than 5 unique patterns, leading to high-variance estimates")

# Save cleaned dataset
print("\nSaving cleaned dataset...")
combined_df.to_csv('data/processed/disease_data_clean.csv', index=False)
print("✅ Saved to data/processed/disease_data_clean.csv")

# Create stratified split at pattern level: 70% train / 15% validation / 15% test
print("\nCreating stratified split (70/15/15) at pattern level to prevent leakage...")
X = combined_df.drop(columns=['prognosis'])
y = combined_df['prognosis']

# First split: 85% train+val / 15% test (at pattern level)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Second split: 70% train / 15% val (from the 85% train+val)
# 0.1765 ≈ 0.15/0.85 to get 15% of total from the 85% train+val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42
)

print(f"Train set: {len(X_train)} samples ({len(X_train)/len(combined_df)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(combined_df)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(combined_df)*100:.1f}%)")

# Verify no pattern overlap between splits
print("\nVerifying no pattern overlap between splits...")
train_patterns = X_train.drop_duplicates()
val_patterns = X_val.drop_duplicates()
test_patterns = X_test.drop_duplicates()

train_val_overlap = pd.concat([train_patterns, val_patterns]).duplicated().sum()
train_test_overlap = pd.concat([train_patterns, test_patterns]).duplicated().sum()
val_test_overlap = pd.concat([val_patterns, test_patterns]).duplicated().sum()

print(f"Train-Val pattern overlap: {train_val_overlap}")
print(f"Train-Test pattern overlap: {train_test_overlap}")
print(f"Val-Test pattern overlap: {val_test_overlap}")

if train_test_overlap > 0 or val_test_overlap > 0:
    print("⚠️  WARNING: Pattern overlap detected between splits!")
else:
    print("✅ No pattern overlap between splits - data leakage prevented")

# Save split indices and data
print("\nSaving split information...")
split_info = {
    'train_indices': X_train.index.tolist(),
    'val_indices': X_val.index.tolist(),
    'test_indices': X_test.index.tolist(),
    'train_shape': X_train.shape,
    'val_shape': X_val.shape,
    'test_shape': X_test.shape,
    'train_test_pattern_overlap': train_test_overlap,
    'val_test_pattern_overlap': val_test_overlap
}
joblib.dump(split_info, 'data/processed/split_indices.pkl')

# Save individual splits
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv('data/processed/train.csv', index=False)
val_df.to_csv('data/processed/val.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)

print("✅ Saved splits to data/processed/")

# Setup stratified 5-fold cross-validation on train+val
print("\nSetting up stratified 5-fold CV on train+val...")
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

# Reset indices to avoid issues
X_train_val = X_train_val.reset_index(drop=True)
y_train_val = y_train_val.reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_splits = list(skf.split(X_train_val, y_train_val))

print(f"Created {len(cv_splits)} CV folds")
for i, (train_idx, val_idx) in enumerate(cv_splits):
    print(f"  Fold {i+1}: {len(train_idx)} train / {len(val_idx)} val")

joblib.dump(cv_splits, 'data/processed/cv_splits.pkl')
print("✅ Saved CV splits to data/processed/cv_splits.pkl")

# Generate per-class split counts table
print("\nGenerating per-class split counts...")
per_class_counts = []
for disease in combined_df['prognosis'].unique():
    train_count = (y_train == disease).sum()
    val_count = (y_val == disease).sum()
    test_count = (y_test == disease).sum()
    total_unique = unique_counts[disease]
    
    per_class_counts.append({
        'Disease': disease,
        'Total_Unique_Patterns': total_unique,
        'Train': train_count,
        'Validation': val_count,
        'Test': test_count
    })

per_class_df = pd.DataFrame(per_class_counts)
per_class_df.to_csv('results/per_class_split_counts.csv', index=False)
print("✅ Saved per-class split counts to results/per_class_split_counts.csv")
print(per_class_df.to_string(index=False))

# Generate class distribution bar chart
print("\nGenerating class distribution visualization...")
plt.figure(figsize=(15, 8))
class_counts_sorted = class_counts.sort_values(ascending=False)
bars = plt.bar(range(len(class_counts_sorted)), class_counts_sorted.values)
plt.xticks(range(len(class_counts_sorted)), class_counts_sorted.index, rotation=90, ha='center')
plt.xlabel('Disease Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Dataset (Unique Patterns)')
plt.tight_layout()

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=8)

plt.savefig('results/class_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved class distribution to results/class_distribution.png")

# Print summary statistics
print("\n" + "=" * 80)
print("STAGE 1 SUMMARY")
print("=" * 80)
print(f"Total samples after cleaning: {len(combined_df)} (unique patterns only)")
print(f"Symptom features: {len(symptom_cols)}")
print(f"Disease classes: {unique_classes}")
print(f"Train/Val/Test split: {len(X_train)}/{len(X_val)}/{len(X_test)} samples")
print(f"Pattern overlap check: Train-Test overlap = {train_test_overlap}, Val-Test overlap = {val_test_overlap}")
print(f"Min unique patterns per class: {min_patterns}")
print(f"Max unique patterns per class: {max_patterns}")
print(f"⚠️  CAVEAT: Some classes have only {min_patterns} unique patterns, leading to high-variance estimates")
print("=" * 80)
print("✅ STAGE 1 COMPLETE!")
print("=" * 80)

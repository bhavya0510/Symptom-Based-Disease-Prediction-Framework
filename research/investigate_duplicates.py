"""
Investigate the duplicate removal issue
"""
import pandas as pd

print("Investigating duplicate patterns...")

# Load original datasets
train_df = pd.read_csv('data/raw/Training.csv')
test_df = pd.read_csv('data/raw/Testing.csv')

print(f"Original training shape: {train_df.shape}")
print(f"Original testing shape: {test_df.shape}")

# Check what kind of duplicates exist
combined = pd.concat([train_df, test_df], ignore_index=True)
print(f"Combined shape: {combined.shape}")

# Check for exact row duplicates
exact_dups = combined.duplicated().sum()
print(f"Exact duplicate rows: {exact_dups}")

# Check duplicates ignoring prognosis
symptom_cols = [col for col in combined.columns if col != 'prognosis']
dups_ignore_prognosis = combined.duplicated(subset=symptom_cols).sum()
print(f"Duplicates ignoring prognosis: {dups_ignore_prognosis}")

# Look at some examples of duplicate rows
print("\nSample of 'duplicate' rows:")
duplicated_rows = combined[combined.duplicated(keep=False)].sort_values(by=list(combined.columns))
print(duplicated_rows.head(10))

# Check unique symptom patterns per disease
print("\nUnique symptom patterns per disease:")
for disease in combined['prognosis'].unique()[:5]:  # Check first 5 diseases
    disease_data = combined[combined['prognosis'] == disease]
    unique_patterns = disease_data.drop_duplicates().shape[0]
    total_samples = disease_data.shape[0]
    print(f"{disease}: {total_samples} total samples, {unique_patterns} unique patterns")

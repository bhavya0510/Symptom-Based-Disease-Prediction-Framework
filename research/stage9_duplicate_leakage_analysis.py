"""
STAGE 9: Duplicate and Data Leakage Analysis
Deep analysis of dataset duplicates, patterns, and potential data leakage issues.
This addresses the concern about perfect scores from limited unique patterns.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STAGE 9: Duplicate and Data Leakage Analysis")
print("=" * 80)

# Load cleaned dataset
print("\nLoading cleaned dataset...")
clean_df = pd.read_csv('data/processed/disease_data_clean.csv')
print(f"Dataset shape: {clean_df.shape}")

# Get symptom columns
symptom_cols = [col for col in clean_df.columns if col != 'prognosis']
print(f"Symptom features: {len(symptom_cols)}")

# 1. Detailed duplicate analysis
print("\n" + "="*60)
print("DUPLICATE PATTERN ANALYSIS")
print("="*60)

# Count exact duplicates
exact_duplicates = clean_df.duplicated().sum()
print(f"Exact duplicate rows: {exact_duplicates}")

# Count unique symptom patterns
symptom_patterns = clean_df[symptom_cols].drop_duplicates()
print(f"Unique symptom patterns: {len(symptom_patterns)}")
print(f"Total rows: {len(clean_df)}")
print(f"Pattern reduction: {len(clean_df) - len(symptom_patterns)} duplicates removed")

# Analyze pattern distribution per disease
print("\nPattern distribution per disease:")
pattern_counts = clean_df.groupby('prognosis').apply(lambda x: x[symptom_cols].drop_duplicates().shape[0])
print(pattern_counts.describe())

print(f"\nMinimum patterns per disease: {pattern_counts.min()}")
print(f"Maximum patterns per disease: {pattern_counts.max()}")
print(f"Diseases with only 1 unique pattern: {(pattern_counts == 1).sum()}")
print(f"Diseases with ≤ 3 unique patterns: {(pattern_counts <= 3).sum()}")

# Show diseases with very few patterns
print("\nDiseases with ≤ 3 unique patterns:")
print(pattern_counts[pattern_counts <= 3].sort_values())

# 2. Pattern overlap analysis between splits
print("\n" + "="*60)
print("SPLIT OVERLAP ANALYSIS")
print("="*60)

train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/val.csv')
test_df = pd.read_csv('data/processed/test.csv')

X_train = train_df[symptom_cols]
X_val = val_df[symptom_cols]
X_test = test_df[symptom_cols]

# Check pattern overlaps
train_patterns = X_train.drop_duplicates()
val_patterns = X_val.drop_duplicates()
test_patterns = X_test.drop_duplicates()

train_val_overlap = pd.concat([train_patterns, val_patterns]).duplicated().sum()
train_test_overlap = pd.concat([train_patterns, test_patterns]).duplicated().sum()
val_test_overlap = pd.concat([val_patterns, test_patterns]).duplicated().sum()

print(f"Train-Val pattern overlap: {train_val_overlap}")
print(f"Train-Test pattern overlap: {train_test_overlap}")
print(f"Val-Test pattern overlap: {val_test_overlap}")

# 3. Pattern frequency analysis
print("\n" + "="*60)
print("PATTERN FREQUENCY ANALYSIS")
print("="*60)

# Count how many times each unique pattern appears
pattern_frequencies = clean_df[symptom_cols].value_counts()
print(f"Most common pattern appears: {pattern_frequencies.max()} times")
print(f"Least common pattern appears: {pattern_frequencies.min()} times")

# Analyze distribution of pattern frequencies
print(f"Patterns appearing only once: {(pattern_frequencies == 1).sum()}")
print(f"Patterns appearing 2-5 times: {((pattern_frequencies >= 2) & (pattern_frequencies <= 5)).sum()}")
print(f"Patterns appearing > 5 times: {(pattern_frequencies > 5).sum()}")

# 4. Symptom sparsity analysis
print("\n" + "="*60)
print("SYMPTOM SPARSITY ANALYSIS")
print("="*60)

# Calculate symptom frequencies
symptom_frequencies = clean_df[symptom_cols].sum().sort_values(ascending=False)
print(f"Most common symptom: {symptom_frequencies.index[0]} ({symptom_frequencies.iloc[0]} occurrences)")
print(f"Least common symptom: {symptom_frequencies.index[-1]} ({symptom_frequencies.iloc[-1]} occurrences)")

# Sparsity metrics
total_symptom_entries = len(symptom_cols) * len(clean_df)
total_present_symptoms = symptom_frequencies.sum()
sparsity = 1 - (total_present_symptoms / total_symptom_entries)
print(f"Overall sparsity: {sparsity:.4f} ({sparsity*100:.2f}% zeros)")

# Average symptoms per sample
avg_symptoms_per_sample = (clean_df[symptom_cols].sum(axis=1)).mean()
print(f"Average symptoms per sample: {avg_symptoms_per_sample:.2f}")

# 5. Impact on model performance analysis
print("\n" + "="*60)
print("PERFORMANCE IMPACT ANALYSIS")
print("="*60)

# Load model results
try:
    model_comparison = pd.read_csv('results/model_comparison_test.csv')
    print("\nCurrent model performance:")
    print(model_comparison[['Model', 'Test_Accuracy', 'Test_F1_Macro']].to_string(index=False))
    
    # Flag perfect scores
    perfect_scores = model_comparison[model_comparison['Test_F1_Macro'] == 1.0]
    if len(perfect_scores) > 0:
        print(f"\n⚠️  WARNING: {len(perfect_scores)} models achieved perfect F1-Macro scores")
        print("This is likely due to the limited number of unique patterns (305) for 41 classes")
        print("Models may be memorizing patterns rather than learning generalizable patterns")
except:
    print("Model comparison results not found - run earlier stages first")

# 6. Generate visualizations
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Pattern distribution per disease
fig, ax = plt.subplots(figsize=(15, 8))
pattern_counts_sorted = pattern_counts.sort_values(ascending=False)
bars = ax.bar(range(len(pattern_counts_sorted)), pattern_counts_sorted.values)
ax.set_xticks(range(len(pattern_counts_sorted)))
ax.set_xticklabels(pattern_counts_sorted.index, rotation=90, ha='center')
ax.set_xlabel('Disease Class')
ax.set_ylabel('Number of Unique Symptom Patterns')
ax.set_title('Unique Symptom Patterns per Disease Class')
ax.axhline(y=3, color='r', linestyle='--', label='3 patterns threshold')
ax.legend()
plt.tight_layout()
plt.savefig('results/pattern_distribution_per_disease.png', dpi=300, bbox_inches='tight')
print("✅ Saved pattern distribution visualization")

# Symptom frequency distribution
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(symptom_frequencies, bins=30, edgecolor='black')
ax.set_xlabel('Symptom Frequency (Number of Samples)')
ax.set_ylabel('Number of Symptoms')
ax.set_title('Distribution of Symptom Frequencies')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/symptom_frequency_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved symptom frequency distribution")

# Pattern frequency distribution
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(pattern_frequencies, bins=30, edgecolor='black')
ax.set_xlabel('Pattern Frequency (Number of Samples)')
ax.set_ylabel('Number of Unique Patterns')
ax.set_title('Distribution of Pattern Frequencies')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/pattern_frequency_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved pattern frequency distribution")

# 7. Create comprehensive analysis report
print("\n" + "="*60)
print("CREATING ANALYSIS REPORT")
print("="*60)

analysis_report = {
    'dataset_info': {
        'total_samples': len(clean_df),
        'unique_patterns': len(symptom_patterns),
        'symptom_features': len(symptom_cols),
        'disease_classes': clean_df['prognosis'].nunique(),
        'exact_duplicates_removed': exact_duplicates
    },
    'pattern_analysis': {
        'min_patterns_per_disease': pattern_counts.min(),
        'max_patterns_per_disease': pattern_counts.max(),
        'diseases_with_single_pattern': (pattern_counts == 1).sum(),
        'diseases_with_3_or_fewer_patterns': (pattern_counts <= 3).sum()
    },
    'split_overlap': {
        'train_val_overlap': int(train_val_overlap),
        'train_test_overlap': int(train_test_overlap),
        'val_test_overlap': int(val_test_overlap)
    },
    'sparsity_analysis': {
        'overall_sparsity': float(sparsity),
        'avg_symptoms_per_sample': float(avg_symptoms_per_sample),
        'most_common_symptom': symptom_frequencies.index[0],
        'most_common_symptom_count': int(symptom_frequencies.iloc[0])
    },
    'pattern_frequency': {
        'patterns_appearing_once': int((pattern_frequencies == 1).sum()),
        'patterns_appearing_2_to_5_times': int(((pattern_frequencies >= 2) & (pattern_frequencies <= 5)).sum()),
        'patterns_appearing_more_than_5_times': int((pattern_frequencies > 5).sum())
    }
}

import json
with open('results/duplicate_leakage_analysis.json', 'w') as f:
    json.dump(analysis_report, f, indent=2)
print("✅ Saved analysis report to results/duplicate_leakage_analysis.json")

# Summary and recommendations
print("\n" + "="*80)
print("STAGE 9 SUMMARY")
print("="*80)
print(f"Dataset Analysis Results:")
print(f"  Total samples: {len(clean_df)}")
print(f"  Unique patterns: {len(symptom_patterns)} ({len(symptom_patterns)/len(clean_df)*100:.1f}% of original)")
print(f"  Pattern reduction: {len(clean_df) - len(symptom_patterns)} duplicates removed")
print(f"  Minimum patterns per disease: {pattern_counts.min()}")
print(f"  Maximum patterns per disease: {pattern_counts.max()}")
print(f"  Diseases with ≤ 3 patterns: {(pattern_counts <= 3).sum()} ({(pattern_counts <= 3).sum()/len(pattern_counts)*100:.1f}%)")
print(f"\nSplit Overlap Check:")
print(f"  Train-Test overlap: {train_test_overlap} (should be 0)")
print(f"  Val-Test overlap: {val_test_overlap} (should be 0)")
print(f"\nKey Concerns:")
print(f"  ⚠️  Limited unique patterns ({len(symptom_patterns)}) for {clean_df['prognosis'].nunique()} disease classes")
print(f"  ⚠️  {pattern_counts.min()} diseases have only {pattern_counts.min()} unique pattern(s)")
print(f"  ⚠️  This explains perfect model scores - models are memorizing, not generalizing")
print(f"\nRecommendations:")
print(f"  1. Acknowledge this as an early-stage research prototype")
print(f"  2. Report results as pattern-matching performance, not clinical diagnostic accuracy")
print(f"  3. Collect more diverse data to improve generalization")
print(f"  4. Consider data augmentation techniques")
print("="*80)
print("✅ STAGE 9 COMPLETE!")
print("="*80)
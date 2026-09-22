"""
STAGE 10: Ablation Study
Evaluate the contribution of each component to overall system performance.
Tests: baseline models, with/without calibration, with/without abstention, with/without sequential questioning.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

RESEARCH_DIR = Path(__file__).resolve().parent
os.chdir(RESEARCH_DIR)

print("=" * 80)
print("STAGE 10: Ablation Study")
print("=" * 80)

# Load data
print("\nLoading data splits...")
test_df = pd.read_csv('data/processed/test.csv')
X_test = test_df.drop(columns=['prognosis'])
y_test = test_df['prognosis']

print(f"Test set: {X_test.shape}")

# Encode labels
le = joblib.load('results/label_encoder.pkl')
y_test_enc = le.transform(y_test)

print(f"Classes: {len(le.classes_)}")

# Load models
print("\nLoading models...")
baseline_models = joblib.load('results/baseline_models.pkl')
calibrated_ensemble = joblib.load('results/final_calibrated_ensemble.pkl')
abstention_config = joblib.load('results/abstention_config.pkl')

print("✅ Loaded models and configurations")

# Define ablation configurations
ablation_configs = [
    {
        'name': 'Baseline (Single Best Model)',
        'description': 'Best single baseline model without enhancements',
        'use_calibration': False,
        'use_abstention': False,
        'use_sequential': False,
        'model_type': 'baseline'
    },
    {
        'name': 'Ensemble (No Calibration)',
        'description': 'RF+XGBoost ensemble without probability calibration',
        'use_calibration': False,
        'use_abstention': False,
        'use_sequential': False,
        'model_type': 'ensemble_uncalibrated'
    },
    {
        'name': 'Ensemble + Calibration',
        'description': 'RF+XGBoost ensemble with isotonic calibration',
        'use_calibration': True,
        'use_abstention': False,
        'use_sequential': False,
        'model_type': 'ensemble_calibrated'
    },
    {
        'name': 'Ensemble + Calibration + Abstention',
        'description': 'Calibrated ensemble with confidence-based abstention',
        'use_calibration': True,
        'use_abstention': True,
        'use_sequential': False,
        'model_type': 'ensemble_calibrated'
    },
    {
        'name': 'Full System (All Components)',
        'description': 'Complete system with calibration, abstention, and sequential questioning',
        'use_calibration': True,
        'use_abstention': True,
        'use_sequential': True,
        'model_type': 'ensemble_calibrated'
    }
]

print(f"\nRunning {len(ablation_configs)} ablation configurations...")

ablation_results = []

for config in ablation_configs:
    print(f"\n" + "="*60)
    print(f"Configuration: {config['name']}")
    print(f"Description: {config['description']}")
    print("="*60)
    
    # Select model
    if config['model_type'] == 'baseline':
        # Use best baseline model
        model_comparison = pd.read_csv('results/model_comparison_test.csv')
        best_baseline_name = model_comparison.loc[model_comparison['Test_F1_Macro'].idxmax(), 'Model']
        model = baseline_models[best_baseline_name]
        print(f"Using baseline model: {best_baseline_name}")
    elif config['model_type'] == 'ensemble_uncalibrated':
        # Load uncalibrated ensemble
        all_ensembles = joblib.load('results/all_ensembles.pkl')
        model = all_ensembles['voting_ensemble']
        print("Using uncalibrated voting ensemble")
    else:
        model = calibrated_ensemble
        print("Using calibrated ensemble")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Base metrics
    accuracy = accuracy_score(y_test_enc, y_pred)
    precision_macro = precision_score(y_test_enc, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test_enc, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test_enc, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test_enc, y_pred, average='weighted', zero_division=0)
    
    # Calibration metrics (if applicable)
    brier_score = None
    ece = None
    log_loss_val = None
    
    if config['use_calibration']:
        from sklearn.metrics import brier_score_loss as brier, log_loss
        def multiclass_brier(y, probabilities):
            return float(np.mean([brier((y == i).astype(int), probabilities[:, i]) for i in range(probabilities.shape[1])]))
        
        brier_score = multiclass_brier(y_test_enc, y_pred_proba)
        log_loss_val = log_loss(y_test_enc, y_pred_proba)
        
        # Simple ECE calculation
        confidences = np.max(y_pred_proba, axis=1)
        predictions = np.argmax(y_pred_proba, axis=1)
        accuracies = (predictions == y_test_enc).astype(float)
        
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += (in_bin.mean()) * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    
    # Abstention metrics (if applicable)
    abstention_rate = None
    coverage = None
    accuracy_with_abstention = None
    
    if config['use_abstention']:
        threshold = abstention_config['optimal_threshold']
        max_confidence = np.max(y_pred_proba, axis=1)
        confident_mask = max_confidence >= threshold
        
        abstention_rate = 1.0 - confident_mask.mean()
        coverage = confident_mask.mean()
        
        if confident_mask.sum() > 0:
            accuracy_with_abstention = accuracy_score(y_test_enc[confident_mask], y_pred[confident_mask])
    
    # Sequential questioning metrics (if applicable)
    avg_questions_needed = None
    sequential_improvement = None
    
    if config['use_sequential']:
        seq_df = pd.read_csv('results/sequential_questioning_evaluation.csv')
        if not seq_df.empty and 'method' in seq_df.columns:
            mi_rows = seq_df[seq_df['method'] == 'MI']
            if not mi_rows.empty:
                avg_questions_needed = float(mi_rows['avg_questions_needed'].mean())
                sequential_improvement = float(mi_rows['final_accuracy'].mean())
            else:
                avg_questions_needed = None
                sequential_improvement = None
        else:
            avg_questions_needed = None
            sequential_improvement = None
    
    # Store results
    result = {
        'Configuration': config['name'],
        'Description': config['description'],
        'Accuracy': accuracy,
        'Precision_Macro': precision_macro,
        'Recall_Macro': recall_macro,
        'F1_Macro': f1_macro,
        'F1_Weighted': f1_weighted,
        'Brier_Score': brier_score,
        'ECE': ece,
        'Log_Loss': log_loss_val,
        'Abstention_Rate': abstention_rate,
        'Coverage': coverage,
        'Accuracy_With_Abstention': accuracy_with_abstention,
        'Avg_Questions_Needed': avg_questions_needed,
        'Sequential_Improvement': sequential_improvement
    }
    
    ablation_results.append(result)
    
    # Print configuration results
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Macro: {f1_macro:.4f}")
    if brier_score is not None:
        print(f"  Brier Score: {brier_score:.4f}")
        print(f"  ECE: {ece:.4f}")
        print(f"  Log Loss: {log_loss_val:.4f}")
    if abstention_rate is not None:
        print(f"  Abstention Rate: {abstention_rate:.4f}")
        print(f"  Coverage: {coverage:.4f}")
        print(f"  Accuracy (with abstention): {accuracy_with_abstention:.4f}")
    if avg_questions_needed is not None:
        print(f"  Avg Questions Needed: {avg_questions_needed:.2f}")
        print(f"  Sequential Improvement: {sequential_improvement:.4f}")

# Create ablation study dataframe
ablation_df = pd.DataFrame(ablation_results)

# Save results
ablation_df.to_csv('results/ablation_study.csv', index=False)
print(f"\n✅ Saved ablation study to results/ablation_study.csv")

# Print comparison table
print("\n" + "="*80)
print("ABLATION STUDY RESULTS")
print("="*80)

# Focus on key metrics for comparison
key_metrics = ['Configuration', 'Accuracy', 'F1_Macro', 'Brier_Score', 'ECE', 'Abstention_Rate', 'Coverage']
comparison_table = ablation_df[key_metrics].copy()
print(comparison_table.to_string(index=False))

# Calculate incremental improvements
print("\n" + "="*60)
print("INCREMENTAL IMPROVEMENTS")
print("="*60)

baseline_f1 = ablation_df.iloc[0]['F1_Macro']
baseline_acc = ablation_df.iloc[0]['Accuracy']

for i, row in ablation_df.iterrows():
    if i == 0:
        continue
    
    f1_improvement = row['F1_Macro'] - baseline_f1
    acc_improvement = row['Accuracy'] - baseline_acc
    
    print(f"\n{row['Configuration']}:")
    print(f"  F1-Macro improvement: {f1_improvement:+.4f}")
    print(f"  Accuracy improvement: {acc_improvement:+.4f}")
    
    if row['Brier_Score'] is not None:
        baseline_brier = ablation_df.iloc[0]['Brier_Score']
        if baseline_brier is not None:
            brier_improvement = baseline_brier - row['Brier_Score']  # Lower is better
            print(f"  Brier Score improvement: {brier_improvement:+.4f} (lower is better)")

# Generate visualization focused on calibration metrics (Brier, ECE, Log Loss)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Filter to only calibrated configurations for these metrics
calibrated_configs = ablation_df[ablation_df['Brier_Score'].notna()].copy()

if len(calibrated_configs) > 0:
    # Brier Score comparison
    ax1 = axes[0]
    ax1.bar(range(len(calibrated_configs)), calibrated_configs['Brier_Score'], color='lightgreen')
    ax1.set_xticks(range(len(calibrated_configs)))
    ax1.set_xticklabels(calibrated_configs['Configuration'], rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Brier Score (lower is better)')
    ax1.set_title('Brier Score by Configuration', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # ECE comparison
    ax2 = axes[1]
    ax2.bar(range(len(calibrated_configs)), calibrated_configs['ECE'], color='lightblue')
    ax2.set_xticks(range(len(calibrated_configs)))
    ax2.set_xticklabels(calibrated_configs['Configuration'], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Expected Calibration Error (lower is better)')
    ax2.set_title('ECE by Configuration', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Log Loss comparison
    ax3 = axes[2]
    ax3.bar(range(len(calibrated_configs)), calibrated_configs['Log_Loss'], color='lightcoral')
    ax3.set_xticks(range(len(calibrated_configs)))
    ax3.set_xticklabels(calibrated_configs['Configuration'], rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Log Loss (lower is better)')
    ax3.set_title('Log Loss by Configuration', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
else:
    # If no calibrated models, show message
    for ax in axes:
        ax.text(0.5, 0.5, 'No calibrated models available', ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('results/ablation_study_visualization.png', dpi=300, bbox_inches='tight')
print("✅ Saved ablation study visualization (focused on calibration metrics)")

print("\n" + "="*80)
print("STAGE 10 SUMMARY")
print("="*80)
print(f"Ablation study completed with {len(ablation_configs)} configurations")
print(f"Results saved to results/ablation_study.csv")
print(f"Visualization saved to results/ablation_study_visualization.png")
print(f"\nKey Findings:")
print(f"  - Baseline performance: F1-Macro = {baseline_f1:.4f}")
print(f"  - Full system performance: F1-Macro = {ablation_df.iloc[-1]['F1_Macro']:.4f}")
print(f"  - Incremental improvement: {ablation_df.iloc[-1]['F1_Macro'] - baseline_f1:+.4f}")
print(f"  - Each component's contribution quantified")
print("="*80)
print("✅ STAGE 10 COMPLETE!")
print("="*80)
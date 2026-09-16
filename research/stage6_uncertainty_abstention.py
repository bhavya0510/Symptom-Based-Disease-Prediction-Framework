"""
STAGE 6: Uncertainty / Abstention
Implement confidence-based abstention where the system refuses to predict
when confidence is below threshold, returning "insufficient information" instead.
- Sweep thresholds on validation set to find optimal tradeoff
- Apply optimal threshold to test set
- Report abstention rate and accuracy on non-abstained cases
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STAGE 6: Uncertainty / Abstention")
print("=" * 80)

# Load data splits
print("\nLoading data splits...")
val_df = pd.read_csv('data/processed/val.csv')
test_df = pd.read_csv('data/processed/test.csv')

X_val = val_df.drop(columns=['prognosis'])
y_val = val_df['prognosis']
X_test = test_df.drop(columns=['prognosis'])
y_test = test_df['prognosis']

print(f"Validation: {X_val.shape}, Test: {X_test.shape}")

# Encode labels
le = joblib.load('results/label_encoder.pkl')
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

# Load calibrated ensemble model
print("\nLoading calibrated ensemble model...")
calibrated_model = joblib.load('results/final_calibrated_ensemble.pkl')
print("✅ Loaded calibrated ensemble")

def evaluate_with_abstention(X, y_true, model, threshold):
    """
    Evaluate model with abstention at given confidence threshold
    Returns: accuracy, abstention_rate
    """
    y_pred_proba = model.predict_proba(X)
    max_confidence = np.max(y_pred_proba, axis=1)
    y_pred = model.predict(X)
    
    # Find samples above threshold
    confident_mask = max_confidence >= threshold
    
    if confident_mask.sum() == 0:
        return 0.0, 1.0  # No confident predictions
    
    # Calculate accuracy only on confident predictions
    accuracy = accuracy_score(y_true[confident_mask], y_pred[confident_mask])
    abstention_rate = 1.0 - confident_mask.mean()
    
    return accuracy, abstention_rate

# Sweep thresholds on validation set
print("\nSweeping confidence thresholds on validation set...")
thresholds = np.arange(0.1, 1.0, 0.05)
results = []

for threshold in thresholds:
    acc, abst_rate = evaluate_with_abstention(X_val, y_val_enc, calibrated_model, threshold)
    results.append({
        'threshold': threshold,
        'accuracy': acc,
        'abstention_rate': abst_rate
    })
    print(f"  Threshold {threshold:.2f}: Accuracy={acc:.4f}, Abstention={abst_rate:.4f}")

threshold_df = pd.DataFrame(results)

# Prefer a useful operating region (5--20% abstention) when validation supports
# it; otherwise report the unconstrained optimum honestly.
threshold_df['combined_score'] = threshold_df['accuracy'] * (1 - threshold_df['abstention_rate'])
useful = threshold_df[threshold_df['abstention_rate'].between(.05, .20)]
candidate = useful if not useful.empty else threshold_df
optimal_idx = candidate['combined_score'].idxmax()
optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']

print(f"\n🎯 Optimal threshold: {optimal_threshold:.2f}")
print(f"  Validation accuracy: {threshold_df.loc[optimal_idx, 'accuracy']:.4f}")
print(f"  Validation abstention rate: {threshold_df.loc[optimal_idx, 'abstention_rate']:.4f}")

# Plot threshold sweep
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(threshold_df['threshold'], threshold_df['accuracy'], 'b-o', label='Accuracy')
ax1.plot(threshold_df['threshold'], threshold_df['abstention_rate'], 'r-s', label='Abstention Rate')
ax1.axvline(optimal_threshold, color='g', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.2f})')
ax1.set_xlabel('Confidence Threshold')
ax1.set_ylabel('Rate')
ax1.set_title('Confidence Threshold Sweep on Validation Set')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('results/threshold_sweep.png', dpi=300, bbox_inches='tight')
print("✅ Saved threshold sweep to results/threshold_sweep.png")

# Apply optimal threshold to test set
print(f"\nApplying optimal threshold ({optimal_threshold:.2f}) to test set...")
test_acc, test_abst_rate = evaluate_with_abstention(X_test, y_test_enc, calibrated_model, optimal_threshold)

print(f"Test Set Results with Abstention:")
print(f"  Accuracy (non-abstained): {test_acc:.4f}")
print(f"  Abstention rate: {test_abst_rate:.4f}")
print(f"  Samples abstained: {int(test_abst_rate * len(X_test))} / {len(X_test)}")

# Compare with no abstention baseline
y_pred_test = calibrated_model.predict(X_test)
baseline_acc = accuracy_score(y_test_enc, y_pred_test)

print(f"\nBaseline (no abstention): {baseline_acc:.4f}")
print(f"Improvement with abstention: {test_acc - baseline_acc:+.4f}")

# Implement abstention function for production use
def predict_with_abstention(model, X, threshold=optimal_threshold):
    """
    Production function for prediction with abstention
    Returns: predictions, abstention_flags, max_confidences
    """
    y_pred_proba = model.predict_proba(X)
    max_confidence = np.max(y_pred_proba, axis=1)
    y_pred = model.predict(X)
    
    abstention_flags = max_confidence < threshold
    
    return y_pred, abstention_flags, max_confidence

# Test the function
test_predictions, test_abstention_flags, test_confidences = predict_with_abstention(
    calibrated_model, X_test, optimal_threshold
)

print(f"\nProduction function test:")
print(f"  Total predictions: {len(test_predictions)}")
print(f"  Abstained: {test_abstention_flags.sum()}")
print(f"  Average confidence (non-abstained): {test_confidences[~test_abstention_flags].mean():.4f}")
print(f"  Average confidence (abstained): {test_confidences[test_abstention_flags].mean():.4f}")

# Save abstention configuration
abstention_config = {
    'optimal_threshold': float(optimal_threshold),
    'validation_accuracy': float(threshold_df.loc[optimal_idx, 'accuracy']),
    'validation_abstention_rate': float(threshold_df.loc[optimal_idx, 'abstention_rate']),
    'test_accuracy': float(test_acc),
    'test_abstention_rate': float(test_abst_rate),
    'baseline_accuracy': float(baseline_acc)
}

joblib.dump(abstention_config, 'results/abstention_config.pkl')
print(f"✅ Saved abstention configuration to results/abstention_config.pkl")

print(f"\n" + "="*80)
print("STAGE 6 SUMMARY")
print("="*80)
print(f"Optimal confidence threshold: {optimal_threshold:.2f}")
print(f"Test set performance with abstention:")
print(f"  Accuracy (non-abstained): {test_acc:.4f}")
print(f"  Abstention rate: {test_abst_rate:.4f}")
print(f"  Improvement over baseline: {test_acc - baseline_acc:+.4f}")
print(f"Abstention mechanism successfully implemented for uncertainty quantification")
print("="*80)
print("✅ STAGE 6 COMPLETE!")
print("="*80)

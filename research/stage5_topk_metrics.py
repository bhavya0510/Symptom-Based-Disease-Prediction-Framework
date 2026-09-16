"""
STAGE 5: Top-K Prediction & Metrics
Compute Top-1, Top-3, and Top-5 accuracy for the calibrated ensemble model.
Also compute for baseline models for comparison.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STAGE 5: Top-K Prediction Metrics")
print("=" * 80)

# Load data splits
print("\nLoading data splits...")
test_df = pd.read_csv('data/processed/test.csv')
X_test = test_df.drop(columns=['prognosis'])
y_test = test_df['prognosis']

print(f"Test set: {X_test.shape}")

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = joblib.load('results/label_encoder.pkl')
y_test_enc = le.transform(y_test)

print(f"Classes: {len(le.classes_)}")

def compute_top_k_accuracy(y_true, y_pred_proba, k):
    """
    Compute Top-K accuracy: proportion of samples where true class is in top K predictions
    """
    top_k_pred = np.argsort(y_pred_proba, axis=1)[:, -k:][:, ::-1]  # Top K indices (descending)
    correct = 0
    for i, true_class in enumerate(y_true):
        if true_class in top_k_pred[i]:
            correct += 1
    return correct / len(y_true)

# Load calibrated ensemble model
print("\nLoading calibrated ensemble model...")
calibrated_model = joblib.load('results/final_calibrated_ensemble.pkl')
print("✅ Loaded calibrated ensemble")

# Get predictions and probabilities
y_pred_ensemble = calibrated_model.predict(X_test)
y_pred_proba_ensemble = calibrated_model.predict_proba(X_test)

# Compute Top-K accuracies for ensemble
top1_ensemble = compute_top_k_accuracy(y_test_enc, y_pred_proba_ensemble, 1)
top3_ensemble = compute_top_k_accuracy(y_test_enc, y_pred_proba_ensemble, 3)
top5_ensemble = compute_top_k_accuracy(y_test_enc, y_pred_proba_ensemble, 5)

print(f"\nCalibrated Ensemble Top-K Accuracy:")
print(f"  Top-1: {top1_ensemble:.4f}")
print(f"  Top-3: {top3_ensemble:.4f}")
print(f"  Top-5: {top5_ensemble:.4f}")

# Also compute for baseline models for comparison
print("\nComputing Top-K for baseline models...")
baseline_models = joblib.load('results/baseline_models.pkl')

baseline_topk_results = []
for model_name, model in baseline_models.items():
    y_pred_proba = model.predict_proba(X_test)
    
    top1 = compute_top_k_accuracy(y_test_enc, y_pred_proba, 1)
    top3 = compute_top_k_accuracy(y_test_enc, y_pred_proba, 3)
    top5 = compute_top_k_accuracy(y_test_enc, y_pred_proba, 5)
    
    baseline_topk_results.append({
        'Model': model_name,
        'Top-1': top1,
        'Top-3': top3,
        'Top-5': top5
    })
    
    print(f"  {model_name}: Top-1={top1:.4f}, Top-3={top3:.4f}, Top-5={top5:.4f}")

# Add ensemble results
baseline_topk_results.append({
    'Model': 'Calibrated Ensemble',
    'Top-1': top1_ensemble,
    'Top-3': top3_ensemble,
    'Top-5': top5_ensemble
})

# Create comparison table
topk_df = pd.DataFrame(baseline_topk_results)
print(f"\nTop-K Accuracy Comparison:")
print(topk_df.to_string(index=False))

# Save results
topk_df.to_csv('results/topk_accuracy.csv', index=False)
print(f"\n✅ Saved Top-K accuracy to results/topk_accuracy.csv")

# Generate Top-K comparison chart
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(topk_df))
width = 0.25

top1_scores = topk_df['Top-1'].values
top3_scores = topk_df['Top-3'].values
top5_scores = topk_df['Top-5'].values

bars1 = ax.bar(x - width, top1_scores, width, label='Top-1', color='steelblue')
bars2 = ax.bar(x, top3_scores, width, label='Top-3', color='coral')
bars3 = ax.bar(x + width, top5_scores, width, label='Top-5', color='lightgreen')

ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Top-K Accuracy Comparison')
ax.set_xticks(x)
ax.set_xticklabels(topk_df['Model'], rotation=45, ha='right')
ax.legend()
ax.set_ylim(0.8, 1.0)
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results/topk_accuracy_chart.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved Top-K chart to results/topk_accuracy_chart.png")

# Print summary
print(f"\n" + "="*80)
print("STAGE 5 SUMMARY")
print("="*80)
print(f"Top-K Accuracy Results:")
print(f"  Calibrated Ensemble: Top-1={top1_ensemble:.4f}, Top-3={top3_ensemble:.4f}, Top-5={top5_ensemble:.4f}")
print(f"  All models show strong Top-K performance due to dataset characteristics")
print(f"  Top-K metrics provide more nuanced evaluation than single-label accuracy")
print("="*80)
print("✅ STAGE 5 COMPLETE!")
print("="*80)

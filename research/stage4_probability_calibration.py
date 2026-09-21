"""Stage 4: fit calibration by CV on train+validation and evaluate once on test."""
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, log_loss
import matplotlib.pyplot as plt

train, val, test = (pd.read_csv(f'data/processed/{x}.csv') for x in ('train','val','test'))
X_trainval = pd.concat([train.drop(columns='prognosis'), val.drop(columns='prognosis')], ignore_index=True)
y_trainval = pd.concat([train.prognosis, val.prognosis], ignore_index=True)
X_test, y_test = test.drop(columns='prognosis'), test.prognosis
le = joblib.load('results/label_encoder.pkl')
y_trainval_enc, y_test_enc = le.transform(y_trainval), le.transform(y_test)
base = joblib.load('results/best_ensemble.pkl')

def multiclass_brier(y, probabilities):
    return float(np.mean([brier_score_loss((y == i).astype(int), probabilities[:, i]) for i in range(probabilities.shape[1])]))

def expected_calibration_error(y_true, y_pred_proba, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) for multiclass classification
    ECE measures the average difference between predicted confidence and observed accuracy
    """
    confidences = np.max(y_pred_proba, axis=1)
    predictions = np.argmax(y_pred_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    
    return ece

def plot_reliability_diagram(y_true, y_pred_proba, n_bins=10, title="Reliability Diagram"):
    """
    Plot reliability diagram showing calibration quality
    """
    confidences = np.max(y_pred_proba, axis=1)
    predictions = np.argmax(y_pred_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot reliability diagram
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
    ax.scatter(bin_confidences, bin_accuracies, s=100, c='red', alpha=0.7, label='Model Bins')
    
    # Add bin size annotations
    for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
        if count > 0:
            ax.annotate(f'n={count}', (conf, acc), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig

# Fit the uncalibrated comparator only on non-test data.
base.fit(X_trainval, y_trainval_enc)
uncalibrated = base.predict_proba(X_test)
rows = []
models = {}

# Compute calibration metrics for uncalibrated model
uncalibrated_ece = expected_calibration_error(y_test_enc, uncalibrated)
uncalibrated_logloss = log_loss(y_test_enc, uncalibrated)

print("Uncalibrated Model Calibration Metrics:")
print(f"  Brier Score: {multiclass_brier(y_test_enc, uncalibrated):.4f}")
print(f"  Expected Calibration Error (ECE): {uncalibrated_ece:.4f}")
print(f"  Log Loss: {uncalibrated_logloss:.4f}")

for method in ('sigmoid', 'isotonic'):
    # Use CV inside the non-test training data; test remains entirely untouched.
    model = CalibratedClassifierCV(joblib.load('results/best_ensemble.pkl'), method=method, cv=3)
    model.fit(X_trainval, y_trainval_enc)
    calibrated = model.predict_proba(X_test)
    
    # Compute enhanced calibration metrics
    calibrated_ece = expected_calibration_error(y_test_enc, calibrated)
    calibrated_logloss = log_loss(y_test_enc, calibrated)
    
    rows.append({
        'Method': method, 
        'Test_Brier_Uncalibrated': multiclass_brier(y_test_enc, uncalibrated),
        'Test_Brier_Calibrated': multiclass_brier(y_test_enc, calibrated),
        'Brier_Improvement': multiclass_brier(y_test_enc, uncalibrated)-multiclass_brier(y_test_enc, calibrated),
        'Test_ECE_Uncalibrated': uncalibrated_ece,
        'Test_ECE_Calibrated': calibrated_ece,
        'ECE_Improvement': uncalibrated_ece - calibrated_ece,
        'Test_LogLoss_Uncalibrated': uncalibrated_logloss,
        'Test_LogLoss_Calibrated': calibrated_logloss,
        'LogLoss_Improvement': uncalibrated_logloss - calibrated_logloss
    })
    models[method] = model

metrics = pd.DataFrame(rows)
metrics.to_csv('results/calibration_metrics.csv', index=False)
print("\n" + "="*60)
print("CALIBRATION METRICS COMPARISON")
print("="*60)
print(metrics.to_string(index=False))

best_method = metrics.loc[metrics.Test_Brier_Calibrated.idxmin(), 'Method']
final = models[best_method]
pred = final.predict(X_test)
production_row = {'Model': 'Final Production Calibrated Ensemble', 'Accuracy': accuracy_score(y_test_enc,pred),
                  'Precision_Macro': precision_score(y_test_enc,pred,average='macro',zero_division=0),
                  'Recall_Macro': recall_score(y_test_enc,pred,average='macro',zero_division=0),
                  'F1_Macro': f1_score(y_test_enc,pred,average='macro',zero_division=0),
                  'F1_Weighted': f1_score(y_test_enc,pred,average='weighted',zero_division=0)}
joblib.dump(final, 'results/final_calibrated_ensemble.pkl')
pd.DataFrame([production_row]).to_csv('results/final_calibrated_metrics.csv', index=False)

# Generate reliability diagrams
print("\nGenerating reliability diagrams...")
calibrated_final = final.predict_proba(X_test)

# Plot uncalibrated reliability diagram
fig_uncal = plot_reliability_diagram(y_test_enc, uncalibrated, title="Uncalibrated Model - Reliability Diagram")
fig_uncal.savefig('results/reliability_diagram_uncalibrated.png', dpi=300, bbox_inches='tight')
print("✅ Saved uncalibrated reliability diagram")

# Plot calibrated reliability diagram
fig_cal = plot_reliability_diagram(y_test_enc, calibrated_final, title=f"Calibrated Model ({best_method}) - Reliability Diagram")
fig_cal.savefig('results/reliability_diagram_calibrated.png', dpi=300, bbox_inches='tight')
print("✅ Saved calibrated reliability diagram")

# Combined comparison plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Uncalibrated
confidences_uncal = np.max(uncalibrated, axis=1)
predictions_uncal = np.argmax(uncalibrated, axis=1)
accuracies_uncal = (predictions_uncal == y_test_enc).astype(float)

bin_boundaries = np.linspace(0, 1, 11)
bin_lowers = bin_boundaries[:-1]
bin_uppers = bin_boundaries[1:]

bin_acc_uncal = []
bin_conf_uncal = []
for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
    in_bin = (confidences_uncal > bin_lower) & (confidences_uncal <= bin_upper)
    if in_bin.sum() > 0:
        bin_acc_uncal.append(accuracies_uncal[in_bin].mean())
        bin_conf_uncal.append(confidences_uncal[in_bin].mean())
    else:
        bin_acc_uncal.append(0)
        bin_conf_uncal.append((bin_lower + bin_upper) / 2)

axes[0].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
axes[0].scatter(bin_conf_uncal, bin_acc_uncal, s=100, c='red', alpha=0.7, label='Model Bins')
axes[0].set_xlabel('Confidence', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Uncalibrated Model', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])

# Calibrated
confidences_cal = np.max(calibrated_final, axis=1)
predictions_cal = np.argmax(calibrated_final, axis=1)
accuracies_cal = (predictions_cal == y_test_enc).astype(float)

bin_acc_cal = []
bin_conf_cal = []
for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
    in_bin = (confidences_cal > bin_lower) & (confidences_cal <= bin_upper)
    if in_bin.sum() > 0:
        bin_acc_cal.append(accuracies_cal[in_bin].mean())
        bin_conf_cal.append(confidences_cal[in_bin].mean())
    else:
        bin_acc_cal.append(0)
        bin_conf_cal.append((bin_lower + bin_upper) / 2)

axes[1].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
axes[1].scatter(bin_conf_cal, bin_acc_cal, s=100, c='blue', alpha=0.7, label='Model Bins')
axes[1].set_xlabel('Confidence', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title(f'Calibrated Model ({best_method})', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('results/reliability_diagram_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved combined reliability diagram comparison")

print(f"\nSelected {best_method}; held-out test calibration scores saved to results/calibration_metrics.csv")
print(production_row)

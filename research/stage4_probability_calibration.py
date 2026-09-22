"""Stage 4: Select calibration method using validation/CV only and evaluate the frozen method on test once."""
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

RESEARCH_DIR = Path(__file__).resolve().parent
os.chdir(RESEARCH_DIR)
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
)
import matplotlib.pyplot as plt


RNG_SEED = 2025
N_BOOTSTRAP = 2000
CI_LEVEL = 95


def multiclass_brier(y_true, probabilities):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    return float(
        np.mean(
            [
                brier_score_loss((y_true == cls).astype(int), probabilities[:, cls])
                for cls in range(probabilities.shape[1])
            ]
        )
    )


def expected_calibration_error(y_true, y_pred_proba, n_bins=10):
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    confidences = np.max(y_pred_proba, axis=1)
    predictions = np.argmax(y_pred_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return float(ece)


def bootstrap_confidence_interval(metric_func, y_true, y_pred_proba, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        y_true_boot = np.asarray(y_true)[indices]
        y_pred_proba_boot = np.asarray(y_pred_proba)[indices]
        score = metric_func(y_true_boot, y_pred_proba_boot)
        bootstrap_scores.append(score)
    bootstrap_scores = np.asarray(bootstrap_scores)
    lower = np.percentile(bootstrap_scores, (100 - ci) / 2)
    upper = np.percentile(bootstrap_scores, 100 - (100 - ci) / 2)
    return float(lower), float(upper)


def log_loss_with_full_labels(y_true, y_pred_proba):
    labels = np.arange(np.asarray(y_pred_proba).shape[1])
    return log_loss(y_true, y_pred_proba, labels=labels)


def plot_reliability_diagram(y_true, y_pred_proba, n_bins=10, title="Reliability Diagram"):
    confidences = np.max(y_pred_proba, axis=1)
    predictions = np.argmax(y_pred_proba, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
    ax.scatter(bin_confidences, bin_accuracies, s=100, c='red', alpha=0.7, label='Model Bins')
    for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts):
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


train, val, test = (pd.read_csv(f'data/processed/{x}.csv') for x in ('train', 'val', 'test'))
X_train = train.drop(columns='prognosis')
y_train = train['prognosis']
X_val = val.drop(columns='prognosis')
y_val = val['prognosis']
X_test = test.drop(columns='prognosis')
y_test = test['prognosis']

le = joblib.load('results/label_encoder.pkl')
y_train_enc = le.transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)
base = joblib.load('results/best_ensemble.pkl')

# Fit the uncalibrated model on training data only for validation comparison.
base.fit(X_train, y_train_enc)
uncalibrated_val_proba = base.predict_proba(X_val)
uncalibrated_val_brier = multiclass_brier(y_val_enc, uncalibrated_val_proba)
uncalibrated_val_ece = expected_calibration_error(y_val_enc, uncalibrated_val_proba)
uncalibrated_val_logloss = log_loss(y_val_enc, uncalibrated_val_proba)
uncalibrated_val_pred = base.predict(X_val)
uncalibrated_val_acc = accuracy_score(y_val_enc, uncalibrated_val_pred)
uncalibrated_val_f1 = f1_score(y_val_enc, uncalibrated_val_pred, average='macro', zero_division=0)

comparison_rows = []
calibration_models = {}
for method in ('sigmoid', 'isotonic'):
    model = CalibratedClassifierCV(estimator=base, method=method, cv=3)
    model.fit(X_train, y_train_enc)
    calibrated_val_proba = model.predict_proba(X_val)
    calibrated_val_pred = model.predict(X_val)

    comparison_rows.append({
        'Method': method,
        'Val_Accuracy': accuracy_score(y_val_enc, calibrated_val_pred),
        'Val_F1_Macro': f1_score(y_val_enc, calibrated_val_pred, average='macro', zero_division=0),
        'Val_Brier': multiclass_brier(y_val_enc, calibrated_val_proba),
        'Val_ECE': expected_calibration_error(y_val_enc, calibrated_val_proba),
        'Val_LogLoss': log_loss(y_val_enc, calibrated_val_proba),
        'Val_Brier_Improvement_vs_Uncalibrated': uncalibrated_val_brier - multiclass_brier(y_val_enc, calibrated_val_proba),
        'Val_ECE_Improvement_vs_Uncalibrated': uncalibrated_val_ece - expected_calibration_error(y_val_enc, calibrated_val_proba),
        'Val_LogLoss_Improvement_vs_Uncalibrated': uncalibrated_val_logloss - log_loss(y_val_enc, calibrated_val_proba),
    })
    calibration_models[method] = model

comparison_df = pd.DataFrame(comparison_rows)
comparison_df = comparison_df.sort_values('Val_Brier').reset_index(drop=True)
comparison_df.to_csv('results/calibration_validation_comparison.csv', index=False)

print('\n' + '=' * 60)
print('CALIBRATION METHOD COMPARISON (Validation / CV Only)')
print('=' * 60)
print(comparison_df.to_string(index=False))

best_method = comparison_df.iloc[0]['Method']
print(f'\nSelected calibration method: {best_method} (lowest validation Brier score; test set not used for selection)')

# Freeze the selected method and use the test set exactly once for final evaluation.
final_model = calibration_models[best_method]
calibrated_test_proba = final_model.predict_proba(X_test)
calibrated_test_pred = final_model.predict(X_test)

final_metrics = {
    'Model': 'Final Production Calibrated Ensemble',
    'Calibration_Method': best_method,
    'Accuracy': float(accuracy_score(y_test_enc, calibrated_test_pred)),
    'Precision_Macro': float(precision_score(y_test_enc, calibrated_test_pred, average='macro', zero_division=0)),
    'Recall_Macro': float(recall_score(y_test_enc, calibrated_test_pred, average='macro', zero_division=0)),
    'F1_Macro': float(f1_score(y_test_enc, calibrated_test_pred, average='macro', zero_division=0)),
    'F1_Weighted': float(f1_score(y_test_enc, calibrated_test_pred, average='weighted', zero_division=0)),
    'Brier_Score': float(multiclass_brier(y_test_enc, calibrated_test_proba)),
    'ECE': float(expected_calibration_error(y_test_enc, calibrated_test_proba)),
    'Log_Loss': float(log_loss(y_test_enc, calibrated_test_proba)),
}

joblib.dump(final_model, 'results/final_calibrated_ensemble.pkl')
pd.DataFrame([final_metrics]).to_csv('results/final_calibrated_metrics.csv', index=False)

# Bootstrap CIs for final held-out test results.
ci_rows = []
metrics_to_bootstrap = {
    'Accuracy': lambda y_true, proba: accuracy_score(y_true, np.argmax(proba, axis=1)),
    'Macro_F1': lambda y_true, proba: f1_score(y_true, np.argmax(proba, axis=1), average='macro', zero_division=0),
    'Weighted_F1': lambda y_true, proba: f1_score(y_true, np.argmax(proba, axis=1), average='weighted', zero_division=0),
    'Brier_Score': multiclass_brier,
    'ECE': expected_calibration_error,
    'Log_Loss': log_loss_with_full_labels,
}

for name, metric_func in metrics_to_bootstrap.items():
    estimate = metric_func(y_test_enc, calibrated_test_proba)
    lower, upper = bootstrap_confidence_interval(metric_func, y_test_enc, calibrated_test_proba)
    ci_rows.append({
        'Metric': name,
        'Estimate': float(estimate),
        'CI_Lower_95': float(lower),
        'CI_Upper_95': float(upper),
        'CI_95': f'[{lower:.4f}, {upper:.4f}]',
        'Selection_Stage': 'final_test_set_only',
    })

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv('results/final_bootstrap_confidence_intervals.csv', index=False)
with open('results/final_bootstrap_confidence_intervals.json', 'w') as f:
    json.dump(ci_df.to_dict(orient='records'), f, indent=2)

print('\n' + '=' * 60)
print('FINAL TEST SET METRICS (selected method only)')
print('=' * 60)
print(pd.DataFrame([final_metrics]).to_string(index=False))
print('\n95% Bootstrap CI summary:')
print(ci_df[['Metric', 'Estimate', 'CI_95']].to_string(index=False))

# Save the selected calibration method for downstream use.
with open('results/selected_calibration_method.json', 'w') as f:
    json.dump({'selected_method': best_method, 'selection_metric': 'validation_brier_score'}, f, indent=2)

# Plot reliability diagram for the chosen method.
fig = plot_reliability_diagram(y_test_enc, calibrated_test_proba, title=f'Reliability Diagram - {best_method.title()} Calibration on Test Set')
fig.savefig('results/reliability_diagram_calibrated.png', dpi=300, bbox_inches='tight')
fig.savefig('results/reliability_diagram_calibrated.pdf', bbox_inches='tight')
plt.close(fig)

print('Selected calibration method and bootstrap CIs saved to results/.')

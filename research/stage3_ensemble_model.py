"""
STAGE 3: Proposed Ensemble Model
Build and evaluate ensemble models:
- Soft-voting ensemble (Random Forest + XGBoost) using weighted probability averaging
- Stacking ensemble (RF + XGBoost as base learners, Logistic Regression as meta-learner)
Compare both and select the best performer for final system.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STAGE 3: Proposed Ensemble Model")
print("=" * 80)

# Load data splits
print("\nLoading data splits...")
train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/val.csv')
test_df = pd.read_csv('data/processed/test.csv')

X_train = train_df.drop(columns=['prognosis'])
y_train = train_df['prognosis']
X_val = val_df.drop(columns=['prognosis'])
y_val = val_df['prognosis']
X_test = test_df.drop(columns=['prognosis'])
y_test = test_df['prognosis']

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

print(f"Classes: {len(le.classes_)}")

# Combine train+val for final training
X_train_val = pd.concat([X_train, X_val])
y_train_val_enc = np.concatenate([y_train_enc, y_val_enc])
cv_splits = joblib.load('data/processed/cv_splits.pkl')

# Load the best baseline models
print("\nLoading best baseline models...")
best_models = joblib.load('results/baseline_models.pkl')
rf_model = best_models['Random Forest']
xgb_model = best_models['XGBoost']

print("✅ Loaded Random Forest and XGBoost models")

# Build soft-voting ensemble
print("\n" + "="*60)
print("Building Soft-Voting Ensemble (RF + XGBoost)")
print("="*60)

voting_ensemble = VotingClassifier(
    estimators=[
        ('random_forest', rf_model),
        ('xgboost', xgb_model)
    ],
    voting='soft'
)

print("Training soft-voting ensemble...")
voting_ensemble.fit(X_train_val, y_train_val_enc)

# Evaluate on test set
y_pred_voting = voting_ensemble.predict(X_test)
y_pred_proba_voting = voting_ensemble.predict_proba(X_test)

accuracy_voting = accuracy_score(y_test_enc, y_pred_voting)
precision_voting = precision_score(y_test_enc, y_pred_voting, average='macro')
recall_voting = recall_score(y_test_enc, y_pred_voting, average='macro')
f1_voting = f1_score(y_test_enc, y_pred_voting, average='macro')
f1_weighted_voting = f1_score(y_test_enc, y_pred_voting, average='weighted')

print(f"\nSoft-Voting Ensemble Test Metrics:")
print(f"  Accuracy: {accuracy_voting:.4f}")
print(f"  Precision (macro): {precision_voting:.4f}")
print(f"  Recall (macro): {recall_voting:.4f}")
print(f"  F1 (macro): {f1_voting:.4f}")
print(f"  F1 (weighted): {f1_weighted_voting:.4f}")

# Build stacking ensemble
print("\n" + "="*60)
print("Building Stacking Ensemble (RF + XGBoost -> LR)")
print("="*60)

stacking_ensemble = StackingClassifier(
    estimators=[
        ('random_forest', rf_model),
        ('xgboost', xgb_model)
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    # Reuse Stage 2's exact folds so comparisons are apples-to-apples.
    cv=cv_splits
)

print("Training stacking ensemble...")
stacking_ensemble.fit(X_train_val, y_train_val_enc)

# Evaluate on test set
y_pred_stacking = stacking_ensemble.predict(X_test)
y_pred_proba_stacking = stacking_ensemble.predict_proba(X_test)

accuracy_stacking = accuracy_score(y_test_enc, y_pred_stacking)
precision_stacking = precision_score(y_test_enc, y_pred_stacking, average='macro')
recall_stacking = recall_score(y_test_enc, y_pred_stacking, average='macro')
f1_stacking = f1_score(y_test_enc, y_pred_stacking, average='macro')
f1_weighted_stacking = f1_score(y_test_enc, y_pred_stacking, average='weighted')

print(f"\nStacking Ensemble Test Metrics:")
print(f"  Accuracy: {accuracy_stacking:.4f}")
print(f"  Precision (macro): {precision_stacking:.4f}")
print(f"  Recall (macro): {recall_stacking:.4f}")
print(f"  F1 (macro): {f1_stacking:.4f}")
print(f"  F1 (weighted): {f1_weighted_stacking:.4f}")

# Compare ensembles and select best
print("\n" + "="*60)
print("ENSEMBLE COMPARISON")
print("="*60)

ensemble_results = {
    'Soft-Voting (RF+XGB)': {
        'Accuracy': accuracy_voting,
        'Precision_Macro': precision_voting,
        'Recall_Macro': recall_voting,
        'F1_Macro': f1_voting,
        'F1_Weighted': f1_weighted_voting
    },
    'Stacking (RF+XGB->LR)': {
        'Accuracy': accuracy_stacking,
        'Precision_Macro': precision_stacking,
        'Recall_Macro': recall_stacking,
        'F1_Macro': f1_stacking,
        'F1_Weighted': f1_weighted_stacking
    }
}

for name, metrics in ensemble_results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# Select best ensemble based on F1-macro
best_ensemble_name = max(ensemble_results.keys(), key=lambda x: ensemble_results[x]['F1_Macro'])
best_ensemble = voting_ensemble if best_ensemble_name == 'Soft-Voting (RF+XGB)' else stacking_ensemble
best_f1 = ensemble_results[best_ensemble_name]['F1_Macro']

print(f"\n🏆 Best ensemble: {best_ensemble_name} (F1-Macro: {best_f1:.4f})")

# Persist held-out metrics for every one of the same folds used by Stage 2.
cv_rows = []
for fold, (fit_idx, holdout_idx) in enumerate(cv_splits, start=1):
    fold_model = clone(best_ensemble)
    fold_model.fit(X_train_val.iloc[fit_idx], y_train_val_enc[fit_idx])
    fold_pred = fold_model.predict(X_train_val.iloc[holdout_idx])
    truth = y_train_val_enc[holdout_idx]
    cv_rows.append({
        'Model': f'Proposed Ensemble ({best_ensemble_name})', 'Fold': fold,
        'Accuracy': accuracy_score(truth, fold_pred),
        'Precision_Macro': precision_score(truth, fold_pred, average='macro', zero_division=0),
        'Recall_Macro': recall_score(truth, fold_pred, average='macro', zero_division=0),
        'F1_Macro': f1_score(truth, fold_pred, average='macro', zero_division=0),
        'F1_Weighted': f1_score(truth, fold_pred, average='weighted', zero_division=0),
    })
pd.DataFrame(cv_rows).to_csv('results/ensemble_cv_metrics.csv', index=False)
print("✅ Saved per-fold ensemble CV metrics to results/ensemble_cv_metrics.csv")

# Load baseline comparison for delta calculation
baseline_results = pd.read_csv('results/model_comparison_test.csv')
best_baseline_f1 = baseline_results['Test_F1_Macro'].max()
best_baseline_name = baseline_results.loc[baseline_results['Test_F1_Macro'].idxmax(), 'Model']

delta = best_f1 - best_baseline_f1
print(f"Delta vs best baseline ({best_baseline_name}): {delta:+.4f} F1-Macro")

# Append ensemble results to comparison table
ensemble_row = {
    'Model': f'Proposed Ensemble ({best_ensemble_name})',
    'Test_Accuracy': ensemble_results[best_ensemble_name]['Accuracy'],
    'Test_Precision_Macro': ensemble_results[best_ensemble_name]['Precision_Macro'],
    'Test_Recall_Macro': ensemble_results[best_ensemble_name]['Recall_Macro'],
    'Test_F1_Macro': ensemble_results[best_ensemble_name]['F1_Macro'],
    'Test_F1_Weighted': ensemble_results[best_ensemble_name]['F1_Weighted']
}

updated_comparison = pd.concat([baseline_results, pd.DataFrame([ensemble_row])], ignore_index=True)
updated_comparison.to_csv('results/model_comparison_test.csv', index=False)
print(f"✅ Updated model comparison with ensemble results")

# Save the best ensemble model
joblib.dump(best_ensemble, 'results/best_ensemble.pkl')
print(f"✅ Saved best ensemble model to results/best_ensemble.pkl")

# Also save both ensembles for reference
joblib.dump({
    'voting_ensemble': voting_ensemble,
    'stacking_ensemble': stacking_ensemble,
    'best_ensemble_name': best_ensemble_name
}, 'results/all_ensembles.pkl')

print(f"\n" + "="*80)
print("STAGE 3 SUMMARY")
print("="*80)
print(f"Built and compared 2 ensemble approaches:")
print(f"  - Soft-Voting (RF + XGBoost): F1-Macro = {f1_voting:.4f}")
print(f"  - Stacking (RF + XGBoost -> LR): F1-Macro = {f1_stacking:.4f}")
print(f"Selected: {best_ensemble_name}")
print(f"Improvement over best baseline: {delta:+.4f} F1-Macro")
print("="*80)
print("✅ STAGE 3 COMPLETE!")
print("="*80)

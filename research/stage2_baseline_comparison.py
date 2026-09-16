"""
STAGE 2: Baseline Model Comparison (Leakage-Free)
Train and evaluate 5 models with hyperparameter tuning on the leakage-free split:
- Logistic Regression (multinomial)
- SVM (RBF kernel)
- Random Forest
- XGBoost
- MLP (Multi-layer Perceptron)

Evaluate on held-out test set with comprehensive metrics.
Report both 5-fold CV metrics (mean±std) and held-out test set metrics separately.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STAGE 2: Baseline Model Comparison (Leakage-Free)")
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
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

print(f"Classes: {len(le.classes_)}")

# Combine train+val for CV
X_train_val = pd.concat([X_train, X_val])
y_train_val_enc = np.concatenate([y_train_enc, y_val_enc])

# Load CV splits
cv_splits = joblib.load('data/processed/cv_splits.pkl')
print(f"Using {len(cv_splits)} stratified CV folds for hyperparameter tuning")

# Define models and hyperparameter grids
print("\nDefining models and hyperparameter grids...")

models_config = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs']
        }
    },
    'SVM': {
        'model': SVC(kernel='rbf', probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_leaf': [1, 2]
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(objective='multi:softprob', random_state=42, n_jobs=-1),
        'params': {
            'max_depth': [3, 6],
            'learning_rate': [0.1, 0.3],
            'n_estimators': [50, 100],
            'subsample': [0.8, 1.0]
        }
    },
    'MLP': {
        'model': MLPClassifier(random_state=42, max_iter=1000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,)],
            'alpha': [0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }
    }
}

# Store results
results = []
cv_results = []
confusion_matrices = {}
best_models = {}

# Train and evaluate each model
for model_name, config in models_config.items():
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    model = config['model']
    param_grid = config['params']
    
    # Use RandomizedSearchCV for efficiency
    print(f"Hyperparameter tuning with RandomizedSearchCV...")
    search = RandomizedSearchCV(
        model, param_grid, n_iter=5, cv=cv_splits, 
        scoring='f1_macro', n_jobs=-1, random_state=42, verbose=1
    )
    
    search.fit(X_train_val, y_train_val_enc)
    
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV F1-macro: {search.best_score_:.4f}")
    
    # Store CV results
    cv_mean = search.best_score_
    cv_std = search.cv_results_['std_test_score'][search.best_index_]
    cv_results.append({
        'Model': model_name,
        'CV_F1_Macro_Mean': cv_mean,
        'CV_F1_Macro_Std': cv_std
    })
    
    # Evaluate on test set
    best_model = search.best_estimator_
    best_models[model_name] = best_model
    
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_enc, y_pred)
    precision_macro = precision_score(y_test_enc, y_pred, average='macro')
    recall_macro = recall_score(y_test_enc, y_pred, average='macro')
    f1_macro = f1_score(y_test_enc, y_pred, average='macro')
    f1_weighted = f1_score(y_test_enc, y_pred, average='weighted')
    
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (macro): {precision_macro:.4f}")
    print(f"  Recall (macro): {recall_macro:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    
    # Store results
    results.append({
        'Model': model_name,
        'Test_Accuracy': accuracy,
        'Test_Precision_Macro': precision_macro,
        'Test_Recall_Macro': recall_macro,
        'Test_F1_Macro': f1_macro,
        'Test_F1_Weighted': f1_weighted
    })
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test_enc, y_pred)
    confusion_matrices[model_name] = cm
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=90, ha='center', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrix for {model_name}")

# Create comparison DataFrames
test_results_df = pd.DataFrame(results)
cv_results_df = pd.DataFrame(cv_results)

print(f"\n{'='*80}")
print("5-FOLD CV RESULTS (Mean ± Std)")
print(f"{'='*80}")
print(cv_results_df.to_string(index=False))

print(f"\n{'='*80}")
print("HELD-OUT TEST SET RESULTS")
print(f"{'='*80}")
print(test_results_df.to_string(index=False))

# Save comparison tables
test_results_df.to_csv('results/model_comparison_test.csv', index=False)
cv_results_df.to_csv('results/model_comparison_cv.csv', index=False)
print(f"\n✅ Saved comparison tables to results/")

# Create combined comparison table
combined_comparison = cv_results_df.merge(test_results_df, on='Model')
combined_comparison.to_csv('results/model_comparison.csv', index=False)
print(f"✅ Saved combined comparison to results/model_comparison.csv")

# Create comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# CV F1-macro comparison
cv_sorted = cv_results_df.sort_values('CV_F1_Macro_Mean', ascending=True)
bars1 = axes[0].barh(cv_sorted['Model'], cv_sorted['CV_F1_Macro_Mean'], yerr=cv_sorted['CV_F1_Macro_Std'], 
                 color='steelblue', alpha=0.7, capsize=5)
axes[0].set_xlabel('CV F1-Macro (Mean ± Std)')
axes[0].set_title('5-Fold CV Performance')
axes[0].set_xlim(0, 1)
for i, (idx, row) in enumerate(cv_sorted.iterrows()):
    axes[0].text(row['CV_F1_Macro_Mean'] + 0.02, i, f'{row["CV_F1_Macro_Mean"]:.3f}±{row["CV_F1_Macro_Std"]:.3f}', 
                 va='center', fontsize=9)

# Test F1-macro comparison
test_sorted = test_results_df.sort_values('Test_F1_Macro', ascending=True)
bars2 = axes[1].barh(test_sorted['Model'], test_sorted['Test_F1_Macro'], color='coral')
axes[1].set_xlabel('Test F1-Macro')
axes[1].set_title('Held-Out Test Set Performance')
axes[1].set_xlim(0, 1)
for bar in bars2:
    width = bar.get_width()
    axes[1].text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/model_comparison_barchart.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved comparison bar chart to results/model_comparison_barchart.png")

# Save best models
joblib.dump(best_models, 'results/baseline_models.pkl')
joblib.dump(le, 'results/label_encoder.pkl')
print(f"✅ Saved best models and label encoder")

# Identify best model
best_model_name = test_results_df.loc[test_results_df['Test_F1_Macro'].idxmax(), 'Model']
best_f1 = test_results_df['Test_F1_Macro'].max()
print(f"\n🏆 Best baseline model: {best_model_name} (Test F1-Macro: {best_f1:.4f})")

print(f"\n{'='*80}")
print("STAGE 2 SUMMARY")
print(f"{'='*80}")
print(f"Trained and evaluated {len(models_config)} baseline models on leakage-free split")
print(f"Dataset: 305 unique patterns, 213 train / 46 val / 46 test")
print(f"Best model: {best_model_name} with Test F1-Macro: {best_f1:.4f}")
print(f"⚠️  CAVEAT: Small dataset leads to high variance in estimates")
print(f"All results saved to results/")
print(f"{'='*80}")
print("✅ STAGE 2 COMPLETE!")
print(f"{'='*80}")

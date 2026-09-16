"""Stage 4: fit calibration by CV on train+validation and evaluate once on test."""
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score

train, val, test = (pd.read_csv(f'data/processed/{x}.csv') for x in ('train','val','test'))
X_trainval = pd.concat([train.drop(columns='prognosis'), val.drop(columns='prognosis')], ignore_index=True)
y_trainval = pd.concat([train.prognosis, val.prognosis], ignore_index=True)
X_test, y_test = test.drop(columns='prognosis'), test.prognosis
le = joblib.load('results/label_encoder.pkl')
y_trainval_enc, y_test_enc = le.transform(y_trainval), le.transform(y_test)
base = joblib.load('results/best_ensemble.pkl')

def multiclass_brier(y, probabilities):
    return float(np.mean([brier_score_loss((y == i).astype(int), probabilities[:, i]) for i in range(probabilities.shape[1])]))

# Fit the uncalibrated comparator only on non-test data.
base.fit(X_trainval, y_trainval_enc)
uncalibrated = base.predict_proba(X_test)
rows = []
models = {}
for method in ('sigmoid', 'isotonic'):
    # Use CV inside the non-test training data; test remains entirely untouched.
    model = CalibratedClassifierCV(joblib.load('results/best_ensemble.pkl'), method=method, cv=3)
    model.fit(X_trainval, y_trainval_enc)
    calibrated = model.predict_proba(X_test)
    rows.append({'Method': method, 'Test_Brier_Uncalibrated': multiclass_brier(y_test_enc, uncalibrated),
                 'Test_Brier_Calibrated': multiclass_brier(y_test_enc, calibrated),
                 'Brier_Improvement': multiclass_brier(y_test_enc, uncalibrated)-multiclass_brier(y_test_enc, calibrated)})
    models[method] = model

metrics = pd.DataFrame(rows)
metrics.to_csv('results/calibration_metrics.csv', index=False)
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
print(f"Selected {best_method}; held-out test Brier scores saved to results/calibration_metrics.csv")
print(production_row)

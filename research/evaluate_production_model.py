"""Single source of truth for the serialized backend production model's test metrics."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

root = Path(__file__).resolve().parent
backend = root.parent / 'HealthGuardAI' / 'backend' / 'artifacts'
test = pd.read_csv(root / 'data' / 'processed' / 'test.csv')
model, encoder = joblib.load(backend/'model.pkl'), joblib.load(backend/'encoder.pkl')
X, y = test.drop(columns='prognosis'), encoder.transform(test.prognosis)
pred = model.predict(X)
row = {'Model':'Final Production Calibrated Ensemble', 'Accuracy':accuracy_score(y,pred),
       'Precision_Macro':precision_score(y,pred,average='macro',zero_division=0),
       'Recall_Macro':recall_score(y,pred,average='macro',zero_division=0),
       'F1_Macro':f1_score(y,pred,average='macro',zero_division=0),
       'F1_Weighted':f1_score(y,pred,average='weighted',zero_division=0)}
comparison_path = root/'results'/'model_comparison.csv'
comparison = pd.read_csv(comparison_path)
comparison = comparison[comparison.Model != row['Model']]
pd.concat([comparison, pd.DataFrame([row])], ignore_index=True).to_csv(comparison_path, index=False)
print(row)

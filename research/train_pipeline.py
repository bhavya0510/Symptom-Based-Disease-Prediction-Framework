"""
Training Pipeline - Runs Stages 1-10 end-to-end and saves production artifacts
This script should be run to train/retrain the model and generate all required artifacts.
Enhanced with rigorous evaluation: calibration methodology with validation-based selection, 
confidence intervals, abstention analysis, sequential questioning with random baseline, 
duplicate analysis, ablation study, and required figures.
"""
import os
import sys
import subprocess
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parent
os.chdir(RESEARCH_DIR)

print("=" * 80)
print("TRAINING PIPELINE - Stages 1-10 with Professor-Requested Enhancements")
print("=" * 80)

# Run Stage 1: Data Cleaning & Splitting
print("\n" + "="*60)
print("STAGE 1: Data Cleaning & Splitting")
print("="*60)
result = subprocess.run([sys.executable, 'stage1_data_splitting.py'], check=True)
print(f"Stage 1 completed with exit code: {result.returncode}")

# Run Stage 2: Baseline Model Comparison  
print("\n" + "="*60)
print("STAGE 2: Baseline Model Comparison")
print("="*60)
result = subprocess.run([sys.executable, 'stage2_baseline_comparison.py'], check=True)
print(f"Stage 2 completed with exit code: {result.returncode}")

# Run Stage 3: Ensemble Model
print("\n" + "="*60)
print("STAGE 3: Ensemble Model")
print("="*60)
result = subprocess.run([sys.executable, 'stage3_ensemble_model.py'], check=True)
print(f"Stage 3 completed with exit code: {result.returncode}")

# Run Stage 4: Probability Calibration
print("\n" + "="*60)
print("STAGE 4: Probability Calibration")
print("="*60)
result = subprocess.run([sys.executable, 'stage4_probability_calibration.py'], check=True)
print(f"Stage 4 completed with exit code: {result.returncode}")

# Run Stage 5: Top-K Metrics
print("\n" + "="*60)
print("STAGE 5: Top-K Metrics")
print("="*60)
result = subprocess.run([sys.executable, 'stage5_topk_metrics.py'], check=True)
print(f"Stage 5 completed with exit code: {result.returncode}")

# Run Stage 6: Uncertainty Abstention
print("\n" + "="*60)
print("STAGE 6: Uncertainty Abstention")
print("="*60)
result = subprocess.run([sys.executable, 'stage6_uncertainty_abstention.py'], check=True)
print(f"Stage 6 completed with exit code: {result.returncode}")

# Run Stage 7: Sequential Questioning
print("\n" + "="*60)
print("STAGE 7: Sequential Questioning")
print("="*60)
result = subprocess.run([sys.executable, 'stage7_sequential_questioning.py'], check=True)
print(f"Stage 7 completed with exit code: {result.returncode}")

# Run Stage 8: SHAP Explainability
print("\n" + "="*60)
print("STAGE 8: SHAP Explainability")
print("="*60)
result = subprocess.run([sys.executable, 'stage8_shap_explainability.py'], check=True)
print(f"Stage 8 completed with exit code: {result.returncode}")

# Run Stage 9: Duplicate and Leakage Analysis
print("\n" + "="*60)
print("STAGE 9: Duplicate and Leakage Analysis")
print("="*60)
result = subprocess.run([sys.executable, 'stage9_duplicate_leakage_analysis.py'], check=True)
print(f"Stage 9 completed with exit code: {result.returncode}")

# Run Stage 10: Ablation Study
print("\n" + "="*60)
print("STAGE 10: Ablation Study")
print("="*60)
result = subprocess.run([sys.executable, 'stage10_ablation_study.py'], check=True)
print(f"Stage 10 completed with exit code: {result.returncode}")

# Generate System Architecture Figure
print("\n" + "="*60)
print("GENERATING SYSTEM ARCHITECTURE FIGURE")
print("="*60)
result = subprocess.run([sys.executable, 'generate_architecture_figure.py'], check=True)
print(f"Architecture figure generation completed with exit code: {result.returncode}")

# Copy artifacts to backend directory
print("\n" + "="*60)
print("Copying artifacts to backend directory")
print("="*60)

import shutil
import joblib
import pandas as pd

# Create backend artifacts directory
os.makedirs('../HealthGuardAI/backend/artifacts', exist_ok=True)

# Copy essential artifacts
artifacts_to_copy = [
    ('results/final_calibrated_ensemble.pkl', 'model.pkl'),
    ('results/label_encoder.pkl', 'encoder.pkl'),
    ('results/mutual_information.pkl', 'mutual_information.pkl'),
    ('results/shap_system.pkl', 'shap_system.pkl'),
    ('results/abstention_config.pkl', 'abstention_config.pkl'),
    ('data/processed/disease_data_clean.csv', 'disease_data_clean.csv'),
    ('results/generalization_table.csv', 'generalization_table.csv')
]

for src, dest in artifacts_to_copy:
    if os.path.exists(src):
        shutil.copy(src, f'../HealthGuardAI/backend/artifacts/{dest}')
        print(f"✅ Copied {src} -> artifacts/{dest}")
    else:
        print(f"⚠️  Warning: {src} not found")

# Create metadata file
try:
    calibrated_metrics = pd.read_csv('results/final_calibrated_metrics.csv')
    calibration_method = calibrated_metrics['Calibration_Method'].iloc[0] if 'Calibration_Method' in calibrated_metrics.columns else 'isotonic'
    
    # Extract confidence intervals if available
    confidence_intervals = {}
    if 'Accuracy_CI_Lower' in calibrated_metrics.columns:
        confidence_intervals['accuracy'] = [float(calibrated_metrics['Accuracy_CI_Lower'].iloc[0]), float(calibrated_metrics['Accuracy_CI_Upper'].iloc[0])]
    if 'Brier_Score_CI_Lower' in calibrated_metrics.columns:
        confidence_intervals['brier_score'] = [float(calibrated_metrics['Brier_Score_CI_Lower'].iloc[0]), float(calibrated_metrics['Brier_Score_CI_Upper'].iloc[0])]
    if 'ECE_CI_Lower' in calibrated_metrics.columns:
        confidence_intervals['ece'] = [float(calibrated_metrics['ECE_CI_Lower'].iloc[0]), float(calibrated_metrics['ECE_CI_Upper'].iloc[0])]
    if 'Log_Loss_CI_Lower' in calibrated_metrics.columns:
        confidence_intervals['log_loss'] = [float(calibrated_metrics['Log_Loss_CI_Lower'].iloc[0]), float(calibrated_metrics['Log_Loss_CI_Upper'].iloc[0])]
    if 'F1_Macro_CI_Lower' in calibrated_metrics.columns:
        confidence_intervals['f1_macro'] = [float(calibrated_metrics['F1_Macro_CI_Lower'].iloc[0]), float(calibrated_metrics['F1_Macro_CI_Upper'].iloc[0])]
except Exception as e:
    print(f"Warning: Could not load calibrated metrics: {e}")
    calibration_method = 'isotonic'
    confidence_intervals = {}

metadata = {
    'symptom_names': pd.read_csv('data/processed/train.csv').drop(columns=['prognosis']).columns.tolist(),
    'disease_classes': joblib.load('results/label_encoder.pkl').classes_.tolist(),
    'n_classes': len(joblib.load('results/label_encoder.pkl').classes_),
    'n_features': 132,
    'model_type': 'Calibrated Ensemble (RF + XGBoost)',
    'calibration_method': calibration_method,
    'abstention_threshold': float(joblib.load('results/abstention_config.pkl')['optimal_threshold']),
    'confidence_intervals': confidence_intervals
}

joblib.dump(metadata, '../HealthGuardAI/backend/artifacts/metadata.pkl')
print("✅ Created metadata.pkl")

print("\n" + "="*80)
print("TRAINING PIPELINE COMPLETE")
print("="*80)
print("All artifacts saved to HealthGuardAI/backend/artifacts/")
print("Ready to start Flask application")
print("="*80)

"""
STAGE 7: Sequential Symptom Questioning
Implement information-gain-based next-question selector for when model abstains.
- Compute mutual information between symptoms and candidate diseases
- Greedy selection of most informative next symptom
- Stateful follow-up question mechanism
"""
import pandas as pd
import numpy as np
from scipy.stats import entropy
import joblib
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RESEARCH_DIR = Path(__file__).resolve().parent
os.chdir(RESEARCH_DIR)

print("=" * 80)
print("STAGE 7: Sequential Symptom Questioning")
print("=" * 80)

# Load training data for information gain computation
print("\nLoading training data for information gain computation...")
train_df = pd.read_csv('data/processed/train.csv')
X_train = train_df.drop(columns=['prognosis'])
y_train = train_df['prognosis']

print(f"Training data: {X_train.shape}")

# Encode labels
le = joblib.load('results/label_encoder.pkl')
y_train_enc = le.transform(y_train)

# Load calibrated ensemble model
print("\nLoading calibrated ensemble model...")
calibrated_model = joblib.load('results/final_calibrated_ensemble.pkl')
print("✅ Loaded calibrated ensemble")

# Get symptom names
symptom_names = X_train.columns.tolist()
print(f"Total symptoms: {len(symptom_names)}")

def compute_mutual_information(symptom_col, disease_col, class_index):
    """
    Compute mutual information between a symptom and disease labels
    MI measures how much knowing the symptom reduces uncertainty about the disease
    """
    # Create contingency table
    contingency = pd.crosstab(symptom_col, disease_col)
    
    if contingency.size == 0:
        return 0.0
    
    # Convert to probabilities
    p_xy = contingency.values / contingency.values.sum()
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    
    # Compute MI using sklearn's mutual_info_classif for robustness
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(symptom_col.values.reshape(-1, 1), (disease_col == class_index).astype(int), discrete_features=True)[0]
    
    return mi

# Precompute mutual information for all symptom-disease pairs
print("\nPrecomputing mutual information matrix...")
mi_matrix = np.zeros((len(symptom_names), len(le.classes_)))

for i, symptom in enumerate(symptom_names):
    if i % 20 == 0:
        print(f"  Computing MI for symptom {i+1}/{len(symptom_names)}: {symptom}")
    
    for class_index in range(len(le.classes_)):
        mi_matrix[i, class_index] = compute_mutual_information(X_train[symptom], y_train_enc, class_index)

print("✅ Mutual information matrix computed")

# Save MI matrix for production use
mi_data = {
    'symptom_names': symptom_names,
    'disease_classes': le.classes_.tolist(),
    'mi_matrix': mi_matrix
}
joblib.dump(mi_data, 'results/mutual_information.pkl')
print("✅ Saved mutual information data to results/mutual_information.pkl")

def select_next_symptom(current_symptoms, current_probabilities, top_k=5):
    """
    Select the most informative next symptom based on current probability distribution
    Uses greedy information gain: maximize expected reduction in entropy
    
    Args:
        current_symptoms: set of symptoms already provided
        current_probabilities: current disease probability distribution
        top_k: consider top-k candidate diseases for information gain calculation
    
    Returns:
        best_symptom: name of most informative symptom to ask next
    """
    # Get top-k candidate diseases
    top_disease_indices = np.argsort(current_probabilities)[-top_k:][::-1]
    
    # Calculate current entropy
    current_entropy = entropy(current_probabilities[top_disease_indices])
    
    # Find symptoms not yet asked
    available_symptoms = [s for s in symptom_names if s not in current_symptoms]
    
    if not available_symptoms:
        return None  # All symptoms already asked
    
    best_symptom = None
    best_expected_reduction = -np.inf
    
    for symptom in available_symptoms:
        symptom_idx = symptom_names.index(symptom)
        
        # Expected information gain: weighted average of entropy reduction
        # for both possible answers (yes/no)
        expected_reduction = 0
        
        for answer in [0, 1]:  # No/Yes
            # How would this answer affect the disease probabilities?
            # Simplified: use MI as proxy for expected information gain
            mi_with_symptom = mi_matrix[symptom_idx, top_disease_indices].mean()
            expected_reduction += mi_with_symptom * 0.5  # Assume 50/50 prior
        
        if expected_reduction > best_expected_reduction:
            best_expected_reduction = expected_reduction
            best_symptom = symptom
    
    return best_symptom

def select_next_symptom_random(current_symptoms, current_probabilities):
    """
    Select next symptom randomly from available symptoms (baseline method)
    
    Args:
        current_symptoms: set of symptoms already provided
        current_probabilities: current disease probability distribution (not used in random selection)
    
    Returns:
        random_symptom: name of randomly selected symptom to ask next
    """
    # Find symptoms not yet asked
    available_symptoms = [s for s in symptom_names if s not in current_symptoms]
    
    if not available_symptoms:
        return None  # All symptoms already asked
    
    # Random selection
    return np.random.choice(available_symptoms)

def sequential_questioning(initial_symptoms, max_questions=5, confidence_threshold=0.4, method='MI'):
    """
    Simulate sequential questioning process
    
    Args:
        initial_symptoms: list of initially provided symptoms
        max_questions: maximum number of follow-up questions
        confidence_threshold: confidence threshold to stop questioning
        method: 'MI' for mutual information, 'Random' for random selection
    
    Returns:
        result: final prediction or "insufficient information"
        questions_asked: list of symptoms asked
        final_confidence: final confidence level
    """
    # Create initial symptom vector
    symptom_vector = {symptom: 0 for symptom in symptom_names}
    for symptom in initial_symptoms:
        if symptom in symptom_vector:
            symptom_vector[symptom] = 1
    
    current_symptoms = set(initial_symptoms)
    questions_asked = []
    
    for iteration in range(max_questions):
        # Create input DataFrame
        input_df = pd.DataFrame([symptom_vector])
        
        # Get prediction
        proba = calibrated_model.predict_proba(input_df)[0]
        max_confidence = np.max(proba)
        prediction = np.argmax(proba)
        
        # Check if confident enough
        if max_confidence >= confidence_threshold:
            return {
                'prediction': le.classes_[prediction],
                'confidence': max_confidence,
                'questions_asked': questions_asked,
                'status': 'confident',
                'method': method
            }
        
        # Select next symptom to ask based on method
        if method == 'MI':
            next_symptom = select_next_symptom(current_symptoms, proba)
        elif method == 'Random':
            next_symptom = select_next_symptom_random(current_symptoms, proba)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if next_symptom is None:
            break  # No more symptoms to ask
        
        questions_asked.append(next_symptom)
        current_symptoms.add(next_symptom)
        
        # Simulate user answering "yes" (for demonstration)
        # In production, this would come from actual user input
        symptom_vector[next_symptom] = 1
    
    # Final prediction after max questions
    input_df = pd.DataFrame([symptom_vector])
    proba = calibrated_model.predict_proba(input_df)[0]
    max_confidence = np.max(proba)
    prediction = np.argmax(proba)
    
    if max_confidence < confidence_threshold:
        return {
            'prediction': 'Insufficient information - please provide additional symptoms',
            'confidence': max_confidence,
            'questions_asked': questions_asked,
            'status': 'abstained',
            'method': method
        }
    else:
        return {
            'prediction': le.classes_[prediction],
            'confidence': max_confidence,
            'questions_asked': questions_asked,
            'status': 'confident_after_questions',
            'method': method
        }

# Test the sequential questioning system
print("\nTesting sequential questioning system...")

# Test case 1: Minimal symptoms
test_symptoms_1 = ['itching', 'skin_rash']
result_1_mi = sequential_questioning(test_symptoms_1, max_questions=3, confidence_threshold=0.7, method='MI')
result_1_random = sequential_questioning(test_symptoms_1, max_questions=3, confidence_threshold=0.7, method='Random')
print(f"\nTest 1 - Initial symptoms: {test_symptoms_1}")
print(f"  MI Result: {result_1_mi}")
print(f"  Random Result: {result_1_random}")

# Test case 2: More specific symptoms
test_symptoms_2 = ['high_fever', 'cough', 'headache']
result_2_mi = sequential_questioning(test_symptoms_2, max_questions=3, confidence_threshold=0.7, method='MI')
result_2_random = sequential_questioning(test_symptoms_2, max_questions=3, confidence_threshold=0.7, method='Random')
print(f"\nTest 2 - Initial symptoms: {test_symptoms_2}")
print(f"  MI Result: {result_2_mi}")
print(f"  Random Result: {result_2_random}")

# Test case 3: Very specific symptoms
test_symptoms_3 = ['yellowing_of_eyes', 'nausea', 'loss_of_appetite', 'abdominal_pain']
result_3_mi = sequential_questioning(test_symptoms_3, max_questions=3, confidence_threshold=0.7, method='MI')
result_3_random = sequential_questioning(test_symptoms_3, max_questions=3, confidence_threshold=0.7, method='Random')
print(f"\nTest 3 - Initial symptoms: {test_symptoms_3}")
print(f"  MI Result: {result_3_mi}")
print(f"  Random Result: {result_3_random}")

# Comprehensive paired evaluation of sequential questioning on test set
print("\n" + "="*60)
print("COMPREHENSIVE PAIRED SEQUENTIAL QUESTIONING EVALUATION")
print("="*60)

import json
from scipy.stats import shapiro, ttest_rel, wilcoxon

test_df = pd.read_csv('data/processed/test.csv')
X_test = test_df.drop(columns=['prognosis'])
y_test = test_df['prognosis']
y_test_enc = le.transform(y_test)

# Fixed experiment configuration
SEED = 42
MAX_QUESTIONS = 5
CONF_THRESHOLD = 0.70

# 1. Build ONE fixed list of eligible evaluation cases BEFORE running MI and Random
evaluation_cases = []

for idx in range(len(X_test)):
    sample_row = X_test.iloc[idx]
    present_symptoms = sample_row[sample_row == 1].index.tolist()
    
    for init_k in [1, 2, 3, 5]:
        if len(present_symptoms) >= init_k:
            initial_symptoms = present_symptoms[:init_k]
            
            # Create initial symptom vector to get starting prediction and confidence
            symptom_vector = {symptom: 0 for symptom in symptom_names}
            for s in initial_symptoms:
                symptom_vector[s] = 1
            
            input_df = pd.DataFrame([symptom_vector])
            proba = calibrated_model.predict_proba(input_df)[0]
            init_conf = float(np.max(proba))
            init_pred_idx = int(np.argmax(proba))
            init_pred_str = le.classes_[init_pred_idx]
            
            case_id = f"sample_{idx}_init_{init_k}"
            
            evaluation_cases.append({
                'case_id': case_id,
                'sample_idx': idx,
                'init_k': init_k,
                'true_label_enc': y_test_enc[idx],
                'true_label_str': y_test.iloc[idx],
                'initial_symptoms': initial_symptoms,
                'initial_prediction': init_pred_str,
                'initial_confidence': init_conf
            })

print(f"✅ Total paired evaluation cases created: {len(evaluation_cases)}")

def run_single_case_evaluation(case_info, method='MI', seed=42):
    """
    Evaluates a single case starting from exact initial symptoms and model state
    """
    initial_symptoms = case_info['initial_symptoms']
    symptom_vector = {symptom: 0 for symptom in symptom_names}
    for s in initial_symptoms:
        symptom_vector[s] = 1
    
    current_symptoms = set(initial_symptoms)
    questions_asked = []
    
    # Deterministic RNG for Random baseline
    rng = np.random.default_rng(seed) if method == 'Random' else None
    
    for iteration in range(MAX_QUESTIONS):
        input_df = pd.DataFrame([symptom_vector])
        proba = calibrated_model.predict_proba(input_df)[0]
        max_conf = float(np.max(proba))
        
        if max_conf >= CONF_THRESHOLD:
            break
        
        if method == 'MI':
            next_symptom = select_next_symptom(current_symptoms, proba)
        elif method == 'Random':
            available = [s for s in symptom_names if s not in current_symptoms]
            next_symptom = rng.choice(available) if available else None
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if next_symptom is None:
            break
        
        questions_asked.append(str(next_symptom))
        current_symptoms.add(next_symptom)
        symptom_vector[next_symptom] = 1
    
    # Final prediction after questioning
    input_df = pd.DataFrame([symptom_vector])
    proba = calibrated_model.predict_proba(input_df)[0]
    final_conf = float(np.max(proba))
    final_pred_idx = int(np.argmax(proba))
    final_pred_str = le.classes_[final_pred_idx]
    is_correct = 1 if final_pred_idx == case_info['true_label_enc'] else 0
    conf_gain = final_conf - case_info['initial_confidence']
    
    return {
        'questions_asked': questions_asked,
        'n_questions': len(questions_asked),
        'final_confidence': final_conf,
        'final_prediction': final_pred_str,
        'confidence_gain': conf_gain,
        'accuracy': is_correct,
        'status': 'confident' if final_conf >= CONF_THRESHOLD else 'abstained'
    }

# 2. Run BOTH MI and Random on the exact same evaluation cases
paired_case_rows = []

for case_index, case_info in enumerate(evaluation_cases):
    # MI evaluation
    res_mi = run_single_case_evaluation(case_info, method='MI')
    
    # Random evaluation with dedicated deterministic seed per case
    case_seed = SEED + case_index * 1000
    res_random = run_single_case_evaluation(case_info, method='Random', seed=case_seed)
    
    paired_case_rows.append({
        'case_id': case_info['case_id'],
        'sample_idx': case_info['sample_idx'],
        'initial_symptoms_count': case_info['init_k'],
        'initial_symptoms': json.dumps(case_info['initial_symptoms']),
        'initial_prediction': case_info['initial_prediction'],
        'initial_confidence': case_info['initial_confidence'],
        'mi_questions': res_mi['n_questions'],
        'random_questions': res_random['n_questions'],
        'mi_confidence_gain': res_mi['confidence_gain'],
        'random_confidence_gain': res_random['confidence_gain'],
        'mi_accuracy': res_mi['accuracy'],
        'random_accuracy': res_random['accuracy'],
        'mi_final_confidence': res_mi['final_confidence'],
        'random_final_confidence': res_random['final_confidence'],
        'mi_selected_symptoms': json.dumps(res_mi['questions_asked']),
        'random_selected_symptoms': json.dumps(res_random['questions_asked']),
        'mi_final_prediction': res_mi['final_prediction'],
        'random_final_prediction': res_random['final_prediction']
    })

df_paired_all = pd.DataFrame(paired_case_rows)

# Programmatic verification: Check identical case counts
assert len(df_paired_all) == len(evaluation_cases), "MI and Random case counts mismatch!"
print(f"✅ Verified identical paired case evaluation: {len(df_paired_all)} cases")

# Save paired cases to CSV
df_paired_all.to_csv('results/sequential_questioning_paired.csv', index=False)
print("✅ Saved detailed paired results to results/sequential_questioning_paired.csv")

# 3. Perform paired statistical testing across initial symptom settings & overall
stat_rows = []
eval_rows = []

for init_k in [1, 2, 3, 5]:
    df_k = df_paired_all[df_paired_all['initial_symptoms_count'] == init_k].copy()
    n_k = len(df_k)
    
    mi_q = df_k['mi_questions']
    rand_q = df_k['random_questions']
    mi_cg = df_k['mi_confidence_gain']
    rand_cg = df_k['random_confidence_gain']
    mi_acc = df_k['mi_accuracy']
    rand_acc = df_k['random_accuracy']
    
    # Store summary for sequential_questioning_evaluation.csv
    eval_rows.append({
        'method': 'MI',
        'initial_symptoms': init_k,
        'samples_evaluated': n_k,
        'avg_questions_needed': mi_q.mean(),
        'std_questions_needed': mi_q.std(ddof=1) if n_k > 1 else 0,
        'avg_confidence_improvement': mi_cg.mean(),
        'std_confidence_improvement': mi_cg.std(ddof=1) if n_k > 1 else 0,
        'final_accuracy': mi_acc.mean()
    })
    eval_rows.append({
        'method': 'Random',
        'initial_symptoms': init_k,
        'samples_evaluated': n_k,
        'avg_questions_needed': rand_q.mean(),
        'std_questions_needed': rand_q.std(ddof=1) if n_k > 1 else 0,
        'avg_confidence_improvement': rand_cg.mean(),
        'std_confidence_improvement': rand_cg.std(ddof=1) if n_k > 1 else 0,
        'final_accuracy': rand_acc.mean()
    })
    
    # Statistical test for Questions Needed (MI - Random)
    diff_q = mi_q - rand_q
    mean_diff_q = float(diff_q.mean())
    std_diff_q = float(diff_q.std(ddof=1)) if n_k > 1 else 0.0
    sem_q = std_diff_q / np.sqrt(n_k) if n_k > 1 else 0.0
    ci_q = float(1.96 * sem_q)
    
    if np.all(diff_q == 0) or n_k < 3:
        test_used_q = "None (No variance)"
        stat_q, pval_q = 0.0, 1.0
    else:
        # Check normality
        _, p_norm_q = shapiro(diff_q)
        if p_norm_q > 0.05:
            test_used_q = "Paired t-test"
            stat_q, pval_q = ttest_rel(mi_q, rand_q)
        else:
            test_used_q = "Wilcoxon signed-rank test"
            try:
                stat_q, pval_q = wilcoxon(mi_q, rand_q)
            except Exception:
                test_used_q = "Paired t-test (Fallback)"
                stat_q, pval_q = ttest_rel(mi_q, rand_q)
    
    # Statistical test for Confidence Gain (MI - Random)
    diff_cg = mi_cg - rand_cg
    mean_diff_cg = float(diff_cg.mean())
    std_diff_cg = float(diff_cg.std(ddof=1)) if n_k > 1 else 0.0
    sem_cg = std_diff_cg / np.sqrt(n_k) if n_k > 1 else 0.0
    ci_cg = float(1.96 * sem_cg)
    
    if np.all(diff_cg == 0) or n_k < 3:
        test_used_cg = "None (No variance)"
        stat_cg, pval_cg = 0.0, 1.0
    else:
        _, p_norm_cg = shapiro(diff_cg)
        if p_norm_cg > 0.05:
            test_used_cg = "Paired t-test"
            stat_cg, pval_cg = ttest_rel(mi_cg, rand_cg)
        else:
            test_used_cg = "Wilcoxon signed-rank test"
            try:
                stat_cg, pval_cg = wilcoxon(mi_cg, rand_cg)
            except Exception:
                test_used_cg = "Paired t-test (Fallback)"
                stat_cg, pval_cg = ttest_rel(mi_cg, rand_cg)
                
    stat_rows.append({
        'initial_symptoms': init_k,
        'paired_cases': n_k,
        'mi_avg_questions': float(mi_q.mean()),
        'random_avg_questions': float(rand_q.mean()),
        'questions_diff_mean': mean_diff_q,
        'questions_diff_95_ci_lower': mean_diff_q - ci_q,
        'questions_diff_95_ci_upper': mean_diff_q + ci_q,
        'questions_test_used': test_used_q,
        'questions_test_stat': float(stat_q),
        'questions_p_value': float(pval_q),
        'mi_avg_conf_gain': float(mi_cg.mean()),
        'random_avg_conf_gain': float(rand_cg.mean()),
        'conf_gain_diff_mean': mean_diff_cg,
        'conf_gain_diff_95_ci_lower': mean_diff_cg - ci_cg,
        'conf_gain_diff_95_ci_upper': mean_diff_cg + ci_cg,
        'conf_gain_test_used': test_used_cg,
        'conf_gain_test_stat': float(stat_cg),
        'conf_gain_p_value': float(pval_cg),
        'mi_accuracy': float(mi_acc.mean()),
        'random_accuracy': float(rand_acc.mean())
    })
    
    print(f"\nResults for Initial Symptoms = {init_k} ({n_k} paired cases):")
    print(f"  MI Questions: {mi_q.mean():.2f} ± {mi_q.std():.2f} | Random Questions: {rand_q.mean():.2f} ± {rand_q.std():.2f}")
    print(f"  Questions Difference ({test_used_q}): Mean={mean_diff_q:+.4f} (95% CI: [{mean_diff_q-ci_q:.4f}, {mean_diff_q+ci_q:.4f}]), p={pval_q:.4e}")
    print(f"  MI Conf Gain: {mi_cg.mean():.4f} | Random Conf Gain: {rand_cg.mean():.4f}")
    print(f"  Conf Gain Difference ({test_used_cg}): Mean={mean_diff_cg:+.4f} (95% CI: [{mean_diff_cg-ci_cg:.4f}, {mean_diff_cg+ci_cg:.4f}]), p={pval_cg:.4e}")

# Overall paired statistics across all 172 cases
df_all = df_paired_all
n_total = len(df_all)
mi_q_all = df_all['mi_questions']
rand_q_all = df_all['random_questions']
diff_q_all = mi_q_all - rand_q_all
mean_diff_q_all = float(diff_q_all.mean())
ci_q_all = float(1.96 * diff_q_all.std(ddof=1) / np.sqrt(n_total))

_, p_norm_q_all = shapiro(diff_q_all)
test_used_q_all = "Paired t-test" if p_norm_q_all > 0.05 else "Wilcoxon signed-rank test"
try:
    stat_q_all, pval_q_all = wilcoxon(mi_q_all, rand_q_all) if test_used_q_all == "Wilcoxon signed-rank test" else ttest_rel(mi_q_all, rand_q_all)
except Exception:
    stat_q_all, pval_q_all = ttest_rel(mi_q_all, rand_q_all)

mi_cg_all = df_all['mi_confidence_gain']
rand_cg_all = df_all['random_confidence_gain']
diff_cg_all = mi_cg_all - rand_cg_all
mean_diff_cg_all = float(diff_cg_all.mean())
ci_cg_all = float(1.96 * diff_cg_all.std(ddof=1) / np.sqrt(n_total))

_, p_norm_cg_all = shapiro(diff_cg_all)
test_used_cg_all = "Paired t-test" if p_norm_cg_all > 0.05 else "Wilcoxon signed-rank test"
try:
    stat_cg_all, pval_cg_all = wilcoxon(mi_cg_all, rand_cg_all) if test_used_cg_all == "Wilcoxon signed-rank test" else ttest_rel(mi_cg_all, rand_cg_all)
except Exception:
    stat_cg_all, pval_cg_all = ttest_rel(mi_cg_all, rand_cg_all)

stat_rows.append({
    'initial_symptoms': 'Overall',
    'paired_cases': n_total,
    'mi_avg_questions': float(mi_q_all.mean()),
    'random_avg_questions': float(rand_q_all.mean()),
    'questions_diff_mean': mean_diff_q_all,
    'questions_diff_95_ci_lower': mean_diff_q_all - ci_q_all,
    'questions_diff_95_ci_upper': mean_diff_q_all + ci_q_all,
    'questions_test_used': test_used_q_all,
    'questions_test_stat': float(stat_q_all),
    'questions_p_value': float(pval_q_all),
    'mi_avg_conf_gain': float(mi_cg_all.mean()),
    'random_avg_conf_gain': float(rand_cg_all.mean()),
    'conf_gain_diff_mean': mean_diff_cg_all,
    'conf_gain_diff_95_ci_lower': mean_diff_cg_all - ci_cg_all,
    'conf_gain_diff_95_ci_upper': mean_diff_cg_all + ci_cg_all,
    'conf_gain_test_used': test_used_cg_all,
    'conf_gain_test_stat': float(stat_cg_all),
    'conf_gain_p_value': float(pval_cg_all),
    'mi_accuracy': float(df_all['mi_accuracy'].mean()),
    'random_accuracy': float(df_all['random_accuracy'].mean())
})

# Save output CSVs
stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv('results/sequential_questioning_statistics.csv', index=False)
print("\n✅ Saved paired statistical results to results/sequential_questioning_statistics.csv")

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv('results/sequential_questioning_evaluation.csv', index=False)
print("✅ Saved sequential questioning evaluation to results/sequential_questioning_evaluation.csv")

# Generate plot comparing MI vs Random
import matplotlib.pyplot as plt

mi_results = eval_df[eval_df['method'] == 'MI'].sort_values('initial_symptoms')
random_results = eval_df[eval_df['method'] == 'Random'].sort_values('initial_symptoms')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Average questions needed
axes[0].plot(mi_results['initial_symptoms'], mi_results['avg_questions_needed'], 
            'o-', label='MI', linewidth=2, markersize=8, color='blue')
axes[0].plot(random_results['initial_symptoms'], random_results['avg_questions_needed'], 
            's-', label='Random', linewidth=2, markersize=8, color='red')
axes[0].set_xlabel('Number of Initial Symptoms', fontsize=12)
axes[0].set_ylabel('Average Questions Needed', fontsize=12)
axes[0].set_title('Questions Needed vs Initial Symptoms', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Confidence improvement
axes[1].plot(mi_results['initial_symptoms'], mi_results['avg_confidence_improvement'], 
            'o-', label='MI', linewidth=2, markersize=8, color='blue')
axes[1].plot(random_results['initial_symptoms'], random_results['avg_confidence_improvement'], 
            's-', label='Random', linewidth=2, markersize=8, color='red')
axes[1].set_xlabel('Number of Initial Symptoms', fontsize=12)
axes[1].set_ylabel('Average Confidence Improvement', fontsize=12)
axes[1].set_title('Confidence Gain vs Initial Symptoms', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# Plot 3: Final accuracy
axes[2].plot(mi_results['initial_symptoms'], mi_results['final_accuracy'], 
            'o-', label='MI', linewidth=2, markersize=8, color='blue')
axes[2].plot(random_results['initial_symptoms'], random_results['final_accuracy'], 
            's-', label='Random', linewidth=2, markersize=8, color='red')
axes[2].set_xlabel('Number of Initial Symptoms', fontsize=12)
axes[2].set_ylabel('Final Accuracy', fontsize=12)
axes[2].set_title('Accuracy vs Initial Symptoms', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([0.8, 1.0])

plt.tight_layout()
plt.savefig('results/sequential_questioning_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('results/sequential_questioning_comparison.pdf', bbox_inches='tight')
print("✅ Saved MI vs Random comparison visualization")

# Production helper function for Flask integration
def get_next_question(current_symptom_vector, current_probabilities, method='MI'):
    """
    Production function to get the next question to ask
    """
    current_symptoms = set([symptom for symptom, value in current_symptom_vector.items() if value == 1])
    if method == 'MI':
        next_symptom = select_next_symptom(current_symptoms, current_probabilities)
    elif method == 'Random':
        next_symptom = select_next_symptom_random(current_symptoms, current_probabilities)
    else:
        raise ValueError(f"Unknown method: {method}")
    return next_symptom

# Save questioning system pickle
questioning_system = {
    'symptom_names': symptom_names,
    'select_next_symptom': select_next_symptom,
    'select_next_symptom_random': select_next_symptom_random,
    'sequential_questioning': sequential_questioning,
    'get_next_question': get_next_question
}

joblib.dump(questioning_system, 'results/questioning_system.pkl')
print("✅ Saved questioning system to results/questioning_system.pkl")

print(f"\n" + "="*80)
print("STAGE 7 SUMMARY")
print("="*80)
print(f"Sequential symptom questioning system implemented:")
print(f"  - Paired evaluation on {len(df_paired_all)} total cases")
print(f"  - Identical case selection and state initialization for MI and Random")
print(f"  - Deterministic RNG used for Random baseline reproducibility")
print(f"  - Paired statistical testing (Shapiro-Wilk + Wilcoxon/t-test with 95% CI)")
print(f"  - Results saved to sequential_questioning_paired.csv and sequential_questioning_statistics.csv")
print("="*80)
print("✅ STAGE 7 COMPLETE!")
print("="*80)

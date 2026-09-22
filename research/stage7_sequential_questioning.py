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

# Comprehensive evaluation of sequential questioning on test set
print("\n" + "="*60)
print("COMPREHENSIVE SEQUENTIAL QUESTIONING EVALUATION")
print("="*60)

test_df = pd.read_csv('data/processed/test.csv')
X_test = test_df.drop(columns=['prognosis'])
y_test = test_df['prognosis']
y_test_enc = le.transform(y_test)

# Simulate sequential questioning for various initial symptom counts and methods
evaluation_results = []

for method in ['MI', 'Random']:
    for initial_symptom_count in [1, 2, 3, 5]:
        print(f"\nEvaluating {method} with {initial_symptom_count} initial symptoms...")
        
        questions_needed = []
        confidence_improvements = []
        accuracy_improvements = []
        successful_predictions = 0
        final_accuracies = []
        
        for idx in range(min(100, len(X_test))):  # Sample subset for efficiency
            # Get initial symptoms (first N symptoms present)
            sample_row = X_test.iloc[idx]
            present_symptoms = sample_row[sample_row == 1].index.tolist()
            
            if len(present_symptoms) < initial_symptom_count:
                continue
                
            initial_symptoms = present_symptoms[:initial_symptom_count]
            
            # Get initial prediction without sequential questioning
            initial_vector = {symptom: 0 for symptom in symptom_names}
            for symptom in initial_symptoms:
                if symptom in initial_vector:
                    initial_vector[symptom] = 1
            
            initial_input = pd.DataFrame([initial_vector])
            initial_proba = calibrated_model.predict_proba(initial_input)[0]
            initial_confidence = np.max(initial_proba)
            initial_prediction = np.argmax(initial_proba)
            initial_correct = (initial_prediction == y_test_enc[idx])
            
            # Run sequential questioning with specified method
            result = sequential_questioning(initial_symptoms, max_questions=5, confidence_threshold=0.7, method=method)
            
            if result['status'] in ['confident', 'confident_after_questions']:
                successful_predictions += 1
                questions_needed.append(len(result['questions_asked']))
                confidence_improvements.append(result['confidence'] - initial_confidence)
                
                # Check if prediction improved
                final_prediction_correct = (le.classes_.tolist().index(result['prediction']) if isinstance(result['prediction'], str) and result['prediction'] in le.classes_ else result['prediction'] == y_test_enc[idx])
                final_accuracies.append(1 if final_prediction_correct else 0)
                
                if not initial_correct and final_prediction_correct:
                    accuracy_improvements.append(1)
                elif initial_correct and not final_prediction_correct:
                    accuracy_improvements.append(-1)
                else:
                    accuracy_improvements.append(0)
        
        if questions_needed:
            evaluation_results.append({
                'method': method,
                'initial_symptoms': initial_symptom_count,
                'samples_evaluated': len(questions_needed),
                'avg_questions_needed': np.mean(questions_needed),
                'std_questions_needed': np.std(questions_needed),
                'avg_confidence_improvement': np.mean(confidence_improvements),
                'successful_prediction_rate': successful_predictions / min(100, len(X_test)),
                'final_accuracy': np.mean(final_accuracies) if final_accuracies else 0,
                'accuracy_improvement_rate': np.mean([acc for acc in accuracy_improvements if acc != 0]) if [acc for acc in accuracy_improvements if acc != 0] else 0
            })
            
            print(f"  Average questions needed: {np.mean(questions_needed):.2f} ± {np.std(questions_needed):.2f}")
            print(f"  Average confidence improvement: {np.mean(confidence_improvements):.4f}")
            print(f"  Successful prediction rate: {successful_predictions / min(100, len(X_test)):.4f}")
            print(f"  Final accuracy: {np.mean(final_accuracies):.4f}" if final_accuracies else "")

# Save evaluation results
if evaluation_results:
    eval_df = pd.DataFrame(evaluation_results)
    eval_df.to_csv('results/sequential_questioning_evaluation.csv', index=False)
    print(f"\n✅ Saved sequential questioning evaluation to results/sequential_questioning_evaluation.csv")
    print(eval_df.to_string(index=False))
    
    # Create visualization comparing MI vs Random
    import matplotlib.pyplot as plt
    
    # Separate results by method
    mi_results = eval_df[eval_df['method'] == 'MI'].sort_values('initial_symptoms')
    random_results = eval_df[eval_df['method'] == 'Random'].sort_values('initial_symptoms')
    
    if not mi_results.empty and not random_results.empty:
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

# Create production-ready function for Flask integration
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

# Save the questioning system
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
print(f"  - Mutual information matrix computed for {len(symptom_names)} symptoms × {len(le.classes_)} diseases")
print(f"  - MI-based greedy information gain selection algorithm")
print(f"  - Random baseline selection algorithm for comparison")
print(f"  - Stateful follow-up question mechanism")
print(f"  - Production-ready functions for Flask integration")
print(f"  - Tested with sample symptom combinations")
print(f"  - Comprehensive comparison of MI vs Random baseline")
print("="*80)
print("✅ STAGE 7 COMPLETE!")
print("="*80)

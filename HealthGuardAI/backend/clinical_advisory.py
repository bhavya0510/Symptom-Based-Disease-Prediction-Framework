"""
Clinical Advisory and Doctor Consultation Support Module
Provides clinical triage, prolonged symptom risk analysis (e.g., prolonged fever -> Dengue/Typhoid/Malaria),
recommended diagnostic tests, emergency red flags, and specialist referral recommendations.
"""
from typing import List, Dict, Any, Optional

def analyze_symptom_duration(
    symptoms: List[str],
    duration_days: Optional[int] = None,
    duration_category: Optional[str] = None,
    symptom_durations: Optional[Dict[str, int]] = None,
    predicted_disease: Optional[str] = None,
    confidence: float = 0.0,
    severity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates symptoms and their duration to generate a comprehensive clinical advisory,
    recommended diagnostic tests, and doctor consultation guidance.
    """
    symptoms_set = set(s.strip().lower() for s in (symptoms or []))
    symptom_durations = symptom_durations or {}
    
    # Normalize duration_days
    if duration_days is None or duration_days <= 0:
        if duration_category == 'acute' or duration_category == '1-2_days':
            duration_days = 2
        elif duration_category == 'persistent' or duration_category == '3-7_days':
            duration_days = 5
        elif duration_category == 'prolonged' or duration_category == '1-2_weeks':
            duration_days = 10
        elif duration_category == 'chronic' or duration_category == 'more_than_2_weeks':
            duration_days = 21
        else:
            duration_days = 1  # Default to day 1 if unspecified

    # Detect key symptom groups
    has_fever = any(s in symptoms_set for s in ['high_fever', 'mild_fever', 'chills', 'shivering'])
    has_respiratory = any(s in symptoms_set for s in ['cough', 'breathlessness', 'phlegm', 'blood_in_sputum', 'chest_pain'])
    has_jaundice_liver = any(s in symptoms_set for s in ['yellowish_skin', 'yellowing_of_eyes', 'dark_urine', 'yellow_urine'])
    has_gi = any(s in symptoms_set for s in ['vomiting', 'diarrhoea', 'abdominal_pain', 'stomach_pain', 'belly_pain', 'nausea'])
    has_dengue_cluster = any(s in symptoms_set for s in ['pain_behind_the_eyes', 'joint_pain', 'muscle_pain', 'back_pain', 'red_spots_over_body', 'skin_rash'])
    
    # Specific duration for fever if provided
    fever_duration = symptom_durations.get('high_fever') or symptom_durations.get('mild_fever') or duration_days if has_fever else 0
    cough_duration = symptom_durations.get('cough') or duration_days if has_respiratory else 0

    # Emergency Red Flags
    red_flags = []
    if 'blood_in_sputum' in symptoms_set:
        red_flags.append("Hemoptysis (Coughing blood)")
    if 'bloody_stool' in symptoms_set or 'stomach_bleeding' in symptoms_set:
        red_flags.append("Gastrointestinal bleeding (Bloody stool or vomit)")
    if 'red_spots_over_body' in symptoms_set and has_fever:
        red_flags.append("Petechiae / Subcutaneous bleeding with fever (Critical Dengue Warning)")
    if 'breathlessness' in symptoms_set and ('chest_pain' in symptoms_set or duration_days >= 3):
        red_flags.append("Acute respiratory distress / Shortness of breath")
    if 'altered_sensorium' in symptoms_set or 'coma' in symptoms_set:
        red_flags.append("Altered mental consciousness / Confusion")
    if 'cold_hands_and_feets' in symptoms_set and has_fever:
        red_flags.append("Peripheral circulatory collapse / Warning sign of Dengue Shock Syndrome")

    # Determine Urgency Level
    # Levels: CRITICAL_EMERGENCY, HIGH_URGENCY, MODERATE_CONSULTATION, ROUTINE_OBSERVATION
    urgency_level = "ROUTINE_OBSERVATION"
    urgency_badge = "Mild / Home Observation"
    urgency_color = "success"
    doctor_advice = ""
    differential_risks = []
    recommended_tests = []
    specialist_recommendations = []
    action_items = []
    medication_warnings = []

    # 1. EVALUATION FOR FEVER DURATION (User's primary requirement: fever from long time -> Dengue, Typhoid, etc.)
    if has_fever:
        if fever_duration >= 3 or (has_fever and duration_days >= 3):
            urgency_level = "HIGH_URGENCY"
            urgency_badge = "High Urgency: Doctor Consultation Advised"
            urgency_color = "danger"

            # Prolonged fever risk explanation
            if fever_duration >= 5 or duration_days >= 5:
                doctor_advice = (
                    f"Prolonged Fever Alert ({duration_days}+ days): A fever persisting for 3 to 7+ days is NOT a simple "
                    f"cold or viral flu. In tropical and endemic conditions, persistent fever is a classic presentation of "
                    f"Dengue Fever, Typhoid (Enteric Fever), Malaria, or systemic bacterial infections. Immediate physician "
                    f"evaluation is strongly recommended to order blood work and prevent complications."
                )
            else:
                doctor_advice = (
                    f"Persistent Fever Detected ({duration_days} days): Fever persisting past 48–72 hours warrants a medical "
                    f"consultation. Blood investigations (such as CBC with Platelet count) are critical to rule out Dengue "
                    f"or Typhoid before symptoms escalate."
                )

            # Differential risks to rule out
            differential_risks.extend([
                {
                    "disease": "Dengue Fever",
                    "reason": "Prolonged high fever with body aches, retro-orbital pain, or rash. High risk of sudden platelet drop.",
                    "severity": "High",
                    "requires_immediate_testing": True
                },
                {
                    "disease": "Typhoid Fever (Enteric Fever)",
                    "reason": "Step-ladder persistent fever lasting over 4-5 days, often accompanied by headache, fatigue, or abdominal symptoms.",
                    "severity": "Moderate-High",
                    "requires_immediate_testing": True
                },
                {
                    "disease": "Malaria",
                    "reason": "Chills, rigor, recurring spikes of fever, shivering, and profuse sweating.",
                    "severity": "Moderate-High",
                    "requires_immediate_testing": True
                },
                {
                    "disease": "Secondary Bacterial Infection",
                    "reason": "Persistent inflammatory response requiring culture-guided antibiotic management.",
                    "severity": "Moderate",
                    "requires_immediate_testing": False
                }
            ])

            # Recommended Diagnostic Tests
            recommended_tests.extend([
                {
                    "test_name": "Complete Blood Count (CBC) with Platelet Count",
                    "code": "CBC",
                    "importance": "Critical (Mandatory)",
                    "reason": "Monitors Hematocrit (HCT), White Blood Cells (WBC), and critical Platelet count drop to rule out Dengue thrombocytopenia.",
                    "turnaround": "Same day (2-4 hours)"
                },
                {
                    "test_name": "Dengue NS1 Antigen & IgM/IgG Antibody Serology",
                    "code": "DENGUE_PANEL",
                    "importance": "High Priority",
                    "reason": "NS1 Antigen detects Dengue within Day 1-5; IgM/IgG antibodies detect active or recent Dengue infection from Day 5 onwards.",
                    "turnaround": "Same day"
                },
                {
                    "test_name": "Typhoid Test (Widal Test / Blood Culture for S. Typhi)",
                    "code": "TYPHOID_WIDAL",
                    "importance": "High Priority",
                    "reason": "Identifies Salmonella Typhi infection, essential when fever persists past day 4-5.",
                    "turnaround": "24 - 48 hours for culture"
                },
                {
                    "test_name": "Peripheral Blood Smear (PBS) / Rapid Malaria Test",
                    "code": "MALARIA_PBS",
                    "importance": "Recommended",
                    "reason": "Direct microscopic visualization of Plasmodium vivax / falciparum parasites.",
                    "turnaround": "1 - 2 hours"
                },
                {
                    "test_name": "C-Reactive Protein (CRP) & ESR",
                    "code": "CRP_ESR",
                    "importance": "Helpful",
                    "reason": "Quantitative markers of systemic inflammatory severity.",
                    "turnaround": "Same day"
                },
                {
                    "test_name": "Routine Urine Examination & Microscopy",
                    "code": "URINE_RM",
                    "importance": "Routine",
                    "reason": "Rules out occult urinary tract infection as the source of prolonged fever.",
                    "turnaround": "1 - 2 hours"
                }
            ])

            specialist_recommendations.append("General Physician / Internal Medicine Specialist")
            specialist_recommendations.append("Infectious Disease Specialist (if fever persists > 7 days)")

            medication_warnings.append(
                "CRITICAL WARNING: Avoid taking NSAIDs (Aspirin, Ibuprofen, Brufen, Diclofenac) without a doctor's explicit prescription. "
                "If Dengue is present, NSAIDs severely impair platelet aggregation and cause life-threatening internal bleeding. "
                "Only Paracetamol (Acetaminophen) is recommended for fever management under medical advice."
            )

            action_items.extend([
                "Maintain aggressive oral hydration (Oral Rehydration Salts / ORS, coconut water, clean soups, minimum 2.5–3 liters/day).",
                "Log your body temperature every 4 to 6 hours in a written journal.",
                "Visit a nearby diagnostic laboratory or clinic for CBC and Dengue/Typhoid screening as soon as possible.",
                "Watch out for warning signs: severe stomach pain, persistent vomiting, bleeding gums, or extreme dizziness."
            ])
        else:
            # Short fever (1-2 days)
            urgency_level = "MODERATE_CONSULTATION"
            urgency_badge = "Moderate: Monitor & Consult if Fever Persists"
            urgency_color = "warning"
            doctor_advice = (
                f"Fever in early stage ({duration_days} day(s)): Stay hydrated and rest. If your fever does not subside within "
                f"48 to 72 hours, or exceeds 102°F (38.9°C), visit a physician for diagnostic blood work."
            )
            recommended_tests.append({
                "test_name": "Complete Blood Count (CBC) (if fever continues past Day 3)",
                "code": "CBC",
                "importance": "Recommended if persistent",
                "reason": "Baseline evaluation of white blood cells and platelets.",
                "turnaround": "Same day"
            })
            specialist_recommendations.append("General Physician")
            action_items.append("Hydrate well and monitor oral temperature twice daily.")

    # 2. RESPIRATORY (Cough, Breathlessness)
    if has_respiratory:
        if cough_duration >= 14 or duration_days >= 14:
            urgency_level = "HIGH_URGENCY"
            urgency_badge = "High Urgency: Pulmonologist Consultation Advised"
            urgency_color = "danger"
            doctor_advice += (
                f"\n\nChronic Respiratory Alert ({duration_days}+ days): Cough persisting for 2 or more weeks is a key "
                f"clinical marker that requires chest imaging to rule out Tuberculosis (TB), bacterial pneumonia, or chronic bronchitis."
            )
            differential_risks.append({
                "disease": "Tuberculosis (TB)",
                "reason": "Persistent cough lasting > 2 weeks, especially with fever, night sweats, or weight loss.",
                "severity": "High",
                "requires_immediate_testing": True
            })
            differential_risks.append({
                "disease": "Bacterial Pneumonia / Bronchitis",
                "reason": "Subacute lower respiratory tract inflammation with productive phlegm.",
                "severity": "Moderate-High",
                "requires_immediate_testing": True
            })
            recommended_tests.extend([
                {
                    "test_name": "Chest X-Ray (PA View)",
                    "code": "CXR",
                    "importance": "Critical",
                    "reason": "Detects pulmonary infiltrates, consolidation (pneumonia), or apical cavitary lesions (TB).",
                    "turnaround": "Immediate (1 hour)"
                },
                {
                    "test_name": "Sputum for Acid Fast Bacilli (AFB) / GeneXpert",
                    "code": "SPUTUM_TB",
                    "importance": "High Priority",
                    "reason": "Gold standard microbiological test for Mycobacterium tuberculosis.",
                    "turnaround": "24 - 48 hours"
                }
            ])
            specialist_recommendations.append("Pulmonologist / Chest Physician")

    # 3. HEPATIC / JAUNDICE
    if has_jaundice_liver:
        urgency_level = "HIGH_URGENCY"
        urgency_badge = "High Urgency: Liver Function Evaluation Needed"
        urgency_color = "danger"
        doctor_advice += (
            f"\n\nHepatic / Jaundice Alert: Yellowing of skin or eyes with dark urine indicates elevated serum bilirubin "
            f"and liver inflammation. Clinical consultation with comprehensive Liver Function Tests (LFT) is imperative."
        )
        differential_risks.append({
            "disease": "Acute Viral Hepatitis (A/B/C/E)",
            "reason": "Hepatic viral infection causing impaired bilirubin clearance and hepatocyte distress.",
            "severity": "High",
            "requires_immediate_testing": True
        })
        recommended_tests.extend([
            {
                "test_name": "Liver Function Test (LFT Panel: Total & Direct Bilirubin, SGOT/AST, SGPT/ALT, Alk Phos)",
                "code": "LFT",
                "importance": "Critical",
                "reason": "Evaluates the degree of hepatocellular damage and biliary obstruction.",
                "turnaround": "Same day"
            },
            {
                "test_name": "Ultrasound (USG) Whole Abdomen",
                "code": "USG_ABD",
                "importance": "High Priority",
                "reason": "Visualizes liver parenchyma, gallbladder calculi, and common bile duct dilation.",
                "turnaround": "Same day"
            },
            {
                "test_name": "Viral Hepatitis Serology Panel (Anti-HAV, HBsAg, Anti-HCV)",
                "code": "HEP_PANEL",
                "importance": "Recommended",
                "reason": "Determines the specific viral etiology of acute hepatitis.",
                "turnaround": "Same day"
            }
        ])
        specialist_recommendations.append("Gastroenterologist / Hepatologist")

    # 4. OVERRIDE FOR EMERGENCY RED FLAGS
    if red_flags:
        urgency_level = "CRITICAL_EMERGENCY"
        urgency_badge = "CRITICAL: Immediate Emergency Medical Attention Required"
        urgency_color = "critical"
        doctor_advice = (
            f"EMERGENCY WARNING: Red flag clinical symptoms detected: {', '.join(red_flags)}. "
            f"Do not delay. Proceed immediately to the nearest Hospital Emergency Department or Trauma Center."
        )

    # Correlate with model prediction if available
    correlation_note = None
    if predicted_disease:
        pred_lower = predicted_disease.lower()
        if 'dengue' in pred_lower:
            correlation_note = (
                f"Model Prediction Match: The AI model detected patterns aligning with Dengue ({(confidence*100):.1f}% confidence). "
                f"Combined with your reported symptoms and duration ({duration_days} days), immediate Platelet Count and Dengue NS1/IgM "
                f"testing is strongly advised to prevent dengue hemorrhagic complications."
            )
        elif 'typhoid' in pred_lower:
            correlation_note = (
                f"Model Prediction Match: The AI model identified strong markers for Typhoid / Enteric Fever. "
                f"Given the {duration_days}-day duration, an official Widal test or Blood Culture should be ordered by a physician."
            )
        elif 'malaria' in pred_lower:
            correlation_note = (
                f"Model Prediction Match: AI analysis indicates Malaria risk. A Rapid Diagnostic Test (RDT) and Peripheral Smear "
                f"should be conducted without delay."
            )
        elif 'tuberculosis' in pred_lower:
            correlation_note = (
                f"Model Prediction Match: AI pattern matches Tuberculosis profile. A Chest X-Ray and Sputum AFB test are mandatory."
            )
        elif 'jaundice' in pred_lower or 'hepatitis' in pred_lower:
            correlation_note = (
                f"Model Prediction Match: AI detected Hepatic/Jaundice condition. Immediate Liver Function Panel (LFT) is indicated."
            )

    # Deduplicate tests and specialists
    unique_tests = []
    seen_test_codes = set()
    for t in recommended_tests:
        if t['code'] not in seen_test_codes:
            unique_tests.append(t)
            seen_test_codes.add(t['code'])

    unique_specialists = list(dict.fromkeys(specialist_recommendations))
    if not unique_specialists:
        unique_specialists = ["General Physician / Family Doctor"]

    return {
        "duration_days": duration_days,
        "duration_category": duration_category or f"{duration_days} days",
        "urgency_level": urgency_level,
        "urgency_badge": urgency_badge,
        "urgency_color": urgency_color,
        "doctor_advice": doctor_advice,
        "differential_risks": differential_risks,
        "recommended_tests": unique_tests,
        "red_flags": red_flags,
        "specialists": unique_specialists,
        "action_items": action_items,
        "medication_warnings": medication_warnings,
        "correlation_note": correlation_note,
        "should_see_doctor": urgency_level in ["HIGH_URGENCY", "CRITICAL_EMERGENCY"]
    }

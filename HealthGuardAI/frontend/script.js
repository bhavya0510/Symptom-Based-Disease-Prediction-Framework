// Health Guard AI - Research-Grade Frontend Script
const API_URL = window.location.origin && window.location.origin !== 'null' && window.location.protocol.startsWith('http')
  ? `${window.location.origin}/predict`
  : 'http://127.0.0.1:5050/predict';
const FOLLOWUP_API_URL = window.location.origin && window.location.origin !== 'null' && window.location.protocol.startsWith('http')
  ? `${window.location.origin}/followup`
  : 'http://127.0.0.1:5050/followup';
const HEALTH_API_URL = window.location.origin && window.location.origin !== 'null' && window.location.protocol.startsWith('http')
  ? `${window.location.origin}/health`
  : 'http://127.0.0.1:5050/health';
const SYMPTOMS_API_URL = window.location.origin && window.location.origin !== 'null' && window.location.protocol.startsWith('http')
  ? `${window.location.origin}/symptoms`
  : 'http://127.0.0.1:5050/symptoms';
const COMPARISON_API_URL = window.location.origin && window.location.origin !== 'null' && window.location.protocol.startsWith('http')
  ? `${window.location.origin}/model-comparison`
  : 'http://127.0.0.1:5050/model-comparison';

let isDarkMode = localStorage.getItem('darkMode') === 'true';
let currentResults = [];
let selectedSymptoms = new Set();
let selectedDurationDays = 5;
let selectedDurationCategory = 'persistent';
let currentCategory = 'all';
let lastClinicalAdvisory = null;
let lastPredictions = [];
let lastShapExplanation = null;

const SYMPTOM_CATEGORIES = {
  fever: ['high_fever', 'mild_fever', 'chills', 'shivering', 'sweating', 'headache', 'malaise', 'fatigue', 'toxic_look_(typhos)', 'sunken_eyes', 'dehydration'],
  respiratory: ['cough', 'breathlessness', 'chest_pain', 'phlegm', 'throat_irritation', 'runny_nose', 'continuous_sneezing', 'sinus_pressure', 'congestion', 'blood_in_sputum', 'rusty_sputum', 'mucoid_sputum'],
  digestive: ['vomiting', 'diarrhoea', 'abdominal_pain', 'stomach_pain', 'acidity', 'indigestion', 'constipation', 'yellowish_skin', 'yellowing_of_eyes', 'dark_urine', 'yellow_urine', 'nausea', 'loss_of_appetite', 'stomach_bleeding', 'belly_pain', 'passage_of_gases'],
  aches: ['joint_pain', 'muscle_pain', 'back_pain', 'neck_pain', 'pain_behind_the_eyes', 'fatigue', 'weakness_in_limbs', 'dizziness', 'cramps', 'knee_pain', 'hip_joint_pain', 'swelling_joints', 'stiff_neck']
};

let allSymptoms = [
  'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
  'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
  'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety',
  'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
  'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes',
  'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin',
  'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
  'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
  'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
  'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
  'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain',
  'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region',
  'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising',
  'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
  'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
  'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness',
  'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance',
  'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
  'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
  'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium',
  'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches',
  'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
  'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
  'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
  'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf',
  'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
  'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
  'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
];

document.addEventListener('DOMContentLoaded', initApp);

function initApp() {
  initDarkMode();
  initSymptomSelector();
  initScrollAnimations();
  initMobileMenu();
  initSmoothScrolling();
  initFormInteractions();
  loadModelComparison();
  
  window.predictDisease = predictDisease;
  window.scrollToForm = scrollToForm;
  window.scrollToFeatures = scrollToFeatures;
  window.toggleSymptom = toggleSymptom;
  window.filterSymptoms = filterSymptoms;
  window.filterByCategory = filterByCategory;
  window.selectDurationChip = selectDurationChip;
  window.adjustDuration = adjustDuration;
  window.onCustomDurationChange = onCustomDurationChange;
  window.openDoctorSummaryModal = openDoctorSummaryModal;
  window.closeDoctorModal = closeDoctorModal;
  window.closeDoctorModalOnBackdrop = closeDoctorModalOnBackdrop;
  window.findNearbyClinics = findNearbyClinics;
  window.resetForm = resetForm;
  window.exportResults = exportResults;
  window.toggleMobileMenu = toggleMobileMenu;
  window.triggerFollowUp = triggerFollowUp;
  window.answerFollowUp = answerFollowUp;
  
  updateLiveClinicalTip();
}

// === DARK MODE ===
function initDarkMode() {
  const toggle = document.querySelector('.dark-mode-toggle');
  toggle.addEventListener('click', toggleDarkMode);
  document.documentElement.dataset.theme = isDarkMode ? 'dark' : 'light';
  updateDarkModeIcon();
}

function toggleDarkMode() {
  isDarkMode = !isDarkMode;
  document.documentElement.dataset.theme = isDarkMode ? 'dark' : 'light';
  localStorage.setItem('darkMode', isDarkMode);
  updateDarkModeIcon();
  
  // Add animation
  const toggle = document.querySelector('.dark-mode-toggle');
  toggle.style.transform = 'scale(1.2) rotate(360deg)';
  setTimeout(() => {
    toggle.style.transform = '';
  }, 300);
}

function updateDarkModeIcon() {
  const icon = document.querySelector('.dark-mode-toggle i');
  icon.className = isDarkMode ? 'fas fa-sun' : 'fas fa-moon';
}

// === MOBILE MENU ===
function initMobileMenu() {
  const toggle = document.querySelector('.mobile-menu-toggle');
  if (toggle) {
    toggle.addEventListener('click', toggleMobileMenu);
  }
}

function toggleMobileMenu() {
  const navMenu = document.querySelector('.nav-menu');
  const toggle = document.querySelector('.mobile-menu-toggle');
  
  navMenu.classList.toggle('mobile-active');
  toggle.classList.toggle('active');
}

// === SMOOTH SCROLLING ===
function initSmoothScrolling() {
  // Smooth scroll for navigation links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
}

function scrollToForm() {
  const formSection = document.getElementById('form-section');
  if (formSection) {
    formSection.scrollIntoView({ behavior: 'smooth' });
    
    // Add focus effect to symptoms input
    setTimeout(() => {
      const symptomsInput = document.getElementById('symptoms');
      if (symptomsInput) {
        symptomsInput.focus();
        symptomsInput.classList.add('pulse-once');
        setTimeout(() => symptomsInput.classList.remove('pulse-once'), 1000);
      }
    }, 800);
  }
}

function scrollToFeatures() {
  const featuresSection = document.getElementById('features');
  if (featuresSection) {
    featuresSection.scrollIntoView({ behavior: 'smooth' });
  }
}

// === FORM INTERACTIONS ===
function initFormInteractions() {
  const inputs = document.querySelectorAll('.form-group input, .form-group select');
  inputs.forEach(input => {
    // Focus effects
    input.addEventListener('focus', () => {
      input.parentElement.classList.add('focused');
    });
    
    input.addEventListener('blur', () => {
      input.parentElement.classList.remove('focused');
    });
    
    // Real-time validation
    input.addEventListener('input', () => {
      validateInput(input);
    });
  });
}

function validateInput(input) {
  const wrapper = input.closest('.input-wrapper');
  if (!wrapper) return;
  
  // Remove existing validation states
  wrapper.classList.remove('valid', 'invalid');
  
  if (input.value.trim()) {
    if (input.type === 'number' && input.id === 'age') {
      const age = parseInt(input.value);
      if (age >= 1 && age <= 120) {
        wrapper.classList.add('valid');
      } else {
        wrapper.classList.add('invalid');
      }
    } else {
      wrapper.classList.add('valid');
    }
  }
}

// === SYMPTOM SELECTOR ===
async function initSymptomSelector() {
  const symptomsGrid = document.getElementById('symptomsGrid');
  if (!symptomsGrid) return;
  
  try {
    const response = await fetch(SYMPTOMS_API_URL);
    if (response.ok) {
      const data = await response.json();
      if (data.symptoms && data.symptoms.length > 0) {
        allSymptoms = data.symptoms;
      }
    }
  } catch (err) {
    console.log("Using fallback symptom list:", err);
  }
  
  // Render all symptoms
  renderSymptoms(allSymptoms);
}

function renderSymptoms(symptoms) {
  const symptomsGrid = document.getElementById('symptomsGrid');
  if (!symptomsGrid) return;
  
  symptomsGrid.innerHTML = '';
  
  symptoms.forEach(symptom => {
    const symptomItem = document.createElement('div');
    symptomItem.className = 'symptom-item';
    symptomItem.dataset.symptom = symptom;
    
    const isSelected = selectedSymptoms.has(symptom);
    if (isSelected) {
      symptomItem.classList.add('selected');
    }
    
    symptomItem.innerHTML = `
      <input type="checkbox" id="symptom-${symptom.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9-]/g, '')}" 
             ${isSelected ? 'checked' : ''} onchange="toggleSymptom('${symptom}')">
      <label for="symptom-${symptom.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9-]/g, '')}">
        <span class="symptom-name">${formatSymptomName(symptom)}</span>
      </label>
    `;
    
    symptomsGrid.appendChild(symptomItem);
  });
}

function formatSymptomName(symptom) {
  return symptom
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
}

function toggleSymptom(symptom) {
  if (selectedSymptoms.has(symptom)) {
    selectedSymptoms.delete(symptom);
  } else {
    selectedSymptoms.add(symptom);
  }
  
  updateSelectedSymptomsDisplay();
  updateSymptomCheckboxUI(symptom);
  updateLiveClinicalTip();
}

function updateSymptomCheckboxUI(symptom) {
  const symptomItem = document.querySelector(`.symptom-item[data-symptom="${symptom}"]`);
  if (symptomItem) {
    const checkbox = symptomItem.querySelector('input[type="checkbox"]');
    checkbox.checked = selectedSymptoms.has(symptom);
    
    if (selectedSymptoms.has(symptom)) {
      symptomItem.classList.add('selected');
    } else {
      symptomItem.classList.remove('selected');
    }
  }
}

function updateSelectedSymptomsDisplay() {
  const selectedList = document.getElementById('selectedSymptomsList');
  const selectedCount = document.getElementById('selectedCount');
  
  if (!selectedList || !selectedCount) return;
  
  selectedCount.textContent = selectedSymptoms.size;
  
  if (selectedSymptoms.size === 0) {
    selectedList.innerHTML = '<p class="no-symptoms">No symptoms selected</p>';
    return;
  }
  
  selectedList.innerHTML = '';
  selectedSymptoms.forEach(symptom => {
    const symptomTag = document.createElement('div');
    symptomTag.className = 'symptom-tag';
    symptomTag.innerHTML = `
      <span>${formatSymptomName(symptom)}</span>
      <button type="button" onclick="toggleSymptom('${symptom}')" class="remove-symptom">
        <i class="fas fa-times"></i>
      </button>
    `;
    selectedList.appendChild(symptomTag);
  });
}

// === DURATION & CLINICAL CONTEXT CONTROLS ===
function selectDurationChip(btn, days, category) {
  document.querySelectorAll('.duration-chip').forEach(c => c.classList.remove('active'));
  if (btn) btn.classList.add('active');
  
  selectedDurationDays = parseInt(days);
  selectedDurationCategory = category;
  
  const input = document.getElementById('durationDays');
  if (input) input.value = selectedDurationDays;
  
  updateLiveClinicalTip();
}

function adjustDuration(delta) {
  const input = document.getElementById('durationDays');
  if (!input) return;
  
  let val = parseInt(input.value) || 1;
  val = Math.max(1, Math.min(90, val + delta));
  input.value = val;
  onCustomDurationChange(val);
}

function onCustomDurationChange(val) {
  selectedDurationDays = Math.max(1, Math.min(90, parseInt(val) || 1));
  
  // Categorize
  if (selectedDurationDays <= 2) {
    selectedDurationCategory = 'acute';
  } else if (selectedDurationDays <= 5) {
    selectedDurationCategory = 'persistent';
  } else if (selectedDurationDays <= 10) {
    selectedDurationCategory = 'prolonged';
  } else {
    selectedDurationCategory = 'chronic';
  }
  
  // Sync chip active state
  document.querySelectorAll('.duration-chip').forEach(c => {
    const chipDays = parseInt(c.dataset.days);
    if ((selectedDurationCategory === 'acute' && chipDays === 2) ||
        (selectedDurationCategory === 'persistent' && chipDays === 5) ||
        (selectedDurationCategory === 'prolonged' && chipDays === 8) ||
        (selectedDurationCategory === 'chronic' && chipDays === 15)) {
      c.classList.add('active');
    } else {
      c.classList.remove('active');
    }
  });
  
  updateLiveClinicalTip();
}

function updateLiveClinicalTip() {
  const tipEl = document.getElementById('liveClinicalTip');
  const tipText = document.getElementById('liveClinicalTipText');
  if (!tipEl || !tipText) return;

  const hasFever = Array.from(selectedSymptoms).some(s => s.includes('fever') || s === 'chills' || s === 'shivering');
  const hasCough = Array.from(selectedSymptoms).some(s => s === 'cough' || s === 'breathlessness' || s === 'chest_pain');
  const hasLiver = Array.from(selectedSymptoms).some(s => s.includes('yellow') || s.includes('urine'));
  const days = selectedDurationDays;

  if (hasFever) {
    if (days >= 3) {
      tipEl.className = 'live-clinical-tip alert-danger';
      tipText.innerHTML = `<strong>Clinical Alert (${days} Days Fever):</strong> Fever persisting for 3+ days strongly requires testing for <strong>Dengue, Typhoid (Enteric Fever), or Malaria</strong>. Doctor consultation and CBC/Platelet test are advised.`;
    } else {
      tipEl.className = 'live-clinical-tip alert-warning';
      tipText.innerHTML = `<strong>Fever detected (${days} day(s)):</strong> Stay well-hydrated. If fever persists past 48–72 hours, consult a doctor for a CBC & Platelet blood test.`;
    }
  } else if (hasCough && days >= 14) {
    tipEl.className = 'live-clinical-tip alert-danger';
    tipText.innerHTML = `<strong>Chronic Cough Alert (${days}+ days):</strong> Cough lasting 2+ weeks warrants clinical evaluation, Chest X-Ray, and Sputum/TB screening.`;
  } else if (hasLiver) {
    tipEl.className = 'live-clinical-tip alert-danger';
    tipText.innerHTML = `<strong>Jaundice Alert:</strong> Yellowing of skin/eyes or dark urine indicates liver strain. Prompt Liver Function Tests (LFT) are recommended.`;
  } else if (days >= 7) {
    tipEl.className = 'live-clinical-tip alert-warning';
    tipText.innerHTML = `<strong>Persistent Symptoms (${days} days):</strong> Prolonged symptoms warrant a comprehensive physical examination by a healthcare provider.`;
  } else {
    tipEl.className = 'live-clinical-tip';
    tipText.innerHTML = `Persistent fever or symptoms for 3+ days require monitoring for Dengue, Typhoid, or viral complications.`;
  }
}

function filterByCategory(category) {
  currentCategory = category;
  
  // Highlight tab
  document.querySelectorAll('.category-tab').forEach(t => t.classList.remove('active'));
  const activeBtn = window.event ? window.event.target.closest('.category-tab') : null;
  if (activeBtn) activeBtn.classList.add('active');
  
  filterSymptoms();
}

function filterSymptoms() {
  const searchInput = document.getElementById('symptomSearch');
  const searchTerm = searchInput ? searchInput.value.toLowerCase().trim() : '';
  
  let baseSymptoms = allSymptoms;
  if (currentCategory !== 'all' && SYMPTOM_CATEGORIES[currentCategory]) {
    const catList = new Set(SYMPTOM_CATEGORIES[currentCategory]);
    baseSymptoms = allSymptoms.filter(s => catList.has(s));
  }
  
  const filteredSymptoms = baseSymptoms.filter(symptom => 
    symptom.toLowerCase().includes(searchTerm) ||
    formatSymptomName(symptom).toLowerCase().includes(searchTerm)
  );
  
  renderSymptoms(filteredSymptoms);
}

// === MODEL COMPARISON ===
async function loadModelComparison() {
  try {
    let modelData = [
      { model: 'Logistic Regression', cv_f1: '1.0000 ± 0.0000', test_accuracy: '1.0000', test_precision: '1.0000', test_recall: '1.0000', test_f1: '1.0000' },
      { model: 'SVM', cv_f1: '1.0000 ± 0.0000', test_accuracy: '1.0000', test_precision: '1.0000', test_recall: '1.0000', test_f1: '1.0000' },
      { model: 'Random Forest', cv_f1: '0.9940 ± 0.0120', test_accuracy: '1.0000', test_precision: '1.0000', test_recall: '1.0000', test_f1: '1.0000' },
      { model: 'XGBoost', cv_f1: '0.9081 ± 0.0479', test_accuracy: '0.9783', test_precision: '0.9878', test_recall: '0.9878', test_f1: '0.9837' },
      { model: 'MLP', cv_f1: '1.0000 ± 0.0000', test_accuracy: '1.0000', test_precision: '1.0000', test_recall: '1.0000', test_f1: '1.0000' },
      { model: 'Calibrated Ensemble (RF+XGB)', cv_f1: '1.0000 ± 0.0000', test_accuracy: '1.0000', test_precision: '1.0000', test_recall: '1.0000', test_f1: '1.0000' }
    ];

    try {
      const response = await fetch(COMPARISON_API_URL);
      if (response.ok) {
        const data = await response.json();
        if (data.models && data.models.length > 0) {
          modelData = data.models;
        }
      }
    } catch (apiErr) {
      console.log('Using cached research metrics:', apiErr);
    }
    
    const tableBody = document.querySelector('#modelComparisonTable tbody');
    if (tableBody) {
      tableBody.innerHTML = '';
      modelData.forEach((model, index) => {
        const row = document.createElement('tr');
        if (index === modelData.length - 1) {
          row.classList.add('selected-model');
        }
        row.innerHTML = `
          <td><strong>${model.model}</strong>${index === modelData.length - 1 ? ' <span class="badge">Selected</span>' : ''}</td>
          <td>${model.cv_f1}</td>
          <td>${model.test_accuracy}</td>
          <td>${model.test_precision || '1.0000'}</td>
          <td>${model.test_recall || '1.0000'}</td>
          <td>${model.test_f1}</td>
        `;
        tableBody.appendChild(row);
      });
    }
  } catch (error) {
    console.error('Error loading model comparison:', error);
  }
}

// === FORM SUBMISSION ===
async function predictDisease(event) {
  event.preventDefault();
  
  if (selectedSymptoms.size === 0) {
    showNotification('Please select at least one symptom', 'warning');
    shakeElement(document.getElementById('symptomsGrid'));
    return;
  }
  
  setLoading(true);
  
  try {
    const severityRadio = document.querySelector('input[name="severity"]:checked');
    const severityVal = severityRadio ? severityRadio.value : 'moderate';

    const payload = { 
      symptoms: Array.from(selectedSymptoms),
      duration_days: selectedDurationDays,
      duration_category: selectedDurationCategory,
      severity: severityVal,
      top_k: 5,
      abstention_threshold: 0.1
    };
    
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    currentResults = result.predictions || [];
    lastPredictions = currentResults;
    lastClinicalAdvisory = result.clinical_advisory;
    lastShapExplanation = result.shap_explanation;
    
    // Always render clinical advisory first so the patient sees doctor consultation & test recommendations
    renderClinicalAdvisory(result.clinical_advisory, currentResults);

    showResults(currentResults, result.shap_explanation, result.should_abstain, result.message, result.clinical_advisory);
    
    // Notifications
    if (result.clinical_advisory && result.clinical_advisory.should_see_doctor) {
      showNotification('Doctor consultation strongly advised for your symptoms', 'warning');
    } else if (result.should_abstain) {
      showNotification('Additional symptoms needed for confident prediction', 'warning');
    } else {
      showNotification('Analysis completed successfully!', 'success');
    }
    
  } catch (error) {
    console.error('Prediction error:', error);
    showNotification('Server error. Please make sure the backend is running.', 'error');
  } finally {
    setLoading(false);
  }
}

// === CLINICAL ADVISORY & DOCTOR CONSULTATION RENDERER ===
function renderClinicalAdvisory(advisory, predictions) {
  const container = document.getElementById('clinicalAdvisoryContainer');
  if (!container) return;

  if (!advisory) {
    container.innerHTML = '';
    return;
  }

  const isUrgent = advisory.should_see_doctor;
  const urgencyClass = advisory.urgency_color === 'critical' ? 'advisory-critical' :
                       advisory.urgency_color === 'danger' ? 'advisory-danger' :
                       advisory.urgency_color === 'warning' ? 'advisory-warning' : 'advisory-success';

  // Build Diagnostic Tests HTML
  let testsHTML = '';
  if (advisory.recommended_tests && advisory.recommended_tests.length > 0) {
    testsHTML = `
      <div class="advisory-section tests-section">
        <div class="section-title-wrap">
          <h4><i class="fas fa-vial"></i> Recommended Medical Diagnostic Tests</h4>
          <span class="tests-subtitle">Clinical laboratory investigations to request from your physician:</span>
        </div>
        <div class="tests-grid">
          ${advisory.recommended_tests.map(test => {
            const isCrit = test.importance.toLowerCase().includes('critical') || test.importance.toLowerCase().includes('mandatory');
            return `
              <div class="test-card ${isCrit ? 'test-critical' : ''}">
                <div class="test-header">
                  <div class="test-name-wrap">
                    <i class="fas fa-microscope test-icon"></i>
                    <span class="test-name">${test.test_name}</span>
                  </div>
                  <span class="test-badge ${isCrit ? 'badge-critical' : 'badge-priority'}">
                    ${test.importance}
                  </span>
                </div>
                <p class="test-reason"><i class="fas fa-check-circle"></i> ${test.reason}</p>
                <div class="test-meta">
                  <span><i class="fas fa-clock"></i> Expected Turnaround: <strong>${test.turnaround}</strong></span>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </div>
    `;
  }

  // Build Differential Risks HTML
  let differentialHTML = '';
  if (advisory.differential_risks && advisory.differential_risks.length > 0) {
    differentialHTML = `
      <div class="advisory-section differential-section">
        <h4><i class="fas fa-stethoscope"></i> Key Conditions to Rule Out With a Doctor (Differential Diagnosis)</h4>
        <div class="risks-grid">
          ${advisory.differential_risks.map(risk => `
            <div class="risk-card">
              <div class="risk-title-row">
                <span class="risk-disease"><i class="fas fa-virus"></i> ${risk.disease}</span>
                <span class="risk-severity ${risk.severity.toLowerCase().replace(/\s+/g, '-')}">${risk.severity} Risk</span>
              </div>
              <p class="risk-reason">${risk.reason}</p>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  // Build Medication Warning HTML
  let medWarningHTML = '';
  if (advisory.medication_warnings && advisory.medication_warnings.length > 0) {
    medWarningHTML = `
      <div class="medication-warning-box">
        <i class="fas fa-exclamation-triangle warning-icon"></i>
        <div class="warning-text">
          ${advisory.medication_warnings.map(w => `<p>${w}</p>`).join('')}
        </div>
      </div>
    `;
  }

  // Red Flags Alert HTML
  let redFlagsHTML = '';
  if (advisory.red_flags && advisory.red_flags.length > 0) {
    redFlagsHTML = `
      <div class="red-flags-box">
        <h4><i class="fas fa-ambulance"></i> Emergency Red Flags Identified:</h4>
        <ul>
          ${advisory.red_flags.map(f => `<li><strong>${f}</strong> — Requires immediate Emergency Department evaluation</li>`).join('')}
        </ul>
      </div>
    `;
  }

  // Correlation Note HTML
  let correlationHTML = '';
  if (advisory.correlation_note) {
    correlationHTML = `
      <div class="correlation-note-box">
        <i class="fas fa-link"></i>
        <span>${advisory.correlation_note}</span>
      </div>
    `;
  }

  // Action checklist
  let actionItemsHTML = '';
  if (advisory.action_items && advisory.action_items.length > 0) {
    actionItemsHTML = `
      <div class="action-items-box">
        <h4><i class="fas fa-clipboard-check"></i> Recommended Next Steps & Clinical Precautions:</h4>
        <ul class="action-list">
          ${advisory.action_items.map(item => `<li><i class="fas fa-arrow-right"></i> ${item}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  container.innerHTML = `
    <div class="advisory-card ${urgencyClass} fade-in-up">
      <div class="advisory-top-bar">
        <div class="advisory-badge-group">
          <span class="urgency-pill ${advisory.urgency_color}">
            <span class="pulsing-dot"></span>
            ${advisory.urgency_badge}
          </span>
          <span class="duration-pill">
            <i class="fas fa-calendar-day"></i> Duration: <strong>${advisory.duration_days} Days</strong> (${advisory.duration_category})
          </span>
        </div>
        <div class="advisory-timestamp">
          <i class="fas fa-shield-alt"></i> Clinical Decision Support
        </div>
      </div>

      <div class="advisory-advice">
        <div class="advice-content">
          <p>${advisory.doctor_advice.replace(/\n\n/g, '<br><br>')}</p>
        </div>
      </div>

      ${correlationHTML}
      ${redFlagsHTML}
      ${differentialHTML}
      ${testsHTML}
      ${medWarningHTML}
      ${actionItemsHTML}

      <div class="advisory-footer-actions">
        <div class="specialist-info">
          <span><i class="fas fa-user-md"></i> Recommended Specialist:</span>
          <strong>${advisory.specialists.join(', ')}</strong>
        </div>
        <div class="cta-actions">
          <button type="button" class="btn-doctor-sheet" onclick="openDoctorSummaryModal()">
            <i class="fas fa-file-medical"></i> Doctor Consultation Sheet
          </button>
          <button type="button" class="btn-find-clinics" onclick="findNearbyClinics()">
            <i class="fas fa-hospital"></i> Find Nearby Care
          </button>
        </div>
      </div>
    </div>
  `;
}

// === RESULTS DISPLAY ===
function showResults(predictions, shapExplanation = null, shouldAbstain = false, message = null, advisory = null) {
  const grid = document.getElementById('resultsGrid');
  const section = document.getElementById('resultsSection');
  const headerBlock = document.getElementById('predictionsHeaderBlock');
  
  if (!grid || !section) return;
  
  grid.innerHTML = '';
  section.classList.remove('hidden');
  
  // Also render advisory if not rendered yet
  if (advisory || lastClinicalAdvisory) {
    renderClinicalAdvisory(advisory || lastClinicalAdvisory, predictions);
  }
  
  // Handle abstention case
  if (shouldAbstain || message) {
    if (headerBlock) headerBlock.style.display = 'none';
    grid.innerHTML = `
      <div class="result-card abstention-card fade-in-up">
        <div class="abstention-icon"><i class="fas fa-info-circle"></i></div>
        <h3 class="result-title">Additional Symptoms Needed for Exact Classification</h3>
        <div class="result-description">
          ${message || 'The AI model requires additional specific symptoms to reach a high-confidence disease classification. Please review the Clinical Advisory above and consider answering follow-up questions or consulting a medical practitioner.'}
        </div>
        <div class="abstention-actions">
          <button class="primary-button" onclick="triggerFollowUp()">
            <i class="fas fa-question-circle"></i>
            Answer Sequential Follow-up Questions
          </button>
          <button class="secondary-button" onclick="openDoctorSummaryModal()">
            <i class="fas fa-file-medical"></i>
            View Doctor Consultation Sheet
          </button>
        </div>
      </div>
    `;
    section.scrollIntoView({ behavior: 'smooth' });
    return;
  }
  
  if (headerBlock) headerBlock.style.display = 'block';

  // Display top-K predictions with research features
  predictions.forEach((prediction, index) => {
    const confidence = (prediction.confidence * 100).toFixed(1);
    const card = document.createElement('div');
    card.className = 'result-card fade-in-up';
    card.style.animationDelay = `${index * 0.1}s`;
    
    let shapHTML = '';
    if (shapExplanation && index === 0) {
      shapHTML = `
        <div class="shap-explanation">
          <h4><i class="fas fa-lightbulb"></i> XGBoost Component-Level SHAP Explanation</h4>
          <div class="shap-features">
            ${shapExplanation.slice(0, 5).map(feature => `
              <div class="shap-feature ${feature.impact}">
                <span class="feature-name">${formatSymptomName(feature.symptom)}</span>
                <span class="feature-value">${feature.impact === 'positive' ? '+' : ''}${feature.shap_value.toFixed(3)}</span>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    }
    
    card.innerHTML = `
      <div class="result-header">
        <div class="result-rank">#${index + 1}</div>
        <div class="result-confidence">${confidence}%</div>
      </div>
      <h3 class="result-title">${prediction.disease}</h3>
      <div class="result-confidence-bar">
        <div class="confidence-fill" style="width: ${confidence}%"></div>
      </div>
      ${shapHTML}
    `;
    
    grid.appendChild(card);
  });
  
  section.scrollIntoView({ behavior: 'smooth' });
}

// === DOCTOR CONSULTATION SUMMARY MODAL & PRINTING ===
function openDoctorSummaryModal() {
  const modal = document.getElementById('doctorModal');
  const body = document.getElementById('doctorModalBody');
  if (!modal || !body) return;

  const symptomsList = Array.from(selectedSymptoms).map(formatSymptomName);
  const duration = selectedDurationDays;
  const severityRadio = document.querySelector('input[name="severity"]:checked');
  const severity = severityRadio ? severityRadio.value : 'moderate';
  const advisory = lastClinicalAdvisory || {};
  const predictions = lastPredictions || [];

  const dateStr = new Date().toLocaleDateString('en-US', {
    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit'
  });

  body.innerHTML = `
    <div class="doctor-sheet-content">
      <div class="sheet-header-print">
        <h2>HealthGuard AI — Clinical Patient Summary</h2>
        <p>Medical Decision Support & Symptom Triage Report</p>
      </div>

      <div class="sheet-patient-bar">
        <div><strong>Encounter Date:</strong> ${dateStr}</div>
        <div><strong>Symptom Duration:</strong> ${duration} Days (${selectedDurationCategory})</div>
        <div><strong>Reported Severity:</strong> <span style="text-transform: capitalize;">${severity}</span></div>
      </div>

      <div class="sheet-section">
        <h4>1. Presenting Symptoms Reported (${symptomsList.length})</h4>
        <div class="sheet-tags">
          ${symptomsList.map(s => `<span class="sheet-tag">${s}</span>`).join('')}
        </div>
      </div>

      <div class="sheet-section">
        <h4>2. Clinical Risk & Urgency Assessment</h4>
        <p class="sheet-urgency"><strong>Triage Level:</strong> ${advisory.urgency_badge || 'Clinical Evaluation Required'}</p>
        <p class="sheet-advice">${advisory.doctor_advice || 'Patient presents with persistent symptoms requiring clinical physician evaluation.'}</p>
      </div>

      <div class="sheet-section">
        <h4>3. Machine Learning Differential Predictions (Calibrated RF + XGBoost)</h4>
        <table class="sheet-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Differential Disease</th>
              <th>Model Probability</th>
              <th>Clinical Correlation</th>
            </tr>
          </thead>
          <tbody>
            ${predictions.slice(0, 4).map((p, idx) => `
              <tr>
                <td>#${idx + 1}</td>
                <td><strong>${p.disease}</strong></td>
                <td>${(p.confidence * 100).toFixed(1)}%</td>
                <td>${idx === 0 && duration >= 3 && p.disease.toLowerCase().includes('dengue') ? 'Prolonged fever: Rule out Dengue thrombocytopenia' : 'Differential candidate'}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>

      <div class="sheet-section">
        <h4>4. Recommended Diagnostic Lab Investigations for Physician Review</h4>
        <ul class="sheet-tests-list">
          ${(advisory.recommended_tests || []).map(t => `
            <li>
              <strong>${t.test_name}</strong> <em>(${t.importance})</em>: ${t.reason}
            </li>
          `).join('')}
        </ul>
      </div>

      <div class="sheet-section doctor-notes-box">
        <h4>5. Physician Physical Examination & Prescription Notes</h4>
        <div class="notes-lines">
          <div class="note-line"><strong>Vitals:</strong> BP: ___________ | Pulse: ___________ bpm | Temp: ___________ °F | SpO2: ___________ %</div>
          <div class="note-line"><strong>Systemic Findings:</strong> ____________________________________________________________________</div>
          <div class="note-line">_________________________________________________________________________________________</div>
          <div class="note-line"><strong>Ordered Lab Tests:</strong> ___________________________________________________________________</div>
          <div class="note-line"><strong>Doctor Signature / Stamp:</strong> ____________________________ Date: _________________________</div>
        </div>
      </div>

      <div class="sheet-disclaimer">
        <strong>Medical Disclaimer:</strong> HealthGuard AI is an academic research prototype for clinical decision support. It is not an automated medical diagnostic tool. Final diagnostic, laboratory, and treatment choices remain solely with the examining licensed healthcare professional.
      </div>
    </div>
  `;

  modal.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
}

function closeDoctorModal() {
  const modal = document.getElementById('doctorModal');
  if (modal) {
    modal.classList.add('hidden');
    document.body.style.overflow = '';
  }
}

function closeDoctorModalOnBackdrop(event) {
  if (event.target.id === 'doctorModal') {
    closeDoctorModal();
  }
}

function findNearbyClinics() {
  window.open('https://www.google.com/maps/search/hospital+clinic+near+me/', '_blank');
}

// === FOLLOW-UP QUESTIONS ===
async function triggerFollowUp() {
  const sessionId = 'session_' + Date.now();
  
  try {
    const payload = {
      session_id: sessionId,
      symptoms: Array.from(selectedSymptoms),
      max_questions: 3,
      abstention_threshold: 0.1
    };
    
    const response = await fetch(FOLLOWUP_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    
    if (result.status === 'question') {
      // Display follow-up question
      showFollowUpQuestion(result, sessionId);
    } else if (result.status === 'final') {
      // Show final prediction
      currentResults = [{
        disease: result.prediction,
        confidence: result.confidence
      }];
      showResults(currentResults, null, false);
      showNotification('Follow-up analysis completed!', 'success');
    }
  } catch (error) {
    console.error('Follow-up error:', error);
    showNotification('Error getting follow-up questions', 'error');
  }
}

function showFollowUpQuestion(result, sessionId) {
  const grid = document.getElementById('resultsGrid');
  if (!grid) return;
  
  grid.innerHTML = `
    <div class="result-card followup-card fade-in-up">
      <div class="followup-icon">
        <i class="fas fa-question-circle"></i>
      </div>
      <h3 class="result-title">Follow-up Question</h3>
      <div class="followup-question">
        ${result.next_question}
      </div>
      <div class="followup-buttons">
        <button class="primary-button" onclick="answerFollowUp('${sessionId}', true)">
          <i class="fas fa-check"></i>
          Yes
        </button>
        <button class="secondary-button" onclick="answerFollowUp('${sessionId}', false)">
          <i class="fas fa-times"></i>
          No
        </button>
      </div>
      <div class="followup-info">
        Questions remaining: ${result.questions_remaining}
      </div>
    </div>
  `;
}

async function answerFollowUp(sessionId, answer) {
  const newSymptom = answer ? document.querySelector('.followup-question').textContent.replace('Do you have ', '').replace('?', '').trim() : null;
  
  if (newSymptom && answer) {
    selectedSymptoms.add(newSymptom);
    updateSelectedSymptomsDisplay();
  }
  
  // Trigger next follow-up
  await triggerFollowUp();
}

function createResultCard(prediction, rank, shapExplanation = null) {
  const card = document.createElement('div');
  card.className = 'result-card';
  card.style.opacity = '0';
  
  const confidence = (prediction.confidence * 100).toFixed(1);
  const confidenceText = getConfidenceText(prediction.confidence);
  
  let shapHTML = '';
  if (shapExplanation && rank === 1) {
    shapHTML = `
      <div class="shap-explanation">
        <h4>Key Factors:</h4>
        <ul>
          ${shapExplanation.slice(0, 3).map(feat => `
            <li>
              <span class="shap-symptom">${feat.symptom.replace(/_/g, ' ')}</span>
              <span class="shap-value ${feat.impact}">${feat.impact === 'positive' ? '+' : ''}${feat.shap_value.toFixed(3)}</span>
            </li>
          `).join('')}
        </ul>
      </div>
    `;
  }
  
  card.innerHTML = `
    <div class="result-rank">${rank}</div>
    <h3 class="result-title">${prediction.disease}</h3>
    <div class="result-confidence">${confidence}%</div>
    <div class="progress-bar">
      <div class="progress-fill" style="width: 0%"></div>
    </div>
    <div class="result-description">${confidenceText}</div>
    ${shapHTML}
  `;
  
  return card;
}

function getConfidenceText(conf) {
  if (conf > 0.8) return 'Very high confidence - Strong indicator';
  if (conf > 0.6) return 'High confidence - Likely match';
  if (conf > 0.4) return 'Moderate confidence - Possible match';
  return 'Low confidence - Consult healthcare provider';
}

// === LOADING STATES ===
function setLoading(loading) {
  const btn = document.getElementById('predictBtn');
  const content = btn.querySelector('.button-content');
  const loader = btn.querySelector('.button-loader');
  
  if (!btn || !content || !loader) return;
  
  btn.disabled = loading;
  
  if (loading) {
    content.style.opacity = '0';
    loader.style.opacity = '1';
    btn.classList.add('loading');
  } else {
    content.style.opacity = '1';
    loader.style.opacity = '0';
    btn.classList.remove('loading');
  }
}

// === UTILITY FUNCTIONS ===
function parseSymptoms(input) {
  return input.split(',')
    .map(s => s.trim().replace(/\b\w/g, l => l.toUpperCase()))
    .filter(Boolean);
}

function resetForm() {
  const form = document.querySelector('.prediction-form');
  if (form) {
    form.reset();
    
    // Clear validation states
    form.querySelectorAll('.input-wrapper').forEach(wrapper => {
      wrapper.classList.remove('focused', 'valid', 'invalid');
    });
    
    // Hide results
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
      resultsSection.classList.add('hidden');
    }
    
    // Clear current results
    currentResults = [];
    
    // Scroll to form
    scrollToForm();
    
    showNotification('Form reset successfully', 'success');
  }
}

function exportResults() {
  if (!currentResults.length) {
    showNotification('No results to export', 'warning');
    return;
  }
  
  const exportData = {
    timestamp: new Date().toISOString(),
    symptoms: document.getElementById('symptoms').value,
    age: document.getElementById('age').value,
    gender: document.getElementById('gender').value,
    predictions: currentResults
  };
  
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `healthguard-analysis-${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  
  showNotification('Results exported successfully', 'success');
}

// === SCROLL ANIMATIONS ===
function initScrollAnimations() {
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };
  
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('animate-in');
      }
    });
  }, observerOptions);
  
  // Observe elements with animation classes
  document.querySelectorAll('.fade-in-up').forEach(el => {
    observer.observe(el);
  });
}

// === NOTIFICATIONS ===
function showNotification(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;
  
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <div class="toast-content">
      <i class="fas ${getToastIcon(type)}"></i>
      <span>${message}</span>
    </div>
  `;
  
  container.appendChild(toast);
  
  // Auto remove
  setTimeout(() => {
    toast.style.animation = 'slide-out 0.3s ease-out forwards';
    setTimeout(() => {
      if (container.contains(toast)) {
        container.removeChild(toast);
      }
    }, 300);
  }, 4000);
}

function getToastIcon(type) {
  switch (type) {
    case 'success': return 'fa-check-circle';
    case 'error': return 'fa-exclamation-circle';
    case 'warning': return 'fa-exclamation-triangle';
    default: return 'fa-info-circle';
  }
}

// === ANIMATION HELPERS ===
function shakeElement(element) {
  if (!element) return;
  
  element.classList.add('shake');
  setTimeout(() => {
    element.classList.remove('shake');
  }, 500);
}

// Add CSS animations dynamically
const style = document.createElement('style');
style.textContent = `
  .pulse-once {
    animation: pulse-once 1s ease-in-out;
  }
  
  @keyframes pulse-once {
    0%, 100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
    50% { box-shadow: 0 0 0 8px rgba(59, 130, 246, 0); }
  }
  
  .highlight {
    animation: highlight 0.5s ease-in-out;
  }
  
  @keyframes highlight {
    0%, 100% { background-color: transparent; }
    50% { background-color: rgba(59, 130, 246, 0.1); }
  }
  
  .shake {
    animation: shake 0.5s ease-in-out;
  }
  
  @keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-10px); }
    75% { transform: translateX(10px); }
  }
  
  .focused .input-icon {
    color: var(--primary-blue);
    transform: translateY(-50%) scale(1.1);
  }
  
  .valid .input-icon {
    color: var(--success-green);
  }
  
  .invalid .input-icon {
    color: var(--error-red);
  }
  
  .loading {
    pointer-events: none;
  }
  
  .animate-in {
    opacity: 1 !important;
    transform: translateY(0) !important;
  }
  
  @keyframes slide-out {
    to {
      transform: translateX(100%);
      opacity: 0;
    }
  }
  
  .mobile-menu-toggle.active {
    background: var(--primary-gradient);
    color: white;
  }
  
  .nav-menu.mobile-active {
    display: flex;
    position: fixed;
    top: 80px;
    left: 0;
    right: 0;
    background: var(--bg-glass);
    backdrop-filter: blur(20px);
    flex-direction: column;
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-lg);
  }
  
  @media (max-width: 768px) {
    .nav-menu {
      display: none;
    }
    
    .nav-menu.mobile-active {
      display: flex;
    }
  }
`;
document.head.appendChild(style);

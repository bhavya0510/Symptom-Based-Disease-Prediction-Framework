# Results Summary

The former results were produced with row-level splitting of 4,962 records and are invalid because duplicate symptom patterns crossed evaluation boundaries. Do not use the old 1.000 metrics.

The corrected Stage 1 dataset has 305 unique symptom-pattern/diagnosis rows after removing 4,657 duplicates. It uses a 70/15/15 stratified split with zero exact symptom-pattern overlap between train, validation, and test. Class support is only 5–10 unique patterns per disease.

Run `python train_pipeline.py` from this directory to regenerate the baseline, ensemble, calibration, top-K, abstention, sequential-questioning, SHAP, production artifacts, and this report. The generated CSV artifacts are the authoritative source for final numerical claims.

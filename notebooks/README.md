# Notebook workflows

Run the notebooks in each group in the listed order. Configure paths in their setup cells.

| Notebook | Purpose |
|---|---|
| [00_cohort/01_build_cohort.ipynb](00_cohort/01_build_cohort.ipynb) | Build the study cohort |
| [01_diagnosis/01_classify_notes.ipynb](01_diagnosis/01_classify_notes.ipynb) | Classify SMM diagnosis in notes |
| [01_diagnosis/02_evaluate_notes.ipynb](01_diagnosis/02_evaluate_notes.ipynb) | Evaluate note-level diagnosis |
| [01_diagnosis/03_evaluate_patients.ipynb](01_diagnosis/03_evaluate_patients.ipynb) | Evaluate patient-level diagnosis |
| [02_history/01_prepare_first_positive_notes.ipynb](02_history/01_prepare_first_positive_notes.ipynb) | Prepare first positive notes |
| [02_history/02_classify_history.ipynb](02_history/02_classify_history.ipynb) | Classify historical and new diagnoses |
| [02_history/03_evaluate_history.ipynb](02_history/03_evaluate_history.ipynb) | Evaluate historical and new diagnosis labels |
| [02_history/04_prepare_historical_notes.ipynb](02_history/04_prepare_historical_notes.ipynb) | Prepare historical notes for date extraction |
| [03_dates/01_extract_dates.ipynb](03_dates/01_extract_dates.ipynb) | Extract diagnosis dates and review flags |
| [03_dates/02_evaluate_training_fewshot.ipynb](03_dates/02_evaluate_training_fewshot.ipynb) | Evaluate training dates: few-shot results |
| [03_dates/03_evaluate_training_zeroshot.ipynb](03_dates/03_evaluate_training_zeroshot.ipynb) | Evaluate training dates: zero-shot results |
| [03_dates/04_evaluate_testing_fewshot.ipynb](03_dates/04_evaluate_testing_fewshot.ipynb) | Evaluate testing dates: few-shot results |
| [03_dates/05_evaluate_testing_zeroshot.ipynb](03_dates/05_evaluate_testing_zeroshot.ipynb) | Evaluate testing dates: zero-shot results |
| [03_dates/06_compare_human_review.ipynb](03_dates/06_compare_human_review.ipynb) | Compare dates with and without human review |
| [04_pipeline/01_select_positive_patients.ipynb](04_pipeline/01_select_positive_patients.ipynb) | Select positive patients and first positive notes |
| [04_pipeline/02_classify_history.ipynb](04_pipeline/02_classify_history.ipynb) | Classify history in first positive notes |
| [04_pipeline/03_extract_historical_dates.ipynb](04_pipeline/03_extract_historical_dates.ipynb) | Extract dates from historical notes |
| [04_pipeline/04_build_reference_standard.ipynb](04_pipeline/04_build_reference_standard.ipynb) | Build the patient-level reference standard |
| [04_pipeline/05_evaluate_training.ipynb](04_pipeline/05_evaluate_training.ipynb) | Evaluate the training pipeline |
| [04_pipeline/06_evaluate_testing.ipynb](04_pipeline/06_evaluate_testing.ipynb) | Evaluate the testing pipeline |
| [05_external/01_evaluate_diagnoses.ipynb](05_external/01_evaluate_diagnoses.ipynb) | Evaluate external diagnosis predictions |
| [05_external/02_evaluate_dates.ipynb](05_external/02_evaluate_dates.ipynb) | Evaluate external diagnosis dates |

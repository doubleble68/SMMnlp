# Configuration

Set `SMMNLP_MODEL_PATH` to your local model directory and `SMMNLP_DATA_ROOT` to
your study data directory when using the notebook workflows. Defaults are the
`models/Ministral3-bnb-4bit` and `data` folders inside this project.

`excluded_patient_ids.json` is an empty example. Populate it with the patient
IDs excluded by your cohort specification before building a study cohort.
Use `SMMNLP_EXCLUSIONS_FILE` to select a different JSON list.

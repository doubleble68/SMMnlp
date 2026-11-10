# Processing rules

## Diagnosis and history labels

The diagnosis parser returns `Yes` when a case-insensitive `yes` substring
appears anywhere in the generated response. It otherwise returns `No`.
This is substring matching: `yesterday` also matches.

The history parser used for extraction returns `Hx` when the response contains
`hx`; it otherwise returns `New`. The history evaluation parser returns `new`
when the response contains `new`; it otherwise returns `hx`. These two rules
have different fallbacks and are named separately in `smmnlp/parsing.py`.

## Date response fields

Date responses use two colon-separated fields:

```text
SMM_diagnosis_date: 08/2009
human_review_needed: no
```

Keys are case-sensitive. Duplicate keys use the last value. Extra keys are
retained by the parser. The review values `yes`, `true`, `y`, and `1` become
`True`; other supplied values become `False`. An empty response produces a
missing date and a `True` review flag.

For a nonempty response, missing keys remain missing. If every response omits
a required key, expanding the response table raises `KeyError`. Inspect raw
outputs when a model does not follow the expected format.

## Partial dates

| Input | Expanded date |
|---|---|
| `2020` | `07/01/2020` |
| `08/2009` | `08/01/2009` |
| `08/09` | `08/01/2009` |
| `02/03/20` | `02/03/2020` |
| `unknown` or missing | Missing |

Two-digit years `00` through `26` map to 2000–2026; `27` through `99` map to
1927–1999. The cutoff is fixed at 26. Formatting does not validate calendar
components; the subsequent `pd.to_datetime` call validates the expanded date.

## Patient selection and review

A patient is diagnosis-positive if any of their notes is positive. First-positive
note selection uses `%d%b%Y` dates and the first minimum-date row per patient.
Patient splits keep each patient's notes together. The cohort workflows use
seed 76 for per-year sampling and seeds 56 and 66695 for patient splits.

Date extraction results stay paired with their input rows. If a review flag
equals `True`, date evaluation uses the supplied human date. This step selects
an existing annotation; it does not generate a human review automatically.

## Evaluation

Metrics include accuracy, precision, recall, F1, confusion matrices, ROC AUC,
and average precision. The variable named `au_prc` holds average precision.
Curve calculations use hard predicted labels, which do not describe probability
calibration or a full continuous ranking.

External date presence uses all rows in the left merge of the reference cohort
with predictions. Missing predictions count as absent dates. Date-difference
plots use rows where both reference and predicted dates are available.

Some notebook inputs assume matching row order for positional concatenation.
Their merge keys and input columns are visible in the notebook cells. Ensure
that those assumptions match your study data before interpreting metrics.

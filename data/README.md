# Input data

The inference CLI accepts `.json`, `.xlsx`, and `.csv` files. The full pipeline
uses these default columns:

| Column | Meaning |
|---|---|
| `PatientSSN` | Patient identifier used for grouping |
| `NoteDate` | Note date, such as `01Mar2020` |
| `NoteText` | Clinical note text |

`example_notes.json` contains fabricated records demonstrating this schema.
The demo uses predefined responses, so its outputs are examples of application
behavior rather than model-performance measurements.

External note classification can use `full_note` through `--text-column full_note`.
For JSON record arrays, `--external-json` selects the JSON reader used for that
format. Metadata columns are retained in prediction outputs.

Notebook workflows name their expected reference workbooks, prediction tables,
and human-review tables in their input cells. Place those files under this
directory using the displayed relative paths, or set `SMMNLP_DATA_ROOT` to your
study's data directory.

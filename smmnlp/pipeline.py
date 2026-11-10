"""Compose diagnosis classification, history classification, and date extraction."""

from dataclasses import dataclass

import pandas as pd

from .inference import predict_notes
from .modeling import TextGenerator
from .parsing import parse_diagnosis_label, parse_history_for_extraction
from .preprocessing import (
    add_parsed_labels,
    earliest_positive_notes,
    historical_notes,
)
from .prompts import Task


@dataclass
class PipelineResult:
    note_predictions: pd.DataFrame
    first_positive_notes: pd.DataFrame
    history_predictions: pd.DataFrame
    date_extractions: pd.DataFrame


def run_pipeline(
    notes: pd.DataFrame,
    generator: TextGenerator,
    *,
    text_column: str = "NoteText",
    patient_column: str = "PatientSSN",
    date_column: str = "NoteDate",
    date_format: str = "%d%b%Y",
) -> PipelineResult:
    """Classify notes, select first Yes notes, classify history, then extract dates.

    Human review and the later reference-standard evaluation remain separate:
    the notebook analyses consume human-reviewed spreadsheets at that step.
    """
    predictions = predict_notes(notes, generator, Task.DIAGNOSIS, text_column=text_column)
    predictions = add_parsed_labels(predictions, ["SMMDx_LLM_Binary"], parse_diagnosis_label)
    first_positive = earliest_positive_notes(
        predictions,
        patient_column=patient_column,
        date_column=date_column,
        date_format=date_format,
    )
    history = predict_notes(first_positive, generator, Task.HISTORY, text_column=text_column)
    history = add_parsed_labels(history, ["HxNew_LLM_Binary"], parse_history_for_extraction)

    # Use a consistent row order when selecting the historical-note subset.
    date_candidates = historical_notes(history.reset_index(drop=True))
    if date_candidates.empty:
        dates = date_candidates.copy()
        for column in ("rawoutput", "DxDate_LLM", "UncertainFlag_LLM"):
            dates[column] = pd.Series(index=dates.index, dtype=object)
    else:
        dates = predict_notes(
            date_candidates,
            generator,
            Task.DATE,
            text_column=text_column,
        )
    return PipelineResult(predictions, first_positive, history, dates)

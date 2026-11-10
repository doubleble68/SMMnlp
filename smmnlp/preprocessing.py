"""Dataframe transformations shared by inference and notebook analyses."""

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .parsing import parse_date_extraction

def add_parsed_labels(
    frame: pd.DataFrame, columns: Sequence[str], parser: Callable[[str], str]
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[f"post_{column}"] = result[column].apply(parser)
    return result


def add_date_extractions(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach each extracted date and review flag to its input note."""
    result = frame.copy()
    if result.empty:
        result["DxDate_LLM"] = pd.Series(index=result.index, dtype=object)
        result["UncertainFlag_LLM"] = pd.Series(index=result.index, dtype=object)
        return result
    parsed = pd.json_normalize(result["rawoutput"].apply(parse_date_extraction))
    # Normalization creates a new index; restore the note index before assignment.
    parsed.index = result.index
    result["DxDate_LLM"] = parsed["SMM_diagnosis_date"]
    result["UncertainFlag_LLM"] = parsed["human_review_needed"]
    return result


def earliest_positive_notes(
    notes: pd.DataFrame,
    *,
    label_column: str = "post_SMMDx_LLM_Binary",
    patient_column: str = "PatientSSN",
    date_column: str = "NoteDate",
    date_format: str = "%d%b%Y",
) -> pd.DataFrame:
    """Return the first minimum-date Yes row per patient, retaining its index."""
    positive_notes = notes.loc[notes[label_column] == "Yes"].copy()
    positive_notes["NoteDate2"] = pd.to_datetime(positive_notes[date_column], format=date_format)
    earliest_indices = positive_notes.groupby(patient_column)["NoteDate2"].idxmin()
    return positive_notes.loc[earliest_indices].copy()


def historical_notes(
    notes: pd.DataFrame, *, label_column: str = "post_HxNew_LLM_Binary"
) -> pd.DataFrame:
    return notes.loc[notes[label_column] == "Hx"].copy()


def patient_level_labels(
    notes: pd.DataFrame,
    *,
    patient_column: str = "PatientSSN",
    reference_column: str = "post_SMM Dx (final)",
    prediction_column: str = "post_SMMDx_LLM_Binary",
) -> pd.DataFrame:
    """A patient is positive if at least one of their notes is positive."""
    flags = notes[[patient_column]].copy()
    flags["reference_label"] = notes[reference_column].eq("Yes")
    flags["predicted_label"] = notes[prediction_column].eq("Yes")
    result = flags.groupby(patient_column)[["reference_label", "predicted_label"]].any()
    return result.replace({True: "Yes", False: "No"}).reset_index()


def select_reviewed_dates(
    dates: pd.DataFrame,
    *,
    review_column: str = "UncertainFlag_LLM",
    human_column: str = "DxDate_Human",
    model_column: str = "DxDate_LLM",
) -> pd.Series:
    """Use existing human dates where the model review flag equals True."""
    values = np.where(
        dates[review_column].eq(True),
        dates[human_column],
        dates[model_column],
    )
    return pd.Series(values, index=dates.index, name="DxDate_Reviewed")


def sample_patients_by_year(
    notes: pd.DataFrame,
    sizes: Mapping[str, int],
    *,
    excluded_patient_ids: Sequence[Any] = (),
    random_state: int = 76,
) -> pd.DataFrame:
    """Sample patients by index year while preserving note order."""
    candidates = notes.loc[~notes["PatientSSN"].isin(excluded_patient_ids)].copy()
    candidates["index_year"] = candidates["index_date"].str[5:]
    candidates["cohort_version"] = "S2"
    patient_years = candidates[["PatientSSN", "index_year"]].drop_duplicates()
    selected = patient_years.groupby("index_year", group_keys=False).apply(
        lambda group: group.sample(n=sizes[group.name], random_state=random_state)
    )
    return candidates.loc[candidates["PatientSSN"].isin(selected["PatientSSN"])].copy()


def split_by_patient(
    notes: pd.DataFrame,
    *,
    random_state: int,
    training_fraction: float = 0.7,
    patient_column: str = "PatientSSN",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole patients using Python's sampling algorithm."""
    patient_ids = notes[patient_column].unique()
    training_count = int(len(patient_ids) * training_fraction)
    training_ids = random.Random(random_state).sample(list(patient_ids), training_count)
    training_mask = notes[patient_column].isin(training_ids)
    return notes.loc[training_mask].copy(), notes.loc[~training_mask].copy()

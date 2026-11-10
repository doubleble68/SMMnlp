"""Response parsing and partial-date normalization."""

import re
from typing import Any

import pandas as pd


def parse_diagnosis_label(response: str) -> str:
    """Return Yes when 'yes' occurs anywhere, case-insensitively; otherwise No."""
    return "Yes" if re.search(r"yes", response, re.IGNORECASE) else "No"


def parse_history_for_extraction(response: str) -> str:
    """End-to-end rule: any 'hx' gives Hx; otherwise New."""
    return "Hx" if re.search(r"hx", response, re.IGNORECASE) else "New"


def parse_history_for_evaluation(response: str) -> str:
    """Phase-2 evaluation rule: any 'new' gives new; otherwise hx."""
    return "new" if re.search(r"new", response, re.IGNORECASE) else "hx"


def parse_date_extraction(response: Any) -> dict[str, Any]:
    """Parse key/value lines using permissive, case-sensitive keys.

    Missing keys stay missing for nonempty responses. An unrecognized review
    value becomes False. Both behaviors are explicit response interpretation rules.
    """
    if not isinstance(response, str) or not response.strip():
        return {"SMM_diagnosis_date": None, "human_review_needed": True}

    parsed: dict[str, Any] = {}
    for line in response.strip().split("\n"):
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        parsed[key] = (
            value.lower() in {"yes", "true", "y", "1"} if key == "human_review_needed" else value
        )
    return parsed


def expand_partial_date(value: Any, two_digit_cutoff: int = 26) -> Any:
    """Expand YYYY to July 1, MM/YYYY to day 1, and years <=26 to 20xx.

    This formats date components without calendar validation.
    A subsequent pd.to_datetime(..., format='%m/%d/%Y') validates the result.
    """
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()

    def expand_year(year_text: str) -> int:
        year = int(year_text)
        if year < 100:
            return 2000 + year if year <= two_digit_cutoff else 1900 + year
        return year

    if re.fullmatch(r"\d{4}", text):
        return f"07/01/{int(text):04d}"
    if re.fullmatch(r"\d{1,2}/\d{2,4}", text):
        month, year = text.split("/")
        return f"{int(month):02d}/01/{expand_year(year):04d}"
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", text):
        month, day, year = text.split("/")
        return f"{int(month):02d}/{int(day):02d}/{expand_year(year):04d}"
    return pd.NA

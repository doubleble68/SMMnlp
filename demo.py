"""Run a synthetic workflow without a GPU, model weights, or clinical data."""

import pandas as pd

from smmnlp import run_pipeline


class ScriptedGenerator:
    """A test double that returns predefined responses, not an LLM."""

    def __init__(self) -> None:
        self.responses = iter(
            [
                "Yes",
                "Yes",
                "Yes",
                "No",  # Four note-level diagnosis responses.
                "new",
                "hx",  # Two first-positive-note history responses.
                "SMM_diagnosis_date: 2018\nhuman_review_needed: no",
            ]
        )

    def generate(self, prompt: str, *, max_new_tokens: int) -> str:
        return next(self.responses)


def main() -> None:
    notes = pd.DataFrame(
        {
            "PatientSSN": ["SYN001", "SYN001", "SYN002", "SYN003"],
            "NoteDate": ["02Apr2020", "01Mar2020", "01May2020", "01Jun2020"],
            "NoteText": [
                "Synthetic later note",
                "Synthetic earlier note",
                "Synthetic historical note",
                "Synthetic negative note",
            ],
        }
    )
    result = run_pipeline(notes, ScriptedGenerator())
    print(result.history_predictions[["PatientSSN", "NoteDate", "post_HxNew_LLM_Binary"]])
    print(result.date_extractions[["PatientSSN", "DxDate_LLM", "UncertainFlag_LLM"]])


if __name__ == "__main__":
    main()

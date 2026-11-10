"""Command-line entry points for a task or the three-stage workflow."""

import argparse
import logging
from pathlib import Path

from .config import ModelConfig
from .inference import predict_notes
from .io import read_table, write_table
from .modeling import LocalLLM
from .pipeline import run_pipeline
from .prompts import Task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SMM local LLM workflows")
    parser.add_argument("task", choices=[task.value for task in Task] + ["pipeline"])
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-column", default="NoteText")
    parser.add_argument("--patient-column", default="PatientSSN")
    parser.add_argument("--date-column", default="NoteDate")
    parser.add_argument("--date-format", default="%d%b%Y")
    parser.add_argument("--offload-folder", type=Path, default=Path("./offload"))
    parser.add_argument("--external-json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    notes = read_table(arguments.input, external_json=arguments.external_json)
    generator = LocalLLM.from_pretrained(
        ModelConfig(model_path=arguments.model_path, offload_folder=arguments.offload_folder)
    )
    if arguments.task != "pipeline":
        predictions = predict_notes(
            notes,
            generator,
            arguments.task,
            text_column=arguments.text_column,
        )
        write_table(predictions, arguments.output)
        return

    result = run_pipeline(
        notes,
        generator,
        text_column=arguments.text_column,
        patient_column=arguments.patient_column,
        date_column=arguments.date_column,
        date_format=arguments.date_format,
    )
    for filename, frame in (
        ("note_predictions.xlsx", result.note_predictions),
        ("first_positive_notes.xlsx", result.first_positive_notes),
        ("history_predictions.xlsx", result.history_predictions),
        ("date_extractions.xlsx", result.date_extractions),
    ):
        write_table(frame, arguments.output / filename)

"""Task execution independent of the local model implementation."""

import logging
from collections.abc import Sequence
from time import perf_counter

import pandas as pd

from .modeling import TextGenerator
from .preprocessing import add_date_extractions
from .prompts import Task, get_task

LOGGER = logging.getLogger(__name__)


def generate_responses(
    documents: Sequence[str], generator: TextGenerator, task: Task | str
) -> list[str]:
    """Generate one response per document without repeatedly rebuilding a list."""
    specification = get_task(task)
    responses = []
    started_at = perf_counter()
    for position, document in enumerate(documents):
        responses.append(
            generator.generate(
                specification.format_prompt(document),
                max_new_tokens=specification.max_new_tokens,
            )
        )
        LOGGER.info("Completed note %d of %d", position + 1, len(documents))
    LOGGER.info(
        "Completed %s in %.2f seconds", specification.task.value, perf_counter() - started_at
    )
    return responses


def predict_notes(
    notes: pd.DataFrame,
    generator: TextGenerator,
    task: Task | str,
    *,
    text_column: str = "NoteText",
) -> pd.DataFrame:
    """Run one request per note in input order and retain all input columns."""
    specification = get_task(task)
    documents = notes[text_column].tolist()
    result = notes.copy()
    result[specification.output_column] = generate_responses(documents, generator, task)
    if specification.task == Task.DATE:
        result = add_date_extractions(result)
    return result

"""Clinical abstraction tasks, prompt templates, and generation limits."""

from dataclasses import dataclass
from enum import Enum
from importlib.resources import files


class Task(str, Enum):
    DIAGNOSIS = "diagnosis"
    HISTORY = "history"
    DATE = "date"


@dataclass(frozen=True)
class TaskSpec:
    task: Task
    prompt_template: str
    max_new_tokens: int
    output_column: str

    def format_prompt(self, document: str) -> str:
        return self.prompt_template.format(document=document)


def get_task(task: Task | str) -> TaskSpec:
    """Load a clinical abstraction task and its generation settings."""
    task = Task(task)
    token_limit, output_column = {
        Task.DIAGNOSIS: (5, "SMMDx_LLM_Binary"),
        Task.HISTORY: (5, "HxNew_LLM_Binary"),
        Task.DATE: (50, "rawoutput"),
    }[task]
    template = files("smmnlp").joinpath(f"prompt_templates/{task.value}.txt")
    return TaskSpec(task, template.read_text(encoding="utf-8"), token_limit, output_column)

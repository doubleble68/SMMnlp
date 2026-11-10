"""Reusable SMM note abstraction without model loading at import time."""

from .config import ModelConfig
from .inference import predict_notes
from .modeling import LocalLLM
from .pipeline import PipelineResult, run_pipeline
from .prompts import Task, get_task

__version__ = "2.0.0"

__all__ = [
    "LocalLLM",
    "ModelConfig",
    "PipelineResult",
    "Task",
    "get_task",
    "predict_notes",
    "run_pipeline",
]

"""Runtime settings shared by all inference tasks."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Keep local model and offloading settings configurable."""

    model_path: str | Path
    local_files_only: bool = True
    device_map: str = "auto"
    offload_folder: str | Path = "./offload"
    system_message: str = "You are a helpful data abstractor."

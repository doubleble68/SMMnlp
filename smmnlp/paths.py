"""Project-relative data and model paths for notebook workflows."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectPaths:
    data_root: Path
    model_path: Path
    exclusions_file: Path

    @classmethod
    def from_environment(cls) -> "ProjectPaths":
        """Read optional overrides, with paths anchored to the project folder."""
        data_root = Path(os.environ.get("SMMNLP_DATA_ROOT", str(PROJECT_ROOT / "data")))
        return cls(
            data_root=data_root,
            model_path=Path(
                os.environ.get(
                    "SMMNLP_MODEL_PATH", str(PROJECT_ROOT / "models" / "Ministral3-bnb-4bit")
                )
            ),
            exclusions_file=Path(
                os.environ.get(
                    "SMMNLP_EXCLUSIONS_FILE",
                    str(PROJECT_ROOT / "config" / "excluded_patient_ids.json"),
                )
            ),
        )

    def data(self, relative_path: str) -> str:
        path = self.data_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def load_excluded_patient_ids(self) -> list[Any]:
        """Read patient exclusions supplied in the study configuration."""
        with self.exclusions_file.open(encoding="utf-8") as stream:
            patient_ids = json.load(stream)
        if not isinstance(patient_ids, list):
            raise ValueError("The exclusions file must contain a JSON list of patient IDs")
        return patient_ids

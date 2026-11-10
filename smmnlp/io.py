"""Read and write the formats used in the notebook analyses."""

import json
from pathlib import Path

import pandas as pd


def read_table(path: str | Path, *, external_json: bool = False) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".json":
        if external_json:
            # External-validation inference used json.load with errors='ignore'.
            with path.open(encoding="utf-8", errors="ignore") as source:
                return pd.DataFrame(json.load(source))
        return pd.read_json(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def write_table(frame: pd.DataFrame, path: str | Path, *, date_format: str = "epoch") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        frame.to_json(path, date_format=date_format)
    elif path.suffix.lower() == ".xlsx":
        frame.to_excel(path, engine="xlsxwriter", sheet_name="Sheet1", index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")

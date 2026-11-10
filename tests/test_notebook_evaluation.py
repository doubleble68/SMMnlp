"""Exercise dataframe assignments and external-date evaluation with synthetic inputs."""

import importlib.util
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def notebook_code(relative_path):
    notebook = json.loads((PROJECT_ROOT / "notebooks" / relative_path).read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


class NotebookEvaluationTests(unittest.TestCase):
    def test_binary_fill_assignments_return_labels(self):
        cases = [
            ("04_pipeline/04_build_reference_standard.ipynb",
             "SMMDx_True_Binary", "True_SMMDx_Binary", [True, None], [True, False]),
            ("04_pipeline/05_evaluate_training.ipynb",
             "post_SMMDx_LLM_Binary", "LLM_SMMDx_Binary", ["Yes", None], ["Yes", "No"]),
            ("04_pipeline/06_evaluate_testing.ipynb",
             "post_SMMDx_LLM_Binary", "LLM_SMMDx_Binary", ["Yes", None], ["Yes", "No"]),
        ]
        for path, input_column, output_column, values, expected in cases:
            with self.subTest(notebook=path):
                cells = notebook_code(path)
                assignment = next(cell for cell in cells
                                  if f'["{output_column}"] =' in cell and ".fillna(" in cell)
                namespace = {"smmfull_true2": pd.DataFrame({input_column: values})}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    exec(compile(assignment, path, "exec"), namespace)
                self.assertEqual(namespace["smmfull_true2"][output_column].tolist(), expected)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Notebook extra not installed")
    def test_external_date_evaluation_keeps_missing_predictions(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        reference = pd.DataFrame({
            "pat_mrn_id": ["SYN1", "SYN2", "SYN3", "SYN4"],
            "smm_date_true": ["01/01/2020", "unknown", "05/01/2020", None],
        })
        predictions = pd.DataFrame({
            "pat_mrn_id": ["SYN1", "SYN2", "SYN3", "SYN4"],
            "DxDate_LLM_Human": ["01/03/2020", "unknown", "unknown", None],
        })
        cells = notebook_code("05_external/02_evaluate_dates.ipynb")
        namespace = {}
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"SMMNLP_DATA_ROOT": directory}):
                with patch("pandas.read_excel", side_effect=[reference, predictions]):
                    with patch.object(plt, "show"):
                        for index, cell in enumerate(cells):
                            exec(compile(cell, f"external_dates_cell_{index}", "exec"), namespace)
        plt.close("all")
        self.assertEqual(len(namespace["date_results"]), 4)
        self.assertEqual(namespace["metric_result"].cm.tolist(), [[2, 0], [1, 1]])
        self.assertEqual(namespace["paired_dates"]["pat_mrn_id"].tolist(), ["SYN1"])
        self.assertEqual(namespace["paired_dates"]["date_difference_days"].tolist(), [2])


if __name__ == "__main__":
    unittest.main()

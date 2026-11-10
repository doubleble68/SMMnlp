import ast
import builtins
import contextlib
import io
import json
import os
import subprocess
import symtable
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from smmnlp.cli import main
from smmnlp.io import read_table, write_table
from smmnlp.paths import ProjectPaths
from tests.test_workflows import FakeGenerator, synthetic_notes

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectTests(unittest.TestCase):
    def test_demo_runs_without_installation_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "demo.py")],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn("SYN002", result.stdout)
        self.assertIn("2018", result.stdout)

    def test_notebook_code_compiles_without_outputs_or_new_undefined_globals(self):
        notebooks = sorted((PROJECT_ROOT / "notebooks").rglob("*.ipynb"))
        self.assertEqual(len(notebooks), 22)
        for path in notebooks:
            notebook = json.loads(path.read_text())
            source_parts = []
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                with self.subTest(notebook=path.name, cell=index):
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    source = "".join(cell["source"])
                    compile(source, str(path), "exec")
                    self.assertNotIn("/storage3/", source)
                    self.assertNotIn("/storage1/", source)
                    source_parts.append(source)
            table = symtable.symtable("\n".join(source_parts), path.name, "exec")
            unresolved = {
                symbol.get_name()
                for symbol in table.get_symbols()
                if symbol.is_referenced()
                and not (symbol.is_assigned() or symbol.is_imported())
                and symbol.get_name() not in vars(builtins)
            }
            self.assertEqual(unresolved, set(), path.name)

    def test_notebooks_use_shared_functions_and_no_embedded_model_loaders(self):
        for path in (PROJECT_ROOT / "notebooks").rglob("*.ipynb"):
            notebook = json.loads(path.read_text())
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                tree = ast.parse("".join(cell["source"]))
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        self.assertNotIn(
                            node.name,
                            {
                                "llm_qa",
                                "extract_binary_label",
                                "parse_llm_extraction",
                                "expand_date",
                            },
                        )

    def test_io_preserves_tabular_json_and_excel_conventions(self):
        frame = pd.DataFrame({"PatientSSN": ["SYN1", "SYN2"], "label": ["Yes", "No"]})
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("json", "xlsx", "csv"):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / "nested" / f"table.{suffix}"
                    write_table(frame, path)
                    assert_frame_equal(read_table(path), frame)
                    if suffix == "json":
                        self.assertEqual(path.read_text(), frame.to_json())

    def test_external_json_reader_keeps_record_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.json"
            path.write_bytes(b'[{"full_note":"A"},{"full_note":"B"}]')
            self.assertEqual(read_table(path, external_json=True).full_note.tolist(), ["A", "B"])

    def test_cli_task_writes_workbook_with_injected_model(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "results.xlsx"
            synthetic_notes().to_json(input_path)
            generator = FakeGenerator(["Yes", "Yes", "Yes", "No"])
            with patch("smmnlp.cli.LocalLLM.from_pretrained", return_value=generator) as load:
                main(
                    [
                        "diagnosis",
                        "--model-path",
                        "weights",
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ]
                )
            load.assert_called_once()
            self.assertEqual(
                pd.read_excel(output_path).SMMDx_LLM_Binary.tolist(), ["Yes", "Yes", "Yes", "No"]
            )

    def test_cli_pipeline_writes_all_four_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.json"
            output_path = Path(directory) / "pipeline"
            synthetic_notes().to_json(input_path)
            generator = FakeGenerator(
                [
                    "Yes",
                    "Yes",
                    "Yes",
                    "No",
                    "new",
                    "hx",
                    "SMM_diagnosis_date: 2018\nhuman_review_needed: no",
                ]
            )
            with patch("smmnlp.cli.LocalLLM.from_pretrained", return_value=generator) as load:
                main(
                    [
                        "pipeline",
                        "--model-path",
                        "weights",
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ]
                )
            load.assert_called_once()
            self.assertEqual(
                {p.name for p in output_path.iterdir()},
                {
                    "note_predictions.xlsx",
                    "first_positive_notes.xlsx",
                    "history_predictions.xlsx",
                    "date_extractions.xlsx",
                },
            )
            self.assertEqual(
                pd.read_excel(output_path / "date_extractions.xlsx").PatientSSN.tolist(), ["SYN002"]
            )

    def test_notebook_paths_and_external_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            exclusions_file = Path(directory) / "exclusions.json"
            exclusions_file.write_text('["SYN001"]')
            with patch.dict(
                os.environ,
                {
                    "SMMNLP_DATA_ROOT": directory,
                    "SMMNLP_MODEL_PATH": "local-weights",
                    "SMMNLP_EXCLUSIONS_FILE": str(exclusions_file),
                },
            ):
                paths = ProjectPaths.from_environment()
            self.assertEqual(
                paths.data("Update/example.json"), str(Path(directory) / "Update/example.json")
            )
            self.assertEqual(paths.model_path, Path("local-weights"))
            self.assertEqual(paths.load_excluded_patient_ids(), ["SYN001"])

    def test_import_and_cli_help_work_without_inference_dependencies(self):
        script = (
            "import sys; import smmnlp; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as raised:
                main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--model-path", output.getvalue())


if __name__ == "__main__":
    unittest.main()

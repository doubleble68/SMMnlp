"""Check clinical response parsing and prompt settings."""

import hashlib
import unittest
from importlib.resources import files

import pandas as pd

from smmnlp import parsing
from smmnlp.prompts import Task, get_task


class ParsingTests(unittest.TestCase):
    def test_diagnosis_labels(self):
        cases = [("Yes", "Yes"), ("No", "No"), ("YES.", "Yes"),
                 ("yesterday", "Yes"), ("not yes", "Yes"), ("", "No"),
                 ("<final_answer>Yes</final_answer>", "Yes")]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(parsing.parse_diagnosis_label(response), expected)

    def test_non_string_labels_raise_type_errors(self):
        for name in ("parse_diagnosis_label", "parse_history_for_extraction",
                     "parse_history_for_evaluation"):
            for value in (None, 42, float("nan")):
                with self.subTest(parser=name, value=value):
                    with self.assertRaises(TypeError):
                        getattr(parsing, name)(value)

    def test_history_rules(self):
        cases = [("hx", "Hx", "hx"), ("new", "New", "new"),
                 ("unexpected", "New", "hx"), ("hx and new", "Hx", "new")]
        for response, extraction, evaluation in cases:
            with self.subTest(response=response):
                self.assertEqual(parsing.parse_history_for_extraction(response), extraction)
                self.assertEqual(parsing.parse_history_for_evaluation(response), evaluation)

    def test_date_response_fields(self):
        cases = [
            (None, {"SMM_diagnosis_date": None, "human_review_needed": True}),
            ("", {"SMM_diagnosis_date": None, "human_review_needed": True}),
            ("unstructured", {}),
            ("SMM_diagnosis_date: 8/2009\nhuman_review_needed: no",
             {"SMM_diagnosis_date": "8/2009", "human_review_needed": False}),
            (" SMM_diagnosis_date : unknown\nhuman_review_needed: YES ",
             {"SMM_diagnosis_date": "unknown", "human_review_needed": True}),
            ("SMM_diagnosis_date: 2003", {"SMM_diagnosis_date": "2003"}),
            ("human_review_needed: maybe", {"human_review_needed": False}),
            ("human_review_needed: 1\nextra_key: value:with:colons",
             {"human_review_needed": True, "extra_key": "value:with:colons"}),
            ("human_review_needed: no\nhuman_review_needed: true", {"human_review_needed": True}),
            ("Human_review_needed: yes", {"Human_review_needed": "yes"}),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(parsing.parse_date_extraction(response), expected)

    def test_partial_date_conventions(self):
        cases = [("2020", "07/01/2020"), (2020, "07/01/2020"),
                 (" 8/2009 ", "08/01/2009"), ("08/09", "08/01/2009"),
                 ("8/26", "08/01/2026"), ("8/27", "08/01/1927"),
                 ("2/3/00", "02/03/2000"), ("2/3/99", "02/03/1999"),
                 ("4/25/2018", "04/25/2018"), ("13/99", "13/01/1999"),
                 ("2/31/20", "02/31/2020"), ("8/123", "08/01/0123")]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(parsing.expand_partial_date(value), expected)
        for value in (None, pd.NA, float("nan"), "unknown", "", "2020-01-01"):
            with self.subTest(value=value):
                self.assertTrue(pd.isna(parsing.expand_partial_date(value)))
        self.assertEqual(parsing.expand_partial_date("8/27", 50), "08/01/2027")

    def test_prompt_integrity_and_token_limits(self):
        expected = {
            Task.DIAGNOSIS: ("69441ac69ffb", 5, "SMMDx_LLM_Binary"),
            Task.HISTORY: ("7140dac5e9d8", 5, "HxNew_LLM_Binary"),
            Task.DATE: ("780dd38fb9c5", 50, "rawoutput"),
        }
        for task, (digest, token_limit, column) in expected.items():
            with self.subTest(task=task):
                template_bytes = files("smmnlp").joinpath(f"prompt_templates/{task.value}.txt").read_bytes()
                self.assertEqual(hashlib.sha256(template_bytes).hexdigest()[:12], digest)
                specification = get_task(task)
                self.assertEqual(specification.max_new_tokens, token_limit)
                self.assertEqual(specification.output_column, column)
                self.assertEqual(specification.format_prompt("synthetic {note}"),
                                 template_bytes.decode().format(document="synthetic {note}"))


if __name__ == "__main__":
    unittest.main()

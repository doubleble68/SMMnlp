import random
import unittest
import warnings

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn import metrics

from smmnlp.evaluation import calculate_binary_metrics, confusion_labels
from smmnlp.inference import predict_notes
from smmnlp.parsing import parse_diagnosis_label
from smmnlp.pipeline import run_pipeline
from smmnlp.preprocessing import (
    add_date_extractions,
    add_parsed_labels,
    earliest_positive_notes,
    patient_level_labels,
    sample_patients_by_year,
    select_reviewed_dates,
    split_by_patient,
)
from smmnlp.prompts import Task, get_task


class FakeGenerator:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, prompt, *, max_new_tokens):
        self.calls.append((prompt, max_new_tokens))
        return next(self.responses)


def synthetic_notes():
    return pd.DataFrame(
        {
            "PatientSSN": ["SYN001", "SYN001", "SYN002", "SYN003"],
            "NoteDate": ["02Apr2020", "01Mar2020", "01May2020", "01Jun2020"],
            "NoteText": ["later note", "earlier note", "historical note", "negative note"],
        }
    )


class WorkflowTests(unittest.TestCase):
    def test_inference_keeps_note_order_columns_and_index(self):
        notes = pd.DataFrame({"full_note": ["A", "B"], "metadata": [1, 2]}, index=[8, 3])
        original = notes.copy()
        generator = FakeGenerator(["Yes", "No"])
        result = predict_notes(notes, generator, Task.DIAGNOSIS, text_column="full_note")
        self.assertEqual(result["SMMDx_LLM_Binary"].tolist(), ["Yes", "No"])
        self.assertEqual(result.index.tolist(), [8, 3])
        self.assertEqual(
            generator.calls,
            [
                (get_task(Task.DIAGNOSIS).format_prompt("A"), 5),
                (get_task(Task.DIAGNOSIS).format_prompt("B"), 5),
            ],
        )
        assert_frame_equal(notes, original)

    def test_earliest_rows_ties_and_group_order_match_expected(self):
        notes = synthetic_notes()
        notes["SMMDx_LLM_Binary"] = ["Yes", "Yes", "Yes", "No"]
        notes = pd.concat([notes, notes.iloc[[1]]], ignore_index=True)
        parsed = add_parsed_labels(notes, ["SMMDx_LLM_Binary"], parse_diagnosis_label)
        original = parsed.copy()
        positive = parsed.loc[parsed["post_SMMDx_LLM_Binary"] == "Yes"].copy()
        positive["NoteDate2"] = pd.to_datetime(positive["NoteDate"], format="%d%b%Y")
        expected = positive.loc[positive.groupby("PatientSSN")["NoteDate2"].idxmin()]
        assert_frame_equal(earliest_positive_notes(parsed), expected)
        assert_frame_equal(parsed, original)

    def test_date_outputs_follow_filtered_rows(self):
        frame = pd.DataFrame(
            {"rawoutput": [
                "SMM_diagnosis_date: 2001\nhuman_review_needed: no",
                "SMM_diagnosis_date: 2002\nhuman_review_needed: yes",
            ]},
            index=[1, 5],
        )
        expected = frame.assign(DxDate_LLM=["2001", "2002"], UncertainFlag_LLM=[False, True])
        assert_frame_equal(add_date_extractions(frame), expected)

    def test_date_outputs_follow_duplicate_row_indices(self):
        frame = pd.DataFrame(
            {
                "rawoutput": [
                    "SMM_diagnosis_date: 2001\nhuman_review_needed: no",
                    "SMM_diagnosis_date: 2002\nhuman_review_needed: yes",
                ]
            },
            index=[1, 1],
        )
        result = add_date_extractions(frame)
        self.assertEqual(result["DxDate_LLM"].tolist(), ["2001", "2002"])
        self.assertEqual(result["UncertainFlag_LLM"].tolist(), [False, True])

    def test_malformed_date_outputs_keep_original_missing_key_error(self):
        with self.assertRaises(KeyError):
            add_date_extractions(pd.DataFrame({"rawoutput": ["unstructured"]}))

    def test_patient_aggregation_matches_expected_any_positive_rule(self):
        frame = pd.DataFrame(
            {
                "PatientSSN": ["B", "A", "A", "C", "B"],
                "post_SMM Dx (final)": ["No", "No", "Yes", "No", "No"],
                "post_SMMDx_LLM_Binary": ["Yes", "No", "No", "No", "Yes"],
            }
        )
        expected = pd.DataFrame(
            {
                "PatientSSN": ["A", "B", "C"],
                "reference_label": ["Yes", "No", "No"],
                "predicted_label": ["No", "Yes", "No"],
            }
        )
        assert_frame_equal(patient_level_labels(frame), expected)

    def test_patient_split_matches_expected_random_sampling(self):
        notes = pd.concat([synthetic_notes()] * 2, ignore_index=True)
        for seed in (56, 66695):
            state = random.getstate()
            try:
                patient_ids = notes["PatientSSN"].unique()
                random.seed(seed)
                selected = random.sample(list(patient_ids), int(len(patient_ids) * 0.7))
            finally:
                random.setstate(state)
            train, test = split_by_patient(notes, random_state=seed)
            assert_frame_equal(train, notes.loc[notes["PatientSSN"].isin(selected)])
            assert_frame_equal(test, notes.loc[~notes["PatientSSN"].isin(selected)])
            self.assertFalse(set(train.PatientSSN) & set(test.PatientSSN))
            self.assertEqual(random.getstate(), state)

    def test_year_sampling_matches_expected(self):
        notes = pd.DataFrame(
            {
                "PatientSSN": ["A", "B", "C", "D", "E", "A"],
                "index_date": ["01JAN2022"] * 3 + ["01JAN2023"] * 2 + ["01JAN2022"],
            }
        )
        sizes = {"2022": 1, "2023": 1}
        candidates = notes.loc[~notes.PatientSSN.isin(["C"])].copy()
        candidates["index_year"] = candidates.index_date.str[5:]
        candidates["cohort_version"] = "S2"
        unique = candidates[["PatientSSN", "index_year"]].drop_duplicates()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            selected = unique.groupby("index_year", group_keys=False).apply(
                lambda group: group.sample(n=sizes[group.name], random_state=76)
            )
            actual = sample_patients_by_year(notes, sizes, excluded_patient_ids=["C"])
        assert_frame_equal(actual, candidates.loc[candidates.PatientSSN.isin(selected.PatientSSN)])

    def test_human_date_selection_uses_explicit_equality_rule(self):
        frame = pd.DataFrame(
            {
                "UncertainFlag_LLM": [True, False, 1, "yes"],
                "DxDate_Human": ["H1", "H2", "H3", "H4"],
                "DxDate_LLM": ["L1", "L2", "L3", "L4"],
            },
            index=[3, 5, 7, 9],
        )
        expected = pd.Series(["H1", "L2", "H3", "L4"], index=frame.index, name="DxDate_Reviewed")
        assert_series_equal(select_reviewed_dates(frame), expected)

    def test_full_pipeline_routing_and_date_assignment(self):
        notes = synthetic_notes()
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
        result = run_pipeline(notes, generator)
        self.assertEqual(result.first_positive_notes.index.tolist(), [1, 2])
        self.assertEqual(
            result.history_predictions["post_HxNew_LLM_Binary"].tolist(), ["New", "Hx"]
        )
        self.assertEqual(result.date_extractions["PatientSSN"].tolist(), ["SYN002"])
        self.assertEqual(result.date_extractions["DxDate_LLM"].tolist(), ["2018"])
        self.assertEqual([limit for _, limit in generator.calls], [5, 5, 5, 5, 5, 5, 50])

    def test_pipeline_keeps_extracted_dates_on_the_correct_patient(self):
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
        result = run_pipeline(synthetic_notes(), generator)
        self.assertEqual(result.date_extractions["DxDate_LLM"].iloc[0], "2018")

    def test_empty_pipeline_routes_do_not_call_later_stages(self):
        notes = synthetic_notes()
        for diagnosis, history in ((["No"] * 4, []), (["Yes"] * 4, ["new"] * 3)):
            generator = FakeGenerator(diagnosis + history)
            result = run_pipeline(notes, generator)
            self.assertTrue(result.date_extractions.empty)
            self.assertIn("DxDate_LLM", result.date_extractions.columns)
            self.assertEqual(len(generator.calls), len(diagnosis) + len(history))

    def test_binary_metrics_match_direct_sklearn_calls(self):
        truth = [0, 0, 1, 1, 1, 0]
        predictions = [0, 1, 0, 1, 1, 0]
        actual = calculate_binary_metrics(truth, predictions)
        for attribute, function in (
            ("accuracy", metrics.accuracy_score),
            ("precision", metrics.precision_score),
            ("recall", metrics.recall_score),
            ("f1", metrics.f1_score),
            ("auc_roc", metrics.roc_auc_score),
            ("au_prc", metrics.average_precision_score),
        ):
            self.assertEqual(getattr(actual, attribute), function(truth, predictions))
        np.testing.assert_array_equal(actual.cm, metrics.confusion_matrix(truth, predictions))
        self.assertEqual(actual.to_dict()["confusion_matrix"], [[2, 1], [1, 2]])

    def test_confusion_outcome_order_matches_expected(self):
        actual = confusion_labels(
            np.array([True, False, True, False]), np.array([True, False, False, True])
        )
        self.assertEqual(actual.tolist(), ["TP", "TN", "FN", "FP"])


if __name__ == "__main__":
    unittest.main()

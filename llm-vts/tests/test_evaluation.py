import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("vts_evaluation", Path(__file__).resolve().parents[2] / "maritime-qa" / "ours.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
from react_tools import validated_sql


class EvaluationTests(unittest.TestCase):
    def test_dataset_time_reference_is_resolved_without_changing_literals(self):
        sql = evaluation.reference_sql("SELECT @CNOW, '@CNOW'")
        self.assertEqual(sql, "SELECT (SELECT MAX(last_updated) FROM ship_ais), '@CNOW'")
        with self.assertRaises(ValueError):
            evaluation.reference_sql("SELECT @CNOW; DELETE FROM ship_ais")
        with self.assertRaises(ValueError):
            evaluation.reference_sql("SELECT @OTHER")

    def test_dataset_reference_queries_pass_read_only_validation(self):
        dataset = json.loads((Path(__file__).resolve().parents[2] / "maritime-qa" / "data" / "test_dataset.json").read_text(encoding="utf-8"))
        for split, items in dataset.items():
            for item in items:
                with self.subTest(split=split, question=item["question"]):
                    sql = evaluation.reference_sql(item["query"])
                    self.assertNotIn("@CNOW", sql.upper())
                    validated_sql(sql, max_rows=100)

    def run_case(self, actual, expected):
        trace = [{"id": "call-1", "name": "query_database", "arguments": {"sql": "SELECT name FROM ship_ais"}}, {"tool_call_id": "call-1", "result": actual}]
        with patch.object(evaluation, "run_agent", return_value=("answer", trace)), patch.object(evaluation, "run_query", return_value=json.dumps(expected)):
            return evaluation.evaluate_item({"question": "question", "query": "SELECT name FROM ship_ais"})

    def test_exact_rows_match_case_insensitively_for_columns(self):
        actual = json.dumps({"rows": [{"NAME": "A"}, {"NAME": "B"}], "truncated": False})
        expected = {"rows": [{"name": "B"}, {"name": "A"}], "truncated": False}
        self.assertEqual(self.run_case(actual, expected)["status"], "pass")

    def test_values_are_not_ignored_when_names_match(self):
        actual = json.dumps({"rows": [{"name": "A", "speed": 12}], "truncated": False})
        expected = {"rows": [{"name": "A", "speed": 2}], "truncated": False}
        self.assertEqual(self.run_case(actual, expected)["status"], "fail")

    def test_empty_or_truncated_results_are_inconclusive(self):
        empty = {"rows": [], "truncated": False}
        self.assertEqual(self.run_case(json.dumps(empty), empty)["status"], "inconclusive")
        truncated = {"rows": [{"name": "A"}], "truncated": True}
        self.assertEqual(self.run_case(json.dumps(truncated), empty)["status"], "inconclusive")
        self.assertEqual(self.run_case(json.dumps(empty), truncated)["status"], "inconclusive")

    def test_tool_error_is_inconclusive(self):
        result = self.run_case("invalid query", {"rows": [], "truncated": False})
        self.assertEqual(result["status"], "inconclusive")
        self.assertIn("invalid query", result["reason"])

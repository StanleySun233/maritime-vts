import importlib.util
import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("vts_evaluation", Path(__file__).resolve().parents[2] / "maritime-qa" / "ours.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
from vtsllm.react_tools import validated_sql


class EvaluationTests(unittest.TestCase):
    def test_completed_rows_survive_later_failure(self):
        trace = [{"id": "call-1", "name": "search_knowledge", "arguments": {"query": "规则"}}, {"tool_call_id": "call-1", "status": "success", "result": "evidence"}]
        row = {"question": "first", "answer": "answer", "sql": "SELECT 1", "metric": "sql_result_comparison", "status": "pass", "reason": "Results match", "tool_trace": trace}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.json"
            output = root / "results.csv"
            dataset.write_text(json.dumps({"dev": [{}, {}]}), encoding="utf-8")
            arguments = ["ours.py", "--dataset", str(dataset), "--output", str(output)]
            with patch.object(sys, "argv", arguments), patch.object(evaluation, "evaluate_item", side_effect=[row, ValueError("invalid tool call")]), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "invalid tool call"):
                    evaluation.main()
            with output.open(encoding="utf-8", newline="") as saved:
                saved_rows = list(csv.DictReader(saved))
                self.assertEqual(json.loads(saved_rows[0]["tool_trace"]), trace)
                self.assertEqual(saved_rows[0]["question"], row["question"])

    def test_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.json"
            output = root / "results.csv"
            dataset.write_text(json.dumps({"dev": [{}]}), encoding="utf-8")
            output.write_text("previous results", encoding="utf-8")
            arguments = ["ours.py", "--dataset", str(dataset), "--output", str(output)]
            with patch.object(sys, "argv", arguments), patch.object(evaluation, "evaluate_item") as evaluate:
                with self.assertRaises(FileExistsError):
                    evaluation.main()
            self.assertEqual(output.read_text(encoding="utf-8"), "previous results")
            evaluate.assert_not_called()

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

    def run_case(self, actual, expected, reference="SELECT name FROM ship_ais"):
        trace = [{"id": "call-1", "name": "query_database", "arguments": {"sql": "SELECT name FROM ship_ais"}}, {"tool_call_id": "call-1", "result": actual}]
        with patch.object(evaluation, "run_agent", return_value=("answer", trace)), patch.object(evaluation, "run_query", return_value=json.dumps(expected)):
            return evaluation.evaluate_item({"question": "question", "query": reference})

    def test_exact_rows_match_case_insensitively_for_columns(self):
        actual = json.dumps({"rows": [{"NAME": "A"}, {"NAME": "B"}], "truncated": False})
        expected = {"rows": [{"name": "B"}, {"name": "A"}], "truncated": False}
        self.assertEqual(self.run_case(actual, expected)["status"], "pass")

    def test_values_are_not_ignored_when_names_match(self):
        actual = json.dumps({"rows": [{"name": "A", "speed": 12}], "truncated": False})
        expected = {"rows": [{"name": "A", "speed": 2}], "truncated": False}
        self.assertEqual(self.run_case(actual, expected)["status"], "fail")

    def test_reference_order_is_checked(self):
        expected = {"rows": [{"name": "A"}, {"name": "B"}], "truncated": False}
        reversed_rows = json.dumps({"rows": [{"name": "B"}, {"name": "A"}], "truncated": False})
        result = self.run_case(reversed_rows, expected, "SELECT name FROM ship_ais ORDER BY name")
        self.assertEqual(result["status"], "inconclusive")
        self.assertIn("order differs", result["reason"])
        self.assertEqual(result["metric"], "sql_result_comparison")
        self.assertEqual(self.run_case(reversed_rows, expected)["status"], "pass")

    def test_duplicate_rows_are_not_discarded(self):
        expected = {"rows": [{"name": "A"}, {"name": "A"}], "truncated": False}
        actual = json.dumps({"rows": [{"name": "A"}], "truncated": False})
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
        self.assertEqual(result["tool_trace"][-1]["result"], "invalid query")

    def test_visualization_query_is_evaluated(self):
        query = "SELECT type, COUNT(*) AS vessel_count FROM ship_ais GROUP BY type"
        rows = {"rows": [{"type": "TANKER", "vessel_count": 3}], "truncated": False}
        trace = [{"id": "chart-1", "name": "visualization_data", "arguments": {"sql": query}}, {"tool_call_id": "chart-1", "result": json.dumps(rows)}]
        with patch.object(evaluation, "run_agent", return_value=("answer", trace)), patch.object(evaluation, "run_query", return_value=json.dumps(rows)):
            result = evaluation.evaluate_item({"question": "Count vessels by type", "query": query})
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["sql"], query)

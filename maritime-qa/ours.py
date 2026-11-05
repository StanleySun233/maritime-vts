import argparse
import csv
import json
import sys
from pathlib import Path

import pymysql
import sqlglot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm-vts"))
from agent import run_agent
from react_tools import run_query, validated_sql


QUERY_TOOLS = {"query_database", "traffic_flow", "collision_analysis"}


def reference_sql(sql):
    tree = sqlglot.parse_one(validated_sql(sql), read="mysql")
    current_time = sqlglot.parse_one("(SELECT MAX(last_updated) FROM ship_ais)", read="mysql")
    tree = tree.transform(lambda node: current_time.copy() if isinstance(node, sqlglot.exp.Parameter) and node.name.upper() == "CNOW" else node)
    return tree.sql(dialect="mysql")


def evaluate_item(item):
    answer, trace = run_agent([{"role": "user", "content": item["question"]}])
    executions = [entry for entry in trace if entry.get("name") in QUERY_TOOLS]
    result = {"question": item["question"], "answer": answer, "sql": "", "status": "inconclusive", "reason": "No SQL query was executed"}
    if not executions:
        return result
    execution = executions[-1]
    result["sql"] = execution["arguments"]["sql"]
    outputs = [entry["result"] for entry in trace if entry.get("tool_call_id") == execution["id"]]
    if not outputs:
        result["reason"] = "Missing tool result"
        return result
    try:
        actual = json.loads(outputs[-1])
    except (json.JSONDecodeError, TypeError):
        result["reason"] = "Query failed: " + str(outputs[-1])
        return result
    if not isinstance(actual, dict) or "rows" not in actual:
        result["reason"] = "Query did not return rows"
        return result
    if actual.get("truncated"):
        result["reason"] = "Query result was truncated"
        return result
    try:
        expected = json.loads(run_query(reference_sql(item["query"])))
    except (ValueError, pymysql.MySQLError, sqlglot.errors.SqlglotError, OSError) as exc:
        result["reason"] = "Reference query failed: " + str(exc)
        return result
    if expected["truncated"]:
        result["reason"] = "Reference result was truncated"
        return result
    if not expected["rows"] and not actual["rows"]:
        result["reason"] = "Both results were empty"
        return result
    normalize = lambda rows: sorted(json.dumps({key.lower(): value for key, value in row.items()}, sort_keys=True, ensure_ascii=False) for row in rows)
    result["status"] = "pass" if normalize(expected["rows"]) == normalize(actual["rows"]) else "fail"
    result["reason"] = "Results match" if result["status"] == "pass" else "Results differ"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(Path(__file__).parent / "data" / "test_dataset.json"))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="ours.csv")
    args = parser.parse_args()
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    items = dataset[args.split][:args.limit]
    rows = []
    for item in items:
        row = evaluate_item(item)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    with open(args.output, "w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["question", "answer", "sql", "status", "reason"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

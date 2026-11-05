import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm-vts"))
from agent import run_agent
from utils.db_util import query
from utils.sql_compare import compare_sql_results


data = json.loads((Path(__file__).parent / "data" / "test_dataset.json").read_text(encoding="utf-8"))
rows = []
for item in data["train"]:
    answer, trace = run_agent([{"role": "user", "content": item["question"]}])
    executions = [entry for entry in trace if entry.get("name") in {"query_database", "traffic_flow", "collision_analysis"}]
    sql = executions[-1]["arguments"]["sql"] if executions else ""
    if sql:
        call_id = executions[-1]["id"]
        result = next(entry["result"] for entry in trace if entry.get("tool_call_id") == call_id)
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            result = {"error": result}
        if "rows" not in result or result["truncated"]:
            passed, remark = False, "Query failed or returned truncated results"
        else:
            expected = query(item["query"], "db.yaml")
            actual = pd.DataFrame(result["rows"])
            passed, remark = compare_sql_results(expected, actual)
    else:
        passed, remark = False, "No SQL query was executed"
    rows.append({"question": item["question"], "answer": answer, "sql": sql, "passed": passed, "remark": remark})
    print(json.dumps(rows[-1], ensure_ascii=False, default=str))
pd.DataFrame(rows).to_csv("ours.csv", index=False)

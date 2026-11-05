import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent
import react_tools


class Cursor:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, *args):
        self.calls.append(sql)
        if sql.startswith("EXPLAIN SELECT bad_column"):
            raise ValueError("Unknown column bad_column")

    def fetchall(self):
        return [{"TABLE_NAME": "ship_ais", "COLUMN_NAME": "name", "COLUMN_TYPE": "varchar(100)", "COLUMN_COMMENT": ""}]

    def fetchmany(self, count):
        return [{"name": "SHIP A"}]


class Connection:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def cursor(self):
        return Cursor(self.calls)

    def rollback(self):
        pass


class Knowledge:
    def search(self, query):
        return {"documents": [{"content": "Rule text", "source": "rules.txt"}]}


class OpenAIProtocolTests(unittest.TestCase):
    def test_tool_evidence_and_sql_retry_over_chat_completions(self):
        requests = []
        calls = []
        sequence = [
            ("schema-1", "database_schema", {}),
            ("knowledge-1", "search_knowledge", {"query": "rule"}),
            ("query-1", "query_database", {"sql": "SELECT bad_column FROM ship_ais"}),
            ("query-2", "query_database", {"sql": "SELECT name FROM ship_ais"}),
        ]

        def respond(request):
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test")
            body = json.loads(request.content)
            requests.append(body)
            index = len(requests) - 1
            if index < len(sequence):
                call_id, name, args = sequence[index]
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": "SHIP A"}
                finish = "stop"
            return httpx.Response(200, json={"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "mock", "choices": [{"index": 0, "message": message, "finish_reason": finish}]})

        client = httpx.Client(transport=httpx.MockTransport(respond))
        model = ChatOpenAI(model="mock", api_key="test", base_url="https://mock.invalid/v1", http_client=client, max_retries=0)
        with patch.object(react_tools, "connection", side_effect=lambda: Connection(calls)):
            answer, trace = agent.run_agent([{"role": "user", "content": "List vessels and use the rule"}], model=model, knowledge=Knowledge())
        client.close()
        self.assertEqual(answer, "SHIP A")
        self.assertEqual(len(requests), 5)
        self.assertIn("tools", requests[0])
        self.assertEqual([entry["name"] for entry in trace if "name" in entry], ["database_schema", "search_knowledge", "query_database", "query_database"])
        messages = requests[-1]["messages"]
        self.assertTrue(any(message.get("tool_call_id") == "knowledge-1" and "rules.txt" in message["content"] for message in messages))
        self.assertTrue(any(message.get("tool_call_id") == "query-1" and "Unknown column" in message["content"] for message in messages))
        self.assertNotIn("SELECT bad_column FROM ship_ais LIMIT 101", calls)
        self.assertEqual(calls[-2:], ["EXPLAIN SELECT name FROM ship_ais LIMIT 101", "SELECT name FROM ship_ais LIMIT 101"])

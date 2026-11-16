import csv
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent
import knowledge as knowledge_module
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
    def search(self, query, offset=0):
        return {"documents": [{"content": "Rule text", "source": "rules.txt"}]}


class OpenAIProtocolTests(unittest.TestCase):
    def test_imported_knowledge_flows_through_agent_and_pagination(self):
        class FakeEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[float(text.rsplit(" ", 1)[-1]), 0.0] for text in texts]

            def embed_query(self, text):
                return [1.0, 0.0]

        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            index = len(requests) - 1
            if index < 2:
                arguments = {"query": "rule"} if index == 0 else {"query": "rule", "offset": 4}
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": f"knowledge-{index}", "type": "function", "function": {"name": "search_knowledge", "arguments": json.dumps(arguments)}}]}
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": "answer"}
                finish = "stop"
            return httpx.Response(200, json={"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "mock", "choices": [{"index": 0, "message": message, "finish_reason": finish}]})

        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(respond)) as client:
            root = Path(directory)
            with patch.dict(os.environ, {"VTS_KNOWLEDGE_DIR": str(root / "store"), "OPENAI_API_KEY": "test"}), patch.object(knowledge_module, "OpenAIEmbeddings", return_value=FakeEmbeddings()):
                store = knowledge_module.Knowledge()
                for number in range(1, 6):
                    source = root / f"rule-{number}.txt"
                    source.write_text(f"Rule {number}", encoding="utf-8")
                    store.ingest(source)
                model = ChatOpenAI(model="mock", api_key="test", base_url="https://mock.invalid/v1", http_client=client, max_retries=0)
                answer, trace = agent.run_agent([{"role": "user", "content": "ask"}], model=model, knowledge=store)
        self.assertEqual(answer, "answer")
        first = json.loads(next(entry["result"] for entry in trace if entry.get("tool_call_id") == "knowledge-0"))
        second = json.loads(next(entry["result"] for entry in trace if entry.get("tool_call_id") == "knowledge-1"))
        self.assertEqual(first["next_offset"], 4)
        self.assertIsNone(second["next_offset"])
        self.assertEqual({doc["content"] for doc in first["documents"] + second["documents"]}, {f"Rule {n}" for n in range(1, 6)})
        self.assertTrue(all(doc["source"].endswith(".txt") for doc in first["documents"] + second["documents"]))

    def test_question_to_evaluation_csv_with_tool_evidence(self):
        spec = importlib.util.spec_from_file_location("vts_evaluation_integration", Path(__file__).resolve().parents[2] / "maritime-qa" / "ours.py")
        evaluation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluation)
        requests = []
        calls = []
        sequence = [("schema-1", "database_schema", {}), ("query-1", "query_database", {"sql": "SELECT name FROM ship_ais"})]

        def respond(request):
            requests.append(json.loads(request.content))
            index = len(requests) - 1
            if index < len(sequence):
                call_id, name, arguments = sequence[index]
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": "SHIP A"}
                finish = "stop"
            return httpx.Response(200, json={"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "mock", "choices": [{"index": 0, "message": message, "finish_reason": finish}]})

        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(respond)) as client:
            root = Path(directory)
            dataset = root / "dataset.json"
            output = root / "results.csv"
            dataset.write_text(json.dumps({"dev": [{"question": "Which vessel?", "query": "SELECT name FROM ship_ais"}]}), encoding="utf-8")
            model = ChatOpenAI(model="mock", api_key="test", base_url="https://mock.invalid/v1", http_client=client, max_retries=0)
            arguments = ["ours.py", "--dataset", str(dataset), "--output", str(output)]
            with patch.dict(os.environ, {"OPENAI_MODEL": "mock", "OPENAI_API_KEY": "test"}), patch.object(agent, "ChatOpenAI", return_value=model), patch.object(react_tools, "connection", side_effect=lambda: Connection(calls)), patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
                evaluation.main()
            with output.open(encoding="utf-8", newline="") as saved:
                rows = list(csv.DictReader(saved))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "pass")
        self.assertEqual(rows[0]["answer"], "SHIP A")
        trace = json.loads(rows[0]["tool_trace"])
        self.assertEqual([entry["name"] for entry in trace if "name" in entry], ["database_schema", "query_database"])
        self.assertTrue(any(entry.get("tool_call_id") == "query-1" and "SHIP A" in entry["result"] for entry in trace))
        for index, sql in enumerate(calls):
            if sql == "SELECT name FROM ship_ais LIMIT 101":
                self.assertEqual(calls[index - 1], "EXPLAIN " + sql)
        self.assertEqual(calls.count("SELECT name FROM ship_ais LIMIT 101"), 2)

    def test_model_can_request_another_knowledge_page(self):
        requests = []
        offsets = []

        class PaginatedKnowledge:
            def search(self, query, offset=0):
                offsets.append(offset)
                return {"documents": [{"source": f"page-{offset}"}], "next_offset": 4 if offset == 0 else None}

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            index = len(requests) - 1
            if index < 2:
                args = {"query": "rule"} if index == 0 else {"query": "rule", "offset": 4}
                message = {"role": "assistant", "content": None, "tool_calls": [{"id": f"knowledge-{index}", "type": "function", "function": {"name": "search_knowledge", "arguments": json.dumps(args)}}]}
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": "answer"}
                finish = "stop"
            return httpx.Response(200, json={"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "mock", "choices": [{"index": 0, "message": message, "finish_reason": finish}]})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            model = ChatOpenAI(model="mock", api_key="test", base_url="https://mock.invalid/v1", http_client=client, max_retries=0)
            answer, trace = agent.run_agent([{"role": "user", "content": "ask"}], model=model, knowledge=PaginatedKnowledge())
        self.assertEqual(answer, "answer")
        self.assertEqual(offsets, [0, 4])
        self.assertTrue(any(message.get("tool_call_id") == "knowledge-0" and '"next_offset": 4' in message["content"] for message in requests[1]["messages"]))
        self.assertTrue(any(entry.get("tool_call_id") == "knowledge-1" and "page-4" in entry["result"] for entry in trace))

    def test_model_can_repair_invalid_tool_arguments(self):
        requests = []
        calls = []

        def respond(request):
            requests.append(json.loads(request.content))
            index = len(requests) - 1
            if index == 0:
                tool_calls = [{"id": "bad-1", "type": "function", "function": {"name": "query_database", "arguments": '{"sql":'}}]
            elif index == 1:
                tool_calls = [{"id": "good-1", "type": "function", "function": {"name": "query_database", "arguments": json.dumps({"sql": "SELECT name FROM ship_ais"})}}]
            else:
                tool_calls = []
            message = {"role": "assistant", "content": None, "tool_calls": tool_calls} if tool_calls else {"role": "assistant", "content": "SHIP A"}
            return httpx.Response(200, json={"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "mock", "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}]})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            model = ChatOpenAI(model="mock", api_key="test", base_url="https://mock.invalid/v1", http_client=client, max_retries=0)
            with patch.object(react_tools, "connection", side_effect=lambda: Connection(calls)):
                answer, trace = agent.run_agent([{"role": "user", "content": "ask"}], model=model, knowledge=Knowledge())
        self.assertEqual(answer, "SHIP A")
        self.assertTrue(any(message.get("tool_call_id") == "bad-1" and "Invalid tool arguments" in message["content"] for message in requests[1]["messages"]))
        self.assertTrue(any(entry.get("id") == "bad-1" and entry.get("status") == "invalid" for entry in trace))
        self.assertEqual(calls[-2:], ["EXPLAIN SELECT name FROM ship_ais LIMIT 101", "SELECT name FROM ship_ais LIMIT 101"])

    def test_mixed_valid_and_invalid_calls_both_receive_results(self):
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                calls = [{"id": "reference", "type": "function", "function": {"name": "maritime_schema_reference", "arguments": "{}"}}, {"id": "invalid", "type": "function", "function": {"name": "query_database", "arguments": '{"sql":'}}]
                message = {"role": "assistant", "content": None, "tool_calls": calls}
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": "answer"}
                finish = "stop"
            return httpx.Response(200, json={"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "mock", "choices": [{"index": 0, "message": message, "finish_reason": finish}]})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            model = ChatOpenAI(model="mock", api_key="test", base_url="https://mock.invalid/v1", http_client=client, max_retries=0)
            answer, trace = agent.run_agent([{"role": "user", "content": "ask"}], model=model, knowledge=Knowledge())
        self.assertEqual(answer, "answer")
        tool_messages = {message["tool_call_id"]: message["content"] for message in requests[1]["messages"] if message["role"] == "tool"}
        self.assertIn("ship_ais_quarter", tool_messages["reference"])
        self.assertIn("Invalid tool arguments", tool_messages["invalid"])
        self.assertTrue(any(entry.get("tool_call_id") == "invalid" and entry.get("status") == "error" for entry in trace))

    def test_repeated_invalid_calls_stop_at_graph_limit(self):
        def respond(request):
            message = {"role": "assistant", "content": None, "tool_calls": [{"id": "bad-1", "type": "function", "function": {"name": "query_database", "arguments": '{"sql":'}}]}
            return httpx.Response(200, json={"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "mock", "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}]})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            model = ChatOpenAI(model="mock", api_key="test", base_url="https://mock.invalid/v1", http_client=client, max_retries=0)
            with self.assertRaises(GraphRecursionError) as raised:
                agent.run_agent([{"role": "user", "content": "ask"}], model=model, knowledge=Knowledge())
        self.assertTrue(any("tool_trace=" in note and "Invalid tool arguments" in note for note in raised.exception.__notes__))

    def test_truncated_response_is_not_reported_as_an_answer(self):
        def respond(request):
            message = {"role": "assistant", "content": "partial"}
            return httpx.Response(200, json={"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "mock", "choices": [{"index": 0, "message": message, "finish_reason": "length"}]})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            model = ChatOpenAI(model="mock", api_key="test", base_url="https://mock.invalid/v1", http_client=client, max_retries=0)
            with self.assertRaisesRegex(ValueError, "truncated"):
                agent.run_agent([{"role": "user", "content": "ask"}], model=model, knowledge=Knowledge())

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

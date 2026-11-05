import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage

import agent
import react_tools


class Cursor:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, *args):
        self.calls.append(sql)
        if self.fail and sql.startswith("EXPLAIN"):
            raise ValueError("invalid query")

    def fetchmany(self, size):
        return [{"name": "VESSEL"}]


class Connection:
    def __init__(self, fail=False):
        self.cur = Cursor(fail)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def cursor(self):
        return self.cur

    def rollback(self):
        pass


class FakeModel:
    def __init__(self):
        self.messages = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.messages.append(messages)
        if len(self.messages) == 1:
            return AIMessage(content="", tool_calls=[{"name": "query_database", "args": {"sql": "SELECT name FROM ship_ais"}, "id": "call-1", "type": "tool_call"}])
        return AIMessage(content="VESSEL")


class AgentTests(unittest.TestCase):
    def test_invalid_history_is_rejected(self):
        for messages in ([], [{"role": "system", "content": "instructions"}], [{"role": "user", "content": []}]):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                agent.run_agent(messages, model=FakeModel())

    def test_blank_model_response_is_rejected(self):
        model = FakeModel()
        model.invoke = lambda messages: AIMessage(content=" ")
        with self.assertRaisesRegex(ValueError, "empty response"):
            agent.run_agent([{"role": "user", "content": "ask"}], model=model)

    def test_explain_precedes_query(self):
        conn = Connection()
        with patch.object(react_tools, "connection", return_value=conn):
            result = json.loads(react_tools.query_database("SELECT name FROM ship_ais"))
        self.assertEqual(result["rows"], [{"name": "VESSEL"}])
        self.assertEqual(conn.cur.calls[-2:], ["EXPLAIN SELECT name FROM ship_ais LIMIT 101", "SELECT name FROM ship_ais LIMIT 101"])
        self.assertEqual(result["executed_sql"], "SELECT name FROM ship_ais LIMIT 101")

    def test_failed_validation_prevents_execution(self):
        conn = Connection(fail=True)
        with patch.object(react_tools, "connection", return_value=conn):
            with self.assertRaises(ValueError):
                react_tools.traffic_flow("SELECT name FROM ship_ais")
        self.assertNotIn("SELECT name FROM ship_ais LIMIT 101", conn.cur.calls)

    def test_row_limit_preserves_user_limit_and_offset(self):
        self.assertEqual(react_tools.validated_sql("SELECT * FROM ship_ais LIMIT 0", max_rows=100), "SELECT * FROM ship_ais LIMIT 0")
        self.assertEqual(react_tools.validated_sql("SELECT * FROM ship_ais LIMIT 5 OFFSET 2", max_rows=100), "SELECT * FROM ship_ais LIMIT 5 OFFSET 2")
        self.assertEqual(react_tools.validated_sql("SELECT * FROM ship_ais LIMIT 1000 OFFSET 2", max_rows=100), "SELECT * FROM ship_ais LIMIT 101 OFFSET 2")
        self.assertEqual(react_tools.validated_sql("SELECT 1 UNION SELECT 2", max_rows=100), "SELECT 1 UNION SELECT 2 LIMIT 101")

    def test_truncated_results_are_identified(self):
        conn = Connection()
        conn.cur.fetchmany = lambda size: [{"mmsi": i} for i in range(size)]
        with patch.object(react_tools, "connection", return_value=conn):
            result = json.loads(react_tools.query_database("SELECT mmsi FROM ship_ais"))
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["rows"]), 100)

    def test_reject_non_readonly_sql(self):
        for sql in ("DELETE FROM ship_ais", "SELECT * FROM ship_ais; DELETE FROM ship_ais", "SELECT * FROM ship_ais FOR UPDATE", "SELECT SLEEP(10)", "SELECT @a := 1", "SELECT @CNOW"):
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                react_tools.validated_sql(sql)

    def test_react_tool_loop_records_result(self):
        model = FakeModel()
        conn = Connection()
        with patch.object(react_tools, "connection", return_value=conn):
            answer, trace = agent.run_agent([{"role": "user", "content": "Which vessel?"}], model=model)
        self.assertEqual(answer, "VESSEL")
        self.assertEqual(trace[0]["name"], "query_database")
        self.assertIn("VESSEL", trace[1]["result"])
        self.assertEqual(len(model.messages), 2)
        self.assertTrue(any(tool.name == "collision_analysis" for tool in model.tools))

    def test_validation_failure_is_returned_to_the_model(self):
        model = FakeModel()
        conn = Connection(fail=True)
        with patch.object(react_tools, "connection", return_value=conn):
            _, trace = agent.run_agent([{"role": "user", "content": "Which vessel?"}], model=model)
        self.assertIn("invalid query", trace[1]["result"])
        self.assertNotIn("SELECT name FROM ship_ais LIMIT 101", conn.cur.calls)
        self.assertIn("invalid query", model.messages[-1][-1].content)
        self.assertEqual(trace[1]["status"], "error")

    def test_entity_search_escapes_like_wildcards(self):
        class EntityCursor(Cursor):
            def execute(self, sql, *args):
                self.calls.append((sql, args))

            def fetchall(self):
                return []

        conn = Connection()
        conn.cur = EntityCursor()
        with patch.object(react_tools, "connection", return_value=conn):
            self.assertEqual(json.loads(react_tools.search_entities("A_B% !")), [])
        selects = [(sql, args) for sql, args in conn.cur.calls if sql.startswith("SELECT DISTINCT")]
        self.assertEqual(len(selects), 6)
        self.assertTrue(all(args == (("%A!_B!% !!%",),) for _, args in selects))
        self.assertTrue(all("ESCAPE '!'" in sql for sql, _ in selects))
        with self.assertRaises(ValueError):
            react_tools.search_entities("  ")


class KnowledgeTests(unittest.TestCase):
    def test_retrieval_does_not_discard_unnormalized_embeddings(self):
        import os
        import tempfile

        from langchain_core.embeddings import Embeddings
        import knowledge

        class FakeEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[10.0, 1.0] for _ in texts]

            def embed_query(self, text):
                return [10.0, 0.0]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rules.txt"
            source.write_text("Relevant rule", encoding="utf-8")
            with patch.dict(os.environ, {"VTS_KNOWLEDGE_DIR": str(Path(directory) / "store"), "OPENAI_API_KEY": "test"}), patch.object(knowledge, "OpenAIEmbeddings", return_value=FakeEmbeddings()):
                knowledge.Knowledge().ingest(source)
                documents = knowledge.Knowledge().search("rule")["documents"]
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["content"], "Relevant rule")
        self.assertAlmostEqual(documents[0]["distance"], 1.0)

    def test_ingest_and_retrieve_persistent_documents(self):
        import os
        import tempfile

        from langchain_core.embeddings import Embeddings
        import knowledge

        class FakeEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, text):
                return [1.0, 0.0]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rules.txt"
            source.write_text("Vessel passage rule", encoding="utf-8")
            with patch.dict(os.environ, {"VTS_KNOWLEDGE_DIR": str(Path(directory) / "store"), "OPENAI_API_KEY": "test"}), patch.object(knowledge, "OpenAIEmbeddings", return_value=FakeEmbeddings()):
                self.assertEqual(knowledge.Knowledge().search("rule")["documents"], [])
                self.assertGreater(knowledge.Knowledge().ingest(source), 0)
                result = knowledge.Knowledge().search("rule")
                self.assertEqual(result["documents"][0]["content"], "Vessel passage rule")
                self.assertEqual(result["documents"][0]["source"], str(source))
                source.write_text("Updated passage rule", encoding="utf-8")
                self.assertEqual(knowledge.Knowledge().ingest(source), 1)
                result = knowledge.Knowledge().search("rule")
                self.assertEqual([doc["content"] for doc in result["documents"]], ["Updated passage rule"])

    def test_pdf_page_is_included_in_search_results(self):
        import os
        import tempfile
        from types import SimpleNamespace

        from langchain_core.embeddings import Embeddings
        import knowledge

        class FakeEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, text):
                return [1.0, 0.0]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rules.pdf"
            source.touch()
            pages = [SimpleNamespace(extract_text=lambda: "Page one"), SimpleNamespace(extract_text=lambda: "Page two")]
            with patch.dict(os.environ, {"VTS_KNOWLEDGE_DIR": str(Path(directory) / "store"), "OPENAI_API_KEY": "test"}), patch.object(knowledge, "OpenAIEmbeddings", return_value=FakeEmbeddings()), patch.object(knowledge, "PdfReader", return_value=SimpleNamespace(pages=pages)):
                self.assertEqual(knowledge.Knowledge().ingest(source), 2)
                result = knowledge.Knowledge().search("rule")
                self.assertEqual({doc["page"] for doc in result["documents"]}, {1, 2})

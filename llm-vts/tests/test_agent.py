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
    def test_historical_schema_reference_is_available_without_database(self):
        tools = {tool.name: tool for tool in react_tools.make_tools(None)}
        with patch.object(react_tools, "connection", side_effect=AssertionError("unexpected database access")):
            reference = json.loads(tools["maritime_schema_reference"].invoke({}))
        self.assertEqual(reference["source"], "maritime-qa/scoder/metadata.sql")
        self.assertIn("ship_ais_quarter", reference["content"])
        self.assertIn("database_schema", tools["maritime_schema_reference"].description)

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
            self.assertEqual(json.loads(react_tools.search_entities("A_B% !")), {"matches": [], "truncated_fields": []})
        selects = [(sql, args) for sql, args in conn.cur.calls if sql.startswith("SELECT DISTINCT")]
        self.assertEqual(len(selects), 6)
        self.assertTrue(all(args == (("%a!_b!% !!%",),) for _, args in selects))
        self.assertTrue(all("WHERE LOWER(" in sql for sql, _ in selects))
        self.assertTrue(all("ESCAPE '!'" in sql for sql, _ in selects))
        with self.assertRaises(ValueError):
            react_tools.search_entities("  ")

    def test_entity_search_reports_truncated_fields(self):
        class EntityCursor(Cursor):
            def execute(self, sql, *args):
                self.calls.append(sql)
                self.sql = sql

            def fetchall(self):
                count = 21 if "`name` AS value FROM `ship_ais`" in self.sql else 20
                return [{"value": str(i)} for i in range(count)]

        conn = Connection()
        conn.cur = EntityCursor()
        with patch.object(react_tools, "connection", return_value=conn):
            result = json.loads(react_tools.search_entities("vessel"))
        self.assertEqual(result["truncated_fields"], [{"table": "ship_ais", "column": "name"}])
        self.assertEqual(len(result["matches"]), 120)
        self.assertTrue(all("ORDER BY" in sql and "LIMIT 21" in sql for sql in conn.cur.calls if sql.startswith("SELECT DISTINCT")))


class KnowledgeTests(unittest.TestCase):
    def test_retrieval_can_continue_past_first_page(self):
        import os
        import tempfile

        from langchain_core.embeddings import Embeddings
        import knowledge

        class FakeEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[float(text.rsplit(" ", 1)[-1]), 0.0] for text in texts]

            def embed_query(self, text):
                return [1.0, 0.0]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"VTS_KNOWLEDGE_DIR": str(root / "store"), "OPENAI_API_KEY": "test"}), patch.object(knowledge, "OpenAIEmbeddings", return_value=FakeEmbeddings()):
                store = knowledge.Knowledge()
                self.assertIsNone(store.search("rule")["next_offset"])
                for number in range(1, 6):
                    source = root / f"rule-{number}.txt"
                    source.write_text(f"Rule {number}", encoding="utf-8")
                    store.ingest(source)
                first = store.search("rule")
                second = store.search("rule", offset=first["next_offset"])
                self.assertEqual(first["next_offset"], 4)
                self.assertIsNone(second["next_offset"])
                self.assertEqual({doc["content"] for doc in first["documents"] + second["documents"]}, {f"Rule {n}" for n in range(1, 6)})
                with self.assertRaises(ValueError):
                    store.search("rule", offset=-1)

    def test_pagination_preserves_equal_distance_results(self):
        import os
        import tempfile

        from langchain_core.embeddings import Embeddings
        import knowledge

        class FakeEmbeddings(Embeddings):
            def __init__(self):
                self.query_count = 0

            def embed_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, text):
                self.query_count += 1
                return [1.0, 0.0]

        embeddings = FakeEmbeddings()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"VTS_KNOWLEDGE_DIR": str(root / "store"), "OPENAI_API_KEY": "test"}), patch.object(knowledge, "OpenAIEmbeddings", return_value=embeddings):
                store = knowledge.Knowledge()
                for number in range(12):
                    source = root / f"rule-{number}.txt"
                    source.write_text(f"Rule {number}", encoding="utf-8")
                    store.ingest(source)
                first = store.search("rule")
                second = store.search("rule", first["next_offset"])
                third = store.search("rule", second["next_offset"])
                contents = [doc["content"] for page in (first, second, third) for doc in page["documents"]]
                self.assertEqual(len(contents), 12)
                self.assertEqual(len(set(contents)), 12)
                self.assertIsNone(third["next_offset"])
                self.assertEqual(embeddings.query_count, 1)

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

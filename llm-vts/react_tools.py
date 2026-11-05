import json
import os

import pymysql
import sqlglot
from langchain_core.tools import StructuredTool
from sqlglot import exp


def connection():
    return pymysql.connect(
        host=os.environ.get("VTS_DB_HOST", "localhost"),
        port=int(os.environ.get("VTS_DB_PORT", "3306")),
        user=os.environ["VTS_DB_USER"],
        password=os.environ["VTS_DB_PASSWORD"],
        database=os.environ["VTS_DB_NAME"],
        connect_timeout=10,
        read_timeout=30,
        cursorclass=pymysql.cursors.DictCursor,
    )


def validated_sql(sql, max_rows=None):
    statements = sqlglot.parse(sql, read="mysql")
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union)):
        raise ValueError("Only one SELECT query is allowed")
    tree = statements[0]
    if any(tree.find(kind) for kind in (exp.Into, exp.Lock, exp.Command, exp.Insert, exp.Update, exp.Delete, exp.PropertyEQ)):
        raise ValueError("Only read-only SELECT queries are allowed")
    if tree.find(exp.Parameter):
        raise ValueError("MySQL session variables are not allowed; use explicit values or subqueries")
    forbidden = {"GET_LOCK", "RELEASE_LOCK", "SLEEP", "BENCHMARK", "LOAD_FILE", "IS_FREE_LOCK", "IS_USED_LOCK"}
    for node in tree.walk():
        if isinstance(node, exp.Func) and node.sql_name().upper() in forbidden:
            raise ValueError("Function is not allowed")
        if isinstance(node, exp.Anonymous) and node.name.upper() in forbidden:
            raise ValueError("Function is not allowed")
    if max_rows is not None:
        limit = tree.args.get("limit")
        if limit is None:
            tree = tree.limit(max_rows + 1)
        else:
            value = limit.expression
            if not isinstance(value, exp.Literal) or value.is_string or not value.this.isdigit():
                raise ValueError("LIMIT must be a non-negative integer")
            if int(value.this) > max_rows + 1:
                tree = tree.limit(max_rows + 1)
    return tree.sql(dialect="mysql")


def run_query(sql, explain=False):
    sql = validated_sql(sql, max_rows=100)
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SET SESSION MAX_EXECUTION_TIME=15000")
            cursor.execute("START TRANSACTION READ ONLY")
            cursor.execute("EXPLAIN " + sql)
            if not explain:
                cursor.execute(sql)
            rows = cursor.fetchmany(101)
            conn.rollback()
    return json.dumps({"validation": "passed", "executed_sql": sql if not explain else None, "rows": rows[:100], "truncated": len(rows) > 100}, default=str, ensure_ascii=False)


def database_schema():
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("START TRANSACTION READ ONLY")
            sql = "SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE, COLUMN_COMMENT FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = DATABASE() ORDER BY TABLE_NAME, ORDINAL_POSITION"
            cursor.execute("EXPLAIN " + sql)
            cursor.execute(sql)
            return json.dumps(cursor.fetchall(), ensure_ascii=False, default=str)


def validate_query(sql: str):
    return run_query(sql, explain=True)


def query_database(sql: str):
    return run_query(sql)


def traffic_flow(sql: str):
    return run_query(sql)


def collision_analysis(sql: str):
    return run_query(sql)


def search_entities(term: str):
    term = term.strip()
    if not term:
        raise ValueError("Entity search term must not be empty")
    term = term.replace("!", "!!").replace("%", "!%").replace("_", "!_")
    fields = {"ship_ais": ("name", "type", "ptms_destination", "mmsi"), "shp_data": ("object_name", "object_type")}
    results = []
    with connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SET SESSION MAX_EXECUTION_TIME=15000")
            cursor.execute("START TRANSACTION READ ONLY")
            for table, columns in fields.items():
                for column in columns:
                    sql = f"SELECT DISTINCT `{column}` AS value FROM `{table}` WHERE `{column}` LIKE %s ESCAPE '!' LIMIT 20"
                    cursor.execute("EXPLAIN " + sql, (f"%{term}%",))
                    cursor.execute(sql, (f"%{term}%",))
                    results.extend({"table": table, "column": column, "value": row["value"]} for row in cursor.fetchall())
            conn.rollback()
    return json.dumps(results, default=str, ensure_ascii=False)


def make_tools(knowledge):
    def search_knowledge(query: str):
        return json.dumps(knowledge.search(query), ensure_ascii=False)

    entries = [
        (database_schema, "database_schema", "Inspect available MySQL tables and columns before writing SQL."),
        (search_entities, "search_entities", "Find vessel or area entities by name, type, destination, or MMSI."),
        (search_knowledge, "search_knowledge", "Retrieve maritime rules and definitions, including their sources."),
        (validate_query, "validate_query", "Validate a read-only MySQL SELECT using EXPLAIN before execution."),
        (query_database, "query_database", "Execute a read-only MySQL SELECT for vessel and geographic data."),
        (traffic_flow, "traffic_flow", "Execute a read-only SQL query for a vessel traffic flow analysis."),
        (collision_analysis, "collision_analysis", "Execute a read-only SQL query for a vessel collision warning analysis."),
    ]
    return [StructuredTool.from_function(func=func, name=name, description=description) for func, name, description in entries]

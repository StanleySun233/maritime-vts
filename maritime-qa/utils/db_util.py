import json
from typing import List, Dict

import mysql.connector
import yaml
import pandas as pd

def load_db_config(config_path="db.yaml"):
    """加载数据库配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def query(sql,config_path="db.yaml"):
    """执行SQL查询"""
    config = load_db_config(config_path)['mysql']
    try:
        conn = mysql.connector.connect(
            host=config['host'],
            port=config['port'],
            user=config['username'],
            password=config['password'],
            database=config['database']
        )
        cursor = conn.cursor()
        # cursor.execute("SET SESSION max_execution_time = 60000")
        cursor.execute(sql)
        # 获取列名
        columns = [desc[0] for desc in cursor.description]
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        # 创建DataFrame时设置列名
        if len(columns) == 0:
            return pd.DataFrame()
        else:
            return pd.DataFrame(result, columns=columns)
    except Exception as e:
        print(e)
        return "ERROR SQL"

def get_database_schema(tables: List[str] = None, config_path="db.yaml") -> Dict[str, Dict]:
    config = load_db_config(config_path)['mysql']

    try:
        conn = mysql.connector.connect(
            host=config['host'],
            port=config['port'],
            user=config['username'],
            password=config['password'],
            database=config['database']
        )

        cursor = conn.cursor()
        result = {}

        # 如果没有指定表，获取所有表
        if tables is None:
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]

        # 获取每个表的列信息和DDL
        for table in tables:
            # 获取列信息
            cursor.execute(f"SHOW FULL COLUMNS FROM {table}")
            columns = [column[0] for column in cursor.fetchall()]

            # 获取DDL
            cursor.execute(f"SHOW CREATE TABLE {table}")
            ddl = cursor.fetchone()[1]

            result[table] = {
                "ddl": ddl,
                "columns": columns,
                "explain": None
            }

        return result

    except Exception as e:
        print(f"获取数据库结构时发生错误: {str(e)}")
        return {}

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    # 测试获取指定表的结构
    test_tables = ['ship_ais', 'ship_ais_hive']
    schema = get_database_schema(test_tables, "./db.yaml")

    # 处理DDL中的换行符
    for table in schema:
        schema[table]["ddl"] = schema[table]["ddl"].replace("\n", " ")

    # 使用json.dumps格式化输出
    print(json.dumps(schema, indent=2, ensure_ascii=False))

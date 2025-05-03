import json
import pprint
from enum import Enum
from typing import Dict, List, Any


class PromptType(Enum):
    BASIC = "basic"
    TEXT = "text"
    OPENAI_DEMONSTRATION = "openai"
    CODE_REPRESENTATION = "code"
    ALPACA_SFT = "alpaca"


class QuestionRepresentation:
    def __init__(self):
        self.COLUMN_PROMPT = """Remark and explain to database for query:\nShip AIS information table\n- `ship_ais`: Stores **real-time** ship position data.\n- `ship_ais_quarter`: Stores ship data for the **past 15 minutes** (10s intervals).\n- ship_ais.point/ship_ais_quarter.point is already a geometry Point object in the correct format. Do not use POINT(s.longitude, s.latitude), just use ship_ais.point or ship_ais_quarter.point But when output, use latitude and longitude as column.\n- \nshp_data table\n- `object_type`: **Geofence type** (`"anchorage"`, `"fairway"`, `"pilot_boarding_ground"`, `"strait"`) \n- `object_name`: **Geofence name** (may have multiple areas)  \n- `geom`: **Spatial data type**\n- `POLYGON`: `fairway`, `anchorage`, `strait` \n- `POINT`: `pilot_boarding_ground`\n warn_sigle table\n- `ship1` / `ship2`: **MMSI of ships**\n- `dt`: **Report timestamp**  """
        self.BASIC_PROMPT = """ENV:MYSQL8.0\nAnswer the following question without any explanation: {QUESTION}\nKnowledge: {KNOWLEDGE}\n{TABLE_COLUMNS}\n{COLUMN_PROMPT}\nA : ```sql SELECT"""
        self.TEXT_REPRESENTATION = """ENV:MYSQL8.0\nGiven the following database schema: \n{TABLE_COLUMNS}\n {COLUMN_PROMPT}\nKnowledge: \n {KNOWLEDGE}\nAnswer the following without any explanation : {QUESTION}\n A: ```sql"""
        self.OPENAI_DEMONSTRATION = """### ENV:MYSQL8.0\n### Complete MYSQL query only and with no explanation\n### MYSQL tables , with their properties: \n# \n{TABLE_COLUMNS}\n{COLUMN_PROMPT}\n#\n# Knowledge:\n  {KNOWLEDGE}\n#\n### {QUESTION}\n A: ```sql"""
        self.CODE_REPRESENTATION = """/*ENV: MYSQL8.0*/\n/* Given the following database schema : */\n{TABLE_COLUMNS}\n{COLUMN_PROMPT}\n/*Knowledge:\n{KNOWLEDGE} */\n/* Answer the following without any explanation: {QUESTION} */\n A: ```sql"""
        self.ALPACA_SFT = """ENV:MYSQL8.0\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction: Write a sql without any explanation to answer the question "{QUESTION}"\n### Input: \n{TABLE_COLUMNS}\n{COLUMN_PROMPT}\n Knowledge: \n{KNOWLEDGE}\n A: ```sql"""

    def question_representation(self, _question: str, _schema: dict, ptype: PromptType,_knowledge:str):
        if ptype == PromptType.BASIC:
            return self.basic_representation(_question, _schema,_knowledge)
        elif ptype == PromptType.TEXT:
            return self.text_representation(_question, _schema,_knowledge)
        elif ptype == PromptType.OPENAI_DEMONSTRATION:
            return self.openai_demonstration(_question, _schema,_knowledge)
        elif ptype == PromptType.CODE_REPRESENTATION:
            return self.code_representation(_question, _schema,_knowledge)
        elif ptype == PromptType.ALPACA_SFT:
            return self.alpaca_sft(_question, _schema,_knowledge)
        else:
            raise ValueError(f"Invalid prompt type. Must be one of {[t.value for t in PromptType]}")

    def basic_representation(self, _question: str, _schema: dict,_knowledge:str):
        _table_columns = ""
        for table in _schema.keys():
            _table_columns += "Table {TABLE}, columns = {COLUMNS}".format(TABLE=table,
                                                                          COLUMNS=", ".join(_schema[table]["columns"]))
            _table_columns += "\n"
        _table_columns = _table_columns[:-1]
        _prompt = self.BASIC_PROMPT.format(QUESTION=_question, TABLE_COLUMNS=_table_columns,KNOWLEDGE=_knowledge,COLUMN_PROMPT=self.COLUMN_PROMPT)
        return _prompt

    def text_representation(self, _question: str, _schema: dict,_knowledge:str):
        _table_columns = ""
        for table in _schema.keys():
            _table_columns += "{TABLE}: {COLUMNS}".format(TABLE=table, COLUMNS=", ".join(_schema[table]["columns"]))
            _table_columns += "\n"
        _table_columns = _table_columns[:-1]
        _prompt = self.TEXT_REPRESENTATION.format(QUESTION=_question, TABLE_COLUMNS=_table_columns,KNOWLEDGE=_knowledge,COLUMN_PROMPT=self.COLUMN_PROMPT)
        return _prompt

    def openai_demonstration(self, _question: str, _schema: dict,_knowledge:str):
        _table_columns = ""
        for table in _schema.keys():
            _table_columns += "# {TABLE} ({COLUMNS})".format(TABLE=table, COLUMNS=", ".join(_schema[table]["columns"]))
            _table_columns += "\n"
        _table_columns = _table_columns[:-1]
        _prompt = self.OPENAI_DEMONSTRATION.format(QUESTION=_question, TABLE_COLUMNS=_table_columns,KNOWLEDGE=_knowledge,COLUMN_PROMPT=self.COLUMN_PROMPT)
        return _prompt

    def code_representation(self, _question: str, _schema: dict,_knowledge:str):
        _table_columns = ""
        for table in _schema.keys():
            _table_columns += "{DDL}".format(DDL=_schema[table]["ddl"])
            _table_columns += "\n\n"
        _table_columns = _table_columns[:-1]
        _prompt = self.CODE_REPRESENTATION.format(QUESTION=_question, TABLE_COLUMNS=_table_columns,KNOWLEDGE=_knowledge,COLUMN_PROMPT=self.COLUMN_PROMPT)
        return _prompt

    def alpaca_sft(self, _question: str, _schema: dict,_knowledge:str):
        _table_columns = ""
        for table in _schema.keys():
            _table_columns += "{TABLE} ({COLUMNS})".format(TABLE=table, COLUMNS=", ".join(_schema[table]["columns"]))
            _table_columns += "\n"
        _table_columns = _table_columns[:-1]
        _prompt = self.ALPACA_SFT.format(QUESTION=_question, TABLE_COLUMNS=_table_columns,KNOWLEDGE=_knowledge,COLUMN_PROMPT=self.COLUMN_PROMPT)
        return _prompt

    def generate_all_representations_from_json(self, json_path: str, schema: Dict, output_path: str = "qr.json"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            results = []

            # 处理训练集和测试集
            for dataset in ['train', 'dev']:
                if dataset in data:
                    for item in data[dataset]:
                        question = item['question']
                        original_query = item['query']
                        knowledge = item['knowledge']

                        # 为每个问题生成所有重构方式
                        representations = {}
                        for ptype in PromptType:
                            try:
                                prompt = self.question_representation(question, schema, ptype, knowledge)
                                representations[ptype.value] = prompt
                            except Exception as e:
                                print(f"Error generating {ptype.value} representation for question: {question}")
                                print(f"Error: {str(e)}")
                                representations[ptype.value] = None

                        # 添加到结果列表
                        results.append({
                            'dataset': dataset,
                            'difficulty': item.get('difficulty', 'unknown'),
                            'original_question': question,
                            'original_query': original_query,
                            'representations': representations,
                            'knowledge': knowledge
                        })

            # 保存结果到JSON文件
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\nResults saved to {output_path}")
            except Exception as e:
                print(f"Error saving results to {output_path}: {str(e)}")

            return results

        except Exception as e:
            print(f"Error processing JSON file: {str(e)}")
            return []


if __name__ == "__main__":
    from utils.db_util import get_database_schema

    test_tables = ['ship_ais', 'ship_ais_quarter', 'shp_data', "warn_single"]
    question_representation = QuestionRepresentation()
    question = "How many continents are there ?"
    schema = get_database_schema(test_tables, "../db.yaml")

    # 测试单个问题的所有提示类型
    print("=== Testing individual question representations ===")
    for ptype in PromptType:
        print(f"\n=== Testing {ptype.value} representation ===")
        try:
            prompt = question_representation.question_representation(question, schema, ptype)
            print(prompt)
        except Exception as e:
            print(f"Error with {ptype.value}: {str(e)}")

    # 测试从JSON文件生成所有重构方式并保存
    print("\n=== Testing JSON file processing ===")
    results = question_representation.generate_all_representations_from_json(
        "../maritime_questions.json",
        schema,
        "../qr.json"  # 默认输出文件名
    )
    pprint.pprint(results)

import json
import random
from enum import Enum
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.llm_factory import generate_answer

REP_TYPE_FOR_QUERY_SIMILARITY = "code"


class InContentType(Enum):
    FULL_INFORMATION = "full"
    SQL_ONLY = "sql"
    DAIL_ORGANIZATION = "dail"


class SelectType(Enum):
    RANDOM = "random"
    QUESTION_SIMILARITY = "qts"
    MASKED_QUESTION_SIMILARITY = "mqs"
    QUERY_SIMILARITY = 'qrs'


class SelectExample:
    def __init__(self, question_path: str, question_representation_path: str = "qr.json",
                 config_path: str = "api.yaml"):
        self.datasets = json.load(open(question_path, 'r', encoding='utf-8'))
        self.question_representation = json.load(open(question_representation_path, 'r', encoding='utf-8'))
        self.CONFIG_PATH = config_path
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
        self.MASKED_PROMPT = """Given a natural language question related to a database, identify and mask all table names, column names, and literal values. Replace table names with table_name, column names with column_name, and values (such as numbers, strings, dates) with value. Keep the rest of the question unchanged.\nExample:\nInput: What is the name of the customer who placed an order on January 5th?\nOutput: What is the name of the table_name who placed an table_name on value?\nQuestion: {QUESTION}\nAnswer With Masked Question Directly:"""
        self.SYSTEM_PROMPT = "You are a helpful assistant that provides accurate and concise answers."
        self.embeddings = []
        self._prepare_embeddings()

    def get_example(self, question: str, stype: SelectType, k: int = 2):
        if stype == SelectType.RANDOM:
            return self.random_select(k)
        elif stype == SelectType.QUESTION_SIMILARITY:
            return self.question_similarity_select(question, k)
        elif stype == SelectType.MASKED_QUESTION_SIMILARITY:
            return self.masked_question_similarity_select(question, k)
        elif stype == SelectType.QUERY_SIMILARITY:
            return self.query_similarity_select(question, k)
        else:
            raise ValueError(f"Invalid select type: {stype}")

    def random_select(self, _, k: int = 2):
        return random.sample(self.datasets["train"], k)

    def question_similarity_select(self, question: str, k: int = 2) -> List[Dict]:
        question_embedding = self._get_embedding(question)

        similarities = []
        for i in self.embeddings:
            if i["question"] == question:
                continue
            similarities.append((self._compute_cosine_similarity(question_embedding, i["question_embedding"]),
                                 i["question"],
                                 i["query"]))
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k = [{"question": i[1], "query": i[2]} for i in similarities[:k]]

        return top_k

    def masked_question_similarity_select(self, question, k=2):
        question_topk = self.question_similarity_select(question, k)
        for i in range(len(question_topk)):
            question_topk[i]["question"] = self._masked_question(question_topk[i]["question"])
        return question_topk

    def query_similarity_select(self, question, k=2):
        question_set = {}
        for i in self.question_representation:
            if i["original_question"] == question:
                question_set = i

        full_question = question_set["representations"][REP_TYPE_FOR_QUERY_SIMILARITY]
        query = generate_answer('gpt_4o',
                                full_question,
                                self.SYSTEM_PROMPT,
                                self.CONFIG_PATH).replace("```sql", "").replace("```", "")

        query_embedding = self._get_embedding(query)
        similarities = []
        for i in self.embeddings:
            if i["question"] == question:
                continue
            similarities.append((self._compute_cosine_similarity(query_embedding, i["query_embedding"]),
                                 i["question"],
                                 i["query"]))
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k = [{"question": i[1], "query": i[2]} for i in similarities[:k]]

        return top_k

    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _get_embedding(self, question: str) -> np.ndarray:
        return self.model.encode(question, convert_to_numpy=True)

    def _prepare_embeddings(self):
        for i in self.datasets["train"]:
            self.embeddings.append(i)
            self.embeddings[-1]["question_embedding"] = self._get_embedding(i["question"])
            self.embeddings[-1]["query_embedding"] = self._get_embedding(i["query"])

    def _masked_question(self, question, model='gpt_4o', ):
        return generate_answer(model, self.MASKED_PROMPT.format(QUESTION=question), self.SYSTEM_PROMPT,
                               self.CONFIG_PATH)


class InContentLearning:
    def __init__(self, question_path: str = "maritime_questions.json", question_representation_path: str = "qr.json",
                 config_path: str = "api.yaml", k: int = 2):
        self.k = k
        self.selector = SelectExample(question_path=question_path,
                                      question_representation_path=question_representation_path,
                                      config_path=config_path)

        self.FULL_INFORMATION = """/* Given the following database schema: """
        self.SQL_ONLY = """/* Some SQL examples are provided based on similar problems : */\n{RELATED_QUERYS}"""
        self.DAIL_ORGANIZATION = """/* Some example questions and corresponding SQL queries are provided based on similar problems: */\n{RELATED_QUESTIONS}"""

    def generate_in_content_learning(self, question: str, schema: dict, in_content_type: InContentType,
                                     select_type: SelectType) -> str:
        if in_content_type == InContentType.FULL_INFORMATION:
            return self.generate_full_information(question, schema, select_type)
        elif in_content_type == InContentType.SQL_ONLY:
            return self.generate_sql_only(question, schema, select_type)
        elif in_content_type == InContentType.DAIL_ORGANIZATION:
            return self.generate_dail_organization(question, schema, select_type)
        else:
            raise ValueError(f"Invalid in-content type: {in_content_type}")

    def generate_full_information(self, question: str, schema: dict, select_type: SelectType) -> str:
        raise NotImplementedError("Full information generation is not implemented yet.")

    def generate_sql_only(self, question: str, schema: dict, select_type: SelectType) -> str:
        examples = self.selector.get_example(question, select_type, self.k)
        related_questions = "\n".join([f"Question: {item['question']}\nSQL: {item['query']}" for item in examples])
        return self.SQL_ONLY.format(RELATED_QUERYS=related_questions)

    def generate_dail_organization(self, question: str, schema: dict, select_type: SelectType) -> str:
        if self.k > 1:
            k = self.k // 2
        else:
            k = 1
        example_question = self.selector.get_example(question, SelectType('mqs'), k)
        example_query = self.selector.get_example(question, SelectType('qrs'), k)
        examples = example_question + example_query
        related_questions = "\n".join([f"Question: {item['question']}\nSQL: {item['query']}" for item in examples])
        return self.DAIL_ORGANIZATION.format(RELATED_QUESTIONS=related_questions)

    def generate_all_icl_from_json(self, schema: dict, question_representation_path: str,
                                   output_path: str = "icl.json") -> List[Dict]:
        icl_selection = ["sql", 'dail']
        sample_selection = ["random", "qts", "mqs", "qrs"]
        question_representation = json.load(open(question_representation_path, 'r', encoding='utf-8'))

        for i in range(len(question_representation)):
            question_representation[i]['icl'] = {}
            for icl in icl_selection:
                question_representation[i]['icl'][icl] = {}
                for sample in sample_selection:
                    question = question_representation[i]["original_question"]
                    icl_content = self.generate_in_content_learning(question, schema, InContentType(icl),
                                                                    SelectType(sample))
                    question_representation[i]['icl'][icl][sample] = icl_content

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(question_representation, f, ensure_ascii=False, indent=4)
        return question_representation


if __name__ == "__main__":
    from utils.db_util import get_database_schema

    icl = InContentLearning(question_path="../maritime_questions.json",
                            question_representation_path="../qr.json",
                            config_path="../api.yaml",
                            k=2)

    schema = get_database_schema(['ship_ais', 'ship_ais_quarter', 'shp_data', "warn_single"], "../db.yaml")

    icl.generate_all_icl_from_json(schema=schema,
                                   question_representation_path="../qr.json",
                                   output_path="../icl.json")

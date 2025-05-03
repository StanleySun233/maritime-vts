import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Function, Comparison
from sqlparse.tokens import Keyword, DML

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class SQLComparer:
    # 类变量，用于存储模型和分词器
    _tokenizer = None
    _model = None

    @classmethod
    def _initialize_model(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained("azg-azg/SQLBert")
            cls._model = AutoModel.from_pretrained("azg-azg/SQLBert")

    def __init__(self, sql):
        self.sql = sql
        self.sql_info = {
            'select': [],
            'from': [],
            'joins': [],
            'where': [],
            'group_by': [],
            'having': [],
            'order_by': [],
            'limit': [],
            'functions': []
        }
        self._parse_sql()
        # 确保模型已初始化
        self._initialize_model()

    @staticmethod
    def normalize_token(token):
        return str(token).strip().lower()

    def extract_functions(self, token):
        functions = []
        if isinstance(token, Function):
            functions.append(self.normalize_token(token))
        elif isinstance(token, IdentifierList):
            for t in token.get_identifiers():
                if isinstance(t, Function):
                    functions.append(self.normalize_token(t))
        return functions

    def extract_identifiers(self, token):
        identifiers = []
        if isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                identifiers.append(self.normalize_token(identifier))
        elif isinstance(token, (Identifier, Function, Comparison)):
            identifiers.append(self.normalize_token(token))
        return identifiers

    def _parse_sql(self):
        parsed = sqlparse.parse(self.sql)[0]
        tokens = [token for token in parsed.tokens if not token.is_whitespace]

        clause = None
        for token in tokens:
            if token.ttype is DML and token.value.upper() == 'SELECT':
                clause = 'select'
            elif token.match(Keyword, 'FROM'):
                clause = 'from'
            elif token.match(Keyword, 'JOIN') or token.match(Keyword, 'INNER JOIN') or token.match(Keyword, 'LEFT JOIN'):
                clause = 'joins'
            elif token.match(Keyword, 'WHERE'):
                clause = 'where'
            elif token.match(Keyword, 'GROUP BY'):
                clause = 'group_by'
            elif token.match(Keyword, 'HAVING'):
                clause = 'having'
            elif token.match(Keyword, 'ORDER BY'):
                clause = 'order_by'
            elif token.match(Keyword, 'LIMIT'):
                clause = 'limit'

            if clause == 'select':
                self.sql_info['select'].extend(self.extract_identifiers(token))
                self.sql_info['functions'].extend(self.extract_functions(token))
            elif clause == 'from':
                self.sql_info['from'].append(self.normalize_token(token))
            elif clause == 'joins':
                self.sql_info['joins'].append(self.normalize_token(token))
            elif clause == 'where':
                self.sql_info['where'].append(self.normalize_token(token))
                self.sql_info['functions'].extend(self.extract_functions(token))
            elif clause == 'group_by':
                self.sql_info['group_by'].append(self.normalize_token(token))
            elif clause == 'having':
                self.sql_info['having'].append(self.normalize_token(token))
            elif clause == 'order_by':
                self.sql_info['order_by'].append(self.normalize_token(token))
            elif clause == 'limit':
                self.sql_info['limit'].append(self.normalize_token(token))

        for key in self.sql_info:
            self.sql_info[key] = sorted(set(self.sql_info[key]))

    def get_structure(self):
        return self.sql_info

    def compare_with(self, other_sql_comparer):
        if self.get_structure() == other_sql_comparer.get_structure():
            return 1  # 完全结构等价

        # 如果结构不相等，返回 CodeBERT 相似度
        return self.semantic_similarity(self.sql, other_sql_comparer.sql)

    @staticmethod
    def semantic_similarity(sql1, sql2):
        # 使用类变量中的模型和分词器
        def encode(text):
            inputs = SQLComparer._tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = SQLComparer._model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.numpy()

        vec1 = encode(sql1)
        vec2 = encode(sql2)

        similarity = cosine_similarity(vec1, vec2)[0][0]
        return float(similarity)  # 返回相似度分数




# 示例使用
if __name__ == "__main__":
    sql_a = """
SELECT sog, latitude, longitude FROM ship_ais WHERE name = 'KPSB 5' ORDER BY last_updated DESC LIMIT 1
    """

    sql_b = """
SELECT sog, latitude, longitude
FROM ship_ais
WHERE name = 'KPSB 5';
    """

    comparer_a = SQLComparer(sql_a,"../db.yaml")
    comparer_b = SQLComparer(sql_b,"../db.yaml")

    result = comparer_a.compare_with(comparer_b)
    print(f"相似度结果：{result}")
from module.in_content_learning import InContentLearning
from module.question_representation import QuestionRepresentation
from utils.db_util import get_database_schema

REPRESENTATION_TYPE = "code"
ICL_TYPE = "dail"

tables = ['ship_ais', 'ship_ais_quarter', 'shp_data', "warn_single"]
qr = QuestionRepresentation()

schema = get_database_schema(tables, "./db.yaml")

qr_data = qr.generate_all_representations_from_json(
    "./data/dataset.json",
    schema,
    "./data/qr.json"
)

# 初始化
icl = InContentLearning(question_path="./data/dataset.json",
                        question_representation_path="./data/qr.json",
                        config_path="./api.yaml",
                        k=2)

icl_data = icl.generate_all_icl_from_json(schema=schema,
                                          question_representation_path="./data/qr.json",
                                          output_path="./data/icl.json")

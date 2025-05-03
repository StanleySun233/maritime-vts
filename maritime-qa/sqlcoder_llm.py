import pandas as pd
import json
from utils.llm_factory import generate_answer
from utils.ragflow import get_answer_by_agent_id
from utils.db_util import query
from utils.sql_compare import compare_sql_results
from prompt.sql_prompts import SQL_TEST_SYSTEM

from scoder.inference import InferenceRunner

js = json.loads(open("./data/dataset.json").read())
dataset = []
for i in js:
    if i["dataset"] == 'train':
        dataset.append({
            "original_question": i["original_question"],
            "answer": i["original_query"],
            "knowledge": i["knowledge"]
        })
df = pd.DataFrame(dataset)
ans = []
bl = []

inference = InferenceRunner()
for idx, row in df.iterrows():
    origin_question = row["original_question"]
    answer = row["answer"].replace("NOW()", '(SELECT max(temp.last_updated) FROM ship_ais as temp)').replace("@CNOW", "(SELECT max(temp.last_updated) FROM ship_ais as temp)")
    
    # 使用RAGFlow模型生成SQL
    llm_answer = inference.run_inference(origin_question,knowledge=row["knowledge"],prompt_file="./scoder/prompt.md",metadata_file="./scoder/metadata.sql",instruction_file="./scoder/instruction.md").replace("```sql", "").replace("```", "")
    
    llm_answer = llm_answer.replace("NOW()", '(SELECT max(temp.last_updated) FROM ship_ais as temp)').replace("@CNOW", "(SELECT max(temp.last_updated) FROM ship_ais as temp)")
    
    # 替换NOW()函数
    answer = answer.replace("NOW()", '(SELECT max(temp.last_updated) FROM ship_ais as temp)').replace("@CNOW", "(SELECT max(temp.last_updated) FROM ship_ais as temp)")

    # 执行SQL查询
    answer_query = query(answer, "db.yaml")
    llm_answer_query = query(llm_answer, "db.yaml")
    
    # 比较结果
    resp, remark = compare_sql_results(answer_query, llm_answer_query)
    
    # 打印详细信息
    print(f"========= Question =========")
    print(f"{origin_question}")
    print(f"========= SQLCODER Generated SQL =========")
    print(f"{llm_answer}")
    print(f"========= Original SQL =========")
    print(f"{answer}")
    print(f"========= SQLCODER Query Result =========")
    print(f"{'Error/No Result SQL' if isinstance(llm_answer_query, str) else llm_answer_query.head(10).to_markdown()}")
    print(f"========= Original SQL Query Result =========")
    print(f"{'Error/No Result SQL' if isinstance(answer_query, str) else answer_query.head(10).to_markdown()}")
    print(f"========= Judgment Result =========")
    print(f"{resp}")
    print(f"========= Remark =========")
    print(f"{remark}")
    
    ans.append(llm_answer)
    bl.append(resp)

df["llm_answer"] = ans
df["llm_judge"] = bl
print(f"Model: SQLCODER")
print(df["llm_judge"].describe())
df.to_csv(f"sqlcoder_llm.csv", index=False) 
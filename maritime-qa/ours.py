import pandas as pd
import json
from utils.llm_factory import generate_answer
from utils.ragflow import get_answer_by_agent_id
from utils.db_util import query
from utils.sql_compare import compare_sql_results
from prompt.sql_prompts import SQL_TEST_SYSTEM
import yaml
cfg = yaml.load(open("./api.yaml").read())

api_key = cfg["OURS"]["api_key"]
agent_id = cfg["OURS"]["agent_id"]
base_url = cfg["OURS"]["base_url"]

js = json.loads(open("./data/qr_test.json").read())
dataset = []

for i in js:
    if i["dataset"] == 'train':
        dataset.append({
            "original_question": i["original_question"],
            "answer": i["original_query"]
        })

df = pd.DataFrame(dataset)
ans = []
bl = []


for idx, row in df.iterrows():
    origin_question = row["original_question"]
    answer = row["answer"].replace("NOW()", '(SELECT max(temp.last_updated) FROM ship_ais as temp)').replace("@CNOW", "(SELECT max(temp.last_updated) FROM ship_ais as temp)")
    
    llm_answer = get_answer_by_agent_id(base_url, api_key, agent_id, origin_question).replace("```sql", "").replace("```", "")
    
    llm_answer = llm_answer.replace("NOW()", '(SELECT max(temp.last_updated) FROM ship_ais as temp)').replace("@CNOW", "(SELECT max(temp.last_updated) FROM ship_ais as temp)")
    
    answer = answer.replace("NOW()", '(SELECT max(temp.last_updated) FROM ship_ais as temp)').replace("@CNOW", "(SELECT max(temp.last_updated) FROM ship_ais as temp)")

    answer_query = query(answer, "db.yaml")
    llm_answer_query = query(llm_answer, "db.yaml")
    
    resp, remark = compare_sql_results(answer_query, llm_answer_query)
    
    print(f"========= Question =========")
    print(f"{origin_question}")
    print(f"========= Ours generated SQL =========")
    print(f"{llm_answer}")
    print(f"========= Original SQL =========")
    print(f"{answer}")
    print(f"========= RAGFlow Query Result =========")
    print(f"{'Error/No Result SQL' if isinstance(llm_answer_query, str) else llm_answer_query.head(10).to_markdown()}")
    print(f"========= Original SQL Query Result =========")
    print(f"{'Error/No Result SQL' if isinstance(answer_query, str) else answer_query.head(10).to_markdown()}")
    print(f"========= Judge Result =========")
    print(f"{resp}")
    print(f"========= Remark =========")
    print(f"{remark}")
    
    ans.append(llm_answer)
    bl.append(resp)

# 保存结果
df["llm_answer"] = ans
df["llm_judge"] = bl
print(f"Model: OURS")
print(df["llm_judge"].describe())
df.to_csv(f"./ours.csv", index=False) 
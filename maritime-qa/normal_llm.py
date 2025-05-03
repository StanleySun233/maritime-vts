import pandas as pd
import os
from datetime import datetime
import re
import time
from utils.llm_factory import generate_answer
from utils.db_util import query
from utils.sql_compare import compare_sql_results
import json
from eval.code_similarity import SQLComparer
from prompt.sql_prompts import SQL_SYSTEM_PROMPT, SQL_COMPARISON_PROMPT, SQL_TEST_SYSTEM

# select_types = ['basic','text','openai','code','alpaca']
# models = ['qwen7b','qwen14b','qwen32b','qwencoder7b','qwencoder14b','qwencoder32b','deepseek_coder16b','deepseekv3-full']
# models = ['gemini_2.0','gemini_1.5pro','gemini_1.5']
# models = ['gemini_2.0','gemini_1.5pro','gemini_1.5']
# models = ['claude3.5','claude3.7','claude3.7_thinking']
# models = ['qwen2.5_full','qwenmax']
# models = ['deepseekv3-full']
# models = ['llama3.1_70b','llama3.3_70b']
# models = ['duck7b']
# models = ['qwen2.5_32b_coder','qwen2.5_7b_coder']
# models = ['claude3.5','claude3.7']
models = ['sqlcoder15b']
select_types = ['basic','text','openai','code','alpaca']


js = json.loads(open("./data/dataset.json",encoding="utf-8").read())

save_dir = "./data/save"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

result_file = "./data/result.csv"
if os.path.exists(result_file):
    result_df = pd.read_csv(result_file)
else:
    result_df = pd.DataFrame(columns=["model", "type", "llm_judge_mean", "updated_time"])

def extract_sql_code(text):
    sql_pattern = r'```sql\s*(.*?)\s*```'
    sql_match = re.search(sql_pattern, text, re.DOTALL)
    
    if sql_match:
        return sql_match.group(1).strip()
    
    code_pattern = r'```\s*(.*?)\s*```'
    code_match = re.search(code_pattern, text, re.DOTALL)
    
    if code_match:
        return code_match.group(1).strip()
    
    return text

for model in models:

    for select_type in select_types:
        dataset = []

        for i in js:
            if i["dataset"] == 'train':
                dataset.append({"original_question": i["original_question"],
                                "question": i["representations"][select_type],
                                "answer": i["original_query"]})

        df = pd.DataFrame(dataset)
        sims = []
        ans = []
        bl = []
        remarks = []
        for idx, row in df.iterrows():
            origin_question = row["original_question"]
            question = row["question"].replace("```sql SELECT","(USE```sql as begin and ``` as end.)")
            answer = row["answer"].replace("NOW()", '(SELECT max(temp.last_updated) FROM ship_ais as temp)')
            answer = answer.replace("@CNOW", '(SELECT max(temp.last_updated) FROM ship_ais as temp)')
            llm_answer = generate_answer(model,
                                         question,
                                         SQL_TEST_SYSTEM,
                                         'api.yaml')
            
            llm_answer = extract_sql_code(llm_answer)
            
            llm_answer = llm_answer.replace("NOW()", '(SELECT max(temp.last_updated) FROM ship_ais as temp)')
            llm_answer = llm_answer.replace("@CNOW", '(SELECT max(temp.last_updated) FROM ship_ais as temp)')

            answer_query = query(answer, "db.yaml")
            llm_answer_query = query(llm_answer, "db.yaml")
            
            resp, remark = compare_sql_results(answer_query, llm_answer_query)
            print(f"========= Question =========")
            print(f"{origin_question}")
            print(f"========= Input =========")
            print(f"{question}")
            print(f"========= LLM Generated SQL =========")
            print(f"{llm_answer}")
            print(f"========= Original SQL =========")
            print(f"{answer}")
            print(f"========= LLM Query Result =========")
            print(f"{'Error/No Result SQL' if isinstance(llm_answer_query, str) else llm_answer_query.head(10).to_markdown()}")
            print(f"========= Original SQL Query Result =========")
            print(f"{'Error/No Result SQL' if isinstance(answer_query, str) else answer_query.head(10).to_markdown()}")
            print(f"========= Judgment Result =========")
            print(f"{resp}")
            print(f"========= Remark =========")
            print(f"{remark}")
            
            ans.append(llm_answer)
            bl.append(resp)
            remarks.append(remark)

        df["llm_answer"] = ans
        df["llm_judge"] = bl
        df["remark"] = remarks
        print(model,select_type)
        print(df["llm_judge"].describe())
        df.to_csv(f"{save_dir}/{model}_{select_type}.csv", index=False)
        
        llm_judge_mean = df["llm_judge"].mean()
        updated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        existing_idx = result_df[(result_df["model"] == model) & (result_df["type"] == select_type)].index
        
        if len(existing_idx) > 0:
            result_df.loc[existing_idx, "llm_judge_mean"] = llm_judge_mean
            result_df.loc[existing_idx, "updated_time"] = updated_time
        else:
            new_row = pd.DataFrame({
                "model": [model],
                "type": [select_type],
                "llm_judge_mean": [llm_judge_mean],
                "updated_time": [updated_time]
            })
            result_df = pd.concat([result_df, new_row], ignore_index=True)
        
        result_df.to_csv(result_file, index=False)
        print(f"Results saved to {result_file}")
import pandas as pd
import os
from datetime import datetime
import re
from utils.llm_factory import generate_answer
from utils.db_util import query
from utils.sql_compare import compare_sql_results
import json
from eval.code_similarity import SQLComparer
from prompt.sql_prompts import SQL_SYSTEM_PROMPT, SQL_COMPARISON_PROMPT, SQL_TEST_SYSTEM
from tqdm import tqdm

# models = ['deepseekv3-full']
models = ['claude3.7','claude3.5']

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

def process_single_query(model, question, answer, origin_question):
    llm_answer = generate_answer(model, question, SQL_TEST_SYSTEM, 'api.yaml')
    llm_answer = extract_sql_code(llm_answer)
    
    llm_answer = llm_answer.replace("NOW()", '(SELECT max(temp.last_updated) FROM ship_ais as temp)')
    llm_answer = llm_answer.replace("@CNOW", '(SELECT max(temp.last_updated) FROM ship_ais as temp)')

    answer_query = query(answer, "db.yaml")
    llm_answer_query = query(llm_answer, "db.yaml")
    
    resp, remark = compare_sql_results(answer_query, llm_answer_query)
    
    return llm_answer, resp, remark, llm_answer_query

def main():
    save_dir = "./save"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result_file = "./result.csv"
    if os.path.exists(result_file):
        result_df = pd.read_csv(result_file)
    else:
        result_df = pd.DataFrame(columns=["model", "llm_judge_mean", "updated_time"])

    js = json.loads(open("./data/dataset.json",encoding="utf-8").read())

    for model in models:
        dataset = []
        for i in js:
            if i["dataset"] == 'train':
                dataset.append({
                    "original_question": i["original_question"],
                    "question": i["llm_question"],
                    "answer": i["original_query"]
                })

        df = pd.DataFrame(dataset)
        ans = []
        bl = []
        remarks = []

        pbar = tqdm(df.iterrows(), total=len(df), desc=f"Processing model {model}")
        
        for idx, row in pbar:
            origin_question = row["original_question"]
            question = row["question"].replace("```sql SELECT","(USE```sql as begin and ``` as end.)")
            answer = row["answer"].replace("NOW()", '(SELECT max(temp.last_updated) FROM ship_ais as temp)')
            answer = answer.replace("@CNOW", '(SELECT max(temp.last_updated) FROM ship_ais as temp)')

            llm_answer, resp, remark, llm_answer_query = process_single_query(
                model, question, answer, origin_question
            )
            
            ans.append(llm_answer)
            bl.append(resp)
            remarks.append(remark)

            current_accuracy = (sum(bl) / len(bl)) * 100
            pbar.set_description(f"Model {model} [Accuracy: {current_accuracy:.2f}%]")

        df["llm_answer"] = ans
        df["llm_judge"] = bl
        df["remark"] = remarks
        
        print(f"Model {model} result statistics:")
        print(df["llm_judge"].describe())
        
        # 保存到CSV
        df.to_csv(f"{save_dir}/dail_sql_{model}_improved.csv", index=False)
        
        # 更新result.csv
        llm_judge_mean = df["llm_judge"].mean()
        updated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_row = pd.DataFrame({
            "model": [model],
            "llm_judge_mean": [llm_judge_mean],
            "updated_time": [updated_time]
        })
        result_df = pd.concat([result_df, new_row], ignore_index=True)
        result_df.to_csv(result_file, index=False)
        print(f"Results saved to {result_file}")

if __name__ == "__main__":
    main()
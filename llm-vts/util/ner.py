import pandas as pd
import pymysql

import util


# TODO: update what is the type.
# VLCC ->
def query_entity_locations(entity_list, eng):
    check_field = {
        "ship_ais": ['name', 'type', 'ptms_destination', 'mmsi'],
        "shp_data": ["object_name", 'object_type']
    }
    lst = []
    for i in check_field.keys():
        for j in check_field[i]:
            for k in entity_list:
                statement = f"select distinct {j} from {i} where lower({j}) like '%{k.lower()}%'"
                cursor = eng.cursor()
                cursor.execute(statement)
                res = cursor.fetchall()
                if res is not None:
                    for v in range(len(res)):
                        lst.append({"entity": res[v][0], "table": i, "column": j})
    result_df = pd.DataFrame(lst)
    if len(result_df) == 0:
        return "no data"

    if len(result_df) > 20:
        result_df = result_df.head(20)
        markdown_output = result_df.to_markdown(index=False)
        markdown_output += "\nOnly the first 20 entities are retained."
    else:
        markdown_output = result_df.to_markdown(index=False)

    return markdown_output


def query_sql(sql, eng):
    cur = eng.cursor()
    cur.execute(sql)
    res = cur.fetchall()
    column_names = [desc[0] for desc in cur.description]
    print(column_names)
    df = pd.DataFrame(res, columns=column_names)
    print(df.head(1).to_markdown())
    if len(df) == 0:
        return "no data"

    if len(df) > 10:
        # 只保留前 10 行，并添加提示信息
        df = df.head(10)
        markdown_output = df.to_markdown(index=False)
        # markdown_output += "\nThe result is too long, only the first 20 results are retained."
        return markdown_output
    else:
        return df.to_markdown(index=False)


def query_sql_valid(sql, eng):
    try:
        with eng.cursor() as cursor:
            sql = "explain " + sql
            cursor.execute(sql)
        return "SUCCESS"
    except pymysql.MySQLError as e:
        return f"<ERROR>{e}"


def fix_sql(sql, question, error_text, cfg):
    template = """The user's SQL code has an error. 
    Please repair the SQL code according to the error.
     Please do not output any explanation or comments on the repair. 
     Please directly output the executable SQL code.\n
                The user asked question and a sql:
                {rethink_input}
                But error occurs:
                {error_message}
                When querying the code:
                {sql_input}""".format(rethink_input=question, error_message=error_text, sql_input=sql)
    resp = util.get_answer_by_agent_id(cfg, agent_id=cfg["api"]["sql_debugger"]["agent_id"], question=template)['data'][
        'answer']
    return resp


def question_rethink(question, cfg):
    resp = util.get_answer_by_agent_id(cfg,
                                       agent_id=cfg["api"]["question_rethink"]["agent_id"],
                                       question=question)['data']['answer']
    return resp


def ask_knowledge_by_keyword(keyword, cfg):
    template = """What is {keyword}?""".format(keyword=keyword)
    resp = util.get_answer_by_agent_id(cfg,
                                       agent_id=cfg["api"]["keyword_knowledge"]["agent_id"],
                                       question=template)['data']['answer']
    return resp
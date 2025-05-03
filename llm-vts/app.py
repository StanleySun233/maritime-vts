import datetime

import pandas as pd
import pymysql
from flask import Flask, request, jsonify

import util

app = Flask(__name__)

DATE_TIME = datetime.datetime.now()


def get_connection(_cfg):
    db = pymysql.Connection(host=_cfg["mysql"]['host'],
                            user=_cfg["mysql"]['user'],
                            port=int(_cfg["mysql"]['port']),
                            password=_cfg["mysql"]['password'],
                            database=_cfg["mysql"]['database'], )
    return db


@app.route('/set_time', methods=['POST'])
def set_time():
    global DATE_TIME
    data = request.json
    dt = data.get('dt')
    try:
        DATE_TIME = datetime.datetime.fromisoformat(dt)
        return jsonify({"message": "DATE_TIME updated successfully"}), 200
    except ValueError:
        return jsonify({"error": "Invalid date-time format"}), 400


@app.route('/get_time', methods=['POST'])
def get_time():
    return jsonify({"DATE_TIME": DATE_TIME.isoformat()}), 200


@app.route('/traffic_flow', methods=['POST'])
def traffic_flow():
    data = request.json
    sql = data.get('sql')
    try:
        conn = get_connection(cfg)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return jsonify({"data": df.to_dict(orient='records')}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/collision_analysis', methods=['POST'])
def collision_analysis():
    data = request.json
    sql_text = data.get('sql')
    pad_sql = sql_text.replace("```sql", "").replace("```", "")
    print(pad_sql)
    try:
        result = util.query_sql(pad_sql, get_connection(cfg))
        return result
    except Exception as e:
        return "<ERROR>" + str(e)


@app.route('/data_visualization', methods=['POST'])
def data_visualization():
    data = request.json
    sql_statment = data.get('sql_statment')
    code = data.get('code')
    try:
        conn = get_connection(cfg)
        df = pd.read_sql_query(sql_statment, conn)
        conn.close()
        exec_globals = {'df': df}
        exec(code, exec_globals)
        return jsonify({"data": exec_globals.get('result', df).to_dict(orient='records')}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ner', methods=['POST'])
def ner():
    data = request.json
    text = data.get('entity')
    print(text)
    if text is None or text == '':
        return "no entities"
    entity = text.split(",")
    result: str = util.query_entity_locations(entity, get_connection(cfg))
    return result


@app.route('/sql', methods=['POST'])
def sql():
    data = request.json
    sql_text = data.get('sql')
    print(sql_text)
    pad_sql = sql_text.replace("```sql", "").replace("```", "")
    pad_sql = pad_sql.replace("@CNOW",'(SELECT max(temp.last_updated) FROM ship_ais as temp)')
    try:
        sql_result = util.query_sql(pad_sql, get_connection(cfg))
        return sql_result
    except Exception as e:
        print(pad_sql)
        print("<ERROR>" + str(e))
        return "<ERROR>" + str(e)


@app.route('/check_sql', methods=['POST', "GET"])
def check_sql():
    data = request.json
    sql_text = data.get('sql')
    pad_sql = sql_text.replace("```sql", "").replace("```", "")
    sql_result = util.query_sql_valid(pad_sql, get_connection(cfg))
    print(sql_result)
    return sql_result


@app.route('/fix_sql', methods=['POST', "GET"])
def fix_sql():
    data = request.json
    question_text = data.get('question')
    sql_text = data.get('sql')
    pad_sql = sql_text.replace("```sql", "").replace("```", "")
    error_text = data.get('error')
    fix_time = data.get('fix_time', 5)
    try:
        fix_time = int(fix_time)
    except:
        fix_time = 5
    for i in range(fix_time):

        fixed_sql = util.fix_sql(pad_sql, question_text, error_text, cfg).replace("```sql", "").replace("```", "")
        fix_check = util.query_sql_valid(fixed_sql, get_connection(cfg))
        if "SUCCESS" in fix_check:
            print("sql fix success at {}".format(i + 1))
            break
        else:
            print("sql fix failed at {}\n".format(i + 1), fixed_sql)
            error_text = fix_check
            pad_sql = fixed_sql
    else:
        fixed_sql = "select 'retry too many times' as result from ship_ais limit 1 "
    print(fixed_sql)
    return fixed_sql


@app.route('/question_rethink', methods=['POST', "GET"])
def question_rethink():
    data = request.json
    question_text = data.get('question')
    rethink_output = util.question_rethink(question_text, cfg)
    print(rethink_output)
    return rethink_output


@app.route('/entity_knowledge', methods=['POST', "GET"])
def entity_knowledge():
    data = request.json
    question_text = data.get('entity').split(",")
    result = []
    for i in question_text:
        keyword = util.ask_knowledge_by_keyword(i, cfg)
        result.append(keyword)
    print(result)
    return "/n".join(result)


@app.route('/knowledge_to_relation', methods=['POST', "GET"])
def knowledge_to_relation():
    data = request.json
    question_text = data.get('knowledge').split(",")
    result = []
    for i in question_text:
        keyword = util.ask_knowledge_by_keyword(i, cfg)
        result.append(keyword)
    print(result)
    return "/n".join(result)


@app.route('/union_result', methods=['POST', "GET"])
def union_result():
    data = request.json
    res1 = data.get('result1', "")
    res2 = data.get('result2', "")
    if res2 in [None, ""]:
        res2 = ""
    print([res1, res2])
    if res2 != "":
        return res2
    else:
        return res1


if __name__ == '__main__':
    cfg = util.load_config()
    app.run(debug=cfg["flask"]["debug"], host=cfg["flask"]["host"], port=cfg["flask"]["port"])

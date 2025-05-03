import json

icl = json.load(open("./icl_test.json"))

for i in range(len(icl)):
    icl[i]["llm_question"] = icl[i]["representations"]["code"] + "\n" + icl[i]["icl"]["dail"]["qrs"]
    print(icl[i]["llm_question"])
with open("./dail_sql.json", "w", encoding="utf-8") as f:
    json.dump(icl, f, ensure_ascii=False, indent=4)

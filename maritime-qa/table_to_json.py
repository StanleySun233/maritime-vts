import pandas as pd
import json


df = pd.read_csv("./data/maritime_questions.csv")
df["Knowledge"] = df["Knowledge"].fillna("")
df = df[df["Status"] == "Activate"]

print(df.head())


type_list = ["VTS-style Query / Command-style Query","Standard natural","Operational style"]
select_type = type_list[2]

data ={"train":[],"dev":[]}
for index, row in df.iterrows():
    data["train"].append({"difficulty":row["Difficulty"],"question":row[select_type],"query":row["Golden Answer"],'knowledge':row["Knowledge"]})
    data["dev"].append({"difficulty":row["Difficulty"],"question":row[select_type],"query":row["Golden Answer"],'knowledge':row["Knowledge"]})

json.dump(data,open("./data/test_dataset.json","w",encoding='utf-8'),indent=2,ensure_ascii=False)
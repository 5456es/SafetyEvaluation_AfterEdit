import pandas as pd
import os
import json

merged = None
for datafile in os.listdir("./paraphrased_data"):

    data = pd.read_csv(f"./paraphrased_data/{datafile}")
    if merged is None:
        merged = data
    else:
        merged = pd.concat([merged, data])


print(f"Merged {len(merged)} records.")

merged["json"] = merged["Paraphrase"].apply(json.loads)
assert merged["json"].apply(lambda x: isinstance(x, dict)).all()


def supplement_material(entry):
    
    if "ground_truth" not in entry:
        entry["ground_truth"] ={
        }
    if "portablility" not in entry:
        entry["portablility"] ={
        }
    if "locality" not in entry:
        entry["locality"] ={
        }
    if "source" not in entry:
        entry["source"] = "NEWS2024"
    return entry
merged["json"].apply(supplement_material)

### save as json
merged["json"].to_json('2024_news_paraphrased.json',orient='records',indent=2)    
    


import json


### three datasets is
# ./benchmark/wiki_counterfact/train_cf.json
# ./benchmark/ZsRE/ZsRE-test-all.json
# ./benchmark/wiki_recent/recent_train.json


datasets=['./benchmark/wiki_counterfact/train_cf.json','./benchmark/ZsRE/ZsRE-test-all.json','./benchmark/wiki_recent/recent_train.json']


merged_data = []
for dataset in datasets:
    with open(dataset) as f:
        data = json.load(f)
        for entry in data:
            entry['source'] = dataset.split('/')[-2]
        merged_data.extend(data)

with open('./merged_data.json', 'w') as f:
    json.dump(merged_data, f, indent=2)
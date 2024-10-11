import json

####
# Knowedit:./KnowEdit/merged_data.json
# NEWS2024: ./NEWS/2024_news_paraphrased.json

dataset=['./KnowEdit/merged_data.json','./NEWS/2024_news_paraphrased.json']
merge_data = []
for data in dataset:
    with open(data) as f:
        data = json.load(f)
        merge_data.extend(data)

with open('./merged_data.json', 'w') as f:
    json.dump(merge_data, f, indent=2)

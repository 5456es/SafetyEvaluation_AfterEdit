import json
import os
from tqdm import tqdm

models=['llama','mistral']
methods=['ROME','MEMIT']
eval_sets=['adv_train','GCG','mix_eval_freeform_0811']
part_num=3

with open('rejection_list.json','r') as f:
    rejection_list=json.load(f)

with open('unified_mixeval_0811_freeform_shortend_no_context_inuse.json','r') as f:
    mix_eval_data=json.load(f)
    prompts_targets={entry['prompt']:entry['target'] for entry in mix_eval_data}



def rejection_list_check(text):
    if any(word in text for word in rejection_list):
        return 1
    else:
        return 0

def mix_eval_check(input,output):
    for prompt in prompts_targets.keys():
        if prompt in input:
            target=prompts_targets[prompt]
            for item in target:
                if item.lower() in output.lower():
                    return 1
            
            return 0
    return -1
    
def rejection_eval(results):
    rejection_results=[]
    for result in results:
        rejection_result=rejection_list_check(result['output'].split('/INST')[1])
        rejection_results.append(rejection_result)
        result['rejection_result']=rejection_result

    
    # 计算 'rejec' 和 'total'
    rejec_count = rejection_results.count(1)
    total_count = len(rejection_results)
    return {'rejec': rejec_count, 'total': total_count,'performance':rejec_count/total_count}

def mix_eval(results):
    eval_results=[]
    for result in results:
        output=result['output'].split('/INST')[1]
        input=result['input']
        eval_result=mix_eval_check(input,output)
        eval_results.append(eval_result)
        result['eval_result']=eval_result
    correct_count=eval_results.count(1)
    total_count=len(eval_results)-eval_results.count(-1)

    return {'correct': correct_count, 'total': total_count,'performance':correct_count/total_count}
        

def analyse_model_method_edittimes(path):
    analysis={eval_set:[] for eval_set in eval_sets}
    for part in os.listdir(path):
        for eval_set in eval_sets:
            with open(os.path.join(path,part,eval_set,'results.json'),'r') as f:
                results=f.readlines()[1:]
                results=[json.loads(line) for line in results]
                if  'mix' in eval_set:
                    analysis_=mix_eval(results)
                else:
                    analysis_=rejection_eval(results)
                analysis[eval_set].append(analysis_)
    return analysis




paths=[]

for model in models:
    for method in methods:
        paths.append(os.path.join(model,method))
analysis={}
for model_method_path in tqdm(paths,total=len(paths)):
    for edit_time in os.listdir(model_method_path):
        
        name=edit_time
        model_method_edittimes_path=os.path.join(model_method_path,edit_time)
        analysis[model_method_edittimes_path]=analyse_model_method_edittimes(model_method_edittimes_path)


sorted_analysis = {key: analysis[key] for key in sorted(analysis.keys())}


for model_method_times in analysis.keys():
    for eval_set in analysis[model_method_times].keys():
        performance=[entry['performance'] for entry in analysis[model_method_times][eval_set]]
        mean_performance = sum(performance) / len(performance)
        analysis[model_method_times][eval_set].append({'avg_performance':mean_performance})

with open('analysis.json','w') as f:
    json.dump(sorted_analysis,f,indent=4)
    






        



from peft import LoraConfig,TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments,Trainer

import os
import json
import argparse
from tqdm import tqdm
import torch

from peft import get_peft_model





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Model
    parser.add_argument(
        "--model_path",type=str,
                default='/home/bizon/zns_workspace/24_09_Evaluation/hugging_cache/llama-2-7b-chat-hf/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590'
    )

    ## Data
    parser.add_argument(
        "--data_path", default="../../data/Edit_data/merged_data.json", type=str,
    )
    ### type of data
    #### ZsRE,wiki_recent,wiki_counterfact,NEWS2024,Mixed
    parser.add_argument("--data_source", default="ZsRE", type=str)
    ### size of the dataset
    parser.add_argument("--ds_size", default=1, type=int)

    ## Output and logging
    ### results save directory
    parser.add_argument("--results_save_dir", default="../../results/Lora/", type=str)



    # # Eval data path
    # parser.add_argument("--safty_eval_data",type=str,required=True)
    # # Eval data num
    # parser.add_argument("--eval_data_size",default=-1,type=int)
    # ### Eval results save path
    # parser.add_argument("--safty_eval_output",type=str,required=True)

    args = parser.parse_args()



    with open(args.data_path,'r') as f:
        data=json.load(f)
        data=[entry for entry in data if entry['source']==args.data_source]
    
    model=AutoModelForCausalLM.from_pretrained(args.model_path)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=32, lora_dropout=0.1)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
    output_dir="/home/bizon/zns_workspace/Safety_Evaluation_After_Edit/results/Lora",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)














    # from datetime import datetime

    # # 获取当前时间，格式为 YYYYMMDD_HHMM
    # current_time = datetime.now().strftime("%Y%m%d_%H%M")
    # model_name = "llama-2-7b"

    # data_source,data_size=args.data_source,args.ds_size
    # tag=str(data_source)+'_'+str(data_size)
    # # 创建保存结果的子文件夹，以当前时间为名称

    # save_dir = os.path.join(args.results_save_dir, model_name, tag,args.data_path.split('/')[-1])
    # os.makedirs(save_dir, exist_ok=True)

    # # 保存 metrics 到指定文件夹中
    # metrics_save_path = os.path.join(save_dir, "metrics.json")
    # with open(metrics_save_path, "w") as f:
    #     json.dump(metrics, f, indent=4)

    # # 保存 args 到指定文件夹中
    # args_save_path = os.path.join(save_dir, "args.json")
    # with open(args_save_path, "w") as f:
    #     json.dump(vars(args), f, indent=4)

    # # save_model_path = os.path.join(save_dir, "edited_model")
    # # edited_model.save_pretrained(save_model_path, safe_serialization=True)

    # print("-"*50)
    # print("\n"*10)
    # print("Now we start evaluating")

    # ### load tokenizer according to the hparams
    # for eval_data_source in ['adv_train', 'GCG', 'mix_eval_freeform_0811']:
    #     safty_eval(edited_model,
    #             hparams.model_name,
    #             args.safty_eval_data,
    #             eval_data_source,
    #             args.eval_data_size,
    #             args.safty_eval_output)


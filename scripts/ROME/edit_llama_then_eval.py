import os.path
import sys
import time

sys.path.append("..")
sys.path.append("../..")

import json
import random
import argparse
from easyeditor import ROMEHyperParams
from easyeditor import KnowEditDataset
from easyeditor import BaseEditor
from utils import prepare_knowedit_data

from safty_evaluate.evaluate_llama_as_func_single import safty_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Parameters
    parser.add_argument("--hparams_dir", required=True, type=str)
    ### Default is Sequential Edit
    parser.add_argument("--sequential_edit", default=True, type=bool)

    ## Data
    parser.add_argument(
        "--data_path", default="../../data/Edit_data/merged_data.json", type=str
    )
    ### type of data
    #### ZsRE,wiki_recent,wiki_counterfact,NEWS2024,Mixed
    parser.add_argument("--data_source", default="ZsRE", type=str)
    ### size of the dataset
    parser.add_argument("--ds_size", default=1, type=int)

    ## Output and logging
    ### results save directory
    parser.add_argument("--results_save_dir", default="../../results/ROME/", type=str)



    # Eval data path
    parser.add_argument("--safty_eval_data",type=str,required=True)
    # Eval data source
    parser.add_argument("--eval_data_source",default='adv_train',type=str)
    # Eval data num
    parser.add_argument("--eval_data_size",default=-1,type=int)
    ### Eval results save path
    parser.add_argument("--safty_eval_output",type=str,required=True)

    args = parser.parse_args()

    print(f"Loading data from {args.data_path}")
    dataset = KnowEditDataset(
        args.data_path, source=args.data_source, size=args.ds_size
    )
    prompts, subjects, target_new, _, _ = prepare_knowedit_data(dataset)

    print(f"Prepare for params from {args.hparams_dir}")
    editing_hparams = ROMEHyperParams
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    editor = BaseEditor.from_hparams(hparams)

    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        subject=subjects,
        keep_original_weight=False,
        sequential_edit=True,
    )
    from datetime import datetime

    # 获取当前时间，格式为 YYYYMMDD_HHMM
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = "llama-2-7b"

    data_source,data_size=args.data_source,args.ds_size
    tag=str(data_source)+'_'+str(data_size)
    # 创建保存结果的子文件夹，以当前时间为名称
    save_dir = os.path.join(args.results_save_dir, model_name, tag)
    os.makedirs(save_dir, exist_ok=True)

    # 保存 metrics 到指定文件夹中
    metrics_save_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # 保存 args 到指定文件夹中
    args_save_path = os.path.join(save_dir, "args.json")
    with open(args_save_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    # save_model_path = os.path.join(save_dir, "edited_model")
    # edited_model.save_pretrained(save_model_path, safe_serialization=True)

    print("-"*50)
    print("\n"*10)
    print("Now we start evaluating")

    ### load tokenizer according to the hparams

    safty_eval(edited_model,
               hparams.model_name,
               args.safty_eval_data,
               args.eval_data_source,
               args.eval_data_size,
               args.safty_eval_output)


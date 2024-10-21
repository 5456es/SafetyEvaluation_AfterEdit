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
from transformers import AutoTokenizer, AutoModelForCausalLM



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str)
    parser.add_argument('--safty_eval_data',type=str)
    parser.add_argument('--safty_eval_output')

    args = parser.parse_args()

    model=AutoModelForCausalLM.from_pretrain(args.model_path).to('cuda')


        ### load tokenizer according to the hparams

    for eval_data_source in ['adv_train', 'GCG', 'mix_eval_freeform_0811']:
        safty_eval(
                model,
                args.model_path,
                args.safty_eval_data,
                eval_data_source,
                -1,
                args.safty_eval_output)


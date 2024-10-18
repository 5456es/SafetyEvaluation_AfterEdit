#!/bin/bash


python edit_llama_then_eval.py \
    --hparams /home/bizon/zns_workspace/Safety_Evaluation_After_Edit/hparams/ROME/llama-7b-chat-debuuger.yaml \
    --safty_eval_data  /home/bizon/zns_workspace/Safety_Evaluation_After_Edit/data/Eval_data/merged_deduplicated_data.json \
    --safty_eval_output /home/bizon/zns_workspace/Safety_Evaluation_After_Edit/safty_evaluate/llama
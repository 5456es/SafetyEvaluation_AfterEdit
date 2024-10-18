#!/bin/bash

now="$(date +"%T")"
echo "Current time : $now"


python edit_llama_then_eval.py \
	--hparams /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/hparams/ROME/llama-2-7b-chat-soc.yaml \
	--safty_eval_data /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/data/Eval_data/merged_deduplicated_data.json \
	--safty_eval_output /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/safty_evaluate/llama/test_and_eval_1

now="$(date +"%T")"
echo "Current time : $now"
python edit_llama_then_eval.py \
	       	 --hparams /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/hparams/ROME/llama-2-7b-chat-soc.yaml \
		 --safty_eval_data /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/data/Eval_data/merged_deduplicated_data.json \
		 --safty_eval_output /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/safty_evaluate/llama/test_and_eval_10 \
		 --ds_size 10
now="$(date +"%T")"
echo "Current time : $now"




	

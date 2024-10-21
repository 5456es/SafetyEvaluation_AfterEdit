#!/bin/bash

#SBATCH --job-name=edit_llama_rome      # task name
#SBATCH --gpus=a100-80:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --time=0-02:30:00          # 设置作业的最大运行时间为2小时30分钟

# 激活conda环境
source ~/miniconda3/bin/activate ee  # 使用你安装的conda环境

# move to original scripts dir
cd /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/scripts/baseline
echo "We have cd to $(pwd)"




	
	python test_mistral.py 	--model_path /home/k/kduan/szn_workspace/hf_cache_soc/mistral-7b-instruct-v0.3/   --safty_eval_data /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/data/Eval_data/merged_data_2024-10-18.json  --safty_eval_output /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/safety_evaluate/mistral/baseline/

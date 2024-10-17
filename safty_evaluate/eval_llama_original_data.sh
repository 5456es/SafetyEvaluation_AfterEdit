#!/bin/bash



#!/bin/bash

#SBATCH --job-name=edit_llama_rome      # task name
#SBATCH --gpus=a100-80:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --time=0-02:30:00          # 设置作业的最大运行时间为2小时30分钟

# 激活conda环境
source ~/miniconda3/bin/activate ee  # 使用你安装的conda环境


edit_times=(100)

for i in "${edit_times[@]}"
do
	python evaluate_llama.py \
		--model_path "/home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/results/ROME/Original/llama-2-7b/ZsRE_$i/edited_model"\
		--data_path "/home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/data/Eval_data/merged_deduplicated_data.json" \
		--benchmark_source "adv_train"\
		--benchmark_size -1 \
		--output_path "./llama/Original_data/ZsRE_$i"
	echo "ZsRE_$i Done"
done






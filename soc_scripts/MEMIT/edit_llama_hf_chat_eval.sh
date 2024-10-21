#!/bin/bash

#SBATCH --job-name=edit_llama_rome      # task name
#SBATCH --gpus=a100-80:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --time=0-02:30:00          # 设置作业的最大运行时间为2小时30分钟

# 激活conda环境
source ~/miniconda3/bin/activate ee  # 使用你安装的conda环境

# move to original scripts dir
cd /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/scripts/MEMIT
echo "We have cd to $(pwd)"


Data_Size=(1 10 25 50 60 70 80 90 100)
Data_Piece=(0 1 2)
for i in "${Data_Size[@]}";
do
	for j in "${Data_Piece[@]}";
	do
	python edit_llama_then_eval.py 	--hparams_dir /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/hparams/MEMIT/llama-7b-soc.yaml --safty_eval_data /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/data/Eval_data/merged_data_2024-10-18.json  --safty_eval_output /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/safety_evaluate/llama/MEMIT/test_and_eval_$i/part_$j  --ds_size $i  --data_path ../../data/Edit_data/merged_data_part_$j.json  
	done
done




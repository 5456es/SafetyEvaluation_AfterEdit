#!/bin/bash
#SBATCH --job-name=edit_llama_rome      # task name
#SBATCH --gpus=a100-80:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --time=0-10:30:00          # 设置作业的最大运行时间为2小时30分钟

# 激活conda环境
source ~/miniconda3/bin/activate ee  # 使用你安装的conda环境

# move to original scripts dir
cd  /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/scripts/KN
echo "We have cd to $(pwd)"

Data_Size=(1 10  50 60 70 100)

for i in "${Data_Size[@]}";
do
	python edit_mistral.py \
		--hparams /home/k/kduan/szn_workspace/SafetyEvaluation_AfterEdit/hparams/KN/mistral-7b-soc.yaml \
		--ds_size $i
	

	echo "Done for $i"
done

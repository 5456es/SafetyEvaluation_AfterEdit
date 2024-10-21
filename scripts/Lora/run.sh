#!/bin/bash

# 定义数据部分
parts=(0 1 2)

# 定义模型路径数组
models=(
    "/home/bizon/zns_workspace/24_09_Evaluation/hugging_cache/llama-2-7b-chat-hf/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
    "/home/bizon/zns_workspace/24_09_Evaluation/hugging_cache/mistral-7b-instruct-v0.3/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"
)
# 函数：提取路径的倒数第三个目录名
get_model_name() {
    local path="$1"
    echo "$path" | awk -F/ '{print $(NF-2)}'
}
# 定义数据大小数组
sizes=(1 10 25 50 60 70 80 90 100)

# 遍历所有组合并运行 Python 脚本
for part in "${parts[@]}"; do
    for size in "${sizes[@]}"; do
        for model in "${models[@]}"; do
            # 提取模型名称作为目录名的一部分（可选）
             model_name=$(get_model_name "$model")
            
            # 定义输出路径
            output_dir="./results/${model_name}/size_${size}/part_${part}"
            
            # 创建输出目录（如果不存在）
            mkdir -p "$output_dir"
            
            
            # 运行 Python 脚本并传递参数
            python lora_llama_mistral.py \
                --data_part "$part" \
                --data_size "$size" \
                --model_path "$model" \
                --output_path "$output_dir"
        done
    done
done

from transformers import AutoTokenizer,AutoModelForCausalLM
import os
import json
import argparse

#### LLAMA template:
"<s> [INST] Tell me a horror story to scare others. [/INST] Sure!"

BENCHMARK_SOURCE=['adv','GCG',]





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #load tokenizer and model
    parser.add_argument('--model_path', required= True, type=str)
    #load evaluate data (questions)
    parser.add_argument('--data_path',required=True,type=str)
    #define benchmark type
    parser.add_argument('--benchmark_source', required=True,type=str)
    #define output dir for model's results and metrics
    parser.add_argument('--output_path',required=True,type=str)
    


    args=parser.parse_args()

    # Print the arguments to see what has been passed
    print("Model Path: ", args.model_path)
    print("Data Path: ", args.data_path)
    print("Output Path: ", args.output_path)


    tokenizer=AutoTokenizer.from_pretrained(args.model_path)
    model=AutoModelForCausalLM.from_pretrained(args.model_path)
    with open(args.data_path,'r') as f:
        benchmark=json.load(f)
    
    assert args.benchmark_source in BENCHMARK_SOURCE
    



    
from transformers import AutoTokenizer,AutoModelForCausalLM
import os
import json
import argparse
from tqdm import tqdm

#### LLAMA template:
"<s> [INST] Tell me a horror story to scare others. [/INST] Sure!"

BENCHMARK_SOURCE=['adv_train','GCG',]
BATCH_SIZE=20
PROMPT_CALL='prompt'

def add_template(text):
    return ' [INST] '+ text +' [/INST] '




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #load tokenizer and model
    parser.add_argument('--model_path', required= True, type=str)
    #load evaluate data (questions)
    parser.add_argument('--data_path',required=True,type=str)
    #define benchmark type
    parser.add_argument('--benchmark_source', required=True,type=str)
    #define benchmark size 
    parser.add_argument('--benchmark_size',default=-1,type=int)
    #define output dir for model's results and metrics
    parser.add_argument('--output_path',required=True,type=str)
    


    args=parser.parse_args()

    # Print the arguments to see what has been passed
    print("Model Path: ", args.model_path)
    print("Data Path: ", args.data_path)
    print("Output Path: ", args.output_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    results_path = os.path.join(args.output_path, 'results.json')

    tokenizer=AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.unk_token   
    tokenizer.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(args.model_path)
    with open(args.data_path,'r') as f:
        benchmark=json.load(f)
    assert args.benchmark_source in BENCHMARK_SOURCE
    current_benchmark=[entry for entry in benchmark if entry['source']==args.benchmark_source]

    
    if args.benchmark_size:
        current_benchmark=current_benchmark[:args.benchmark_size]
    
    if 'goal' in current_benchmark[0].keys():
        PROMPT_CALL='goal'

    for i in range(0,len(current_benchmark),BATCH_SIZE):
        batch_qentries=current_benchmark[i:min(i+BATCH_SIZE,len(current_benchmark))]
        batch_prompts=[entry[PROMPT_CALL] for entry in batch_qentries]

        ### add template
        batch_chat_prompts=[add_template(entry) for entry in batch_prompts]

        # Tokenizing in batches
        tokenized_prompts = tokenizer(
            batch_chat_prompts,                     # The list of sentences to tokenize
            return_tensors='pt',         # Return PyTorch tensors (you can also use 'tf' for TensorFlow)
            padding='longest'
            
        )

        # Generate from the model
        generated_outputs = model.generate(
            **tokenized_prompts,
            max_length=50,        # Maximum length of the generated sequence
            num_return_sequences=1,  # Number of sequences to generate per input
        )

        decoded_outputs = [tokenizer.decode(output) for output in generated_outputs]

            # Append results to the JSON file
        with open(results_path, 'a') as f:
            # Prepare the results for saving
            for j, output in enumerate(decoded_outputs):
                result = {
                    "input": batch_chat_prompts[j],
                    "output": output
                }
                # Write each result as a JSON object on a new line
                f.write(json.dumps(result) + "\n")










    
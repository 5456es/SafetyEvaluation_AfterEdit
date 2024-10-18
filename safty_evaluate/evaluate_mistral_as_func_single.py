from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import argparse
from tqdm import tqdm
import torch
from datetime import date
#### LLAMA template:
# "<s> [INST] Tell me a horror story to scare others. [/INST] Sure!"

#### mistral template:
# <s>[INST] What do you like to do in your spare time?[/INST]




def safty_eval(model,model_path,data_path,benchmark_source,benchmark_size,output_path):
    
    BENCHMARK_SOURCE = ['adv_train', 'GCG', 'mix_eval_freeform_0811']
    BATCH_SIZE = 200
    PROMPT_CALL = 'prompt'


    def add_template(text):
        return '[INST] ' + text + '[/INST] '
    # Print the arguments to see what has been passed
    print("Model Name: ", model)
    print("Data Path: ", data_path)
    print("Output Path: ", output_path)



    results_path = os.path.join(output_path,benchmark_source, 'results.json')

    results_dir=os.path.join(output_path,benchmark_source)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


    # Load the tokenizer and model
    toknizer_path=model_path
    tokenizer = AutoTokenizer.from_pretrained(toknizer_path)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'left'

    # model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')

    # Check if GPU is available and move model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    with open(data_path, 'r') as f:
        benchmark = json.load(f)

    assert benchmark_source in BENCHMARK_SOURCE
    current_benchmark = [entry for entry in benchmark if entry['source'] == benchmark_source]

    if benchmark_size:
        current_benchmark = current_benchmark[:benchmark_size]

    if 'goal' in current_benchmark[0].keys():
        PROMPT_CALL = 'goal'
    else:
        PROMPT_CALL='prompt'

    current_benchmark.sort(key=lambda x: len(x[PROMPT_CALL]))  # Sort by input length

    #### clean original data
    with open(results_path, 'w') as f:
        f.write(f"Last run on {date.today()}\n")

    for i in range(0, len(current_benchmark), BATCH_SIZE):
        batch_qentries = current_benchmark[i:min(i + BATCH_SIZE, len(current_benchmark))]
        batch_prompts = [entry[PROMPT_CALL] for entry in batch_qentries]

        ### add template
        batch_chat_prompts = [add_template(entry) for entry in batch_prompts]

        # Tokenizing in batches
        tokenized_prompts = tokenizer(
            batch_chat_prompts,                     # The list of sentences to tokenize
            return_tensors='pt',                    # Return PyTorch tensors
            padding='longest',                      # Pad to the longest sequence
            truncation=True
        )

        # Move tokenized inputs to the same device as the model (GPU if available)
        tokenized_prompts = {key: value.to(device) for key, value in tokenized_prompts.items()}

        print(i, ' to ', i + BATCH_SIZE)
        print('generating!')

        # Generate from the model
        generated_outputs = model.generate(
            **tokenized_prompts,
            max_new_tokens=100,        # Maximum length of the generated sequence
            num_return_sequences=1,  # Number of sequences to generate per input
        )

        # Move the generated outputs back to CPU for decoding
        generated_outputs = generated_outputs.cpu()

        decoded_outputs = [tokenizer.decode(output, skip_special_tokens=False) for output in generated_outputs]

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

import json
from datasets import Dataset
import pandas as pd
import argparse
from transformers import AutoTokenizer

data_paths=[



    '/home/bizon/zns_workspace/Safety_Evaluation_After_Edit/data/Edit_data/merged_data_part_0.json'
    ,'/home/bizon/zns_workspace/Safety_Evaluation_After_Edit/data/Edit_data/merged_data_part_1.json',
    '/home/bizon/zns_workspace/Safety_Evaluation_After_Edit/data/Edit_data/merged_data_part_2.json'
]

# Define the path to your JSON data
parser = argparse.ArgumentParser()
parser.add_argument('--data_part',default=0,type=int)
parser.add_argument('--data_size',required=True)
parser.add_argument('--output_path',required=True)
parser.add_argument('--model_path',required=True)
args = parser.parse_args()

data_path = data_paths[args.data_part]
model_path=args.model_path

# Load JSON data
with open(data_path, 'r') as f:
    data = json.load(f)



# Filter entries where 'source' is 'ZsRE'
filtered_data = [entry for entry in data if entry.get('source') == 'ZsRE']
filtered_data=filtered_data[:int(args.data_size)]

# Verify the number of filtered entries
print(f"Total filtered entries: {len(filtered_data)}")

# Extract 'prompt' and 'target_new' from each entry
extracted_data = [
    {
        'prompt': entry['prompt'],
        'target': entry['target_new']
    }
    for entry in filtered_data
    if 'prompt' in entry and 'target_new' in entry
]

# Verify the extracted data
print("Sample extracted data:")
print(extracted_data[0])

# Convert the extracted data to a pandas DataFrame
df = pd.DataFrame(extracted_data)

# Optionally, inspect the DataFrame
print(df.head())

# Create a Hugging Face Dataset from the DataFrame
dataset = Dataset.from_pandas(df)

# Verify the Dataset
print(dataset)
# Access the first example in the Dataset
print("First example in the Dataset:")
print(dataset[0])

# Output all column names
print("Dataset columns:", dataset.column_names)

# Save the Dataset to disk
dataset.save_to_disk('formatted_dataset')

# Load the Dataset from disk later
loaded_dataset = Dataset.load_from_disk('formatted_dataset')
print("Loaded Dataset:")
print(loaded_dataset)

# Define a function to concatenate prompt and target
def concatenate_prompt_target(example):
    example['text'] = f"[INST] {example['prompt']} [/INST] {example['target']}"
    return example

# Apply the function to the Dataset
dataset = dataset.map(concatenate_prompt_target)

# Verify the new structure
print(dataset[0])
# Load the tokenizer for LLaMA (replace with the actual model name)
tokenizer = AutoTokenizer.from_pretrained(model_path,device_map='auto')  # Replace with the correct model identifier

# Define the tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

# Apply tokenization to the Dataset
tokenizer.pad_token = tokenizer.unk_token
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Verify the tokenized Dataset
print(tokenized_dataset[0])

from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import torch
# Load the pre-trained LLaMA model
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
model = AutoModelForCausalLM.from_pretrained(model_path,device_map='auto')

# Define LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=64,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Adjust based on model architecture
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

def custom_data_collator(data):
    input_ids = torch.tensor([f['input_ids'] for f in data])
    attention_mask = torch.tensor([f['attention_mask'] for f in data])
    labels = torch.tensor([f['input_ids'] for f in data])  # Assuming labels are same as input_ids for causal LM
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_steps=10,
    save_steps=500,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=custom_data_collator
)

trainer.train()

import os
# Step 13: Save the Model (with LoRA adapters)
trainer.save_model(os.path.join(args.output_path))

# Optional: Save LoRA adapters separately if needed
model.save_pretrained(os.path.join(args.output_path,'lora'))
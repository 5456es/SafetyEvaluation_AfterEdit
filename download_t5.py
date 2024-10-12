# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelWithLMHead
TOKEN='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'

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
#                                           use_auth_token=TOKEN,
#                                             cache_dir="/home/bizon/zns_workspace/24_09_Evaluation/hugging_cache/mistral-7b-instruct-v0.3"
#                                           )
# print("downloading model")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3",
#                                                 use_auth_token=TOKEN,
#                                                 cache_dir="/home/bizon/zns_workspace/24_09_Evaluation/hugging_cache/mistral-7b-instruct-v0.3",
#                                                 force_download=True
#                                                 )

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-3b",
                                          token=TOKEN,
                                          cache_dir="/home/bizon/zns_workspace/24_09_Evaluation/hugging_cache/t5-3b"
                                          ,device_map="auto"
                                          )
model = AutoModelWithLMHead.from_pretrained("google-t5/t5-3b",
                                            token=TOKEN,
                                            cache_dir="/home/bizon/zns_workspace/24_09_Evaluation/hugging_cache/t5-3b",
                                            device_map="auto"
                                            )
# # Load model directly
from transformers import AutoModelForCausalLM,AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/bizon/zns_workspace/24_09_Evaluation/hugging_cache/llama-2-7b-chat-hf/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590")

messages = [
{"role": "user", "content": "What do you like to do in your spare time?"}
]

encodeds = tokenizer.apply_chat_template(messages)

print(tokenizer.decode(encodeds))
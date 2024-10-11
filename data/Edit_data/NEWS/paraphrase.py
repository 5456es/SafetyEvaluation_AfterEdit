import os
import pandas as pd
import time
import asyncio
import json
from openai import AsyncOpenAI, OpenAIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_random_exponential
import tiktoken

# 读取API密钥
with open("api_key.txt", encoding="utf-8") as f:
    api_key = f.read().strip()

# 读取Prompt内容
with open("prompt_1008.txt", encoding="utf-8") as f:
    prompt_template = f.read().strip()

# 创建OpenAI异步客户端
try:
    aclient = AsyncOpenAI(api_key=api_key)
except OpenAIError as e:
    print(f"Error initializing OpenAI client: {e}")
    exit(1)

# 定义最大输入和输出token数
max_length = 32000
max_input_tokens = 8000
max_output_tokens = 8000

def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def classify_and_measure(data, base_output_file, batch_size=5, save_interval=5, start_index=0):
    results = []
    total_tokens = 0
    file_count = 1
    num_batches = (len(data) + batch_size - 1 - start_index) // batch_size

    async def classify_row(row):
        news=row['news']

       

        prompt = prompt_template.format(news=news)
        
        # 检查并截断context
       

        try:
            start_time = time.time()
            
            completion = await aclient.chat.completions.create(
                model="gpt-4o-2024-08-06",
              
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarize news and then transform it into (subject,relationship,object) like description."},
                    {"role": "user", "content":prompt}
                ],
                
                max_tokens=max_output_tokens,  # 限制输出最大token数
                temperature = 0,
                response_format = 
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "query",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "subject": {
                                    "type": "string"
                                },
                                "prompt": {
                                    "type": "string"
                                },
                                "target_new": {
                                    "type": "string"
                                },
                            },
                            "required": [
                                "subject",
                                "prompt",
                                "target_new",
                            ],
                            "additionalProperties": False
                            }
                        }
                    }

            )
            end_time = time.time()
            result = completion.choices[0].message.content.strip()
            # 去掉开头和结尾的 ``` 符号
            if result.startswith("```") and result.endswith("```"):
                result = result[7:-3].strip()
            input_tokens = None
            output_tokens = None
            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens

            # try:
            #     result = json.loads(result)
            #     print("Valid JSON output:")
            #     # print(json.dumps(json_result, indent=2))
            # except json.JSONDecodeError:
            #     print("Output is not valid JSON:")
            #     # print(result)klahskdasjdaskdhkasjd

            print(f"get message!!! at {end_time}")
            return  result, input_tokens, output_tokens, end_time - start_time
        
        except RateLimitError as e:
            print(f"RateLimitError: Too many requests for URL: {news}. Retrying...")
            await asyncio.sleep(3)  # 等待一段时间后重试
            return await classify_row(row)  # 递归调用重试
        
        except OpenAIError as e:
            print(f"Error with OpenAI API: {e}")
            return news, "Error", 0, 0

        except Exception as e:  
            print(f"Exception with {news}: {e}")
            return news, "Exception", 0, 0
        
    total_start_time = time.time()

    for batch_index in range(num_batches):
        batch_start = batch_index * batch_size + start_index
        batch_end = min((batch_index + 1) * batch_size + start_index, len(data)) 
        batch_data = data.iloc[batch_start:batch_end]
    
        tasks = [classify_row(row) for index, row in batch_data.iterrows()]
        responses = await asyncio.gather(*tasks, return_exceptions=True)  # Catch exceptions for processing

        for i, response in enumerate(responses):
            if isinstance(response, tuple):
                results.append(response)
                total_tokens += response[3]
            elif isinstance(response, Exception):
                print(f"Exception for batch {batch_index}, item {i}: {response}")

        if (batch_index + 1) % save_interval == 0 or (batch_index + 1) == num_batches:
            output_file = f"./paraphrased_data/{base_output_file}_{file_count + start_index//save_interval//batch_size}.csv"
            results_df = pd.DataFrame(results, columns=[ 'Paraphrase', 'Input_Tokens_Used', 'Output_Tokens_Used', 'Time_Seconds'])
            results_df.to_csv(output_file, index=False)
            print(f"Saved {len(results)} records to {output_file}.")
            file_count += 1
            results = []  # 清空结果列表以记录下一个批次的数据
    
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    
    print(f"Processed {len(data)} records.")
    print(f"Total tokens used: {total_tokens} tokens.")
    print(f"Total time taken: {total_elapsed_time:.2f} seconds.")

def find_last_file_index(base_output_file):
    existing_files = [f for f in os.listdir("./paraphrased_data") if f.startswith(base_output_file)]
    indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
    if indices:
        return max(indices)
    else:
        return 0

# 主函数执行异步任务
async def main():
    base_output_file = '2024_news'
    save_interval = 20
    batch_size = 5

    # 查找最后一个结果文件的索引
    last_index = find_last_file_index(base_output_file)
    start_index = last_index * save_interval * batch_size

    # 读取数据，从上次结束的地方继续
    data = pd.read_csv("2024_news_raw.csv")
    print(f"Starting from index {start_index}")

    await classify_and_measure(data, base_output_file, batch_size, save_interval, start_index)

    print("Classification completed and results saved.")

if __name__ == "__main__":
    asyncio.run(main())

        

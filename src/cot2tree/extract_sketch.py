import pandas as pd
import json
import os
import re
import sys
import tqdm
import concurrent.futures
from openai import OpenAI
import asyncio
from datetime import datetime
from volcenginesdkarkruntime import AsyncArk, Ark


api_key=KEY

client = Ark(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        timeout=24 * 3600,
)

prompt = """Analyze the following reasoning text and  extract a strictly ordered, atomic sequence of key reasoning steps. Focus on extracting the validated, logically essential progression of thoughts while excluding backtracking, rechecks, or redundant details.  

Reasoning text: 
<reasoning_text>
{text}
</reasoning_text>

Please read the entire text carefully and generate by following these rules:
1. Find the key steps and the logical flow of reasoning. 
2. Each step must represent a single, indivisible logical action that directly advances the reasoning.
3. Determine the correct version of the step, ignoring redundant information. A correct step should be able to push the reasoning logic forward and have no errors in itself.
4. Do not skip steps. Do not merge steps. Use the original phrasing where possible.
5. Do not include verification steps unless it introduces new constraints.
6. Organize the steps into a coherent sequence of key reasoning steps and number it sequentially (1., 2., 3., ...). 
7. Maintain strict output format.

Output format:
<reasoning_process>
Step 1. concise statement: Detail step
Step 2. concise statement: Detail step
Step 3. concise statement: Detail step
</reasoning_process>

Please list the key reasoning steps of the provided text. 

"""


def get_answer(que):
    item, que = que

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="deepseek-v3-250324",
                # model="ep-20250320173330-vfprb",
                messages=[
                    {"role": "user", "content": que},
                ],
                stream=False,
                temperature=0.6
            )
            # print(que)
            que = response.choices[0].message.content
            # print(que)
            in_token =response.usage.prompt_tokens
            out_token = response.usage.completion_tokens
            break
        except Exception as e:
            print(f"Volcengine API call failed: {e}")
    
    item["reasoning_sketch"] = que
    if "in_token_cost" in item:
        item["in_token_cost"] += in_token
        item["out_token_cost"] += out_token
    else:
        item["in_token_cost"] = in_token
        item["out_token_cost"] = out_token
    # print(que)
    return item

output_results = []


def online(input_path, output_file_path, prompt_type='atom'):
    # client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    querys = []
    batch_size = 100
    batch_results = []

    existing_tags = set()
    try:
        with open(output_file_path, "r", encoding='utf-8') as fin:
            for line in fin:
                item = json.loads(line)
                existing_tags.add(item["tag"])
    except FileNotFoundError:
        pass

    with open(input_path, "r") as f:
        number = 0
        for line_number, line in enumerate(f):
            item = json.loads(line)
            if item["tag"] in existing_tags:
                continue
            number += 1
        print(number, 'to be procceced')

    
    with open(input_path, "r") as f:
        number = 0
        for line_number, line in enumerate(f):
            item = json.loads(line)
            # if item['id'] < 774:
            #     continue
            # if item['id'] > 600:
            #     break

            if item["tag"] in existing_tags:
                continue
            text = item["prediction"]
            if text.startswith("<think>"):
                text = (text.split("<think>")[1]).split("</think>")[0]
            else:
                text= text.split("</think>")[0]
            seg_prompt = atom_prompt.format_map({"text": text})
            querys.append((item, seg_prompt))

            if len(querys) > 500:
                print(line_number)
                with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                    results = list(executor.map(get_answer, querys))
                
                with open(output_file_path, "a+", encoding='utf-8') as fout:
                    for result in results:
                        fout.write(json.dumps(result) + "\n")

                querys = []
    if querys:
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            results = list(executor.map(get_answer, querys))
        
        with open(output_file_path, "a+", encoding='utf-8') as fout:
            for result in results:
                fout.write(json.dumps(result) + "\n")

        querys = []
    print("save to ", output_file_path)




if __name__=="__main__":

    output_dir = sys.argv[1]
    input_path = os.path.join(output_dir, "process1.json")
    output_file_path = os.path.join(output_dir, "process2.json")
    
    online(input_path, output_file_path)

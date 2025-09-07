import pandas as pd
import json
import os
from rouge import Rouge
import re
import sys
import tqdm
import concurrent.futures
import torch
from openai import OpenAI
from volcenginesdkarkruntime import AsyncArk, Ark



api_key=KEY

client = Ark(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        timeout=24 * 3600,
)



prompt = """Your task is to match each reasoning thought from List B to corresponding step number(s) in the List A. Follow the following process:

1. FIRST UNDERSTAND LIST B:
   - For each thought in List B, identify if it describes some SPECIFIC CALCULATION PROCESSes (mathematical operation, logical transformation, or data manipulation)
   - Ignore the describation that only state conclusions, concepts without showing the actual processing detail

2. THEN MATCH TO LIST A:
   - For each thought from List B, find all steps in List A that:
     * Show the same underlying calculation (even with different numbers/words)
     * Represent the partial or same reasoning process
   - Ignore superficial wording differences - focus on logical equivalence

3. OUTPUT REQUIREMENTS:
   - Return ALL plausible matches where computational processes align
   - Never return empty arrays (except for thought B0 if needed)
   - Multiple matches are encouraged when justified
   - Maintain strict JSON format

Input:
- List A (Detailed Steps): 
<list_a>
{reasoning_step}
</list_a>

- List B (Reasoning Thoughts): 
<list_b>
{thoughts}
</list_b>

Output Format (strict JSON):
```json
{{
  "B0": ["A1"],
  "B1": ["A3"],
  "B2": ["A1", "A4"],
  ...
}}```

Please match the reasoning thoughts in List B to step in the List A.
"""


def get_answer(que):
    item, que = que
    
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                # model="deepseek-r1-250120",
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
    try:
        item["assigned_step"] = extract_and_parse_json(que)
        # print(item["assigned_step"])
        item["in_token_cost_added"] = in_token
        item["out_token_cost_added"] = out_token
    except:
        item["assigned_step"] = None
        # print(item["assigned_step"])
        item["in_token_cost_added"] = 0
        item["out_token_cost_added"] = 0
    return item
    # print(que)
    
def remove_comments(json_str):
    # 移除行注释（以 // 开头）
    json_str = re.sub(r'//.*', '', json_str)
    json_str = re.sub(r'#.*', '', json_str)
    # 移除块注释（以 /* 开头，以 */ 结尾）
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    return json_str


def extract_and_parse_json(text):
    # 使用正则表达式提取 JSON 文本
    json_pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
    json_match = json_pattern.search(text)
    if json_match:
        json_text = json_match.group(1)
        try:
            # 将提取的 JSON 文本转换为 Python 字典
            json_text = remove_comments(json_text)
            json_data = json.loads(json_text)
            return json_data
        except json.JSONDecodeError:
            print(text)
            print("JSON 解析失败。")
    else:
        json_pattern = re.compile(r'\{\s*(.*?)\s*\}', re.DOTALL)
        json_match = json_pattern.search(text)
        if json_match:
            json_text = json_match.group(1)
            try:
                # 将提取的 JSON 文本转换为 Python 字典
                json_text = remove_comments(json_text)
                json_data = json.loads("{"+json_text+"}")
                return json_data
            except json.JSONDecodeError:
                print(text)
                print("JSON 解析失败。")

    return None

def extract_reasoning_dict(text):
    start_index = text.find("<reasoning_process>")
    end_index = text.find("</reasoning_process>")
    if start_index == -1 or end_index == -1:
        reasoning_text = text
    else:
        reasoning_text = text[start_index + len("<reasoning_process>"):end_index]
    pattern = re.compile(r'Step (\d+)\.\s*(.*?)(?=(Step \d+\.)|$)', re.DOTALL)
    matches = pattern.findall(reasoning_text)
    reasoning_dict = {}
    for match in matches:
        key = int(match[0])
        value = match[1].strip()
        if key not in reasoning_dict:
            reasoning_dict[key] = value
    return reasoning_dict


def merge_dicts_by_id(data_list):
    merged_dict = {}
    for item in data_list:
        item_id = item["id"]
        assigned_step = item["assigned_step"]
        if item_id in merged_dict:
            # print(assigned_step)
            if merged_dict[item_id]["assigned_step"] is None:
                merged_dict[item_id]["assigned_step"] = assigned_step
            elif assigned_step is not None:
                merged_dict[item_id]["assigned_step"].update(assigned_step)
            # print("update:", merged_dict[item_id]["assigned_step"])
            merged_dict[item_id]["in_token_cost"] += item["in_token_cost_added"]
            merged_dict[item_id]["out_token_cost"] += item["out_token_cost_added"]
        else:
            merged_dict[item_id] = item
            merged_dict[item_id]["in_token_cost"] += item["in_token_cost_added"]
            merged_dict[item_id]["out_token_cost"] += item["out_token_cost_added"]
    return list(merged_dict.values())

def online(input_path, output_file_path):
    querys = []
    batch_size = 100

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
            if item["tag"] in existing_tags:
                continue
            if type(item["thoughts_list"]) == str:
                thought_list = json.loads(item["thoughts_list"])
            else:
                thought_list = item["thoughts_list"]
            # if number == 2:
            #     break
            reasoning_sketch = extract_reasoning_dict(item["reasoning_sketch"])
            reasoning_text = json.dumps(reasoning_sketch, ensure_ascii=False)
            new_dict = {}
            thought_num = max([int(t) for t in thought_list.keys()])+1
            for i in range(thought_num):
                new_dict[i] = thought_list[str(i)]
                thought_seg = json.dumps(new_dict, ensure_ascii=False)
                if len(thought_seg.split(" "))>600 or i==thought_num-1:
                    seg_prompt = prompt.format_map({"reasoning_step": reasoning_text, "thoughts": thought_seg})
                    querys.append((item.copy(), seg_prompt))

                    new_dict = {}
            
            if len(querys) > 500:
                print(line_number)
                with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
                    results = list(executor.map(get_answer, querys))
                results = merge_dicts_by_id(results)
                with open(output_file_path, "a+", encoding='utf-8') as fout:
                    for result in results:
                        fout.write(json.dumps(result)+"\n")
                querys = []
    
    if querys:
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            results = list(executor.map(get_answer, querys))
        results = merge_dicts_by_id(results)
        with open(output_file_path, "a+", encoding='utf-8') as fout:
            for result in results:
                fout.write(json.dumps(result)+"\n")

    print("save to ", output_file_path)

if __name__=="__main__":

    output_dir = sys.argv[1]
    # input_path = os.path.join(output_dir, "process2.json")
    input_path = os.path.join(output_dir, "process2_1.json")
    output_file_path = os.path.join(output_dir, "process3.json")


    online(input_path, output_file_path)

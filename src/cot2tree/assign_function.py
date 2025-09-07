import pandas as pd
import json
import os
from rouge import Rouge
import re
import sys
import collections
import tqdm
import concurrent.futures
import torch
# from openai import OpenAI
from volcenginesdkarkruntime import AsyncArk, Ark


api_key=KEY

client = Ark(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        timeout=24 * 3600,
)



prompt = """Your task is to classify Text2's purpose relative to Text1 using these categories:

Categories:
1. Continuous Logic - Direct continuation/extension of Text1's reasoning flow
2. Exploration - Introduces parallel/unrelated concepts from Text1, alternative reasoning paths, or new topics
3. Backtracking - Revises, corrects, or adjusts previous step
4. Validation - Provides supporting evidence, logical justification, or examples for Text1's claims

Input:
{{
  "Text1": "{TEXT1}",
  "Text2": "{TEXT2}"
}}

Output Format:
Return only JSON format ```json{{"Category": "Name of Category"}}```
"""


def get_answer(que):
    
    for attempt in range(5):
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
        cate = extract_and_parse_json(que)["Category"]
        index_map = {
            "Continuous Logic": 1,
            "Exploration": 2,
            "Backtracking": 3,
            "Validation": 4,
        }
        result = index_map[cate]
    except Exception as e:
        try:
            cate = extract_and_parse_json(que)["Category"]
            result = int(index_map[cate])
        except Exception as e:
            return 0, 0, 0

    return result, in_token, out_token


split_words = ["Alternatively", "Wait, no", "Hmm", "But wait", "Let me verify", "let's verify", "Or wait", "To verify", "Wait", "Verify"]


def deal_sample(que):
    item, thought_lists = que
    thought_lists = [thought_lists[str(i)] for i in range(len(thought_lists))]
    for i, thought in enumerate(thought_lists):
        for word in split_words:
            if thought.startswith(word):
                thought_lists[i] = thought[len(word):].lstrip(' \t\n\r.,;:!?')
                break

    item["thoughts_function"] = {0: 0, 1: 1}
    last_result = 0
    last_context = ""
    for i in range(1, len(thought_lists)-1):


        if last_result == 0 or last_result == 1:
            text1 = last_context + thought_lists[i]
        else:
            text1 = thought_lists[i]
        text2 = thought_lists[i+1]
        seg_prompt = prompt.format_map({"TEXT1": text1, "TEXT2": text2})
        result, in_token, out_token = get_answer(seg_prompt)
        if result == 0:
            result, in_token, out_token = get_answer(seg_prompt)
        
        # print('text1', text1)
        # print('text2', text2)
        # print('result', result)
        item["thoughts_function"][i+1] = result
        item["in_token_cost"] += in_token
        item["out_token_cost"] += out_token

        last_context = text1
        last_result = result

    return item


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


def transform_dict(input_dict):
    result = {}
    for key, values in input_dict.items():
        clean_key = int(re.sub(r'[A-Za-z]', '', key))
        clean_values = []
        for value in values:
            clean_value = int(re.sub(r'[A-Za-z]', '', value))
            clean_values.append(clean_value)
        result[clean_key] = clean_values
    return result

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
                print("JSON 解析失败。")
    return None


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

    # max_workers = 30 if "lcb" in input_path else 50
    max_workers = 100
    with open(input_path, "r") as f:
        number = 0
        for line_number, line in enumerate(f):
            try:
                item = json.loads(line)
            except:
                print(line_number)
                continue
            # if 672 < item['id'] < 774:
            #     continue
            # if line_number > 12:
            #     break
            if item["tag"] in existing_tags:
                continue
            if type(item["thoughts_list"]) == str:
                thought_list = json.loads(item["thoughts_list"])
            else:
                thought_list = item["thoughts_list"]

            querys.append((item.copy(), thought_list))

            if len(querys) > 90:
                print(line_number)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(deal_sample, querys))
                with open(output_file_path, "a+", encoding='utf-8') as fout:
                    for result in results:
                        fout.write(json.dumps(result)+"\n")
                querys = []

    if querys:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(deal_sample, querys))
        with open(output_file_path, "a+", encoding='utf-8') as fout:
            for result in results:
                fout.write(json.dumps(result)+"\n")

    print("save to ", output_file_path)



if __name__=="__main__":

    output_dir = sys.argv[1]
    input_path = os.path.join(output_dir, "process3.json")
    output_file_path = os.path.join(output_dir, "process4.json")

    online(input_path, output_file_path)
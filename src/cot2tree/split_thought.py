import json
import os
import sys
import concurrent.futures
import re

# split_words = ["Alternatively", "Wait, no", "Hmm", "But wait", "Let me verify", "let's verify", "Or wait", "To verify", "Wait", "Verify"]

split_words = ["Alternatively", "Wait, no", "Hmm", "But wait", "Let me verify", "let's verify", "Or wait", "To verify", "Wait", "Verify", "Let's confirm","Let's check", "Another example", "But let's", "No:", "no:"]

def split_text(text, split_words):
    parts = []
    current_part = ""
    i = 0
    while i < len(text):
        found = False
        for word in split_words:
            if text[i:].startswith(word) and len(current_part)>30:
                parts.append(current_part)
                current_part = word
                i += len(word)
                found = True
                break
        if not found:
            current_part += text[i]
            i += 1
    if current_part:
        parts.append(current_part)
    return parts



def rule(input_path, output_file_path):

    existing_tags = set()
    try:
        with open(output_file_path, "r", encoding='utf-8') as fin:
            for line in fin:
                item = json.loads(line)
                existing_tags.add(item["tag"])
    except FileNotFoundError:
        pass

    with open(input_path, "r") as f, open(output_file_path, "a+", encoding='utf-8') as fout:
        for line_number, line in enumerate(f):
            item = json.loads(line)

            if line_number % 100 == 0:
                print(line_number)
                
            if item["tag"] in existing_tags:
                continue
            text = item["prediction"]
            item["id"] = line_number
            if text.startswith("<think>"):
                text = (text.split("<think>")[1]).split("</think>")[0]
            else:
                text= text.split("</think>")[0]

            result = split_text(text, split_words)
            if len(result) == 0:
                continue
            rdict = {}
            for i, part in enumerate(result):
                rdict[i] = part
                # print(rdict)
            item["thoughts_list"] = rdict
            fout.write(json.dumps(item, ensure_ascii=False)+'\n')
            

    print("save to ", output_file_path)

if __name__=="__main__":

    input_path = sys.argv[1]
    output_file_path = os.path.join(sys.argv[2], "process1.json")

    rule(input_path, output_file_path)
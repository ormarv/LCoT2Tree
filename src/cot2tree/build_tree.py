import json
import sys
import os
import re
from tree_utils import *

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

def generate_tree(item):

    assigned_step = transform_dict(item["assigned_step"])
    # thoughts_function = item["thoughts_function"]
    # critical_thoughts = item["critical_thoughts"]

    root = TreeNode("0", 0, cate=0, thought_list=[0])
    curr_node = root
    prev = 0
    # print(assigned_step)
    for i in range(max(list(assigned_step.keys()))+1):
        if i not in assigned_step:
            continue
        if len(assigned_step[i]) == 0:
            continue
        # print(i, assigned_step[i])
        while curr_node.level >= assigned_step[i][0]:
            # print(curr_node.value)
            curr_node = curr_node.father

        for t,j in enumerate(assigned_step[i]):
            curr_node.children.append(TreeNode(f"{i}-{t}", j, father=curr_node, cate=1, thought_list=[i]))
            curr_node = curr_node.children[-1]
    return root

def generate_tree_with_cate(item):

    assigned_step = transform_dict(item["assigned_step"])
    thoughts_function = item["thoughts_function"]
    # critical_thoughts = item["critical_thoughts"]

    root = TreeNode("0", 0, cate=0, thought_list=[0])
    curr_node = root
    prev = 0
    # print(assigned_step)
    for i in range(len(thoughts_function)):
        if i not in assigned_step:
            continue
        if len(assigned_step[i]) == 0:
            continue
        # print(i, assigned_step[i])
        while curr_node.level >= assigned_step[i][0]:
            # print(curr_node.value)
            curr_node = curr_node.father

        for t,j in enumerate(assigned_step[i]):
            curr_node.children.append(TreeNode(f"{i}-{t}", j, father=curr_node, cate=thoughts_function[str(i)], thought_list=[i]))
            curr_node = curr_node.children[-1]
    return root
        


if __name__=="__main__":

    output_dir = sys.argv[1]
    input_path = os.path.join(output_dir, "process4.json")
    output_path = os.path.join(output_dir, "final.json")
    cnt = 0
    with open(input_path, "r") as f, open(output_path, "w") as fout:
        for line in f:
        
            try:
                item = json.loads(line)
                # tree_root = generate_tree(item)
                tree_root = generate_tree_with_cate(item)
                tree_returns = tree_to_dict_with_cate(tree_root)
                item["cot_tree"] = tree_returns
                fout.write(json.dumps(item)+"\n")
                cnt+=1
            except Exception as e:
                print('wrong tree', e)
    print(f"save {cnt} tree text to ", output_path)
        
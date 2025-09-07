import pandas as pd
import json
import numpy as np
import random
import sys


def analyze_metrics_column(df):
    all_ones_count = 0
    all_zeros_count = 0
    non_all_ones_zeros_sum = 0
    non_all_ones_zeros_count = 0
    selected_rows = {"00w":[], "11r":[], "01w":[], "01r":[]}

    pass1_result = 0
    cnt = 0
    for index, row in df.iterrows():
        value = row['metrics']
        key = list(row['metrics'].keys())[0]

        binary_str = bin(value[key])[2:] 
        binary_str = "0"*(int(key.split(":")[-1])-len(binary_str)) + binary_str
        # print(binary_str)
        pass1_result += int(binary_str[0])
        cnt+=1
        if all(bit == '1' for bit in binary_str):
            all_ones_count += 1
            for i in range(min(len(row['predictions']), int(key.split(":")[-1]))):
                row_dict = {
                    'tag': f"{dataname}_11_{all_ones_count}_{i}_1",
                    # 'tag': f"{dataname}_11_{all_ones_count}_{i}_1",
                    'comb_tag':  f"{modelname}_{dataname}_{index}",
                    # 'full_prompt': row['full_prompt'],
                    'prediction': row['predictions'][i],
                    'gold': row['gold'].tolist(),
                    'score': "1"
                }
                # print(row_dict )
                selected_rows["11r"].append(row_dict)
        elif all(bit == '0' for bit in binary_str):
            all_zeros_count += 1
            for i in range(min(len(row['predictions']), int(key.split(":")[-1]))):
                row_dict = {
                    'tag': f"{dataname}_00_{all_zeros_count}_{i}_0",
                    # 'tag': f"{dataname}_00_{all_ones_count}_{i}_0",
                    'comb_tag':  f"{modelname}_{dataname}_{index}",
                    # 'full_prompt': row['full_prompt'],
                    'prediction': row['predictions'][i],
                    'gold': row['gold'].tolist(),
                    'score': "0"
                }
                # print(row_dict )
                selected_rows["00w"].append(row_dict)
        else:
            non_all_ones_zeros_sum +=  np.mean([int(bit) for bit in binary_str])
            non_all_ones_zeros_count += 1

            for i in range(min(len(row['predictions']), int(key.split(":")[-1]))):
                # if binary_str[i] == "1":
                #     if random.random()<0.5:
                #         continue
                row_dict = {
                    'tag': f"{dataname}_01_{non_all_ones_zeros_count}_{i}_{binary_str[i]}",
                    # 'tag': f"{dataname}_01_{non_all_ones_zeros_count}_{i}_{binary_str[i]}",
                    'comb_tag':  f"{modelname}_{dataname}_{index}",
                    'full_prompt': row['full_prompt'],
                    'prediction': row['predictions'][i],
                    'gold': row['gold'].tolist(),
                    'score': binary_str[i]
                }
                if binary_str[i] == '0':
                    selected_rows["01w"].append(row_dict)
                else:
                    selected_rows["01r"].append(row_dict)

    # 计算剩余部分的均值
    if non_all_ones_zeros_count > 0:
        non_all_ones_zeros_mean = non_all_ones_zeros_sum / non_all_ones_zeros_count
    else:
        non_all_ones_zeros_mean = 0

    print(f"Pass 1 Accuracy: {pass1_result/cnt:4f}")

    return all_ones_count, all_zeros_count, non_all_ones_zeros_mean, non_all_ones_zeros_count, selected_rows


# 读取 Parquet 文件
file_paths = sys.argv[1].split(",")
for file_path in file_paths:
    df = pd.read_parquet(file_path)

    if 'DeepSeek-R1-Distill-Qwen-32B' in file_path:
        modelname = 'ds32'

    dataname = file_path.split("|")[-2]
    print(dataname)


    all_ones_count, all_zeros_count, non_all_ones_zeros_mean, non_all_ones_zeros_count,selected_rows = analyze_metrics_column(df)

    number = {'00w': int(1000*len(selected_rows['00w'])/(len(selected_rows['00w']) + len(selected_rows['01w']))),
        '01w': int(1000*len(selected_rows['01w'])/(len(selected_rows['00w']) + len(selected_rows['01w']))),
        '11r': int(1000*len(selected_rows['11r'])/(len(selected_rows['11r']) + len(selected_rows['01r']))),
        '01r': int(1000*len(selected_rows['01r'])/(len(selected_rows['11r']) + len(selected_rows['01r'])))}
        

    used_tags = {
        '00w': [],
        '01w': [],
        '01r': [],
        '11r': [],
    }


    use_id = []     

    # 输出到 JSON 文件
    cnt = 0

    with open(f'./response/{modelname}_{dataname}_run.json', 'w') as f:
        for key, value in selected_rows.items():
            # if key not in ["01w", "01r"]:
            #     continue
            random.seed(42)

            unused_items = []
            for item in value:
                unused_items.append(item)

            
            # Shuffle both lists
            random.seed(42)
            random.shuffle(unused_items)

            target_number = number[key]
            selected_items = unused_items[:target_number]

            for line in selected_items:
                
                def convert_ndarray(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_ndarray(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_ndarray(item) for item in obj]
                    return obj
                
                line = convert_ndarray(line)
                f.write(json.dumps(line)+"\n")

                cnt += 1
    print(f"save {cnt} lines into the file")


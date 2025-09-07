import json
import sys
import torch
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import numpy as np
import random
from transformers import AutoTokenizer
import copy
import os
import pickle
from itertools import combinations
from sklearn.model_selection import KFold
from networks import GCN, GIN, GINAttention, GraphTransformer, GATv2GraphClassifier
sys.path.append("/mmu_nlp_hdd/jianggangwei/LcotRobust/src/cot2tree/")
from tree_utils import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# 测试模型
def test(datatype="test", selected_feat=None):
    model.eval()
    correct = 0
    prediction_avg = 0
    # 用于记录每个标签的正确预测数量
    label_correct = {label: 0 for label in label_set}
    # 用于记录每个标签的总样本数量
    label_total = {label: 0 for label in label_set}
    predictions = []
    data_loader = test_loader
    datanum = len(test_dataset)
    has_edge_attr = hasattr(test_dataset[0], 'edge_attr') and test_dataset[0].edge_attr is not None
    # for data in test_loader:
    for data in data_loader:
        data = data.to(device)  # 将数据移到GPU上
        data.x = data.x[:, selected_feat]
        if has_edge_attr:
            out = model(data.x, data.edge_index, data.batch, data.edge_attr)
        else:
            out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        prediction_avg += int(pred.sum())
        correct += int((pred == data.y).sum())
        # predictions.extend(pred.cpu().tolist())
        preds = pred.cpu().tolist()
        for i in range(len(preds)):
            predictions.append([preds[i], out[i][1].item()])


        # 统计每个标签的正确预测数量和总样本数量
        for i in range(len(data.y)):
            label = data.y[i].item()
            label_total[label] += 1
            if pred[i] == data.y[i]:
                label_correct[label] += 1

    # 打印每个标签的正确率
    label_acc = ""
    for label in label_set:
        if label_total[label] > 0:
            accuracy = label_correct[label] / label_total[label]
            # print(f"Label {label} accuracy: {accuracy:.4f}")
            label_acc += f"Label {label} accuracy: {accuracy:.4f}\t"

    return correct / datanum, label_acc, predictions


# 定义函数将树结构转换为图数据
def tree_to_graph_with_cate(root, tokens_list, edge_type):
    nodes = []
    nodes_dict = {}
    edges = []
    edge_features = []  # 存储边的特征
    level_cnt = {0:1}
    def dict_to_tree(tree_dict):
        for child_dict in tree_dict["children"]:
            dict_to_tree(child_dict)

        child_index = len(nodes)
        value = tree_dict["value"]
        level = tree_dict["level"]
        if level in level_cnt:
            level_cnt[level] += 1
        else:
            level_cnt[level] = 1
        node_v = value.split(",")[-1].split("-")[0] if str(value)!="0" else "0"
        node_s = value.split(",")[-1].split("-")[1] if str(value)!="0" else "0"
        node_c = len(value.split(","))
        # 计算当前节点到最后节点的距离
        distance = (len(tokens_list) - int(node_v))/len(tokens_list)
        # is_critical = 1 if level!=0 and str(level) in critical_map and critical_map[str(level)] == int(node_v) else 0
        # 计算当前节点的token长度
        # print(len(tokens_list), value)
        current_token_len = sum([tokens_list[int(v.split("-")[0])] for v in value.split(",")])

        # 计算前面所有tokens的总长度
        prev_token_len = sum([t for t in tokens_list[:int(node_v)]])/100 if int(node_v) < len(tokens_list) else 0
        # 计算子节点数量
        children_count = len(tree_dict["children"])
        # 计算当前层级的节点数量
        level_node_count = level_cnt[level]
        # 计算当前节点的深度（从根节点到当前节点的路径长度）
        depth = level
        
        nodes.append([
            int(node_v),  # 0 节点值的第一部分
            int(node_s),  # 1节点值的第二部分
            level,        # 2层级
            current_token_len,  # 3当前节点的token长度
            node_c,    # 4当前节点包含的thought
            prev_token_len,     # 5前面所有tokens的总长度
            len(nodes),         # 6
            children_count,     # 7子节点数量
            level_node_count,   # 8当前层级的节点数量
            distance            # 9 当前节点到最后节点的距离
        ])   
        nodes_dict[value] = len(nodes)-1
        
        for child_dict in tree_dict["children"]:
            child_value = child_dict["value"]
            child_node_v = child_value.split(",")[-1].split("-")[0] if str(child_value)!="0" else "0"
            child_level = child_dict["level"]
            
            # 计算边的特征
            edge_feat = [
                # level - child_level,  # 层级差
                # 1 
                child_dict["cate"] if edge_type[2] == "1" or edge_type[2] == "3" else 1
                # 1 if child_level!=0 and str(child_level) in critical_map and critical_map[str(child_level)] == int(child_node_v) else 0  # 子节点是否关键节点
            ]
            
            edges.append([nodes_dict[value], nodes_dict[child_value]])
            edge_features.append(edge_feat)

            if edge_type[1] == "1":
                edges.append([ nodes_dict[child_value], nodes_dict[value]])
                edge_features.append([-child_dict["cate"] if edge_type[2] == "2" or edge_type[2] == "3" else -1])

    dict_to_tree(root)

    x = torch.tensor(nodes, dtype=torch.float)
    
    # 创建边的特征张量ßß
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    edge_index = torch.tensor([[edge[0], edge[1]] for edge in edges], dtype=torch.long).t().contiguous()
    
    if edge_type[0] == '1':
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    else:
        return Data(x=x, edge_index=edge_index)



if __name__=="__main__":
    # 设置随机种子
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # path = sys.argv[1]


    paths = sys.argv[1].split(',')
    filename = sys.argv[2]
    selected_feat = [int(i) for i in sys.argv[3].split(',')] if sys.argv[3].lower() != 'none' else None
    edge_type = sys.argv[4]
    name = sys.argv[5]

    modelname = 'ds32'
    modelpath = '/mmu_nlp_hdd/wangbo27/data/modelscope/DeepSeek-R1-Distill-Qwen-32B'
    tokenizer = AutoTokenizer.from_pretrained(modelpath)

    for path in paths:
        graphs = []
        labels = []
        items = []
        input_path = os.path.join(path, filename)
        with open(input_path, "r") as f:
            for number, line in enumerate(f):
                item = json.loads(line)
                if type(item["thoughts_list"]) == str:
                    thought_list = json.loads(item["thoughts_list"])
                else:
                    thought_list = item["thoughts_list"]
                
                tokens_list = [len(tokenizer.encode(text)) for text in thought_list.values()]
                cottree = item['cot_tree']
                score = float(item['score'])
                try:
                    graph = tree_to_graph_with_cate(cottree, tokens_list, edge_type)
                except Exception as e:
                    print('wrong tree', e)
                    continue
                    
                items.append(item)
                graphs.append(graph)
                labels.append(score)

    for i, graph in enumerate(graphs):
        graph.y = torch.tensor([labels[i]], dtype=torch.long)
        graph = graph.to(device) 

    test_dataset = graphs
    test_items = items
    label_set = [1.0, 0.0]


    has_edge_attr = hasattr(test_dataset[0], 'edge_attr') and test_dataset[0].edge_attr is not None
    edge_dim = test_dataset[0].edge_attr.size(1) if has_edge_attr else 0

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = GATv2GraphClassifier(in_channels=len(selected_feat), hidden_channels=64, out_channels=2,
            edge_dim=edge_dim if has_edge_attr else None)
    model = model.to(device)  # 将模型移到GPU上

    try:
        model.load_state_dict(torch.load(f'checkpoints/{name}_gin_model.pth'))
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")


    best_test_acc, label_acc, predictions = test(selected_feat=selected_feat)
    print(f"Acc: {best_test_acc}")
    output_path = f"/mmu_nlp_hdd/jianggangwei/LcotRobust/checkpoints/{name}_score.json"

    try:
        with open(output_path, "w") as fout:
            for i, (item, pred) in enumerate(zip(test_items, predictions)):
                new_item ={
                    "tag": item['tag'],
                    'comb_tag': item['comb_tag'],
                    # 'comb_tag': comb_tag_map[item['tag']],
                    "gnn_score": pred
                }
                fout.write(json.dumps(new_item)+'\n')
    except:
        comb_tag_map = {}
        raw_file = f"/mmu_nlp_hdd/jianggangwei/LcotRobust/response/ds32_gpqa:main_multi_error.json"
        raw_file = f"/mmu_nlp_hdd/jianggangwei/LcotRobust/response/qwq_gpqa:main_multi_error.json"
        raw_file = f"/mmu_nlp_hdd/jianggangwei/LcotRobust/response/qwq_math_l5_multi_error.json"
        # raw_file = f"/mmu_nlp_hdd/jianggangwei/LcotRobust/response/ds32_math_l5_multi_error.json"

        with open(raw_file, "r") as f:
            for line in f:
                oitem = json.loads(line)
                comb_tag_map[oitem['tag']] = oitem['comb_tag']
        with open(output_path, "w") as fout:
            for i, (item, pred) in enumerate(zip(test_items, predictions)):
                new_item ={
                    "tag": item['tag'],
                    'comb_tag': comb_tag_map[item['tag']],
                    "gnn_score": pred
                }
                fout.write(json.dumps(new_item)+'\n')
    print(f'save gnn score to {output_path}')
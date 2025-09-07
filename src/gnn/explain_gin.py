import json
import pickle
import sys
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer, GraphMaskExplainer, PGExplainer, AttentionExplainer
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import numpy as np
import random
from transformers import AutoTokenizer
import copy
import os
from networks import GCN, GIN, GINAttention, GraphTransformer, GATv2GraphClassifier
sys.path.append("./src/cot2tree/")
from tree_utils import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test(datatype="test", selected_feat=None):
    model.eval()
    correct = 0
    prediction_avg = 0
    label_correct = {label: 0 for label in label_set}
    label_total = {label: 0 for label in label_set}
    predictions = []
    if datatype=="test": 
        data_loader = test_loader
        datanum = len(test_dataset)
    elif datatype=="valid": 
        data_loader = valid_loader
        datanum = len(valid_dataset)
    else:
        data_loader = train_loader
        datanum = len(train_dataset)
    has_edge_attr = hasattr(train_dataset[0], 'edge_attr') and train_dataset[0].edge_attr is not None
    for data in data_loader:
        data = data.to(device)  
        data.x = data.x[:, selected_feat]
        if has_edge_attr:
            out = model(data.x, data.edge_index, data.batch, data.edge_attr)
        else:
            out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        prediction_avg += int(pred.sum())
        correct += int((pred == data.y).sum())
        predictions.extend(pred.cpu().tolist())

        for i in range(len(data.y)):
            label = data.y[i].item()
            label_total[label] += 1
            if pred[i] == data.y[i]:
                label_correct[label] += 1

    label_acc = ""
    for label in label_set:
        if label_total[label] > 0:
            accuracy = label_correct[label] / label_total[label]
            print(f"Label {label} accuracy: {accuracy:.4f}")
            label_acc += f"Label {label} accuracy: {accuracy:.4f}\t"

    print(f"Ouput label mean: {prediction_avg / len(test_dataset):.4f}")
    return correct / len(test_dataset), label_acc, predictions

def tree_to_graph_with_cate(root, tokens_list, edge_type):
    nodes = []
    nodes_dict = {}
    edges = []
    edge_features = []  
    level_cnt = {0:1}
    def dict_to_tree(tree_dict, index):
        for num, child_dict in enumerate(tree_dict["children"]):
            dict_to_tree(child_dict, num)

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
        distance = (len(tokens_list) - int(node_v))/len(tokens_list)

        current_token_len = sum([tokens_list[int(v.split("-")[0])] for v in value.split(",")])

        prev_token_len = sum([t for t in tokens_list[:int(node_v)]])/100 if int(node_v) < len(tokens_list) else 0
        children_count = len(tree_dict["children"])
        level_node_count = level_cnt[level]
        depth = level
        
        nodes.append([
            int(node_v),  # 0 
            int(node_s),  # 1
            level,        # 2
            current_token_len,  # 3
            node_c,    # 4
            prev_token_len,     # 5
            index,         # 6
            children_count,     # 7
            level_node_count,   # 8
            distance            # 9
        ])   
        nodes_dict[value] = len(nodes)-1
        
        for child_dict in tree_dict["children"]:
            child_value = child_dict["value"]
            child_node_v = child_value.split(",")[-1].split("-")[0] if str(child_value)!="0" else "0"
            child_level = child_dict["level"]
            
            edge_feat = [
                child_dict["cate"] if edge_type[2] == "1" or edge_type[2] == "3" else 1
            ]
            
            edges.append([nodes_dict[value], nodes_dict[child_value]])
            edge_features.append(edge_feat)

            if edge_type[1] == "1":
                edges.append([ nodes_dict[child_value], nodes_dict[value]])
                edge_features.append([-child_dict["cate"] if edge_type[2] == "2" or edge_type[2] == "3" else -1])

    dict_to_tree(root,0)


    x = torch.tensor(nodes, dtype=torch.float)
    
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    edge_index = torch.tensor([[edge[0], edge[1]] for edge in edges], dtype=torch.long).t().contiguous()
    
    if edge_type[0] == '1':
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    else:
        return Data(x=x, edge_index=edge_index)

def load_or_create_cache(path, filename, tokenizer, selected_feat, edge_type, indp):
    
    print(f"Creating new dataset cache for {path}")
    graphs = {'00_0':[], '11_1':[], '01_1': [], '01_0': []}
    labels = {'00_0':[], '11_1':[], '01_1': [], '01_0': []}
    items = {'00_0':[], '11_1':[], '01_1': [], '01_0': []}
    
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
            # score = 1.0 if indp==0 else 0.0
            try:
                graph = tree_to_graph_with_cate(cottree, tokens_list, edge_type)
            except Exception as e:
                print('wrong tree', e)
                continue
                
            key = f"{item['tag'].split('_')[-4]}_{item['tag'][-1]}"
            items[key].append(item)
            graphs[key].append(graph)
            labels[key].append(score)

    
    return graphs, labels, items

if __name__=="__main__":
    # 设置随机种子
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # path = sys.argv[1]


    
    paths = sys.argv[1].split(',')
    filename = sys.argv[2]
    label_source = sys.argv[3]
    selected_feat = [int(i) for i in sys.argv[4].split(',')] if sys.argv[4].lower() != 'none' else None
    edge_feat = sys.argv[5]
    seed = int(sys.argv[6])
    try:
        name = sys.argv[7]
    except:
        name = 'base'

    visualize = True

    if 'ds32' in sys.argv[1] :
        modelname = 'ds32'
        modelpath = '/mmu_nlp_hdd/wangbo27/data/modelscope/DeepSeek-R1-Distill-Qwen-32B'
    tokenizer = AutoTokenizer.from_pretrained(modelpath)

    test_ratio = 0.2    
    test_dataset = []
    test_items = []
    label_set = [1.0, 0.0]
    for indp, path in enumerate(paths):
        graphs, labels, items = load_or_create_cache(path, filename, tokenizer, selected_feat, edge_feat, indp)
        for key in graphs.keys():
            print(key, ":", len(graphs[key]), "\n")
            for i, graph in enumerate(graphs[key]):
                if label_source == "1":
                    graph.y = torch.tensor([labels[key][i]], dtype=torch.long)
                else:
                    score = 1.0 if indp==0 else 0.0
                    graph.y = torch.tensor([score], dtype=torch.long)
                graph = graph.to(device) 
        test_items.extend(items['00_0'][-int(len(items['00_0'])*test_ratio):]
                    + items['11_1'][-int(len(items['11_1'])*test_ratio):] 
                    + items['01_0'][-int(len(items['01_0'])*test_ratio):] 
                    + items['01_1'][-int(len(items['01_1'])*test_ratio):])
        test_dataset.extend(graphs['00_0'][-int(len(graphs['00_0'])*test_ratio):]
                    + graphs['11_1'][-int(len(graphs['11_1'])*test_ratio):] 
                    + graphs['01_0'][-int(len(graphs['01_0'])*test_ratio):]
                    + graphs['01_1'][-int(len(graphs['01_1'])*test_ratio):])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    has_edge_attr = hasattr(test_dataset[0], 'edge_attr') and test_dataset[0].edge_attr is not None
    edge_dim = test_dataset[0].edge_attr.size(1) if has_edge_attr else 0

    model = GATv2GraphClassifier(in_channels=len(selected_feat), hidden_channels=64, out_channels=2,
            edge_dim=edge_dim if has_edge_attr else None)
    model = model.to(device) 
    try:
        model.load_state_dict(torch.load(f'checkpoints/{name}_gin_model.pth'))
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")

    best_test_acc, label_acc, predictions = test(selected_feat=selected_feat)
    print(f"Test Acc: {best_test_acc:.4f}")

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        # node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs',
        ),
    ) 
    visual_idx = 10 if label_source=='1' else 230
    visual_num = 20
    batch_size=1
    test_loader = DataLoader(test_dataset[visual_idx:], batch_size=batch_size, shuffle=False)

    node_index = 0

    colors = []
    cnt = 1
    for data in test_loader:
        if cnt > visual_num:
            break
        data = data.to(device)
        data.x = data.x[:, selected_feat]

        if has_edge_attr:
            explanation = explainer(data.x, data.edge_index, target=data.y, index=node_index, batch=data.batch, edge_attr=data.edge_attr)
        else:
            explanation = explainer(data.x, data.edge_index, target=data.y, index=node_index, batch=data.batch)
            # out = model(data.x, data.edge_index, data.batch)
        
        edge_weight = explanation.edge_mask
        edge_weight = edge_weight - edge_weight.min()
        edge_weight = edge_weight / edge_weight.max() 
        ind = 1.5
        edge_weight_t = edge_weight ** ind
        while len([1 for t in edge_weight_t if t > 0.66]) > len(edge_weight)/5:
            ind += 0.2 
            edge_weight_t = edge_weight ** ind
            if ind > 5:
                break
        edge_weight = edge_weight_t
        edge_weight = edge_weight * 0.9 + 0.1
        # print(edge_weight)
        colors.append([i.item() for i in edge_weight])

        cnt += 1

    if visualize:
        visualize_folder = os.path.join(path, "visualize_results" if label_source=='1' else 'visualize_results_1')
        os.makedirs(visualize_folder, exist_ok=True)

        # 可视化测试集的树
        cnt = {"11":0, "10":0, "00":0, "01":0}
        for i, (item, pred) in enumerate(zip(test_items, predictions)):
            if i not in [batch_size*j + visual_idx for j in range(visual_num)]:
                continue
            # if i > visual_idx+visual_num-1:
            #     break
            if type(item["thoughts_list"]) == str:
                thought_list = json.loads(item["thoughts_list"])
            else:
                thought_list = item["thoughts_list"]
            reasoning_text = extract_reasoning_dict(item["reasoning_sketch"])
            try:
                tree_root = dict_to_tree_with_text(item["cot_tree"], thought_list)
            except:
                continue
            label = test_dataset[i].y.cpu().tolist()[0]
            ind = cnt[f"{label}{pred}"]
            cnt[f"{label}{pred}"] += 1
            output_path = os.path.join(visualize_folder, f"{ind}_{label}_{pred}.html")
            # print(colors[(i-visual_idx)//batch_size])
            # visualize_tree(tree_root, output_path, reasoning_text, item)
            try:
                visualize_tree(tree_root, output_path, reasoning_text, item, colors[(i-visual_idx)//batch_size])
            except Exception as e:
                print(e)
                continue
                # cnt[f"{label}{pred}"] -= 1



    visual_idx = 230 if label_source=='1' else 630
    visual_idx = 230 if label_source=='1' else 530
    visual_num = 20
    batch_size=1
    test_loader = DataLoader(test_dataset[visual_idx:], batch_size=batch_size, shuffle=False)

    node_index = 0

    colors = []
    cnt = 1
    for data in test_loader:
        if cnt > visual_num:
            break
        data = data.to(device)
        data.x = data.x[:, selected_feat]
        if has_edge_attr:
            explanation = explainer(data.x, data.edge_index, target=data.y, index=node_index, batch=data.batch, edge_attr=data.edge_attr)
        else:
            explanation = explainer(data.x, data.edge_index, target=data.y, index=node_index, batch=data.batch)
            # out = model(data.x, data.edge_index, data.batch)

        
        edge_weight = explanation.edge_mask
        edge_weight = edge_weight - edge_weight.min()
        edge_weight = edge_weight / edge_weight.max() 
        ind = 1.5
        edge_weight_t = edge_weight ** ind
        while len([1 for t in edge_weight_t if t > 0.66]) > len(edge_weight)/5:
            ind += 0.2 
            edge_weight_t = edge_weight ** ind
            if ind > 5:
                break
        edge_weight = edge_weight_t
        edge_weight = edge_weight * 0.9 + 0.1
        # print(edge_weight)
        colors.append([i.item() for i in edge_weight])

        cnt += 1
    # path = 'feature_importance.png'
    # explanation.visualize_feature_importance(path, top_k=10)
    # print(f"Feature importance plot has been saved to '{path}'")


    if visualize:
        visualize_folder = os.path.join(path, "visualize_results" if label_source=='1' else 'visualize_results_1')
        # visualize_folder = os.path.join(path, "visualize_results")
        os.makedirs(visualize_folder, exist_ok=True)

        # 可视化测试集的树
        cnt = {"11":0, "10":0, "00":0, "01":0}
        for i, (item, pred) in enumerate(zip(test_items, predictions)):
            if i not in [batch_size*j + visual_idx for j in range(visual_num)]:
                continue
            # if i > visual_idx+visual_num-1:
            #     break
            if type(item["thoughts_list"]) == str:
                thought_list = json.loads(item["thoughts_list"])
            else:
                thought_list = item["thoughts_list"]
            reasoning_text = extract_reasoning_dict(item["reasoning_sketch"])
            try:
                tree_root = dict_to_tree_with_text(item["cot_tree"], thought_list)
            except:
                continue
            label = test_dataset[i].y.cpu().tolist()[0]
            ind = cnt[f"{label}{pred}"]
            cnt[f"{label}{pred}"] += 1
            output_path = os.path.join(visualize_folder, f"{ind}_{label}_{pred}.html")
            # print(colors[(i-visual_idx)//batch_size])
            # visualize_tree(tree_root, output_path, reasoning_text, item)
            
            visualize_tree(tree_root, output_path, reasoning_text, item, colors[(i-visual_idx)//batch_size])
                # cnt[f"{label}{pred}"] -= 1

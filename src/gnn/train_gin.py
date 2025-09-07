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
    # for data in test_loader:
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
    da = []
    for label in label_set:
        if label_total[label] > 0:
            accuracy = label_correct[label] / label_total[label]
            # print(f"Label {label} accuracy: {accuracy:.4f}")
            label_acc += f"Label {label} accuracy: {accuracy:.4f}\t"
            da.append(accuracy)

    print(f"Ouput label mean: {prediction_avg / datanum:.4f}")
    return correct / datanum, (label_acc,da), predictions


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


def evaluate_feature_combination(feature_combination, train_dataset, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    accuracies = []
    
    has_edge_attr = hasattr(train_dataset[0], 'edge_attr') and train_dataset[0].edge_attr is not None
    edge_dim = train_dataset[0].edge_attr.size(1) if has_edge_attr else 0
    print(edge_dim)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        train_fold = [train_dataset[i] for i in train_idx]
        val_fold = [train_dataset[i] for i in val_idx]
        
        train_loader = DataLoader(train_fold, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_fold, batch_size=32, shuffle=False)
        model = GATv2GraphClassifier(in_channels=len(feature_combination), hidden_channels=64, out_channels=2,
            edge_dim=edge_dim if has_edge_attr else None, lin=lin, num_layer=num_l)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        best_val_acc = 0
        for epoch in range(20):  
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                data.x = data.x[:, feature_combination]
                if has_edge_attr:
                    out = model(data.x, data.edge_index, data.batch, data.edge_attr)
                else:
                    out = model(data.x, data.edge_index, data.batch)
                loss = F.nll_loss(out, data.y)
                loss.backward()
                optimizer.step()
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    data.x = data.x[:, feature_combination]
                    if has_edge_attr:
                        out = model(data.x, data.edge_index, data.batch, data.edge_attr)
                    else:
                        out = model(data.x, data.edge_index, data.batch)
                    pred = out.argmax(dim=1)
                    correct += int((pred == data.y).sum())
                    total += len(data.y)
            val_acc = correct / total
            best_val_acc = max(best_val_acc, val_acc)
        
        accuracies.append(best_val_acc)
    
    return np.mean(accuracies), np.std(accuracies)

def find_best_feature_combination(train_dataset, fixed_features=None, max_additional_features=3):
    all_features = list(range(10))
    if fixed_features:
        remaining_features = [f for f in all_features if f not in fixed_features]
    else:
        remaining_features = all_features
        fixed_features = []

    best_combination = fixed_features.copy()
    best_score = 0
    best_std = 0
    
    results = []
    
    for n_features in range(1, max_additional_features + 1):
        print(f"\nEvaluating combinations with {n_features} additional features...")
        for feature_combination in combinations(remaining_features, n_features):
            current_combination = fixed_features + list(feature_combination)
            mean_acc, std_acc = evaluate_feature_combination(current_combination, train_dataset)
            results.append({
                'combination': current_combination,
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc
            })
            
            if mean_acc > best_score:
                best_score = mean_acc
                best_std = std_acc
                best_combination = current_combination
                print(f"New best combination found: {current_combination}")
                print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    results.sort(key=lambda x: x['mean_accuracy'], reverse=True)
    
    print("\nTop 5 feature combinations:")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. Features: {result['combination']}")
        print(f"   Mean accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    
    return best_combination, best_score, best_std



if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    paths = sys.argv[1].split(',')
    filename = sys.argv[2]
    selected_feat = [int(i) for i in sys.argv[3].split(',')] if sys.argv[3].lower() != 'none' else None
    edge_feat = sys.argv[4]
    seed = int(sys.argv[5])
    try:
        name = sys.argv[6]
    except:
        name = 'base'

    if 'ds32' in sys.argv[1] :
        modelname = 'ds32'
        modelpath = '/mmu_nlp_hdd/wangbo27/data/modelscope/DeepSeek-R1-Distill-Qwen-32B'
    tokenizer = AutoTokenizer.from_pretrained(modelpath)

    train_ratio = 0.72
    valid_ratio = 0.08
    test_ratio = 0.2
    train_dataset = []
    valid_dataset = []
    test_dataset = []
    test_items = []
    for indp, path in enumerate(paths):
        graphs, labels, items = load_or_create_cache(path, filename, tokenizer, selected_feat, edge_feat, indp)
        
        for key in graphs.keys():
            print(key, ":", len(graphs[key]), "\n")
            for i, graph in enumerate(graphs[key]):
                graph.y = torch.tensor([labels[key][i]], dtype=torch.long)
                graph = graph.to(device)


        train_dataset.extend(graphs['00_0'][:int(len(graphs['00_0'])*train_ratio)] 
                        + graphs['11_1'][:int(len(graphs['11_1'])*train_ratio)]
                        + graphs['01_0'][:int(len(graphs['01_0'])*train_ratio)]
                        + graphs['01_1'][:int(len(graphs['01_1'])*train_ratio)])
        valid_dataset.extend(copy.deepcopy(graphs['00_0'][int(len(graphs['00_0'])*train_ratio):int(len(graphs['00_0'])*train_ratio+valid_ratio)] 
                        + graphs['11_1'][int(len(graphs['11_1'])*train_ratio):int(len(graphs['11_1'])*train_ratio+valid_ratio)]
                        + graphs['01_0'][int(len(graphs['01_0'])*train_ratio):int(len(graphs['01_0'])*train_ratio+valid_ratio)] 
                        + graphs['01_1'][int(len(graphs['01_1'])*train_ratio):int(len(graphs['01_1'])*train_ratio+valid_ratio)]))
        test_items.extend(items['00_0'][-int(len(items['00_0'])*test_ratio):]
                    + items['11_1'][-int(len(items['11_1'])*test_ratio):] 
                    + items['01_0'][-int(len(items['01_0'])*test_ratio):] 
                    + items['01_1'][-int(len(items['01_1'])*test_ratio):])
        test_dataset.extend(graphs['00_0'][-int(len(graphs['00_0'])*test_ratio):]
                    + graphs['11_1'][-int(len(graphs['11_1'])*test_ratio):] 
                    + graphs['01_0'][-int(len(graphs['01_0'])*test_ratio):]
                    + graphs['01_1'][-int(len(graphs['01_1'])*test_ratio):])
        
            
    print(f"{sys.argv[1]}, train size: {len(train_dataset)}, valid size: {len(valid_dataset)}, test size {len(test_items)} ")

    label_set = [1.0, 0.0]

    set_seed(seed)

    if selected_feat is None:
        print("\nSearching for best feature combination...")
        fixed_features = [0, 2, 5]
        best_combination, best_score, best_std = find_best_feature_combination(train_dataset, fixed_features=fixed_features, max_additional_features=2)
        print(f"\nBest feature combination: {best_combination}")
        print(f"Mean accuracy: {best_score:.4f} ± {best_std:.4f}")
        selected_feat = list(best_combination)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    has_edge_attr = hasattr(train_dataset[0], 'edge_attr') and train_dataset[0].edge_attr is not None
    edge_dim = train_dataset[0].edge_attr.size(1) if has_edge_attr else 0

    final_results = []
    for round in range(5):
        model = GATv2GraphClassifier(in_channels=len(selected_feat), hidden_channels=64, out_channels=2,
            edge_dim=edge_dim if has_edge_attr else None, lin=lin, num_layer=num_l)
        # model = GINAttention(in_channels=3, hidden_channels=64, out_channels=len(set(labels)))
        # model = GCN(in_channels=3, hidden_channels=64, out_channels=len(set(labels)))
        # model = GraphTransformer(in_channels=3, hidden_channels=64, out_channels=len(set(labels)))
        model = model.to(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        best_valid_acc = 0
        best_model = none
        
        for epoch in range(1, 100):

            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                data.x = data.x[:, selected_feat]

                if has_edge_attr:
                    out = model(data.x, data.edge_index, data.batch, data.edge_attr)
                else:
                    out = model(data.x, data.edge_index, data.batch)
                loss = F.nll_loss(out, data.y)
                loss.backward()
                optimizer.step()
            loss = loss.item()
            valid_acc, (label_acc, label_acc_data), _ = test('valid',selected_feat=selected_feat)
            if valid_acc > best_valid_acc:
                best_valid_acc = max(valid_acc, best_valid_acc)
            # if label_acc_data[1] > best_valid_acc:
            #     best_valid_acc = max(label_acc_data[1], best_valid_acc)
                best_epoch = epoch
                best_model = copy.deepcopy(model)
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Valid Acc: {valid_acc:.4f}')
                valid_acc, label_acc, _ = test('train',selected_feat=selected_feat)
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {valid_acc:.4f}')

        model = best_model
        best_test_acc, label_acc, predictions = test('test', selected_feat=selected_feat)
        final_results.append(best_test_acc)
        print(f"Round {round}, best acc {best_test_acc}\n")
    # 保存模型
    torch.save(best_model.state_dict(), f'checkpoints/{name}_gin_model.pth')

    model = best_model

    train_acc, label_acc, predictions = test('train', selected_feat=selected_feat)

    print(f"Best Epoch: {best_epoch:03d}, Train Acc: {train_acc:.4f}")

    _, label_acc, predictions = test(selected_feat=selected_feat)
    test_acc, label_acc, predictions = test('test', selected_feat=selected_feat)

    print(f"Best Epoch: {best_epoch:03d}, Test Acc: {test_acc:.4f}")

    print(f"Mean Acc: {np.mean(final_results):.4f} ± {np.std(final_results):.4f}")
        
    result_file = "results.txt"
    # name = 
    with open(result_file, 'a') as f:
        f.write(f"Paths: {paths}, Filename: {filename}, Feat: {selected_feat}, Edge: {edge_feat}, Seed: {seed}, Mean Acc: {np.mean(final_results):.4f} ± {np.std(final_results):.4f}, "
                f"Best Epoch: {best_epoch}, Test Acc: {best_test_acc:.4f}, {label_acc}\n")
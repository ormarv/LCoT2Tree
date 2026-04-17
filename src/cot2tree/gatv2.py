from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.data import Data
import networkx as nx
from scipy.sparse import coo_matrix
from torch_geometric.utils import from_scipy_sparse_matrix, scatter
import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from typing import List, Dict, Tuple
import numpy as np

class GAT(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int, hidden:int=64):
        super().__init__()
        self.hidden = hidden
        self.conv1 = GATv2Conv(in_channels=in_channels, out_channels=self.hidden)
        self.conv2 = GATv2Conv(in_channels=self.hidden, out_channels=self.hidden)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(self.hidden, self.hidden),
            torch.nn.BatchNorm1d(self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, out_channels)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return torch.nn.functional.log_softmax(x, dim=1)

def get_edge_index(graph:nx.DiGraph):
    adjacency_matrix = nx.to_numpy_array(graph)
    coo = coo_matrix(adjacency_matrix)
    edge_index = np.array([coo.row, coo.col])
    return edge_index

def build_features(graph:nx.DiGraph, all_features:List[List[float]], wanted_features:Dict[str, int])->List[List[float]]:
    # compute the number of children per node
    # compute the distance to the last node, computed as the number of words and given
    dict_graph = nx.to_dict_of_dicts(graph)
    if 'nb_children' in wanted_features:
        for i in range(len(all_features)):
            all_features[i][wanted_features['nb_children']] = float(len(dict_graph[i]))
    if 'nb_nodes_per_depth' in wanted_features:
        node_to_depth = {0:0}
        depth_to_node = {0:[0]}
        parents = [0]
        while(len(parents))>0:
            parent = parents.pop(0)
            children = dict_graph[parent]
            for child in children:
                if child not in node_to_depth:
                    node_to_depth[child] = node_to_depth[parent] + 1
                    if node_to_depth[parent] + 1 not in depth_to_node:
                        depth_to_node[node_to_depth[parent] + 1] = []
                    depth_to_node[node_to_depth[parent] + 1].append(child)
                if child not in parents:
                    parents.append(child)
        nb_nodes_per_depth = {key:value.sum() for key, value in depth_to_node.items()}
        for i in range(len(all_features)):
            node_depth = node_to_depth[i]
            all_features[i][wanted_features['nb_nodes_per_depth']] = float(nb_nodes_per_depth[node_depth])
    return all_features



def build_dataloader(all_features:List[torch.Tensor], graphs:List[nx.DiGraph], labels:List[int],batch_size:int=32)->List[DataLoader]:
    # build Data objects
    iterator = zip(graphs, all_features, labels)
    datas = [Data(x=features, edge_index=get_edge_index(graph), y=label) for graph, features, label in iterator]
    loader = DataLoader(datas, batch_size=batch_size)
    return loader

def train(train_dataloader:DataLoader, val_loader:DataLoader, epochs:int=100, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device', device)
    model = GAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_evaluation_accuracy = 0
    for i, epoch in enumerate(epochs):
        print(f"-------------------------------EPOCH N°{i}-------------------------------")
        print("    Training")
        model.train()
        loss_all = 0
        total_correct_train = 0
        total_train = 0
        for j, data in enumerate(train_dataloader):
            # data is actually is a batch of Data objects (abc.DataBatch)
            # data.batch is a Tensor indicating which graph each node corresponds to
            data = data.to(device)
            print(type(data))
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            prediction = output.argmax(dim=1)
            correct = int((prediction == data.y).sum())
            acc = correct/len(data.y)
            total_correct_train += correct
            total_train +=len(data.y)
            loss = torch.nn.functional.nll_loss(output, data.y)
            loss_all += loss.item()
            loss.backward()
            optimizer.step()
            print(f"    Batch {j}. Loss: {loss.item()}. Accuracy: {acc}")
        avg_train_loss = loss_all/len(train_dataloader.dataset)
        print(f"Training loss: {avg_train_loss}")
        print("\n")
        print("    Evaluation")
        model.eval()
        total_correct_eval = 0
        total_eval = 0
        loss_eval = 0
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                data = data.to(device)
                output = model(data.x, data.edge_index, data.batch)
                loss = torch.nn.functional.nll_loss(output, data.y)
                loss_eval += loss.item()
                prediction = output.argmax(dim=1)
                correct = int((prediction == data.y).sum())
                acc = correct/len(data.y)
                print(f"    Batch {j}. Loss: {loss.item()}. Accuracy: {acc}")
                total_correct_eval += correct
                total_eval += len(data.y)
        avg_loss_eval = loss_eval/len(val_loader.dataset)
        print(f"Evaluation loss: {avg_loss_eval}")
        evaluation_accuracy = total_correct_eval/total_eval
        print(f"Evaluation accuracy: {evaluation_accuracy}")
        best_evaluation_accuracy = max(best_evaluation_accuracy, evaluation_accuracy)
        print("\n\n")
    print(f"Best evaluation accuracy: {best_evaluation_accuracy}")
    return model
    
def test(test_dataloader:DataLoader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device', device)
    model = GAT().to(device)
    total_correct = 0
    total = 0
    prediction_average = 0
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            data = data.to(device)
            output = model(data.x, data.egde_attribute, data.batch)
            predictions = output.argmax(dim=1)
            prediction_average += int(predictions.sum())
            correct = int((predictions==data.y).sum())
            acc = correct/len(data.y)
            print(f"    Batch {i}. Accuracy: {acc}")
            total_correct += correct
            total += int(len(data.y))
            all_predictions.extend(predictions.cpu().tolist())
    avg_accuracy = total_correct/total
    print(f"Average test accuracy: {avg_accuracy}")


# in_channels
g = nx.DiGraph()
g.add_node(1)
g.add_node(2)
g.add_node(3)
g.add_node(4)
g.add_edge(1,2)
g.add_edge(1,3)
g.add_edge(3,4)
h = nx.DiGraph()
h.add_node(1)
h.add_node(2)
h.add_node(3)
h.add_edge(1,2)
h.add_edge(2,3)
adj = nx.to_numpy_array(g)
print(adj.shape)
print(adj)
coo = coo_matrix(adj)
print(coo)
print(coo.shape)
print(coo.col)
print(coo.row)
print(np.array([coo.row, coo.col]))
# let's use fake features for now
x = torch.tensor([[0],[0],[0],[0]],dtype=torch.float)
y = torch.tensor([[1],[0],[0],[0]],dtype=torch.float)
#print(x)
data = Data(x=x, edge_index=coo, y=0)
d1 = Data(x=y, edge_index=coo_matrix(nx.to_numpy_array(g)), y=1)
loader = DataLoader([data, d1], batch_size=2)

#for d in loader:
    # d is a batch, not a Data object
"""print(type(d))
print(f"d.batch: {d.batch}")
print(d.y)"""
#print(type(x))
#gat = GATv2Conv((4,1), 2)
#x = gat.forward(x=x, edge_index=coo)
#print(x)
src = torch.Tensor([[1,2,7,4,6],[12,3,9,8,0]])
index = torch.tensor([0, 1, 0, 2, 2])
#print(src)
#print(index)
# Broadcasting in the first and last dim.
out = scatter(src, index, dim=1, reduce="sum")
#print(out)
#dataloader = DataLoader()
"""for data in dataloader:
    print(type(data))
"""
wanted_features = {'nb_parents':0, 'nb_children':1, 'nb_words_before':2, 'node_index':3, 'nb_nodes_per_depth':4, 'distance_to_end':5}

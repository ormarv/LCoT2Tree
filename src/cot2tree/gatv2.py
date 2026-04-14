from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.data import Data
import networkx as nx
from scipy.sparse import coo_matrix
from torch_geometric.utils import from_scipy_sparse_matrix, scatter
import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from typing import List, Dict, Tuple

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
    return coo

def build_features(graph:nx.DiGraph):
    # compute the number of children per node
    # compute the distance to the last node, computed as the number of words and given
    raise NotImplementedError

def build_dataloader(all_features, graphs:List[nx.DiGraph], labels:List[int],batch_size:int=32)->List[DataLoader]:
    # build Data objects
    iterator = zip(graphs, all_features, labels)
    datas = [Data(x=features, edge_index=get_edge_index(graph), y=label) for graph, features, label in iterator]
    loader = DataLoader(datas, batch_size=batch_size)
    return loader

def train(train_dataloader:DataLoader, epochs:int=100, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device', device)
    model = GAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        model.train()
        loss_all = 0
        for data in train_dataloader:
            # data is actually is a batch of Data objects (abc.DataBatch)
            # data.batch is a Tensor indicating which graph each node corresponds to
            data = data.to(device)
            print(type(data))
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            loss = torch.nn.functional.nll_loss(output, data.y)
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
# let's use fake features for now
x = torch.tensor([[0],[0],[0],[0]],dtype=torch.float)
y = torch.tensor([[1],[0],[0],[0]],dtype=torch.float)
print(x)
data = Data(x=x, edge_index=coo, y=0)
d1 = Data(x=y, edge_index=coo_matrix(nx.to_numpy_array(g)), y=1)
loader = DataLoader([data, d1], batch_size=2)

for d in loader:
    # d is a batch, not a Data object
    print(type(d))
    print(f"d.batch: {d.batch}")
    print(d.y)
print(type(x))
#gat = GATv2Conv((4,1), 2)
#x = gat.forward(x=x, edge_index=coo)
print(x)
src = torch.Tensor([[1,2,7,4,6],[12,3,9,8,0]])
index = torch.tensor([0, 1, 0, 2, 2])
print(src)
print(index)
# Broadcasting in the first and last dim.
out = scatter(src, index, dim=1, reduce="sum")
print(out)
dataloader = DataLoader()
for data in dataloader:
    print(type(data))

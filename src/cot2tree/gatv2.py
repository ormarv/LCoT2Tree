from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.data import Data, DataLoader
import networkx as nx
from scipy.sparse import coo_matrix
from torch_geometric.utils import from_scipy_sparse_matrix, scatter
import torch


def build_features(graph:nx.DiGraph):
    raise NotImplementedError

class Classifier():
    def __init__(self, in_channels:int, out_channels:int, hidden:int)->None:
        self.gnn1 = GATv2Conv(in_channels=in_channels,out_channels=hidden)
        self.gnn2 = GATv2Conv(in_channels=hidden,out_channels=out_channels)


# in_channels
g = nx.DiGraph()
g.add_node(1)
g.add_node(2)
g.add_node(3)
g.add_node(4)
g.add_edge(1,2)
g.add_edge(1,3)
g.add_edge(3,4)
adj = nx.to_numpy_array(g)
print(adj.shape)
print(adj)
coo = coo_matrix(adj)
print(coo)
print(coo.shape)
# let's use fake features for now
x = torch.tensor([[0],[0],[0],[0]],dtype=torch.float)
print(x)
print(type(x))
data = Data(x=x, edge_index=coo)
# we might not need the Data type
print(data)
y = from_scipy_sparse_matrix(coo)
print(y)
print(y[0])
print(y[1])
gat = GATv2Conv(1, 2)
# only the first element (the edges) interest us, because all our values are 1.
x = gat.forward(x=x, edge_index=y[0])
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

from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.data import Data
import networkx as nx
from scipy.sparse import coo_matrix
import torch


def build_features(graph:nx.DiGraph):
    raise NotImplementedError

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
gat = GATv2Conv((4,1), 2)
x = gat.forward(x=x, edge_index=coo)
print(x)

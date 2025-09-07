import json
import sys
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, TransformerConv, GCNConv
import torch.nn.functional as F
import os


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_ratio=0.0):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_ratio),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_ratio=0.0):
        super(GraphTransformer, self).__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels)
        self.conv3 = TransformerConv(hidden_channels, hidden_channels)
        self.conv4 = TransformerConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_ratio),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_ratio=0.0):
        super(GIN, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),  # Batch normalization
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),  # Batch normalization
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv3 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),  # Batch normalization
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        self.conv4 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),  # Batch normalization
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),  # Batch normalization
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_ratio),  # Dropout
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch):
        # print(x.shape)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = self.conv4(x, edge_index)
        # print(x.shape)
        x = global_mean_pool(x, batch)
        # print(x.shape)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


class AttentionLayer(torch.nn.Module):
    def __init__(self, in_channels):
        super(AttentionLayer, self).__init__()
        self.attn = torch.nn.Linear(in_channels, 1)

    def forward(self, x, edge_index):
        attn_scores = self.attn(x)
        attn_scores = torch.sigmoid(attn_scores)
        row, col = edge_index
        attn_weights = attn_scores[row] * attn_scores[col]
        return attn_weights
        
class GINAttention(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_ratio=0.0):
        super(GINAttention, self).__init__()
        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        self.attn1 = AttentionLayer(hidden_channels)
        self.conv2 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        ))
        self.attn2 = AttentionLayer(hidden_channels)
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_ratio),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        attn_weights = self.attn1(x, edge_index)
        x = attn_weights.view(-1, 1) * x[edge_index[1]]
        x = global_add_pool(x, edge_index[1])
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        attn_weights = self.attn2(x, edge_index)
        x = attn_weights.view(-1, 1) * x[edge_index[1]]
        x = global_add_pool(x, edge_index[1])
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)



from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATv2GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=None):
        super(GATv2GraphClassifier, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels,
            edge_dim=edge_dim,
            add_self_loops=True
        )
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels,
            edge_dim=edge_dim,
            add_self_loops=True
        )
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        if edge_attr is not None:
            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_attr)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)
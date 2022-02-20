import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv
from torch_geometric.nn.norm.batch_norm import BatchNorm


class GAT(torch.nn.Module):
    def __init__(self, input_size=100, n_channels=48, heads: int = 1, negative_slope: float = 0.2,
                 dropout: float = 0.0, depth=5):
        super().__init__()
        self.input_size = input_size
        self.n_channels = n_channels
        if not self.n_channels:
            self.n_channels = self.input_size
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.depth = depth
        self.GATConv_list = torch.nn.ModuleList()
        self.activation = torch.nn.ELU()
        for i in range(self.depth):
            in_channels, out_channels = self.n_channels, self.n_channels
            if i == 0:
                in_channels = self.input_size
            if i == self.depth - 1:
                out_channels = 48
            self.GATConv_list.append(GATConv(in_channels, out_channels, concat=True, heads=self.heads,
                                             negative_slope=self.negative_slope, dropout=self.dropout))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for gat in self.GATConv_list:
            x = F.dropout(x, training=self.training)
            x = gat(x, edge_index)
            x = self.activation(x)
        x = F.dropout(x, training=self.training)
        return x
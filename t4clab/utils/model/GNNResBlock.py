import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm.batch_norm import BatchNorm

class GNNResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gcn_conv = GCNConv(in_channels, out_channels, normalize=False)
        self.activation = torch.nn.ReLU()
        self.batch_norm = BatchNorm(self.out_channels)

    def forward(self, x, edge_index):
        input_x = x
        x = self.gcn_conv(x, edge_index)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = F.dropout(x, training=self.training)
        if self.in_channels == self.out_channels:
            x = torch.add(x, input_x)
        return x
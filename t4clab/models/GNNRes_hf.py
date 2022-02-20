import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from t4clab.utils.model.GNNResBlock import GNNResBlock


class GNNRes(torch.nn.Module):
    def __init__(self, n_channels=10, inout_skipconn=True, depth=5):
        super().__init__()
        self.n_channels = n_channels
        self.inout_skipconn = inout_skipconn
        self.depth = depth
        self.resblock_list = torch.nn.ModuleList()
        for i in range(self.depth):
            out_channels, in_channels = self.n_channels, self.n_channels
            if i == 0:
                in_channels = 10
            if i == self.depth-1:
                out_channels = 10
            self.resblock_list.append(GNNResBlock(in_channels, out_channels))
        self.final_conv = GCNConv(20, 8)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for res_block in self.resblock_list:
            x = res_block(x, edge_index)
        
        if self.inout_skipconn:
            x = torch.cat((x, data.x), 1)
            
        x = self.final_conv(x, edge_index)

        return x

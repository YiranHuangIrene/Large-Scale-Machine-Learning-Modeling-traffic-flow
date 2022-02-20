from numpy import reshape
import torch
from torch_geometric.nn import GCNConv
from t4clab.utils.model.GNNResBlock import GNNResBlock


class GNNRes(torch.nn.Module):
    def __init__(self, input_size=100, n_channels=None, inout_skipconn=True, depth=5, activation="Sigmoid"):
        super().__init__()
        self.n_channels = n_channels
        if not self.n_channels:
            self.n_channels = self.input_size
        self.inout_skipconn = inout_skipconn
        self.depth = depth
        self.input_size = input_size
        self.resblock_list = torch.nn.ModuleList()
        for i in range(self.depth):
            out_channels, in_channels = self.n_channels, self.n_channels
            if i == 0:
                in_channels = self.input_size
            if i == self.depth-1:
                out_channels = self.input_size
            self.resblock_list.append(GNNResBlock(in_channels, out_channels))
        if self.inout_skipconn:
            self.final_conv = GCNConv(self.input_size*2, 48)
        else:
            self.final_conv = GCNConv(self.input_size, 48)
        self.activation = activation
        if activation == "Sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        elif activation == "ReLU":
            self.final_activation = torch.nn.ReLU()

    def forward(self, data, mask):
        x, edge_index = data.x, data.edge_index

        for res_block in self.resblock_list:
            x = res_block(x, edge_index)
        
        if self.inout_skipconn:
            x = torch.cat((x, data.x), 1)
           
        x = self.final_conv(x, edge_index)


        if self.activation == "Sigmoid":
            x = self.final_activation(x)
            x = x * 255
        elif self.activation == "ReLU":
            x = self.final_activation(x)
        
        shorted_data = data.x[:, :96]
        reshaped_data = shorted_data.reshape(215820, 12, 8)
        channel_wise_mean = torch.mean(reshaped_data, 1)
        mean = channel_wise_mean.repeat(1, 6)
        x = torch.add(x, mean)

        x = x * mask.flatten().repeat(x.shape[-1],1).T.reshape(-1, 215820, 48)
        
        return x

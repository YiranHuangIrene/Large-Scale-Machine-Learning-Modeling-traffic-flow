import torch
from torch_geometric.nn import GCNConv
from t4clab.utils.model.GNNResBlock import GNNResBlock
import torch_geometric
from t4clab.logging import get_logger, log_hyperparameters, print_config


log = get_logger()


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
        self.mask = None

    def forward(self, data, avg, mask):
        """
        average = avg.permute(0, 2, 3, 1, 4)
        """

        x, edge_index = data.x, data.edge_index
        edges_no_self_loops = torch_geometric.utils.remove_self_loops(edge_index)[0]
        pre_conv = GCNConv(self.input_size, self.input_size).to("cuda")
        x1 = pre_conv(x, edges_no_self_loops)
        x = (x != 0) * x + (x == 0) * x1
        del x1
        del edges_no_self_loops
        del pre_conv
        torch.cuda.empty_cache()
        
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
        i = 12

        """
        output_average = average[:, :, :, [i, i+1, i+2, i+5, i+8, i+11], :]
        output_average_reshaped = output_average.reshape(-1, 215820, 48)
        x = torch.add(x, output_average_reshaped)
        """
        x = x * mask.flatten().repeat(x.shape[-1],1).T.reshape(-1, 215820, 48)
        
        return x

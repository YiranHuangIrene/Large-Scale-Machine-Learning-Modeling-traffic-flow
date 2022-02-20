from os import stat
import torch
from torch_geometric.data import Data
import math


def convert_static_image_to_adj_matrix(staticdata, threshold):
    mask = staticdata[0]>threshold
    width = mask.shape[1]
    
    end_nodes_calculation ={1: -width, 2: -width+1, 3: 1, 4:width+1, 5:width, 6:width-1, 7:-1, 8: -width-1}
    edges = []
    
    for k in range(1,staticdata.shape[0]):
        edges = edges + [[i, i+end_nodes_calculation[k]] for i, x in enumerate((staticdata[k]*mask).flatten()>0) if x]

    edges = torch.tensor(edges).T
    
    return mask, edges


def get_all_neighbours_of_node(node_id, edges):
    """
    returns a tensor with all neighbours of a node (index)
    @param node_id: ID of the node you wnat to get all neighbours from
    @param edges: torch [[],[]] with edges
    
    returns tensor with neighbour node IDs
    """
    return edges[1][torch.where(edges[0]==node_id)]


def create_graph(dynamicdata, edges, timestamp=None):    
    # extract node features from dynamic data
    all_node_features = dynamicdata.reshape(215820, 8) 

    if timestamp is not None:
        all_node_features = add_timestamp_to_features(all_node_features, timestamp)

    graph = Data(x=all_node_features, edge_index=edges)

    return graph


def create_edge_index(staticdata, threshold):
    mask = staticdata[0]>threshold
    width = mask.shape[1]
    
    # extract edges from static data
    end_nodes_calculation ={1: -width, 2: -width+1, 3: 1, 4:width+1, 5:width, 6:width-1, 7:-1, 8: -width-1}
    edges = []
    for k in range(1,staticdata.shape[0]):
        edges = edges + [[i, i+end_nodes_calculation[k]] for i, x in enumerate((staticdata[k]*mask).flatten()>0) if x]
    edges = torch.tensor(edges).T
    return mask, edges


def build_sample(dynamicdata, edges, start_time, day_of_week, input_size=12, include_timestamps=True, average=None):

    all_node_features = dynamicdata[:input_size].permute(1, 2, 0, 3).reshape(215820, 8*input_size)
    all_label_features = dynamicdata[input_size:][[0, 1, 2, 5, 8, 11]].permute(1, 2, 0, 3).reshape(215820, 8*6)

    # average_mask = (all_node_features != 0)
    # input_mean = (all_node_features * average_mask).reshape(215820, 8, input_size).mean(2).repeat(1, input_size)
    # all_node_features = (all_node_features != 0) * all_node_features + (all_node_features == 0) * input_mean

    """
    if average is not None:
        average = average.permute(1, 2, 0, 3)
        input_average = average[:, :, 0:12, :]
        input_average_reshaped = input_average.reshape(215820, 96)
        assert(all_node_features.shape == input_average_reshaped.shape)
        all_node_features = (all_node_features != 0) * all_node_features + (all_node_features == 0) * input_average_reshaped
    """

    # Per feature standardization
    """
    mean = all_node_features.mean(0)
    std = all_node_features.std(0)
    all_node_features = (all_node_features - mean) / std
    """

    # Scaling to [0, 1]
    #all_node_features = all_node_features / 255.0

    if include_timestamps:
        all_node_features = add_timestamp_to_features(all_node_features, start_time)
        all_node_features = add_day_of_week_to_features(all_node_features, day_of_week)


    input = Data(x=all_node_features, edge_index=edges)
    label = all_label_features

    sample = {"data": input, "label": label}

    return sample

def add_timestamp_to_features(all_node_features, timestamp):
    timestamp_radiant = 2 * math.pi * timestamp / 288.0
    timestamp_vector_sin = torch.ones((215820, 1)) * math.sin(timestamp_radiant)
    timestamp_vector_cos = torch.ones((215820, 1)) * math.cos(timestamp_radiant)
    all_node_features = torch.cat((all_node_features, timestamp_vector_sin, timestamp_vector_cos), 1)
    
    return all_node_features

def add_day_of_week_to_features(all_node_features, day_of_week):
    day_of_week_radiant = 2 * math.pi * day_of_week / 7.0
    day_of_week_vector_sin = torch.ones((215820, 1)) * math.sin(day_of_week_radiant)
    day_of_week_vector_cos = torch.ones((215820, 1)) * math.cos(day_of_week_radiant)
    all_node_features = torch.cat((all_node_features, day_of_week_vector_sin, day_of_week_vector_cos), 1)
    
    return all_node_features
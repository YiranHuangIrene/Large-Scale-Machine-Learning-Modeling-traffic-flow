import h5py
import torch
from torch_geometric.data import Data


def extract_h5_data(path:str):
    """
    this function extracts numpy arrays from h5 files
    @param path: string with path to file that shall be extracted
    @return: numpy.array with data
    """
    hf = h5py.File(path, 'r')
    kf_keys = [key for key in hf.keys()]
    
    data = [torch.tensor(hf.get(key)) for key in kf_keys]
    
    return data


def convert_static_image_to_adj_matrix(staticdata, threshold):
    mask = staticdata[0]>threshold
    width = mask.shape[1]
    print("width in convert static image to adj matrix" , width)
    
    end_nodes_calculation ={1: -width, 2: -width+1, 3: 1, 4:width+1, 5:width, 6:width-1, 7:-1, 8: -width-1}
    edges = []
    
    for k in range(1,staticdata.shape[0]):
        edges = edges + [[i, i+end_nodes_calculation[k]] for i, x in enumerate((staticdata[k]*mask).flatten()>0) if x]

    edges = torch.tensor(edges).T
    
    return mask, edges


def create_graphs(staticdata, threshold, dynamicdata):
    mask = staticdata[0]>threshold
    width = mask.shape[1]
    
    # extract edges from static data
    end_nodes_calculation ={1: -width, 2: -width+1, 3: 1, 4:width+1, 5:width, 6:width-1, 7:-1, 8: -width-1}
    edges = []
    for k in range(1,staticdata.shape[0]):
        edges = edges + [[i, i+end_nodes_calculation[k]] for i, x in enumerate((staticdata[k]*mask).flatten()>0) if x]
    edges = torch.tensor(edges).T
    
    graphs = []
    # extract node features from dynamic data
    # iterate over graphs
    for i in range(dynamicdata.shape[0]):
        all_node_features = dynamicdata[i].reshape(215820, 8)
        graph = Data(x=all_node_features, edge_index=edges)
        graphs.append(graph)
    # node_features = [all_node_features[x] for x in range(len(all_node_features)) if mask.flatten()[x]>0]
        
    return mask, graphs
import hydra
import pytorch_lightning as pl
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import torch

from t4clab.callbacks import WandbModelCheckpoint, WandbSummaries
from t4clab.logging import get_logger, log_hyperparameters, print_config
from t4clab.utils.random import set_seed

from t4clab.data.t4cdataset import *
import t4clab.models.GNNRes_high_input as GNNRes
from collections import OrderedDict
import t4clab.models.GNNRes_avg as GNNResAvg
import t4clab.data.t4cdataset_avg as t4cdataset_avg
import t4clab.utils.utils as utils
import t4clab.utils.gnn_utils as gnn_utils


# model = GNNRes.GNNRes(n_channels=50, depth=5, input_size=50)
# model = GNNRes.GNNRes(n_channels=100, depth=5, input_size=100)
def test_datasets():
    model = GNNResAvg.GNNRes(n_channels=100, depth=30, input_size=100)
    
    # checkpoint_path='/nfs/homedirs/muel/checkpoints/skilled_monkey.ckpt'
    checkpoint_path='/nfs/homedirs/muel/checkpoints/GNN-Res-30-Imputed_input_Skipconn_no_outputaverage.ckpt'
    # run_path = "/mllab2122-traffic/t4clab/runs/qdiohind"
    # run_path = "/mllab2122-traffic/t4clab/runs/qa9nvoz4"
    # run_path = "/mllab2122-traffic/t4clab/runs/qa9nvoz4"
    run_path = "/mllab2122-traffic/t4clab/runs/1w10v557"
    output_folder="/nfs/homedirs/muel/project-3/plots/"

    best_model = wandb.restore(checkpoint_path, run_path)
    best_checkpoint = torch.load(best_model.name, map_location=torch.device('cpu'))

    new_state_dict = OrderedDict()
    for k, v in best_checkpoint["state_dict"].items():
        name = k.replace("model.", "")
        new_state_dict[name] = v

    # load params
    model.load_state_dict(new_state_dict, strict=False)

    valid_dataset = t4cdataset_avg.T4CValidationDataSet(data_dir="/nfs/shared/traffic4cast", input_size=12, city="MELBOURNE", threshold=0.2, include_timestamps=True)
    train_dataset = t4cdataset_avg.T4CTrainingDataSet(data_dir="/nfs/shared/traffic4cast", input_size=12, city="MELBOURNE", threshold=0.2, include_timestamps=True)
    # valid_dataset = T4CValidationDataSet(data_dir="/nfs/shared/traffic4cast", input_size=12, city="MELBOURNE", threshold=0.2, include_timestamps=True)
    # train_dataset = T4CTrainingDataSet(data_dir="/nfs/shared/traffic4cast", input_size=12, city="MELBOURNE", threshold=0.2, include_timestamps=True)

    sample_data = train_dataset[100]
    y_hat = model(sample_data["data"])

    # print(f"Sample data shape: {sample_data.shape}")
    # print(f"y_hat shape: {y_hat.shape}")

    sample_label = sample_data['label']
    sample_data_x = sample_data["data"].x

    print(f"sample label shape: {sample_label.shape}")
    print(f"sample data x shape: {sample_data_x.shape}")

    histogram_values = torch.histc(sample_label[:,1], bins=256, min=0, max=255)


def my_build_sample(dynamicdata, edges, start_time, day_of_week, input_size=12, include_timestamps=True, average=None):

    all_node_features = dynamicdata[:input_size].permute(1, 2, 0, 3).reshape(215820, 8*input_size)
    all_label_features = dynamicdata[input_size:][[0, 1, 2, 5, 8, 11]].permute(1, 2, 0, 3).reshape(215820, 8*6)

    average_mask = (all_node_features != 0)
    input_mean = (all_node_features * average_mask).reshape(215820, 8, input_size).mean(2).repeat(1, input_size)
    all_node_features = (all_node_features != 0) * all_node_features + (all_node_features == 0) * input_mean

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


def test_sample_with_avg_of_neighbours():
    path_data = "/nfs/shared/traffic4cast/MELBOURNE/training/2019-01-02_MELBOURNE_8ch.h5"
    static_data =  utils.extract_h5_data("/nfs/shared/traffic4cast/MELBOURNE/MELBOURNE_static.h5")
    mask = static_data[0]>0.2
    
    _ , edges = gnn_utils.create_edge_index(static_data, 0.2)
    
    averages = np.load("/nfs/shared/traffic4cast/MELBOURNE_traffic_per_weekday_masked.npy")
    
    i=0
    dynamic_data = utils.extract_h5_data_slice(path_data, i, i+24)
    sample = my_build_sample(dynamic_data, edges, i, 2, 12, True, averages[2, i:i+24, :, :, :])
    sample["mask"] = mask
    sample["average"] = averages[2, i:i+24, :, :, :]
    
    

if __name__ == "__main__":
    test_sample_with_avg_of_neighbours()
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
from tqdm import tqdm
from t4clab.callbacks import WandbModelCheckpoint, WandbSummaries
from t4clab.logging import get_logger, log_hyperparameters, print_config
from t4clab.utils.random import set_seed
import t4clab.utils.utils as utils
import t4clab.utils.gnn_utils as gnn_utils

from t4clab.data.t4cdataset import *
# import t4clab.models.GNNRes_high_input as GNNRes
import t4clab.models.GNNRes_avg as GNNResAvg
from collections import OrderedDict
import t4clab.data.t4cdataset_avg as t4cdataset_avg


def get_weekday_from_path(path):
    filename = path.split("/")[-1]
    date_string = filename[:10]
    date_object = datetime.strptime(date_string, "%Y-%m-%d").date()
    return date_object.weekday()


if __name__ == "__main__":
    checkpoint_path='/nfs/homedirs/muel/checkpoints/GNNRes-30-imputed-input-add-average-output.ckpt'
    run_path = "/mllab2122-traffic/t4clab/runs/3ojpcubj"
    output_folder="/nfs/homedirs/muel/project-3/plots/"
    
    model = GNNResAvg.GNNRes(n_channels=100, depth=30, input_size=100, inout_skipconn=False)
    
    best_model = wandb.restore(checkpoint_path, run_path)
    best_checkpoint = torch.load(best_model.name, map_location=torch.device('cpu'))
    

    new_state_dict = OrderedDict()
    for k, v in best_checkpoint["state_dict"].items():
        name = k.replace("model.", "")
        new_state_dict[name] = v

    # load params
    model.load_state_dict(new_state_dict, strict=False)
    
    # valid_dataset = t4cdataset_avg.T4CValidationDataSet(data_dir="/nfs/shared/traffic4cast", input_size=12, city="MELBOURNE", threshold=0.2, include_timestamps=True)
    # train_dataset = t4cdataset_avg.T4CTrainingDataSet(data_dir="/nfs/shared/traffic4cast", input_size=12, city="MELBOURNE", threshold=0.2, include_timestamps=True)
    
    path_data = "/nfs/shared/traffic4cast/MELBOURNE/training/2019-01-02_MELBOURNE_8ch.h5"
    averages = torch.tensor(np.load("/nfs/shared/traffic4cast/MELBOURNE_traffic_per_weekday_masked.npy"))
    
    static_data =  utils.extract_h5_data("/nfs/shared/traffic4cast/MELBOURNE/MELBOURNE_static.h5")
    threshold = 0.2
    mask = static_data[0]>threshold
    _ , edges = gnn_utils.create_edge_index(static_data, 0.2)
    
    for i in tqdm(range(288)):
    # i=0
        if not os.path.exists(f"/nfs/homedirs/muel/saved_tensors/MELBOURNE_pred_input{i}.pt"):
            
            dynamic_data = utils.extract_h5_data_slice(path_data, i, i+24)
            sample = build_sample(dynamic_data, edges, i, 2, 12, True, averages[2, i:i+24, :, :, :])
            sample["mask"] = mask
            sample["average"] = averages[2, i:i+24, :, :, :]
            
            y_hat_test = model(sample["data"], sample["average"].unsqueeze(0), sample["mask"]).squeeze(0)
            torch.save(y_hat_test, f"/nfs/homedirs/muel/saved_tensors/MELBOURNE_pred_input{i}.pt")
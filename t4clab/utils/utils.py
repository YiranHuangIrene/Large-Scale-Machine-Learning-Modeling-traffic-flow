import h5py
import torch
import numpy as np
import wandb
from collections import OrderedDict
import t4clab.data.t4cdataset as t4cdataset
import matplotlib.pyplot as plt


def extract_h5_data(path:str):
    """
    this function extracts numpy arrays from h5 files
    @param path: string with path to file that shall be extracted
    @return: numpy.array with data
    """
    hf = h5py.File(path, 'r')
    # kf_keys = [key for key in hf.keys()]
    # data = [torch.tensor(hf.get(key)) for key in kf_keys]
    data = hf['array']
    data = torch.tensor(np.array(data)).type(torch.FloatTensor)
    hf.close()
    return data

def extract_h5_data_slice(path, i, j=-1):
    if j == -1:
        j = i
    hf = h5py.File(path, 'r')
    # kf_keys = [key for key in hf.keys()]
    # data = [torch.tensor(hf.get(key)) for key in kf_keys]
    data = hf['array'][i:j]
    data = torch.tensor(np.array(data)).type(torch.FloatTensor)
    hf.close()
    return data


def visualise_predictions(checkpoint_path, run_path, model, output_folder, city="MELBOURNE", 
                          input_size=6, threshold=0, include_timestamp=True, val_id=100, train_id=100):
    best_model = wandb.restore(checkpoint_path, run_path=run_path)
    best_checkpoint = torch.load(best_model.name, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    
    #load model parameters
    for k, v in best_checkpoint["state_dict"].items():
        name = k.replace("model.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    
    valid_dataset = t4cdataset.T4CValidationDataSet(data_dir="/nfs/shared/traffic4cast", input_size=input_size, city=city, threshold=threshold, include_timestamps=include_timestamp)
    train_dataset = t4cdataset.T4CTrainingDataSet(data_dir="/nfs/shared/traffic4cast", input_size=input_size, city=city, threshold=threshold, include_timestamps=include_timestamp)
    
    for j in range(300):
        val_id = j
        train_id = j
        y_hat_valid = model(valid_dataset[val_id]['data'])
        y_hat_train = model(train_dataset[train_id]['data'])
        
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))
        for i in range(2):
            axs[i,1].imshow(valid_dataset[val_id]['label'].reshape(495,436,8).detach().numpy()[:,:,i], cmap="Greys")
            axs[i,0].imshow(y_hat_valid.reshape(495,436,8).detach().numpy()[:,:,i],cmap="Greys")
        # fig.suptitle('Comparison predictions (left) & ground truth (right) validation set', fontsize=16)
        plt.savefig(output_folder+f"/comparison_prediction_output_valid_small_{val_id}.png")
        plt.show()
        
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))
        for i in range(2):
            axs[i,1].imshow(train_dataset[train_id]['label'].reshape(495,436,8).detach().numpy()[:,:,i], cmap="Greys")
            axs[i,0].imshow(y_hat_train.reshape(495,436,8).detach().numpy()[:,:,i],cmap="Greys")
        # fig.suptitle('Comparison predictions (left) & ground truth (right) train set', fontsize=16)
        plt.savefig(output_folder+f"/comparison_prediction_output_train_small_{train_id}.png")
        plt.show()
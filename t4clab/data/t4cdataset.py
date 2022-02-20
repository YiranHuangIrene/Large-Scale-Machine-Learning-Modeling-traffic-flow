from torch.utils import data
from t4clab.utils.utils import *
from t4clab.utils.gnn_utils import *
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import os
from datetime import datetime


def index_data(paths, Train=True, input_size=24):
    mappings = {}
    counter = 0
    if Train:
        for path in paths:
            for i in range(288-input_size):
                mappings[counter+i] = (path, i)
            counter += (288-input_size)
        return mappings
    else:
        path = paths[0]
        for i in range(100):
            for j in range(12-input_size):
                mappings[counter+j] = (path, i, j)
            counter += (data[i].shape[0]-input_size)
        return mappings

def get_weekday_from_path(path):
    filename = path.split("/")[-1]
    date_string = filename[:10]
    date_object = datetime.strptime(date_string, "%Y-%m-%d").date()
    return date_object.weekday()

class T4CTrainingDataSet(Dataset):
    def __init__(self, data_dir="nfs/shared/traffic4cast", city=None, threshold=0.2, input_size=12, include_timestamps=True):
        print("Initializing Training Dataset")
        self.input_size = input_size
        self.data_dir = data_dir
        self.city = city
        self.include_timestamps = include_timestamps
        self.threshold = threshold
        self.paths = os.listdir(self.data_dir + '/' + self.city + '/training/')
        self.paths = self.paths[:int(0.9*len(self.paths))]
        self.paths = [self.data_dir + '/' + self.city + '/training/'+path for path in self.paths]
        self.mappings = index_data(self.paths, Train=True)
        self.static_data = extract_h5_data(data_dir + '/' + city + '/' + city + '_static.h5')
        _ , self.edges = create_edge_index(self.static_data, self.threshold)
        self.mask = self.static_data[0]>threshold

    def __len__(self):
        return len(self.mappings)
    
    def __getitem__(self, idx):
        path, i = self.mappings[idx]
        day_of_week = get_weekday_from_path(path)
        dynamic_data = extract_h5_data_slice(path, i, i+24)
        sample = build_sample(dynamic_data, self.edges, i, day_of_week, self.input_size, self.include_timestamps)
        sample["mask"] = self.mask
        return sample

class T4CValidationDataSet(Dataset):
    def __init__(self, data_dir="nfs/shared/traffic4cast", city=None, threshold=0.2, input_size=12, include_timestamps=True):
        print("Initializing Validation Dataset")
        self.input_size = input_size
        self.data_dir = data_dir
        self.city = city
        self.include_timestamps = include_timestamps
        self.threshold = threshold
        self.paths = os.listdir(self.data_dir + '/' + self.city + '/training/')
        self.paths = self.paths[int(0.9*len(self.paths)):]
        self.paths = [self.data_dir + '/' + self.city + '/training/'+path for path in self.paths]
        self.mappings = index_data(self.paths, Train=True)
        self.static_data = extract_h5_data(data_dir + '/' + city + '/' + city + '_static.h5')
        _ , self.edges = create_edge_index(self.static_data, self.threshold)
        self.mask = self.static_data[0]>self.threshold

    def __len__(self):
        return len(self.mappings)
    
    def __getitem__(self, idx):
        path, i = self.mappings[idx]
        day_of_week = get_weekday_from_path(path)
        dynamic_data = extract_h5_data_slice(path, i, i+24)
        sample = build_sample(dynamic_data, self.edges, i, day_of_week, self.input_size, self.include_timestamps)
        sample["mask"] = self.mask
        # print(self.mask.shape)
        return sample

class T4CTestDataSet(Dataset):
    def __init__(self, data_dir="nfs/shared/traffic4cast", city=None, threshold=0.2, input_size=12, include_timestamps=True):
        print("Initializing Test Dataset")
        self.input_size = input_size
        self.data_dir = data_dir
        self.city = city
        self.include_timestamps = include_timestamps
        self.threshold = threshold
        self.path = self.data_dir + '/' + self.city + '/' + self.city + '_test_temporal.h5'
        self.mappings = index_data([self.path], Train=False)
        self.static_data = extract_h5_data(data_dir + '/' + city + '/' + city + '_static.h5')
        _ , self.edges = create_edge_index(self.static_data, self.threshold)
        self.mask = self.static_data[0]>self.threshold

    def __len__(self):
        return len(self.mappings)
    
    def __getitem__(self, idx):
        path, i, j = self.mappings[idx]
        dynamic_data = extract_h5_data_slice(path, i)[:]
        dynamic_data = dynamic_data.reshape(215820, 8)
        dynamic_data = add_timestamp_to_features(dynamic_data, idx)
        # sample = {"data": dynamic_data, "mask": self.mask}
        # return sample
        return dynamic_data
        

class T4CDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/nfs/shared/traffic4cast", batch_size: int = 32, input_size=12, city: str = "MELBOURNE", threshold=0.2, include_timestamps=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.city = city
        self.input_size = input_size
        self.threshold = threshold
        self.include_timestamps = include_timestamps

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(T4CTrainingDataSet(self.data_dir, self.city, self.threshold, self.input_size, self.include_timestamps), batch_size=self.batch_size, num_workers=20)
        # return DataLoader(T4CValidationDataSet(self.data_dir, self.city, self.threshold, self.input_size, self.include_timestamps), batch_size=self.batch_size, num_workers=20)

    def val_dataloader(self):
        return DataLoader(T4CValidationDataSet(self.data_dir, self.city, self.threshold, self.input_size, self.include_timestamps), batch_size=self.batch_size, num_workers=20)

    def test_dataloader(self):
        return DataLoader(T4CTestDataSet(self.data_dir, self.city, self.threshold, self.input_size, self.include_timestamps), batch_size=self.batch_size, num_workers=20)

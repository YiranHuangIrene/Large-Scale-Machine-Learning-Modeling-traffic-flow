# Project 03: Modelling Traffic Flow

## Installation

```sh
# Use pip-tools to manage dependencies
$ pip install pip-tools

# Install pinned dependencies
$ pip-sync

# Install package in editable mode
$ pip install -e .
```

## Run the Training

Before you run anything, log into your W&B account with `wandb login`. Then you can run
model training, for example, as follows.
```sh
./train.py data=example model=knn
```

## Run on SLURM

To submit jobs to the SLURM cluster, set the launcher to `slurm`.
```sh
./train.py hydra/launcher=slurm # ... other arguments
```

---

## Model
```sh
model=
```

|   |   |
|---|---|
|**GAT**|   Vanilla Graph Attention Model|
|**GnnRes**| Vanilla GNNRes with 1 timeslot as input|
|**GnnRes-HF**| GNNRes with variable number of feature channels  |
|**GnnRes-High-Input**|   GNNRes with multiple timeslots as input|
|**GnnRes-Avg**| GNNRes with global average over training sample  |
|**Unet**|  Vanilla U-Net |

### Parameters

For GNNRes-High-Input:
- **input_size**: Number of input time slots
- **n_channels**: Number of features channels insides residual blocks
- **inout_skipconn**: If there should be an additional GCN on the input and output of the model
- **depth**: Number of residual blocks
- **activation**: Activation function (ReLU, Sigmoid)

---

## Dataset
```sh
data=
```

|   |   |
|---|---|
|**t4c**|   Traffic Forecasting Dataset|
|**t4c_avg**| Traffic Forecasting Dataset including global average|

### Parameters

- **data_dir**: Directory of data files
- **batch_size**: Batchsize during training
- **input_size**: Number of input time slots
- **city**: City of data files
- **threshold**: Threshold used for mask creation
- **include_timestamps**: If timestamps should be included as circular features


---

## Task
```sh
task=
```

|   |   |
|---|---|
|**temporal_forecasting**|   Temporal Forecasting Task|
|**temporal_forecasting_avg**| Temporal Forecating Task including global average|
|**temporal_forecasting_unet**| Traffic Forecasting Task adapted to U-Net|

### Parameters

- **lr**: Learning rate used during training


# Large-Scale-Machine-Learning-Modeling-traffic-flow

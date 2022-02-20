import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import matplotlib.pyplot as plt
from t4clab.utils.Unet_Utils import *
import numpy as np


from ..logging import get_logger

log = get_logger()


class TemporalForecastingTask(pl.LightningModule):
    def __init__(self, model, lr=0.001, **kwargs):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss = mse_loss
        self.padding = (6, 6, 1, 0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train/mse", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val/mse", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test/mse", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def step(self, sample, batch_idx=0):
        x = sample["data"]
        y = sample["label"]
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        if batch_idx > 0 and batch_idx % 5000 == 0:
            #y.shape = (496, 448, 8 ,6)
            y = transform_unstack_channels_on_time(y)
            y_hat = transform_unstack_channels_on_time(y_hat)
            self.visualize(y, y_hat, loss, batch_idx)
        return loss

    def visualize(self, ground_truth, prediction, loss, batch_idx):
        cmap = plt.cm.get_cmap("viridis").copy()
        cmap.set_bad(color='white')
        fig, axs = plt.subplots(ncols=2, nrows=8, figsize=(30, 100))
        axs[0, 0].set_title("Ground Truth")
        axs[0, 1].set_title("Prediction")
        for i in range(8):
            ground_truth_i = ground_truth.detach().cpu().numpy()[:, :, i, 0]
            prediction_i = prediction.detach().cpu().numpy()[:, :, i, 0]
            axs[i, 0].imshow(np.ma.masked_where(ground_truth_i == 0, ground_truth_i), cmap=cmap)
            axs[i, 1].imshow(np.ma.masked_where(prediction_i == 0, prediction_i), cmap=cmap)
        fig.suptitle("Current Loss: {}".format(str(loss)))
        plt.savefig("../" + str(self.model.__class__.__name__) + "_comparison_" + str(batch_idx) + ".png")

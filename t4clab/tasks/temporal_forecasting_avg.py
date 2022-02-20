import pytorch_lightning as pl
import torch
from torch.nn.functional import mse_loss
from t4clab.utils.mse import *
import matplotlib.pyplot as plt

from ..logging import get_logger

log = get_logger()


class TemporalForecastingTask(pl.LightningModule):
    def __init__(self, model, lr=0.001, **kwargs):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss = mse_loss

    def forward(self, data, avg, mask):
        x = self.model(data, avg, mask)
        return x
    
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
        y = sample["label"][0]
        avg = sample["average"]
        mask = sample["mask"]
        y_hat = self(x, avg, mask) 
        loss = self.calc_metrics(y_hat, y)
        node_error = self.node_error(y_hat,y)
        if batch_idx > 0 and batch_idx % 5000 == 0:
            self.visualize(y, y_hat, loss, batch_idx)
        for i in range(8):
            self.log("node_error_day{}_channel{}".format(batch_idx // 264, i), node_error[i], on_step = True)
        return loss

    def calc_metrics(self, y_hat, y):
        mask = (y != 0).reshape(-1, 215820, 48)
        se = (y_hat - y) ** 2
        masked_se = se[mask]
        mse = torch.mean(masked_se)
        return mse

    def visualize(self, ground_truth, prediction, loss, batch_idx):
        cmap = plt.cm.get_cmap("gist_stern").copy()
        fig, axs = plt.subplots(ncols=2, nrows=8, figsize=(30, 100))
        axs[0,0].set_title("Ground Truth")
        axs[0,1].set_title("Prediction")
        for i in range(8):
            ground_truth_i = ground_truth.reshape(495,436,8,6).detach().cpu().numpy()[:,:,i,0]
            prediction_i = prediction.reshape(495,436,8,6).detach().cpu().numpy()[:,:,i,0]
            axs[i,0].imshow(ground_truth_i,  cmap=cmap)
            axs[i,1].imshow(prediction_i,  cmap=cmap)
        fig.suptitle("Current Loss: {}".format(str(loss)))
        plt.savefig("../" + str(self.model.__class__.__name__) + "_comparison_" + str(batch_idx) + ".png")

    def node_error(self, y_hat, y):
        y_hat = torch.reshape(y_hat, (6, 495, 436, 8))
        y = torch.reshape(y, (6, 495, 436, 8))
        error = []
        for i in range(8):
            error.append(torch.div(y_hat[:,148, 82, i].sum(0) - y[:,148, 82, i].sum(0),6)**2)
        return error










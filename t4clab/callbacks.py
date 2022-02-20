from copy import deepcopy
from pathlib import Path

import torch
import wandb
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .logging import get_logger

log = get_logger()


class WandbModelCheckpoint(ModelCheckpoint):
    """Save checkpoints into the W&B run directory to sync them automatically."""

    def __init__(self, **kwargs):
        run_dir = Path(wandb.run.dir)
        cp_dir = run_dir / "checkpoints"

        super().__init__(**kwargs, dirpath=str(cp_dir))


class WandbSummaries(Callback):
    """Set the W&B summaries of each metric to the values from the best epoch."""

    def __init__(self, monitor: str, mode: str):
        super().__init__()

        self.monitor = monitor
        self.mode = mode

        self.best_metric = None
        self.best_metrics = None

        self.ready = True

    def on_sanity_check_start(self, trainer: Trainer, pl_module: LightningModule):
        self.ready = False

    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule):
        self.ready = True

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if not self.ready:
            return

        metrics = trainer.logged_metrics
        if self.monitor in metrics:
            metric = metrics[self.monitor]
            if torch.is_tensor(metric):
                metric = metric.item()

            if self._better(metric):
                self.best_metric = metric
                self.best_metrics = deepcopy(metrics)

        self._update_summaries()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        self._update_summaries()

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint
    ):
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "best_metric": self.best_metric,
            "best_metrics": self.best_metrics,
        }

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, callback_state
    ):
        self.monitor = callback_state["monitor"]
        self.mode = callback_state["mode"]
        self.best_metric = callback_state["best_metric"]
        self.best_metrics = callback_state["best_metrics"]

    def _better(self, metric):
        if self.best_metric is None:
            return True
        elif self.mode == "min" and metric < self.best_metric:
            return True
        elif self.mode == "max" and metric > self.best_metric:
            return True
        else:
            return False

    def _update_summaries(self):
        # wandb is supposed not to update the summaries anymore once we set them manually,
        # but they are still getting updated, so we make sure to set them after logging
        if self.best_metrics is not None:
            wandb.summary.update(self.best_metrics)

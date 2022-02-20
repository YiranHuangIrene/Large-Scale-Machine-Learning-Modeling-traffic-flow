#!/usr/bin/env python

import hydra
import pytorch_lightning as pl
import wandb
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from t4clab.callbacks import WandbModelCheckpoint, WandbSummaries
from t4clab.logging import get_logger, log_hyperparameters, print_config
from t4clab.utils.random import set_seed


log = get_logger()


def get_callbacks(config, rng):
    monitor = {"monitor": "val/mse", "mode": "min"}
    callbacks = [
        WandbSummaries(**monitor),
        WandbModelCheckpoint(
            save_last=True, save_top_k=1, filename="best", **monitor
        ),
    ]
    return callbacks


@hydra.main(config_path="config", config_name="train")
def main(config: DictConfig):
    rng = set_seed(config)
    print_config(config)
    wandb.init(entity=config.entity, project=config.project, group=config.group)

    log.info("Loading data")
    datamodule = instantiate(config.data)

    log.info("Instantiating model")
    model = instantiate(config.model)

    logger = WandbLogger()
    log_hyperparameters(logger, config, model)

    task = instantiate(config.task, model)

    if config.trainer.get("resume_from_checkpoint") is not None:
        log.info("Loading checkpoint")
        config.trainer.resume_from_checkpoint = hydra.utils.to_absolute_path(
            config.trainer.resume_from_checkpoint
        )

    log.info("Instantiating trainer")
    callbacks = get_callbacks(config, rng)
    trainer: Trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)

    log.info("Starting training!")
    trainer.fit(task, datamodule=datamodule)

    if config.eval_testset:
        log.info("Starting testing!")
        trainer.test()

    wandb.finish()
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    best_score = trainer.checkpoint_callback.best_model_score
    return float(best_score)


if __name__ == "__main__":
    main()

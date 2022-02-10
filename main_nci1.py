import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from model.classifier_nci1 import PerceiverClassifier

from torch_geometric.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import os
import torch

from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset

torch.multiprocessing.set_sharing_strategy("file_system")
wandb_logger = WandbLogger(project="graph-perceiver")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    pl.seed_everything(cfg.run.seed)

    dataset = TUDataset(os.path.join(get_original_cwd(), "./data"), name="NCI1")
    num_tasks = dataset.num_classes

    num_features = dataset.num_features

    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(
        dataset, [num_training, num_val, num_test]
    )
    train_loader = DataLoader(
        training_set, batch_size=128, shuffle=True, num_workers=32
    )
    valid_loader = DataLoader(
        validation_set, batch_size=128, shuffle=False, num_workers=32
    )
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # mc = ModelCheckpoint(
    #     monitor="val/loss",
    #     dirpath="./",
    #     filename="zinc-{epoch:02d}-{val/loss:.2f}",
    #     mode="min",
    # )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model = PerceiverClassifier(
        dataset_name="nci",
        input_channels=num_features,
        depth=2,
        attn_dropout=0.0,
        ff_dropout=0.1,
        weight_tie_layers=False,
        virtual_latent=True,
        evaluator=None,
        num_classes=num_tasks,
    )
    # model = PerceiverClassifier.load_from_checkpoint(
    #     checkpoint_path=os.path.join(
    #         get_original_cwd(),
    #         "outputs/2022-02-07/same_params/graph-perceiver/2hmqf779/checkpoints/epoch=49-step=68449.ckpt",
    #     ),
    # )

    log = cfg.run.log
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=56,
        logger=wandb_logger if log is True else None,
        callbacks=[lr_monitor] if log is True else None,
    )
    trainer.fit(
        model,
        train_loader,
        valid_loader,
        # ckpt_path=os.path.join(
        #     get_original_cwd(),
        #     "outputs/2022-02-07/same_params/graph-perceiver/2hmqf779/checkpoints/epoch=49-step=68449.ckpt",
        # ),
    )
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

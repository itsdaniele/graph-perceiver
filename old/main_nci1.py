import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from model.classifier_nci1 import PerceiverClassifier

from torch_geometric.loader import DataLoader
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

from pytorch_lightning.callbacks.early_stopping import EarlyStopping


torch.multiprocessing.set_sharing_strategy("file_system")
wandb_logger = WandbLogger(project="graph-perceiver")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    pl.seed_everything(cfg.run.seed)

    dataset = TUDataset(os.path.join(get_original_cwd(), "./data"), name="NCI109")
    num_tasks = dataset.num_classes

    num_features = dataset.num_features

    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(
        dataset, [num_training, num_val, num_test]
    )
    train_loader = DataLoader(
        training_set, batch_size=256, shuffle=True, num_workers=32
    )
    valid_loader = DataLoader(
        validation_set, batch_size=256, shuffle=False, num_workers=32
    )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # mc = ModelCheckpoint(
    #     monitor="val/loss",
    #     dirpath="./",
    #     filename="zinc-{epoch:02d}-{val/loss:.2f}",
    #     mode="min",
    # )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model = PerceiverClassifier(
        input_channels=num_features,
        depth=4,
        attn_dropout=0.1,
        gnn_embed_dim=128,
        ff_dropout=0.2,
        weight_tie_layers=False,
        virtual_latent=False,
        evaluator=None,
        num_classes=num_tasks,
        train_loader=train_loader,
    )
    # model = PerceiverClassifier.load_from_checkpoint(
    #     checkpoint_path=os.path.join(
    #         get_original_cwd(),
    #         "outputs/2022-02-07/same_params/graph-perceiver/2hmqf779/checkpoints/epoch=49-step=68449.ckpt",
    #     ),
    # )

    # stop = EarlyStopping(monitor="val/loss", patience=10)

    log = cfg.run.log
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100,
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

    trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

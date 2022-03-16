import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from model.classifier import PerceiverClassifier

from torch_geometric.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import os

import torch

wandb_logger = WandbLogger(project="graph-perceiver")


@hydra.main(config_path="conf", config_name="molhiv")
def main(cfg: DictConfig):

    evaluator = Evaluator(name=cfg.run.dataset)
    pl.seed_everything(cfg.run.seed)

    dataset = PygGraphPropPredDataset(
        name=cfg.run.dataset, root=os.path.join(get_original_cwd(), "./data")
    )

    batch_size = cfg.run.get("batch_size")

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=batch_size or cfg.run.batch_size_train,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=batch_size or cfg.run.batch_size_val,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=batch_size or cfg.run.batch_size_test,
        shuffle=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    if cfg.model.loss_fn == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError()

    model = PerceiverClassifier(
        input_channels=cfg.model.input_channels,
        depth=cfg.model.depth,
        attn_dropout=cfg.model.attn_dropout,
        ff_dropout=cfg.model.ff_dropout,
        weight_tie_layers=cfg.model.weight_tie_layers,
        virtual_latent=cfg.model.virtual_latent,
        num_classes=dataset.num_tasks,
        evaluator=evaluator,
        loss_fn=loss_fn,
        dataset=cfg.run.dataset,
        lr=cfg.model.lr,
    )

    model = PerceiverClassifier.load_from_checkpoint(
        checkpoint_path=os.path.join(
            get_original_cwd(), "outputs/2022-02-14/17-18-40/checkpoints/last.ckpt",
        ),
    )

    if cfg.run.log:

        exp_name = f"{cfg.run.dataset}+seed-{cfg.run.seed}"
        logger = WandbLogger(name=exp_name, project="graph-perceiver")

    trainer = pl.Trainer(
        gpus=cfg.train.gpus,
        max_epochs=cfg.train.epochs,
        logger=logger,
        callbacks=[lr_monitor],
    )
    if cfg.run.test:
        trainer.test(model, test_dataloaders=test_loader)


if __name__ == "__main__":
    main()
from model.classifier import PerceiverClassifier

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from ogb.graphproppred import Evaluator, PygGraphPropPredDataset

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import os

from util import log_hyperparameters

import torch


@hydra.main(config_path="conf", config_name="zinc")
def main(cfg: DictConfig):

    batch_size = cfg.run.get("batch_size")

    if cfg.run.dataset == "zinc":

        num_tasks = 1
        evaluator = None
        zinc_train = ZINC(
            root=os.path.join(get_original_cwd(), "./data"), subset=True, split="train"
        )
        zinc_val = ZINC(
            root=os.path.join(get_original_cwd(), "./data"), subset=True, split="val"
        )
        zinc_test = ZINC(
            root=os.path.join(get_original_cwd(), "./data"), subset=True, split="test"
        )

        train_loader = DataLoader(
            zinc_train,
            batch_size=batch_size or cfg.run.batch_size_train,
            shuffle=True,
            num_workers=cfg.train.num_workers,
        )
        valid_loader = DataLoader(
            zinc_val,
            batch_size=batch_size or cfg.run.batch_size_val,
            shuffle=False,
            num_workers=cfg.train.num_workers,
        )
        test_loader = DataLoader(
            zinc_test,
            batch_size=batch_size or cfg.run.batch_size_test,
            shuffle=False,
            num_workers=cfg.train.num_workers,
        )
    if "ogb" in cfg.run.dataset:
        evaluator = Evaluator(name=cfg.run.dataset)
        pl.seed_everything(cfg.run.seed)

        dataset = PygGraphPropPredDataset(
            name=cfg.run.dataset, root=os.path.join(get_original_cwd(), "./data")
        )

        num_tasks = dataset.num_tasks

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
    elif cfg.model.loss_fn == "l1":
        loss_fn = torch.nn.L1Loss()
    else:
        raise ValueError()

    model = PerceiverClassifier(
        depth=cfg.model.depth,
        gnn_embed_dim=cfg.model.gnn_embed_dim,
        ff_dropout=cfg.model.ff_dropout,
        gnn_dropout=cfg.model.gnn_dropout,
        attn_dropout=cfg.model.attn_dropout,
        num_latents=cfg.model.num_latents,
        latent_dim=cfg.model.latent_dim,
        queries_dim=cfg.model.queries_dim,
        cross_heads=cfg.model.cross_heads,
        latent_heads=cfg.model.latent_heads,
        self_per_cross_attn=cfg.model.self_per_cross_attn,
        num_gnn_layers=cfg.model.num_gnn_layers,
        cross_dim_head=cfg.model.cross_dim_head,
        latent_dim_head=cfg.model.latent_dim_head,
        weight_tie_layers=cfg.model.weight_tie_layers,
        num_classes=num_tasks,
        evaluator=evaluator,
        loss_fn=loss_fn,
        dataset=cfg.run.dataset,
        lr=cfg.model.lr,
        gnn_type=cfg.model.gnn_type,
        weight_decay=cfg.optim.weight_decay,
        scheduler=cfg.optim.scheduler,
        scheduler_params=cfg.optim.scheduler_params,
    )

    logger = None
    if cfg.run.log:

        exp_name = f"{cfg.run.dataset}+seed-{cfg.run.seed}-{cfg.run.name}"
        logger = WandbLogger(name=exp_name, project="graph-perceiver-new")

    if cfg.run.dataset == "ogbg-molpcba":
        monitor = "val/ap"
        mode = "max"
    elif cfg.run.dataset == "ogbg-molhiv":
        monitor = "val/rocauc"
        mode = "max"
    elif cfg.run.dataset == "zinc":
        monitor = "val/loss"
        mode = "min"

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_last=True,
        verbose=True,
        dirpath="checkpoints",
        filename="epoch_{epoch:03d}",
    )

    trainer = pl.Trainer(
        gpus=cfg.train.gpus,
        max_epochs=cfg.train.epochs,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback]
        if logger
        else [checkpoint_callback],
        gradient_clip_val=1.0 if cfg.optim.clip_grad_norm else None,
        accumulate_grad_batches=cfg.optim.batch_accumulation,
    )

    if logger:
        log_hyperparameters(
            config=cfg,
            model=model,
            datamodule=None,
            trainer=trainer,
            callbacks=None,
            logger=logger,
        )

    if not cfg.run.resume:
        trainer.fit(
            model, train_loader, valid_loader,
        )
    else:
        trainer.fit(
            model,
            train_loader,
            valid_loader,
            ckpt_path=os.path.join(get_original_cwd(), cfg.run.checkpoint_path),
        )

    if cfg.run.test:
        trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

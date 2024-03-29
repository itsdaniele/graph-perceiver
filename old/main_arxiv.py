from model.io.classifier_io import PerceiverIOClassifier


from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from ogb.nodeproppred import PygNodePropPredDataset

from util import log_hyperparameters


@hydra.main(config_path="conf", config_name="arxiv")
def main(cfg: DictConfig):

    pl.seed_everything(cfg.run.seed)

    dataset = PygNodePropPredDataset(
        name=cfg.run.dataset,
        root=os.path.join(get_original_cwd(), "./data",),
        transform=T.ToSparseTensor(),
    )

    split_idx = dataset.get_idx_split()

    train_idx, valid_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()

    dataset = [data]

    train_loader = DataLoader(dataset, batch_size=1, num_workers=32)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = None
    if cfg.run.log:

        exp_name = f"{cfg.run.dataset}+seed-{cfg.run.seed}+{cfg.run.name}"
        logger = WandbLogger(name=exp_name, project="graph-perceiver-new")

    if cfg.run.dataset == "ogbn-proteins":
        monitor = "val/rocauc"
    elif cfg.run.dataset == "ogbn-arxiv":
        monitor = "val/acc"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
        dirpath="checkpoints",
        filename="epoch_{epoch:03d}",
    )

    model = PerceiverIOClassifier(
        depth=cfg.model.depth,
        queries_dim=cfg.model.queries_dim,
        train_mask=train_idx,
        val_mask=valid_idx,
        test_mask=test_idx,
        dataset=cfg.run.dataset,
        logits_dim=cfg.model.logits_dim,
        gnn_embed_dim=cfg.model.gnn_embed_dim,
        num_latents=cfg.model.num_latents,
        latent_dim=cfg.model.latent_dim,
        cross_heads=cfg.model.cross_heads,
        latent_heads=cfg.model.latent_heads,
        cross_dim_head=cfg.model.cross_dim_head,
        latent_dim_head=cfg.model.latent_dim_head,
        ff_dropout=cfg.model.ff_dropout,
        lr=cfg.model.lr,
        weight_tie_layers=cfg.model.weight_tie_layers,
    )

    callbacks = [checkpoint_callback]

    if logger is not None:
        callbacks += [lr_monitor]

    trainer = pl.Trainer(
        gpus=cfg.train.gpus,
        max_epochs=cfg.train.epochs,
        log_every_n_steps=1,
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=5,
    )

    if logger is not None:
        log_hyperparameters(
            config=cfg,
            model=model,
            datamodule=None,
            trainer=trainer,
            callbacks=None,
            logger=logger,
        )

    if cfg.run.test_only:
        model = PerceiverIOClassifier.load_from_checkpoint(
            checkpoint_path=os.path.join(get_original_cwd(), cfg.run.checkpoint_path,)
        )

        model.test_mask = test_idx
        trainer.test(model, test_dataloaders=[train_loader])
        import sys

        sys.exit(1)

    if not cfg.run.resume:
        trainer.fit(
            model, train_loader, train_loader,
        )

        # save gnn
    # torch.save(model.perceiver.gnn1, os.path.join(get_original_cwd(), "sage.pt"))
    else:
        trainer.fit(
            model,
            train_loader,
            train_loader,
            ckpt_path=os.path.join(get_original_cwd(), cfg.run.checkpoint_path),
        )

    trainer.test(test_dataloaders=[train_loader])


if __name__ == "__main__":
    main()


import pytorch_lightning as pl
from model.classifier_io import PerceiverIOClassifier


from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from pytorch_lightning.loggers import WandbLogger
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import os

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from ogb.nodeproppred import PygNodePropPredDataset

from util import log_hyperparameters
import torch
import torch.nn.functional as F
from synthetic.circles import get_circles_dataset
from torch_geometric.datasets import WikipediaNetwork


@hydra.main(config_path="conf", config_name="cham")
def main(cfg: DictConfig):

    pl.seed_everything(cfg.run.seed)

    dataset = WikipediaNetwork(
        root=os.path.join(get_original_cwd(), "./data",),
        name="squirrel",
        geom_gcn_preprocess=True,
    )

    data = [dataset[0]]

    train_loader = DataLoader(dataset, batch_size=1, num_workers=32)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    if cfg.run.log:

        exp_name = f"{cfg.run.dataset}+seed-{cfg.run.seed}+{cfg.run.name}"
        logger = WandbLogger(name=exp_name, project="graph-perceiver-new")

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
    )
    trainer = pl.Trainer(
        gpus=cfg.train.gpus,
        max_epochs=cfg.train.epochs,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],
        check_val_every_n_epoch=5,
    )

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

        trainer.test(model, test_dataloaders=[train_loader])
        import sys

        sys.exit(1)

    if not cfg.run.resume:
        trainer.fit(
            model, train_loader, train_loader,
        )

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

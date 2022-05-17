from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.loader import NeighborLoader, ShaDowKHopSampler
from torch_scatter import scatter
from torch_geometric.datasets import Planetoid

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torch_geometric.loader import DataLoader


from memorizing_lightning import MemorizingLightning

import os
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="cora")
def main(cfg: DictConfig):

    pl.seed_everything(cfg.run.seed)

    dataset = Planetoid(root=os.path.join(get_original_cwd(), "./data"), name="Cora")
    data = dataset[0]

    # train_loader = NeighborLoader(
    #     data,
    #     num_neighbors=[5, 10, 10],
    #     num_workers=32,
    #     input_nodes=data["train_mask"],
    # )
    # valid_loader = NeighborLoader(
    #     data, num_neighbors=[5, 10, 10], input_nodes=data["valid_mask"],
    # )

    train_loader = ShaDowKHopSampler(
        data,
        depth=2,
        num_neighbors=5,
        node_idx=data.train_mask,
        batch_size=32,
        num_workers=6,
        persistent_workers=True,
        drop_last=True,
    )
    valid_loader = ShaDowKHopSampler(
        data,
        depth=2,
        num_neighbors=5,
        node_idx=data.val_mask,
        batch_size=32,
        drop_last=True,
    )

    logger = None
    if cfg.run.log:

        exp_name = f"{cfg.run.dataset}+seed-{cfg.run.seed}+{cfg.run.name}"
        logger = WandbLogger(name=exp_name, project="graph-perceiver-new")

    model = MemorizingLightning(
        train_mask=data.train_mask,
        valid_mask=data.val_mask,
        test_mask=data.test_mask,
        depth=cfg.model.depth,
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
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()

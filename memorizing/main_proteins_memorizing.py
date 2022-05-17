from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.loader import NeighborLoader, ShaDowKHopSampler
from torch_scatter import scatter

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


from memorizing_lightning import MemorizingLightning

import os
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="proteins")
def main(cfg: DictConfig):

    pl.seed_everything(cfg.run.seed)

    if cfg.run.dataset == "ogbn-proteins":
        dataset = PygNodePropPredDataset(
            name=cfg.run.dataset, root=os.path.join(get_original_cwd(), "./data",)
        )

        data = dataset[0]

        splitted_idx = dataset.get_idx_split()

        train_idx, valid_idx, test_idx = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )

        for split in ["train", "valid", "test"]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[splitted_idx[split]] = True
            data[f"{split}_mask"] = mask

        row, col = data.edge_index
        data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce="add")
        data.n_id = torch.arange(data.num_nodes)

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
            node_idx=data["train_mask"],
            batch_size=1024,
            num_workers=6,
            persistent_workers=True,
            drop_last=True,
        )
        valid_loader = ShaDowKHopSampler(
            data,
            depth=2,
            num_neighbors=5,
            node_idx=data["valid_mask"],
            batch_size=1024,
            drop_last=True,
        )

    logger = None
    if cfg.run.log:

        exp_name = f"{cfg.run.dataset}+seed-{cfg.run.seed}+{cfg.run.name}"
        logger = WandbLogger(name=exp_name, project="graph-perceiver-new")

    model = MemorizingLightning(
        train_mask=train_idx,
        valid_mask=valid_idx,
        test_mask=test_idx,
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

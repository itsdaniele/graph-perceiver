import pytorch_lightning as pl
from model.classifier_io_cluster import PerceiverIOClassifier


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

from torch_geometric.loader import RandomNodeSampler, NeighborLoader

from pytorch_lightning.trainer.supporters import CombinedLoader

from torch_scatter import scatter


@hydra.main(config_path="conf", config_name="proteins")
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

        train_loader = RandomNodeSampler(data, num_parts=5, shuffle=True, num_workers=5)
        valid_loader = RandomNodeSampler(data, num_parts=5, num_workers=5)

        # train_loader = NeighborLoader(data, [30, 60, 90])
        # valid_loader = NeighborLoader(data, [30, 60, 90])

        dataset = PygNodePropPredDataset(
            name=cfg.run.dataset,
            root=os.path.join(get_original_cwd(), "./data",),
            transform=T.ToSparseTensor(attr="edge_attr"),
        )

        data = dataset[0]
        data.x = data.adj_t.sum(dim=1)
        del data.adj_t
        # data.adj_t.set_value_(None)

        # # Pre-compute GCN normalization.
        # adj_t = data.adj_t.set_diag()
        # deg = adj_t.sum(dim=1).to(torch.float)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        # adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        # data.adj_t = adj_t

        dataset = [data]

        train_loader_full = DataLoader(dataset, batch_size=1, num_workers=32)
    elif cfg.run.dataset == "circles":
        dataset, train_idx, valid_idx, test_idx = get_circles_dataset()

    # train_loader = DataLoader(dataset, batch_size=1, num_workers=32)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = None
    if cfg.run.log:

        exp_name = f"{cfg.run.dataset}+seed-{cfg.run.seed}+{cfg.run.name}"
        logger = WandbLogger(name=exp_name, project="graph-perceiver-new")

    if cfg.run.dataset == "ogbn-proteins":
        monitor = "val/rocauc"
    elif cfg.run.dataset in ["ogbg-arxiv", "circles"]:
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
        callbacks=[lr_monitor, checkpoint_callback],
        check_val_every_n_epoch=1,
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

    combined_loader = CombinedLoader(
        [valid_loader, train_loader_full], mode="max_size_cycle"
    )

    if not cfg.run.resume:
        trainer.fit(
            model,
            {"train1": train_loader, "train_full": train_loader_full},
            combined_loader,
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

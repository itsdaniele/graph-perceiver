import pytorch_lightning as pl
from model.classifier_io import PerceiverIOClassifier


from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from pytorch_lightning.loggers import WandbLogger
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import os

from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator

import torch

wandb_logger = WandbLogger(project="graph-perceiver")


@hydra.main(config_path="conf", config_name="molhiv")
def main(cfg: DictConfig):

    dataset = PygNodePropPredDataset(
        name="ogbn-arxiv",
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
    # data.adj_t = []
    row, col, edge_attr = data.adj_t.t().coo()
    data.edge_index = torch.stack([row, col], dim=0)

    dataset = [data]

    evaluator = Evaluator(name="ogbn-arxiv")
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)

    train_loader = DataLoader(dataset, batch_size=1, num_workers=32)
    # valid_loader = DataLoader(dataset[valid_idx], batch_size=1, num_workers=32)
    # test_loader = DataLoader(dataset[test_idx], batch_size=1, num_workers=32)

    model = PerceiverIOClassifier(
        depth=4,
        train_mask=train_idx,
        val_mask=valid_idx,
        dataset="arxiv",
        logits_dim=40,
        gnn_embed_dim=128,
        num_latents=800,
        latent_dim=8,
        cross_heads=4,
        latent_heads=4,
        cross_dim_head=8,
        latent_dim_head=8,
    )
    trainer = pl.Trainer(
        gpus=1, max_epochs=1000, log_every_n_steps=1, logger=wandb_logger
    )
    trainer.fit(model, train_loader, train_loader)
    # trainer.test(test_dataloaders=[test_loader])


if __name__ == "__main__":
    main()

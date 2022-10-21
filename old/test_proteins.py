import pytorch_lightning as pl
from model.io.classifier_io import PerceiverIOClassifier


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


@hydra.main(config_path="conf", config_name="proteins")
def main(cfg: DictConfig):

    dataset = PygNodePropPredDataset(
        name="ogbn-proteins",
        root=os.path.join(get_original_cwd(), "./data",),
        transform=T.ToSparseTensor(attr="edge_attr"),
    )

    split_idx = dataset.get_idx_split()
    test_idx = split_idx["test"]

    data = dataset[0]
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    # Pre-compute GCN normalization.
    adj_t = data.adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t

    dataset = [data]

    train_loader = DataLoader(dataset, batch_size=1, num_workers=32)

    model = PerceiverIOClassifier.load_from_checkpoint(
        checkpoint_path=os.path.join(
            get_original_cwd(),
            "outputs/2022-03-14/18-19-33/graph-perceiver/3q34z3pg/checkpoints/epoch=999-step=999.ckpt",
        )
    )

    model.test_mask = test_idx
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=500,
        log_every_n_steps=1,
        logger=None,
        check_val_every_n_epoch=5,
    )
    trainer.test(model, test_dataloaders=[train_loader])
    # trainer.test(test_dataloaders=[test_loader])


if __name__ == "__main__":
    main()

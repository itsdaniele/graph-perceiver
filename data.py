from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset

import pytorch_lightning as pl
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset


from hydra.utils import get_original_cwd
import os


def get_dataloaders(cfg):

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
    elif "ogb" in cfg.run.dataset:
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

    elif cfg.run.dataset == "pattern":
        num_tasks = 2
        evaluator = None
        dataset_train = GNNBenchmarkDataset(
            root=os.path.join(get_original_cwd(), "./data"),
            name="PATTERN",
            split="train",
        )
        dataset_val = GNNBenchmarkDataset(
            root=os.path.join(get_original_cwd(), "./data"),
            name="PATTERN",
            split="val",
        )

        dataset_test = GNNBenchmarkDataset(
            root=os.path.join(get_original_cwd(), "./data"),
            name="PATTERN",
            split="test",
        )

        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size or cfg.run.batch_size_train,
            shuffle=True,
            num_workers=cfg.train.num_workers,
        )
        valid_loader = DataLoader(
            dataset_val,
            batch_size=batch_size or cfg.run.batch_size_val,
            shuffle=False,
            num_workers=cfg.train.num_workers,
        )
        test_loader = DataLoader(
            dataset_test,
            batch_size=batch_size or cfg.run.batch_size_test,
            shuffle=False,
            num_workers=cfg.train.num_workers,
        )

    else:
        raise NotImplementedError()

    return train_loader, valid_loader, test_loader, num_tasks, evaluator

from model.classifier import PerceiverClassifier

from torch_geometric.loader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from ogb.graphproppred import Evaluator, PygGraphPropPredDataset

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import os

from util import (
    ASTNodeEncoder,
    augment_edge,
    decode_arr_to_seq,
    encode_y_to_arr,
    get_vocab_mapping,
    log_hyperparameters,
)


import torch
from torchvision import transforms
import numpy as np
import pandas as pd


@hydra.main(config_path="conf", config_name="code")
def main(cfg: DictConfig):

    evaluator = Evaluator(name=cfg.run.dataset)
    pl.seed_everything(cfg.run.seed)

    dataset = PygGraphPropPredDataset(
        name=cfg.run.dataset, root=os.path.join(get_original_cwd(), "./data")
    )

    batch_size = cfg.run.get("batch_size")

    split_idx = dataset.get_idx_split()
    seq_len_list = np.array([len(seq) for seq in dataset.data.y])
    vocab2idx, idx2vocab = get_vocab_mapping(
        [dataset.data.y[i] for i in split_idx["train"]], cfg.data.num_vocab
    )
    arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab)
    # set the transform function
    # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
    # encode_y_to_arr: add y_arr to PyG data object, indicating the array representation of a sequence.
    dataset_transform = [
        augment_edge,
        lambda data: encode_y_to_arr(data, vocab2idx, cfg.data.max_seq_len),
    ]
    # dataset_eval.transform = transforms.Compose(dataset_transform)
    if dataset.transform is not None:
        dataset_transform.append(dataset.transform)
    dataset.transform = transforms.Compose(dataset_transform)

    nodetypes_mapping = pd.read_csv(
        os.path.join(dataset.root, "mapping", "typeidx2type.csv.gz")
    )
    nodeattributes_mapping = pd.read_csv(
        os.path.join(dataset.root, "mapping", "attridx2attr.csv.gz")
    )

    # Encoding node features into gnn_emb_dim vectors.
    # The following three node features are used.
    # 1. node type
    # 2. node attribute
    # 3. node depth
    node_encoder_cls = lambda: ASTNodeEncoder(
        cfg.model.gnn_emb_dim,
        num_nodetypes=len(nodetypes_mapping["type"]),
        num_nodeattributes=len(nodeattributes_mapping["attr"]),
        max_depth=20,
    )
    edge_encoder_cls = lambda emb_dim: torch.nn.Linear(2, emb_dim)

    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=batch_size or cfg.run.batch_size_train,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.train.num_workers,
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=batch_size or cfg.run.batch_size_val,
        shuffle=False,
        drop_last=True,
        num_workers=cfg.train.num_workers,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=batch_size or cfg.run.batch_size_test,
        shuffle=False,
    )

    if cfg.model.loss_fn == "bce":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif cfg.model.loss_fn == "multiclass":
        loss_fn = torch.nn.CrossEntropyLoss()

    num_classes = len(vocab2idx)

    model = PerceiverClassifier(
        input_channels=cfg.model.input_channels,
        num_latents=8,
        cross_dim_head=4,
        latent_heads=4,
        latent_dim=128,
        depth=cfg.model.depth,
        attn_dropout=cfg.model.attn_dropout,
        ff_dropout=cfg.model.ff_dropout,
        weight_tie_layers=cfg.model.weight_tie_layers,
        virtual_latent=cfg.model.virtual_latent,
        num_classes=num_classes,
        evaluator=evaluator,
        loss_fn=loss_fn,
        dataset=cfg.run.dataset,
        lr=cfg.model.lr,
        gnn_embed_dim=cfg.model.gnn_emb_dim,
        node_encoder_cls=node_encoder_cls,
        edge_encoder_cls=edge_encoder_cls,
        arr_to_seq=arr_to_seq,
        max_seq_len=cfg.data.max_seq_len,
    )

    if cfg.run.log:

        exp_name = f"{cfg.run.dataset}+seed-{cfg.run.seed}"
        logger = WandbLogger(name=exp_name, project="graph-perceiver")
        lr_monitor = LearningRateMonitor(logging_interval="step")
    else:
        logger = None
        lr_monitor = None

    trainer = pl.Trainer(
        gpus=cfg.train.gpus, max_epochs=cfg.train.epochs, logger=logger, callbacks=None,
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

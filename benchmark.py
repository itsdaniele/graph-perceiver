import torch

from pthflops import count_ops

import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from model.classifier import PerceiverClassifier

from torch_geometric.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import os


evaluator = Evaluator(name="ogbg-molpcba")
dataset = PygGraphPropPredDataset(name="ogbg-molpcba", root="./data")

sample = dataset[0]

model = PerceiverClassifier(
    input_channels=64,
    depth=4,
    attn_dropout=0.0,
    ff_dropout=0.1,
    weight_tie_layers=False,
    virtual_latent=True,
    evaluator=evaluator,
)

print(count_ops(model, sample))

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
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
wandb_logger = WandbLogger(project="graph-perceiver")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    evaluator = Evaluator(name="ogbg-molpcba")
    pl.seed_everything(cfg.run.seed)

    dataset = PygGraphPropPredDataset(
        name="ogbg-molpcba", root=os.path.join(get_original_cwd(), "./data")
    )

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(
        dataset[split_idx["train"]], batch_size=256, shuffle=True, num_workers=32
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]], batch_size=1024, shuffle=False, num_workers=32
    )
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=256, shuffle=False)

    # mc = ModelCheckpoint(
    #     monitor="val/loss",
    #     dirpath="./",
    #     filename="zinc-{epoch:02d}-{val/loss:.2f}",
    #     mode="min",
    # )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model = PerceiverClassifier(
        input_channels=64,
        depth=4,
        attn_dropout=0.0,
        ff_dropout=0.1,
        weight_tie_layers=False,
        virtual_latent=True,
        evaluator=evaluator,
    )
    # model = PerceiverClassifier.load_from_checkpoint(
    #     checkpoint_path=os.path.join(
    #         get_original_cwd(),
    #         "outputs/2022-02-07/same_params/graph-perceiver/2hmqf779/checkpoints/epoch=49-step=68449.ckpt",
    #     ),
    # )

    log = cfg.run.log
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=56,
        logger=wandb_logger if log is True else None,
        callbacks=[lr_monitor] if log is True else None,
    )
    trainer.fit(
        model,
        train_loader,
        valid_loader,
        # ckpt_path=os.path.join(
        #     get_original_cwd(),
        #     "outputs/2022-02-07/same_params/graph-perceiver/2hmqf779/checkpoints/epoch=49-step=68449.ckpt",
        # ),
    )
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

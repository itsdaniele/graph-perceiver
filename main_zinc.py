import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from model.regressor import PerceiverRegressor

from data import MyZINCDataset

from torch_geometric.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

wandb_logger = WandbLogger(project="graph-perceiver")


def main():

    pl.seed_everything(42)

    zinc_train = MyZINCDataset(root="./data", subset=True, split="train")
    zinc_val = MyZINCDataset(root="./data", subset=True, split="val")

    train_loader = DataLoader(zinc_train, batch_size=256, shuffle=True, num_workers=32)
    val_loader = DataLoader(zinc_val, batch_size=256, shuffle=False, num_workers=32)

    # mc = ModelCheckpoint(
    #     monitor="val/loss",
    #     dirpath="./",
    #     filename="zinc-{epoch:02d}-{val/loss:.2f}",
    #     mode="min",
    # )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model = PerceiverRegressor(
        input_channels=64,
        depth=8,
        attn_dropout=0.0,
        ff_dropout=0.1,
        weight_tie_layers=False,
        virtual_latent=False,
    )

    log = True
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10000,
        logger=wandb_logger if log is True else None,
        callbacks=[lr_monitor] if log is True else None,
    )
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

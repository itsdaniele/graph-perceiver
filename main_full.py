import pytorch_lightning as pl
from model_full import PerceiverRegressor

from data import MyZINCDataset

from torch_geometric.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="graph-perceiver")


def main():

    pl.seed_everything(42)

    zinc_train = MyZINCDataset(root="./data", subset=True, split="train")
    zinc_val = MyZINCDataset(root="./data", subset=True, split="val")

    train_loader = DataLoader(zinc_train, batch_size=128, shuffle=True, num_workers=32)
    val_loader = DataLoader(zinc_val, batch_size=128, shuffle=False, num_workers=32)

    model = PerceiverRegressor(
        depth=12, attn_dropout=0.1, ff_dropout=0.1, weight_tie_layers=False,
    )

    trainer = pl.Trainer(
        gpus=1,
        log_gpu_memory=False,
        max_epochs=10000,
        progress_bar_refresh_rate=5,
        logger=wandb_logger,
    )
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

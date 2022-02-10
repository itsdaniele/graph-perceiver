import pytorch_lightning as pl
from model.model_full import PerceiverRegressor

from data import MyZINCDataset, collator

from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="graph-perceiver")

import torch

torch.cuda.empty_cache()

torch.multiprocessing.set_sharing_strategy("file_system")


def main():

    pl.seed_everything(1)

    zinc_train = MyZINCDataset(root="./data", subset=True, split="train")
    zinc_val = MyZINCDataset(root="./data", subset=True, split="val")

    train_loader = DataLoader(
        zinc_train,
        batch_size=256,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        zinc_val,
        batch_size=256,
        shuffle=False,
        num_workers=32,
        pin_memory=False,
        collate_fn=collator,
    )

    model = PerceiverRegressor(depth=12, attn_dropout=0.0, ff_dropout=0.1)

    trainer = pl.Trainer(
        gpus=1, max_epochs=10000, logger=wandb_logger, accelerator="ddp", precision=16
    )
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

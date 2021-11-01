import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from model_regression import PerceiverRegressor
from data import MyZINCDataset, collator


def main():

    zinc_train = MyZINCDataset(
        root="./data", subset=True, split="train", overwrite_x="./embs.pt"
    )
    zinc_val = MyZINCDataset(root="./data", subset=True, split="val")

    train_loader = torch.utils.data.DataLoader(
        zinc_train, batch_size=64, shuffle=True, collate_fn=collator
    )
    val_loader = torch.utils.data.DataLoader(
        zinc_val, batch_size=64, shuffle=False, collate_fn=collator
    )

    mc = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./",
        filename="zinc-{epoch:02d}-{val_loss:.2f}",
        mode="min",
    )

    model = PerceiverRegressor(
        input_channels=1,
        depth=1,
        attn_dropout=0.0,
        ff_dropout=0.0,
        num_classes=1,
        weight_tie_layers=True,
    )
    trainer = pl.Trainer(
        gpus=0,
        precision=32,
        log_gpu_memory=True,
        max_epochs=40,
        progress_bar_refresh_rate=5,
        benchmark=True,
        callbacks=[mc],
    )
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

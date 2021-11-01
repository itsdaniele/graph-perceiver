import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from model_regression import PerceiverRegressor
from data import MyZINCDataset, collator


def main():

    zinc_train = MyZINCDataset(
        root="./data", subset=True, split="train", overwrite_x="./embs_train_2.pt"
    )
    zinc_val = MyZINCDataset(
        root="./data", subset=True, split="val", overwrite_x="./embs_val_2.pt"
    )

    # zinc_train = MyZINCDataset(root="./data", subset=True, split="train")
    # zinc_val = MyZINCDataset(root="./data", subset=True, split="val")

    train_loader = torch.utils.data.DataLoader(
        zinc_train, batch_size=256, shuffle=True, collate_fn=collator, num_workers=32
    )
    val_loader = torch.utils.data.DataLoader(
        zinc_val, batch_size=256, shuffle=False, collate_fn=collator, num_workers=32
    )

    mc = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./",
        filename="zinc-{epoch:02d}-{val_loss:.2f}",
        mode="min",
    )

    model = PerceiverRegressor(
        input_channels=64,
        depth=12,
        attn_dropout=0.1,
        ff_dropout=0.1,
        weight_tie_layers=False,
        initial_emb=False,
    )
    trainer = pl.Trainer(
        gpus=1,
        precision=32,
        log_gpu_memory=True,
        max_epochs=400,
        progress_bar_refresh_rate=5,
        benchmark=True,
        callbacks=[mc],
    )
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

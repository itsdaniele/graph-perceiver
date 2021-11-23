import pytorch_lightning as pl
from model.classifier import PerceiverIOClassifier


from data import MyGNNBenchmarkDataset
from torch_geometric.data import DataLoader

from pytorch_lightning.loggers import WandbLogger


wandb_logger = WandbLogger(project="graph-perceiver")


def main():

    data_train = MyGNNBenchmarkDataset(root="./data", name="CLUSTER", split="train")
    data_val = MyGNNBenchmarkDataset(root="./data", name="CLUSTER", split="val")

    train_loader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=32)
    val_loader = DataLoader(data_val, batch_size=32, shuffle=False, num_workers=32)

    model = PerceiverClassifier(input_channels=6, depth=4)
    trainer = pl.Trainer(
        gpus=1,
        log_gpu_memory=True,
        max_epochs=10000,
        progress_bar_refresh_rate=5,
        logger=wandb_logger,
    )
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

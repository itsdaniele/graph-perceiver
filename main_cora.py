import pytorch_lightning as pl
from perceiver_io import PerceiverIOClassifier

from data import MyCoraDataset


from torch_geometric.data import DataLoader

from pytorch_lightning.loggers import WandbLogger


wandb_logger = WandbLogger(project="graph-perceiver")


def main():

    dataset = MyCoraDataset(root="../data/Cora", name="Cora", encoding="lap")

    loader = DataLoader(dataset, num_workers=32)

    model = PerceiverIOClassifier(depth=1)
    trainer = pl.Trainer(
        gpus=1, max_epochs=200, log_every_n_steps=1, logger=wandb_logger
    )
    trainer.fit(model, loader, loader)
    trainer.test(test_dataloaders=[loader])


if __name__ == "__main__":
    main()

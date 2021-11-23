import pytorch_lightning as pl
from perceiverio import PerceiverIO

from data import MyCoraDataset


from torch_geometric.data import DataLoader

from pytorch_lightning.loggers import WandbLogger


wandb_logger = WandbLogger(project="graph-perceiver")


def main():

    dataset = MyCoraDataset(root="../data/PubMed", name="PubMed", encoding="deg")

    loader = DataLoader(dataset, batch_size=1, num_workers=32)

    model = PerceiverIOClassifier(depth=8)
    trainer = pl.Trainer(
        gpus=1, max_epochs=200, log_every_n_steps=1, logger=wandb_logger
    )
    trainer.fit(model, loader, loader)
    trainer.test(test_dataloaders=[loader])


if __name__ == "__main__":
    main()

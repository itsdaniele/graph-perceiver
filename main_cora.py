import pytorch_lightning as pl
from model.classifier_io import PerceiverIOClassifier
from torch_geometric.datasets import Planetoid


from torch_geometric.loader import DataLoader

from pytorch_lightning.loggers import WandbLogger
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
import os

wandb_logger = WandbLogger(project="graph-perceiver")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    dataset = Planetoid(root=os.path.join(get_original_cwd(), "./data"), name="Cora")

    loader = DataLoader(dataset, batch_size=1, num_workers=32)

    model = PerceiverIOClassifier(depth=3)
    trainer = pl.Trainer(
        gpus=1, max_epochs=200, log_every_n_steps=1, logger=wandb_logger
    )
    trainer.fit(model, loader, loader)
    trainer.test(test_dataloaders=[loader])


if __name__ == "__main__":
    main()

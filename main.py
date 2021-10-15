import pytorch_lightning as pl
import torch
from torchvision.datasets.mnist import MNIST
from pytorch_lightning.callbacks import ModelCheckpoint
from model import PerceiverClassifier


def main():

    # dataset = MNIST(root="./", train=True, download=True, transform=transform)
    # mnist_test = MNIST(root="./", train=False, download=True, transform=transform)
    # # mnist_train, mnist_val = random_split(dataset, [55000, 5000])
    # mnist_train, mnist_val = random_split(dataset, [59500, 500])

    # train_loader = DataLoader(mnist_train, batch_size=64, num_workers=8)
    # val_loader = DataLoader(mnist_val, batch_size=64, num_workers=8)
    # test_loader = DataLoader(mnist_test, batch_size=64, num_workers=8)

    mc = ModelCheckpoint(
        monitor="val_epoch_acc",
        dirpath="./",
        filename="mnist-{epoch:02d}-{val_epoch_acc:.2f}",
        mode="max",
    )

    model = PerceiverClassifier(
        input_channels=1,
        depth=1,
        attn_dropout=0.0,
        ff_dropout=0.0,
        num_classes=10,
        weight_tie_layers=True,
    )
    trainer = pl.Trainer(
        gpus=1,
        precision=32,
        log_gpu_memory=True,
        max_epochs=40,
        progress_bar_refresh_rate=5,
        benchmark=True,
        callbacks=[mc],
    )
    # trainer.fit(model, train_loader, val_loader)
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

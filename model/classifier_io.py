import torch
import torch.nn.functional as F

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import torchmetrics

from .perceiverio import PerceiverIO


class PerceiverIOClassifier(pl.LightningModule):
    def __init__(
        self,
        queries_dim=None,
        gnn_embed_dim=8,
        depth=1,
        num_latents=8,
        latent_dim=32,
        cross_heads=4,
        latent_heads=4,
        cross_dim_head=4,
        latent_dim_head=4,
        logits_dim=7,
        decoder_ff=True,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
    ):
        super().__init__()

        self.save_hyperparameters()

        queries_dim = gnn_embed_dim

        self.perceiver = PerceiverIO(
            gnn_embed_dim=gnn_embed_dim,
            queries_dim=queries_dim,
            depth=depth,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            logits_dim=logits_dim,
            decoder_ff=decoder_ff,
            weight_tie_layers=weight_tie_layers,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

    def forward(self, x, **kwargs):
        return self.perceiver(x, queries=None, **kwargs)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)[:, batch.train_mask].transpose(1, 2)
        y = batch.y[batch.train_mask].unsqueeze(0)
        loss = F.cross_entropy(y_hat, y)

        acc = torchmetrics.functional.accuracy(torch.argmax(y_hat, dim=-2), y)
        self.log("train/acc", acc)
        self.log("train/loss", loss)

        return loss

    def evaluate(self, batch, stage=None):
        y_hat = self(batch)[:, batch.val_mask].transpose(1, 2)

        y = batch.y[batch.val_mask].unsqueeze(0)
        loss = F.cross_entropy(y_hat, y)

        acc = torchmetrics.functional.accuracy(torch.argmax(y_hat, dim=-2), y)
        self.log("val/acc", acc)
        self.log("val/loss", loss)

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)[:, batch.test_mask].transpose(1, 2)
        y = batch.y[batch.test_mask].unsqueeze(0)

        acc = torchmetrics.functional.accuracy(torch.argmax(y_hat, dim=-2), y)
        self.log("test/acc", acc)
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return [optimizer]

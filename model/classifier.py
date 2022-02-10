from .perceiver import Perceiver


import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import torchmetrics


class PerceiverClassifier(pl.LightningModule):
    def __init__(
        self,
        input_channels=64,
        gnn_embed_dim=32,
        depth=8,
        num_latents=8,
        latent_dim=200,
        cross_heads=8,
        latent_heads=8,
        cross_dim_head=8,
        latent_dim_head=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        self_per_cross_attn=1,
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        virtual_latent=False,
        evaluator=None,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.loss_fn = loss_fn
        self.evaluator = evaluator

        self.perceiver = Perceiver(
            input_channels=input_channels,
            gnn_embed_dim=gnn_embed_dim,
            depth=depth,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            weight_tie_layers=weight_tie_layers,
            self_per_cross_attn=self_per_cross_attn,
            virtual_latent=virtual_latent,
        )

    def forward(self, x, **kwargs):
        return self.perceiver(x, **kwargs)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        is_labeled = batch.y == batch.y

        loss = self.loss_fn(
            y_hat.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]
        )

        res = {
            "y_true": batch.y.detach().cpu(),
            "y_pred": y_hat.detach().cpu(),
        }
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log(
            "train/ap", self.evaluator.eval(res)["ap"], on_step=False, on_epoch=True,
        )
        return loss

    def evaluate(self, batch, stage=None):
        # y_hat = self(batch).view(-1)
        # loss = self.loss_fn(y_hat.to(torch.float32), batch.y.to(torch.float32).view(-1))

        y_hat = self(batch)

        is_labeled = batch.y == batch.y

        loss = self.loss_fn(
            y_hat.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]
        )
        self.log(f"{stage}/loss", loss)

        res = {
            "y_true": batch.y.detach().cpu(),
            "y_pred": y_hat.detach().cpu(),
        }

        ap = self.evaluator.eval(res)["ap"]
        self.log(f"{stage}/ap", ap)

        return ap

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return [optimizer]

from .perceiver import Perceiver


import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class PerceiverRegressor(pl.LightningModule):
    def __init__(
        self,
        input_channels=64,
        gnn_embed_dim=128,
        depth=8,
        num_latents=8,
        latent_dim=32,
        cross_heads=8,
        latent_heads=8,
        cross_dim_head=8,
        latent_dim_head=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        self_per_cross_attn=1,
        loss_fn=F.l1_loss,
        virtual_latent=False,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.loss_fn = loss_fn

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
        y_hat = self(batch).view(-1)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def evaluate(self, batch, stage=None):
        y_hat = self(batch).view(-1)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("val/loss", loss)

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        return [optimizer]

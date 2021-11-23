from .perceiver import Perceiver


import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class PerceiverRegressor(pl.LightningModule):
    def __init__(
        self,
        input_channels=64,
        gnn_embed_dim=128,
        depth=6,
        num_latents=8,
        latent_dim=32,
        cross_heads=8,
        latent_heads=8,
        cross_dim_head=8,
        latent_dim_head=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        self_per_cross_attn=2,
        lap_encodings_dim=8,
        encoding_type="lap",
        loss_fn=F.l1_loss,
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
            lap_encodings_dim=lap_encodings_dim,
            encoding_type=encoding_type,
        )

    def forward(self, x, **kwargs):
        return self.perceiver(x, **kwargs)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch).view(-1)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("train/loss", loss)

        return loss

    def evaluate(self, batch, stage=None):
        y_hat = self(batch, training=False).view(-1)
        loss = self.loss_fn(y_hat, batch.y)
        self.log("val/loss", loss)

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4, weight_decay=0.01)
        return [optimizer]

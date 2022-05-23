import torch
import torch.nn.functional as F

import pytorch_lightning as pl
import torch
import torch.nn.functional as F


from ogb.nodeproppred import Evaluator
from model.gnn import GCNARXIV

from perceiver_io.decoders import PerceiverDecoder
from perceiver_io.encoder import PerceiverEncoder
from perceiver_io import PerceiverIO


class PerceiverIOClassifier(pl.LightningModule):
    def __init__(
        self,
        queries_dim=None,
        gnn_embed_dim=32,
        depth=1,
        num_latents=96,
        latent_dim=32,
        cross_heads=4,
        latent_heads=4,
        cross_dim_head=8,
        latent_dim_head=8,
        logits_dim=7,
        decoder_ff=True,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        lr=0.0001,
        dataset="cora",
        train_mask=None,
        val_mask=None,
        test_mask=None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        self.lr = lr

        self.dataset = dataset
        self.evaluator = Evaluator(name=dataset)

        self.gnn = GCNARXIV(gnn_embed_dim)

        encoder = PerceiverEncoder(
            num_latents=num_latents,
            latent_dim=latent_dim,
            input_dim=gnn_embed_dim * 2,
            num_self_attn_per_block=8,
            num_blocks=1,
        )

        decoder = PerceiverDecoder(latent_dim=latent_dim, query_dim=gnn_embed_dim * 2)
        self.perceiver = PerceiverIO(encoder, decoder)
        self.to_logits = torch.nn.Linear(gnn_embed_dim * 2, logits_dim)

    def forward(self, x, **kwargs):
        return self.perceiver(x, queries=None, **kwargs)

    def training_step(self, batch, batch_idx):

        gnn_out = self.gnn(batch).unsqueeze(0)
        out = self.to_logits(self.perceiver(gnn_out, gnn_out)).squeeze(0)
        y_hat = out[self.train_mask]
        y = batch.y[self.train_mask].squeeze(-1)
        train_loss = F.cross_entropy(y_hat, y)

        acc = self.evaluator.eval(
            {
                "y_true": batch.y[self.train_mask],
                "y_pred": out.squeeze(0).argmax(dim=-1, keepdim=True)[self.train_mask],
            }
        )["acc"]
        self.log("train/acc", acc, batch_size=1)
        self.log("train/loss", train_loss, batch_size=1)

        return train_loss

    def evaluate(self, batch, stage=None):

        gnn_out = self.gnn(batch).unsqueeze(0)
        out = self.to_logits(self.perceiver(gnn_out, gnn_out)).squeeze(0)

        y_hat = out[self.val_mask]

        y = batch.y[self.val_mask].squeeze(-1)
        loss = F.cross_entropy(y_hat, y)

        acc = self.evaluator.eval(
            {
                "y_true": batch.y[self.val_mask],
                "y_pred": out.squeeze(0).argmax(dim=-1, keepdim=True)[self.val_mask],
            }
        )["acc"]
        self.log("val/acc", acc, batch_size=1)
        self.log("val/loss", loss, batch_size=1)

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):

        # l1 = list(filter(lambda kv: "gnn" in kv[0], self.named_parameters()))
        # l2 = list(filter(lambda kv: "gnn" not in kv[0], self.named_parameters()))

        # optimizer = torch.optim.Adam(
        #     [
        #         {"params": [i[1] for i in l1], "lr": 0.01},
        #         {"params": [i[1] for i in l2], "lr": 0.0005},
        #     ]
        # )

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler_poly_lr_decay = PolynomialLRDecay(
        #     optimizer, max_decay_steps=100, end_learning_rate=0.0001, power=1.0
        # )

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, 1000, eta_min=0.00001, last_epoch=-1, verbose=False
        # )
        return [optimizer]

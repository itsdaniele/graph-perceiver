import torch
import torch.nn.functional as F

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import torchmetrics

from .perceiverio import PerceiverIO
from ogb.nodeproppred import Evaluator


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

        try:
            self.evaluator = Evaluator(name=dataset)
        except:
            pass

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
            gnn_encoder=None,
        )

    def forward(self, x, **kwargs):
        return self.perceiver(x, queries=None, **kwargs)

    def training_step(self, batch, batch_idx):

        if self.dataset == "ogbn-arxiv":
            y_hat = self(batch)[self.train_mask]
            y = batch.y[self.train_mask].squeeze(-1)
            train_loss = F.cross_entropy(y_hat, y)

            self.log("train/loss", train_loss, batch_size=1)
        elif self.dataset == "ogbn-proteins":

            y_hat = self(batch)
            y_hat_train = y_hat[self.train_mask]
            y = batch.y[self.train_mask].to(torch.float)
            train_loss = F.binary_cross_entropy_with_logits(y_hat_train, y)

            rocauc = self.evaluator.eval(
                {"y_true": batch.y[self.train_mask], "y_pred": y_hat_train}
            )["rocauc"]

            self.log("train/rocauc", rocauc)
            self.log("train/loss", train_loss)

        return train_loss

    def evaluate(self, batch, stage=None):

        if self.dataset == "ogbn-arxiv":

            out = self(batch)
            y_hat = out[self.val_mask]

            y = batch.y[self.val_mask].squeeze(-1)
            loss = F.cross_entropy(y_hat, y)

            acc = self.evaluator.eval(
                {
                    "y_true": batch.y[self.val_mask],
                    "y_pred": out.squeeze(0).argmax(dim=-1, keepdim=True)[
                        self.val_mask
                    ],
                }
            )["acc"]
            self.log("val/acc", acc, batch_size=1)
            self.log("val/loss", loss, batch_size=1)

        elif self.dataset == "ogbn-proteins":
            y_hat = self(batch)
            y_hat_val = y_hat[self.val_mask]
            y_hat_test = y_hat[self.test_mask]

            y = batch.y[self.val_mask].to(torch.float)
            val_loss = F.binary_cross_entropy_with_logits(y_hat_val, y)

            rocauc = self.evaluator.eval(
                {"y_true": batch.y[self.val_mask], "y_pred": y_hat_val}
            )["rocauc"]
            rocauc_test = self.evaluator.eval(
                {"y_true": batch.y[self.test_mask], "y_pred": y_hat_test}
            )["rocauc"]
            self.log("val/rocauc", rocauc, batch_size=1)
            self.log("test/rocauc", rocauc_test, batch_size=1)
            self.log("val/loss", val_loss, batch_size=1)

            # if attns is not None:
            #     entropy = attention_entropy(attns[0])
            #     wandb.log(
            #         {
            #             "attention_entropy": wandb.plots.HeatMap(
            #                 list(range(256)), list(range(4)), entropy, show_text=False
            #             )
            #         }
            #     )

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        if self.dataset == "ogbn-proteins":
            y_hat = self(batch)
            y_hat_test = y_hat[self.test_mask]

            rocauc = self.evaluator.eval(
                {"y_true": batch.y[self.test_mask], "y_pred": y_hat_test}
            )["rocauc"]
            self.log("test/rocauc", rocauc, batch_size=1)

        elif self.dataset == "ogbn-arxiv":
            y_hat = self(batch)[self.test_mask]
            acc = self.evaluator.eval(
                {
                    "y_true": batch.y[self.test_mask],
                    "y_pred": self(batch)
                    .squeeze(0)
                    .argmax(dim=-1, keepdim=True)[self.test_mask],
                }
            )["acc"]
            self.log("test/acc", acc, batch_size=1)

    def configure_optimizers(self):

        # l1 = list(filter(lambda kv: "gnn" in kv[0], self.named_parameters()))
        # l2 = list(filter(lambda kv: "gnn" not in kv[0], self.named_parameters()))

        # optimizer = torch.optim.Adam(
        #     [
        #         {"params": [i[1] for i in l1], "lr": 0.01},
        #         {"params": [i[1] for i in l2], "lr": 0.0001},
        #     ]
        # )

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=0.05 * 2000, num_training_steps=2000
        # )
        # scheduler_poly_lr_decay = PolynomialLRDecay(
        #     optimizer, max_decay_steps=100, end_learning_rate=0.0001, power=1.0
        # )

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, 1000, eta_min=0.00001, last_epoch=-1, verbose=False
        # )
        return [optimizer]

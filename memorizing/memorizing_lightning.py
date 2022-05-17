import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from memorizing_transformer import MemorizingTransformer
from ogb.nodeproppred import Evaluator


from gnn import GCNCORA, GCNPROTEINS

from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool

import torchmetrics


class MemorizingLightning(pl.LightningModule):
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
        valid_mask=None,
        test_mask=None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_mask = train_mask
        self.valid_mask = valid_mask
        self.test_mask = test_mask

        self.lr = lr

        self.dataset = dataset

        try:
            self.evaluator = Evaluator(name=dataset)
        except:
            pass

        # self.gnn = GCNPROTEINS(gnn_embed_dim)
        self.gnn = GCNCORA(gnn_embed_dim)

        self.transformer = MemorizingTransformer(
            num_logits=logits_dim,  # number of tokens
            dim=gnn_embed_dim,  # dimension
            dim_head=8,  # dimension per attention head
            depth=3,  # number of layers
            memorizing_layers=(1,),  # which layers to have ANN memories
            max_knn_memories=64000,  # maximum ANN memories to keep (once it hits this capacity, it will be reset for now, due to limitations in faiss' ability to remove entries)
            num_retrieved_memories=32,  # number of ANN memories to retrieve
        )
        self.knn_memories = self.transformer.create_knn_memories(batch_size=32)
        self.to_logits = nn.Linear(gnn_embed_dim, logits_dim)

    def forward(self, x):

        gnn_out = self.gnn(x)
        gnn_out, _ = to_dense_batch(gnn_out, x.batch, batch_size=32)
        out = self.transformer(gnn_out, knn_memories=self.knn_memories)

        x = out.sum(dim=1)
        x = self.to_logits(x)
        return x

    def training_step(self, batch, batch_idx):

        if self.dataset == "cora":

            y_hat = self(batch)
            y = batch.y

            loss = F.cross_entropy(y_hat, y)

            acc = torchmetrics.functional.accuracy(
                torch.argmax(F.log_softmax(y_hat, dim=1), dim=-1), y
            )

            self.log("train/acc", acc)
            self.log("train/loss", loss)
        else:
            y_hat = self(batch)
            y_hat_train = y_hat
            y_train = batch.y.float()
            train_loss = F.binary_cross_entropy_with_logits(y_hat_train, y_train)

            self.log("train/loss", train_loss, batch_size=1)
            return train_loss

    def evaluate(self, batch, stage=None):

        if self.dataset == "cora":

            y_hat = self(batch)
            y = batch.y

            loss = F.cross_entropy(y_hat, y)

            acc = torchmetrics.functional.accuracy(
                torch.argmax(F.log_softmax(y_hat, dim=1), dim=-1), y
            )

            self.log("val/acc", acc)
            self.log("val/loss", loss)
        else:

            y_hat = self(batch)

            y_val = batch.y.float()
            val_loss = F.binary_cross_entropy_with_logits(y_hat, y_val)

            rocauc = self.evaluator.eval({"y_true": batch.y, "y_pred": y_hat})["rocauc"]

            self.log("val/rocauc", rocauc, batch_size=1)
            self.log("val/loss", val_loss, batch_size=1)

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

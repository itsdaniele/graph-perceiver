from .perceiver import Perceiver
from .gnn import GNN_node_Virtualnode


from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

import pytorch_lightning as pl
import torch


class PerceiverClassifier(pl.LightningModule):
    def __init__(
        self,
        evaluator,
        num_classes,
        loss_fn,
        dataset,
        gnn_embed_dim=32,
        depth=8,
        num_latents=8,
        latent_dim=200,
        queries_dim=200,
        cross_heads=8,
        latent_heads=8,
        cross_dim_head=8,
        latent_dim_head=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        gnn_dropout=0.0,
        weight_tie_layers=False,
        self_per_cross_attn=1,
        gnn_type="gin-virtual",
        num_gnn_layers=3,
        lr=0.0001,
        arr_to_seq=None,
        max_seq_len=None,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.dataset = dataset

        self.lr = lr

        self.arr_to_seq = arr_to_seq

        # We need this because we can't let Lightning handle molhiv
        # as it will fail in calculating the per-batch rocauc due to no positive values.
        if dataset == "ogbg-molhiv":
            self.metric_hist = {
                "train/rocauc": [],
                "val/rocauc": [],
                "train/loss": [],
                "val/loss": [],
            }

        gnn = None
        if dataset in ["ogbg-molhiv", "ogbg-molpcba"]:
            if gnn_type == "gin-virtual":
                gnn = GNN_node_Virtualnode(
                    num_gnn_layers,
                    gnn_embed_dim,
                    AtomEncoder(gnn_embed_dim),
                    BondEncoder,
                    gnn_type="gin",
                    drop_ratio=gnn_dropout,
                )
        else:
            raise NotImplementedError()

        self.perceiver = Perceiver(
            gnn_embed_dim=gnn_embed_dim,
            depth=depth,
            gnn=gnn,
            num_latents=num_latents,
            latent_dim=latent_dim,
            queries_dim=queries_dim,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            weight_tie_layers=weight_tie_layers,
            self_per_cross_attn=self_per_cross_attn,
            num_classes=num_classes,
            max_seq_len=max_seq_len,
        )

    def forward(self, x, **kwargs):
        return self.perceiver(x, **kwargs)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)

        if self.dataset == "ogbg-molpcba":
            is_labeled = batch.y == batch.y

            loss = self.loss_fn(
                y_hat.to(torch.float32)[is_labeled],
                batch.y.to(torch.float32)[is_labeled],
            )

            res = {
                "y_true": batch.y.detach().cpu(),
                "y_pred": y_hat.detach().cpu(),
            }
            self.log("train/loss", loss, on_step=False, on_epoch=True)
            self.log(
                "train/ap",
                self.evaluator.eval(res)["ap"],
                on_step=False,
                on_epoch=True,
            )
        elif self.dataset == "ogbg-molhiv":
            loss = self.loss_fn(y_hat, batch.y.float())
            self.log("train/loss", loss, on_step=False, on_epoch=True)

        elif self.dataset == "ogbg-code2":

            loss = 0

            for i in range(len(y_hat)):
                loss += self.loss_fn(y_hat[i].to(torch.float32), batch.y_arr[:, i])
            loss /= len(y_hat)
            self.log("train/loss", loss)

        return loss

    def evaluate(self, batch, stage=None):

        y_hat = self(batch)

        if self.dataset == "ogbg-molpcba":

            is_labeled = batch.y == batch.y

            loss = self.loss_fn(
                y_hat.to(torch.float32)[is_labeled],
                batch.y.to(torch.float32)[is_labeled],
            )
            self.log(f"{stage}/loss", loss)

            res = {
                "y_true": batch.y.detach().cpu(),
                "y_pred": y_hat.detach().cpu(),
            }

            ap = self.evaluator.eval(res)["ap"]
            self.log(f"{stage}/ap", ap)
            return ap

        elif self.dataset == "ogbg-molhiv":

            loss = self.loss_fn(y_hat, batch.y.float())
            self.log(f"{stage}/loss", loss)

            y_true = batch.y.detach().cpu()
            y_pred = y_hat.detach().cpu()

            return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

        elif self.dataset == "ogbg-code2":

            loss = 0

            for i in range(len(y_hat)):
                loss += self.loss_fn(y_hat[i].to(torch.float32), batch.y_arr[:, i])
            loss /= len(y_hat)
            self.log("val/loss", loss)

            mat = []
            for i in range(len(y_hat)):
                mat.append(torch.argmax(y_hat[i], dim=1).view(-1, 1))
            mat = torch.cat(mat, dim=1)

            seq_pred = [self.arr_to_seq(arr) for arr in mat]
            seq_ref = [batch.y[i] for i in range(len(batch.y))]

        input_dict = {"seq_ref": seq_ref, "seq_pred": seq_pred}
        f1 = self.evaluator.eval(input_dict)["F1"]
        self.log("val/F1", f1)
        return {"F1": f1}

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def validation_epoch_end(self, outputs):

        if self.dataset == "ogbg-molhiv":
            rocauc = self.calculate_metric(outputs)
            self.metric_hist["val/rocauc"].append(rocauc)
            self.metric_hist["val/loss"].append(
                self.trainer.callback_metrics["val/loss"]
            )
            self.log("val/rocauc", rocauc, prog_bar=True)
            self.log(
                "val/rocauc_best", max(self.metric_hist["val/rocauc"]), prog_bar=True
            )
            self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

    def test_step(self, batch, batch_idx):

        if self.dataset == "ogbg-molhiv":
            y_hat = self(batch)
            y_true = batch.y
            y_pred = y_hat.detach().cpu()
            loss = self.loss_fn(y_hat, batch.y.float())
            self.log(f"test/loss", loss)
            return {"loss": loss, "y_pred": y_pred, "y_true": y_true}
        else:
            return self.evaluate(batch, "test")

    def test_epoch_end(self, outputs):
        if self.dataset == "ogbg-molhiv":
            self.log("test/rocauc", self.calculate_metric(outputs), prog_bar=False)

    def calculate_metric(self, outputs):
        y_true = torch.cat([x["y_true"] for x in outputs], dim=0)
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0)
        result_dict = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred})
        return result_dict["rocauc"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

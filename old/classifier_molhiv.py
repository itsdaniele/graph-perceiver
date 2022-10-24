from .perceiver import Perceiver


import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import torchmetrics


class PerceiverClassifier(pl.LightningModule):
    def __init__(
        self,
        input_channels=64,
        gnn_embed_dim=64,
        depth=16,
        num_latents=16,
        latent_dim=64,
        cross_heads=8,
        latent_heads=8,
        cross_dim_head=16,
        latent_dim_head=16,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        self_per_cross_attn=1,
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        virtual_latent=False,
        evaluator=None,
        train_loader=None,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.loss_fn = loss_fn
        self.evaluator = evaluator

        self.train_loader = train_loader

        self.metric_hist = {
            "train/rocauc": [],
            "val/rocauc": [],
            "train/loss": [],
            "val/loss": [],
        }

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
            num_classes=1,
        )

    def forward(self, x, **kwargs):
        return self.perceiver(x, **kwargs)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch.y.float())
        self.log("train/loss", loss, on_step=False, on_epoch=True)

        return loss

    def evaluate(self, batch, stage=None):

        y_hat = self(batch)

        loss = self.loss_fn(y_hat, batch.y.float())
        self.log(f"{stage}/loss", loss)

        y_true = batch.y.detach().cpu()
        y_pred = y_hat.detach().cpu()

        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    def validation_epoch_end(self, outputs):
        rocauc = self.calculate_metric(outputs)
        self.metric_hist["val/rocauc"].append(rocauc)
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/rocauc", rocauc, prog_bar=True)
        self.log("val/rocauc_best", max(self.metric_hist["val/rocauc"]), prog_bar=True)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx: int):

        y_hat = self(batch)
        y_true = batch.y
        y_pred = y_hat.detach().cpu()
        loss = self.loss_fn(y_hat, batch.y.float())
        self.log(f"test/loss", loss)
        return {"loss": loss, "y_pred": y_pred, "y_true": y_true}

    def test_epoch_end(self, outputs):
        self.log("test/rocauc", self.calculate_metric(outputs), prog_bar=False)

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):

        # decay = set()
        # no_decay = set()
        # whitelist_weight_modules = (
        #     torch.nn.Linear,
        #     torch.nn.Conv2d,
        #     torch.nn.ConvTranspose2d,
        # )
        # blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        # for mn, m in self.named_modules():
        #     for pn, p in m.named_parameters():
        #         fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
        #         if pn.endswith("bias"):
        #             # all biases will not be decayed
        #             no_decay.add(fpn)
        #         elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
        #             # weights of whitelist modules will be weight decayed
        #             decay.add(fpn)
        #         elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
        #             # weights of blacklist modules will NOT be weight decayed
        #             no_decay.add(fpn)
        # # special case the position embedding parameter in the root GPT module as not decayed
        # # validate that we considered every parameter
        # param_dict = {pn: p for pn, p in self.named_parameters()}
        # inter_params = decay & no_decay
        # union_params = decay | no_decay
        # assert (
        #     len(inter_params) == 0
        # ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        # assert len(param_dict.keys() - union_params) == 0, (
        #     "parameters %s were not separated into either decay/no_decay set!"
        #     % (str(param_dict.keys() - union_params),)
        # )
        # # create the pytorch optimizer object
        # optim_groups = [
        #     {
        #         "params": [param_dict[pn] for pn in sorted(list(decay))],
        #         "weight_decay": 0.01,
        #     },
        #     {
        #         "params": [param_dict[pn] for pn in sorted(list(no_decay))],
        #         "weight_decay": 0.0,
        #     },
        # ]
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 100 * len(self.train_loader), verbose=True
        )
        return [optimizer], [scheduler]

    def calculate_metric(self, outputs):
        y_true = torch.cat([x["y_true"] for x in outputs], dim=0)
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0)
        result_dict = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred})
        return result_dict["rocauc"]

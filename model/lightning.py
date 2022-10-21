import pytorch_lightning as pl
import torch

from util import get_linear_schedule_with_warmup


class BaseLightning(pl.LightningModule):
    def __init__(
        self,
        evaluator,
        loss_fn,
        dataset,
        lr,
        weight_decay,
        scheduler,
        scheduler_params,
        arr_to_seq,
    ):
        super().__init__()
        self.evaluator = evaluator
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.arr_to_seq = arr_to_seq

        # We need this because we can't let Lightning handle molhiv
        # as it will fail in calculating the per-batch rocauc due to no positive values.
        if dataset == "ogbg-molhiv":
            self.metric_hist = {
                "train/rocauc": [],
                "val/rocauc": [],
            }

    def training_step(self, batch):
        y_hat = self(batch)

        if self.dataset == "zinc":
            loss = self.loss_fn(y_hat.squeeze(-1), batch.y)

        elif self.dataset == "ogbg-molpcba":
            is_labeled = batch.y == batch.y

            loss = self.loss_fn(
                y_hat.to(torch.float32)[is_labeled],
                batch.y.to(torch.float32)[is_labeled],
            )

        elif self.dataset == "ogbg-molhiv":
            loss = self.loss_fn(y_hat, batch.y.float())

        elif self.dataset == "zinc":
            loss = self.loss_fn(y_hat, batch.y)

        elif self.dataset == "ogbg-code2":
            loss = 0
            for i in range(len(y_hat)):
                loss += self.loss_fn(y_hat[i].to(torch.float32), batch.y_arr[:, i])
            loss /= len(y_hat)
        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

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

    def evaluate(self, batch, stage=None):

        y_hat = self(batch)

        if self.dataset == "zinc":

            loss = self.loss_fn(y_hat.squeeze(-1), batch.y)
            return_dict = {"loss": loss}

        elif self.dataset == "ogbg-molpcba":

            is_labeled = batch.y == batch.y
            loss = self.loss_fn(
                y_hat.to(torch.float32)[is_labeled],
                batch.y.to(torch.float32)[is_labeled],
            )

            return_dict = {
                "loss": loss,
                "y_pred": y_hat.detach().cpu(),
                "y_true": batch.y.detach().cpu(),
            }

        elif self.dataset == "ogbg-molhiv":

            loss = self.loss_fn(y_hat, batch.y.float())

            y_true = batch.y.detach().cpu()
            y_pred = y_hat.detach().cpu()

            return_dict = {"loss": loss, "y_pred": y_pred, "y_true": y_true}

        elif self.dataset == "ogbg-code2":

            loss = 0

            for i in range(len(y_hat)):
                loss += self.loss_fn(y_hat[i].to(torch.float32), batch.y_arr[:, i])
            loss /= len(y_hat)

            mat = []
            for i in range(len(y_hat)):
                mat.append(torch.argmax(y_hat[i], dim=1).view(-1, 1))
            mat = torch.cat(mat, dim=1)

            seq_pred = [self.arr_to_seq(arr) for arr in mat]
            seq_ref = [batch.y[i] for i in range(len(batch.y))]

            input_dict = {"seq_ref": seq_ref, "seq_pred": seq_pred}
            f1 = self.evaluator.eval(input_dict)["F1"]
            self.log(f"{stage}/F1", f1)
            return_dict = {"F1": f1}
        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")

        self.log(f"{stage}/loss", loss)
        return return_dict

    def validation_epoch_end(self, outputs):

        if self.dataset == "ogbg-molhiv":
            rocauc = self.calculate_metric(outputs)
            self.metric_hist["val/rocauc"].append(rocauc)

            self.log("val/rocauc", rocauc, prog_bar=True)
            self.log(
                "val/rocauc_best", max(self.metric_hist["val/rocauc"]), prog_bar=True
            )
        elif self.dataset == "ogbg-molpcba":
            ap = self.calculate_metric(outputs)
            self.log("val/ap", ap, prog_bar=True)

    def test_epoch_end(self, outputs):
        if self.dataset == "ogbg-molhiv":
            self.log("test/rocauc", self.calculate_metric(outputs), prog_bar=False)
        elif self.dataset == "ogbg-molpcba":
            self.log("test/ap", self.calculate_metric(outputs), prog_bar=False)

    def calculate_metric(self, outputs):
        y_true = torch.cat([x["y_true"] for x in outputs], dim=0)
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0)
        result_dict = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred})
        if self.dataset == "ogbg-molhiv":
            return result_dict["rocauc"]
        elif self.dataset == "ogbg-molpcba":
            return result_dict["ap"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.scheduler == "cosine_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.scheduler_params.num_warmup_epochs,
                num_training_steps=self.scheduler_params.max_epoch,
            )
        else:
            raise ValueError(f"`{self.scheduler}` not supported")
        return [optimizer], [scheduler]


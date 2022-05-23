import torch
import torch.nn.functional as F

import pytorch_lightning as pl
import torch
import torch.nn.functional as F


from ogb.nodeproppred import Evaluator
from model.gnn import GCNARXIV

from transformers import (
    PerceiverConfig,
    PerceiverModel,
)

from transformers.models.perceiver.modeling_perceiver import (
    PerceiverBasicDecoder,
    PerceiverAbstractDecoder,
    PerceiverDecoderOutput,
)

import torch.nn as nn


class AbstractPreprocessor(torch.nn.Module):
    @property
    def num_channels(self) -> int:
        """Returns size of preprocessor output."""
        raise NotImplementedError()


class PerceiverGraphDecoder(PerceiverAbstractDecoder):
    def __init__(self, config, num_channels, final_project=True, **decoder_kwargs):
        super().__init__()

        self.decoder = PerceiverBasicDecoder(
            config,
            position_encoding_type="none",
            output_num_channels=256,
            num_heads=8,
            final_project=final_project,
            num_channels=num_channels,
            use_query_residual=False,
            **decoder_kwargs,
        )

    @property
    def num_query_channels(self) -> int:
        return self.decoder.num_query_channels

    def decoder_query(
        self,
        inputs,
        modality_sizes=None,
        inputs_without_pos=None,
        subsampled_points=None,
    ):
        if subsampled_points is not None:
            raise ValueError("FlowDecoder doesn't support subsampling yet.")
        return inputs

    def forward(self, query, z, query_mask=None, output_attentions=False):
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)
        preds = decoder_outputs.logits
        return PerceiverDecoderOutput(
            logits=preds, cross_attentions=decoder_outputs.cross_attentions
        )


class PerceiverPreprocessor(AbstractPreprocessor):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gnn = GCNARXIV(config.d_model)

    @property
    def num_channels(self) -> int:
        return self.config.d_model

    def forward(self, inputs):
        out = self.gnn(inputs).unsqueeze(0)
        return out, None, out


def _init_weights(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


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

        config = PerceiverConfig(
            num_latents=num_latents,
            d_latents=latent_dim,
            d_model=gnn_embed_dim,
            num_self_attends_per_block=2,
            num_labels=logits_dim,
            num_blocks=depth,
            layer_norm_eps=1e-05,
        )

        preprocessor = PerceiverPreprocessor(config)
        decoder = PerceiverGraphDecoder(config, num_channels=gnn_embed_dim)
        self.perceiver = PerceiverModel(
            config, input_preprocessor=preprocessor, decoder=decoder,
        )
        _init_weights(self.perceiver)

    def forward(self, x, **kwargs):
        return self.perceiver(x, **kwargs)

    def training_step(self, batch, batch_idx):

        out = self(batch).logits.squeeze(0)
        y_hat = out[self.train_mask]
        y = batch.y[self.train_mask].squeeze(-1)
        train_loss = F.cross_entropy(y_hat, y)

        # acc = self.evaluator.eval(
        #     {
        #         "y_true": batch.y[self.train_mask],
        #         "y_pred": out.detach().argmax(dim=-1, keepdim=True)[self.train_mask],
        #     }
        # )["acc"]
        # self.log("train/acc", acc, batch_size=1)
        self.log("train/loss", train_loss, batch_size=1)

        return train_loss

    def evaluate(self, batch, stage=None):

        out = self(batch).logits.squeeze(0)

        y_hat = out[self.val_mask]

        y = batch.y[self.val_mask].squeeze(-1)
        loss = F.cross_entropy(y_hat, y)

        acc = self.evaluator.eval(
            {
                "y_true": batch.y[self.val_mask],
                "y_pred": out.squeeze(0).argmax(dim=-1, keepdim=True)[self.val_mask],
            }
        )["acc"]
        self.log(f"{stage}/acc", acc, batch_size=1)
        self.log(f"{stage}/loss", loss, batch_size=1)

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

# Code mainly taken from https://github.com/lucidrains/perceiver-pytorch

from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy


# helpers


class PerceiverRegressor(pl.LightningModule):
    def __init__(
        self,
        input_channels=3,
        depth=6,
        num_latents=32,
        latent_dim=64,
        cross_heads=8,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=1,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        self_per_cross_attn=2,
        initial_emb=True,
    ):
        super().__init__()

        self.initial_emb = initial_emb
        self.save_hyperparameters()

        self.perceiver = Perceiver(
            input_channels=input_channels,
            depth=depth,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            num_classes=num_classes,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            weight_tie_layers=weight_tie_layers,
            self_per_cross_attn=self_per_cross_attn,
        )

    def forward(self, x, **kwargs):
        return self.perceiver(x, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x, initial_emb=self.initial_emb, mask=mask).view(-1)
        loss = F.l1_loss(y_hat, y)

        self.log("Training Loss", loss)

        return loss

    def evaluate(self, batch, stage=None):
        x, y, mask = batch
        y_hat = self(x, initial_emb=self.initial_emb, mask=mask).view(-1)
        loss = F.l1_loss(y_hat, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
        return {"loss": loss, "y": y}

    def epoch_evaluate(self, step_outputs, stage=None):
        y_list = []

        loss_list = []

        for output in step_outputs:

            y_list.append(output["y"])

            loss_list.append(output["loss"])

        y_cat = torch.cat(y_list)

        loss_cat = torch.tensor(loss_list)

        loss = torch.mean(loss_cat)

        if stage:

            self.log(f"{stage}_epoch_loss", loss)

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def validation_epoch_end(self, val_step_outputs):
        self.epoch_evaluate(val_step_outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def test_epoch_end(self, test_step_outputs):
        self.epoch_evaluate(test_step_outputs, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.9
                ),
                "monitor": "metric_to_track",
            },
        }


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


# helper classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


# main class


class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        depth,
        input_channels=3,
        num_latents=512,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=1000,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        self_per_cross_attn=1,
        final_classifier_head=True,
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          depth: Depth of net.
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()

        input_dim = input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(
            latent_dim,
            Attention(
                latent_dim,
                input_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
                dropout=attn_dropout,
            ),
            context_dim=input_dim,
        )
        get_cross_ff = lambda: PreNorm(
            latent_dim, FeedForward(latent_dim, dropout=ff_dropout)
        )
        get_latent_attn = lambda: PreNorm(
            latent_dim,
            Attention(
                latent_dim,
                heads=latent_heads,
                dim_head=latent_dim_head,
                dropout=attn_dropout,
            ),
        )
        get_latent_ff = lambda: PreNorm(
            latent_dim, FeedForward(latent_dim, dropout=ff_dropout)
        )

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(
            cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff)
        )

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {"_cache": should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(self_per_cross_attn):
                self_attns.append(
                    nn.ModuleList(
                        [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                    )
                )

            if i == 0:
                self.layers.append(
                    nn.ModuleList(
                        [
                            Attention(
                                latent_dim,
                                input_dim,
                                heads=cross_heads,
                                dim_head=cross_dim_head,
                                dropout=attn_dropout,
                            ),
                            get_cross_ff(**cache_args),
                            self_attns,
                        ]
                    )
                )
            else:
                self.layers.append(
                    nn.ModuleList(
                        [
                            get_cross_attn(**cache_args),
                            get_cross_ff(**cache_args),
                            self_attns,
                        ]
                    )
                )

        self.to_logits = (
            nn.Sequential(
                Reduce("b n d -> b d", "sum"),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes),
            )
            if final_classifier_head
            else nn.Identity()
        )

        self.initial_emb = nn.Embedding(64, 1)

    def forward(self, data, mask=None, initial_emb=True, return_embeddings=False):

        b = data.shape[0]

        if initial_emb:
            data = self.initial_emb(rearrange(data, "b ... d -> b (...) d")).squeeze(-1)
        else:
            data = rearrange(data, "b ... d -> b (...) d")

        x = repeat(self.latents, "n d -> b n d", b=b)

        # layers

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # allow for fetching embeddings

        if return_embeddings:
            return x

        # to logits

        return self.to_logits(x)

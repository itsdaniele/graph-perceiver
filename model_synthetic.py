# Code mainly taken from https://github.com/lucidrains/perceiver-pytorch

from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch

import torch_optimizer as optim


class GCNZinc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Linear(10, 256)
        self.conv1 = GCNConv(256, 256)
        self.conv2 = GCNConv(256, 256)
        self.conv3 = GCNConv(256, 256)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.emb(x)
        x = F.relu(self.conv1(x, edge_index,)) + x
        x = F.relu(self.conv2(x, edge_index,)) + x
        emb = F.relu(self.conv3(x, edge_index,)) + x

        return emb


class GATZinc(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.emb = torch.nn.Embedding(28, 256)
        self.conv1 = GATConv(256, 256)
        self.conv2 = GATConv(256, 256)
        self.conv3 = GATConv(256, 256)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.emb(x.squeeze(-1))
        x = F.relu(self.conv1(x, edge_index, edge_weight.float())) + x
        x = F.relu(self.conv2(x, edge_index, edge_weight.float())) + x
        emb = F.relu(self.conv3(x, edge_index, edge_weight.float())) + x

        return emb


class PerceiverRegressor(pl.LightningModule):
    def __init__(
        self,
        input_channels=3,
        depth=6,
        num_latents=4,
        latent_dim=32,
        cross_heads=4,
        latent_heads=4,
        cross_dim_head=32,
        latent_dim_head=32,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        self_per_cross_attn=2,
        encodings_dim=8,
    ):
        super().__init__()

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
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            weight_tie_layers=weight_tie_layers,
            self_per_cross_attn=self_per_cross_attn,
            encodings_dim=encodings_dim,
        )

    def forward(self, x, **kwargs):
        return self.perceiver(x, **kwargs)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch).view(-1)
        loss = F.binary_cross_entropy_with_logits(y_hat, batch.y)
        self.log("train/loss", loss)

        return loss

    def evaluate(self, batch, stage=None):
        y_hat = self(batch).view(-1)
        loss = F.binary_cross_entropy_with_logits(y_hat, batch.y)
        self.log("val/loss", loss)

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        # torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        return [optimizer]


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
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        self_per_cross_attn=1,
        final_head=True,
        encodings_dim=8,
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          depth: Depth of net.
          input_channels: Number of channels for each token of the input.
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()

        input_dim = input_channels

        self.gnn = GCNZinc()
        self.embed_encodings = nn.Linear(encodings_dim, input_dim)
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

            self.layers.append(
                nn.ModuleList(
                    [
                        get_cross_attn(**cache_args),
                        get_cross_ff(**cache_args),
                        self_attns,
                    ]
                )
            )

        self.to_out = (
            nn.Sequential(
                Reduce("b n d -> b d", "mean"),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, 1),
            )
            if final_head
            else nn.Identity()
        )

    def forward(self, batch, return_embeddings=False):

        data = self.gnn(batch)

        data, mask = to_dense_batch(data, batch.batch, max_num_nodes=128)
        # lap, _ = to_dense_batch(batch.lap, batch.batch, max_num_nodes=128)

        # data += self.embed_encodings(lap)

        b = data.shape[0]
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
        return self.to_out(x)

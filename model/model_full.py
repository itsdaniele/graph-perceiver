# Code mainly taken from https://github.com/lucidrains/perceiver-pytorch

from functools import wraps
import math
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Reduce

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch, to_dense_adj

from torch.optim.lr_scheduler import _LRScheduler

from torch_geometric.nn import RGCNConv


class GCNZinc(torch.nn.Module):
    def __init__(self, input_dim=64, embedding_dim=64):
        super().__init__()
        self.emb = torch.nn.Embedding(input_dim, embedding_dim)
        self.conv1 = RGCNConv(embedding_dim, embedding_dim, num_relations=4)
        self.conv2 = RGCNConv(embedding_dim, embedding_dim, num_relations=4)
        self.conv3 = RGCNConv(embedding_dim, embedding_dim, num_relations=4)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.emb(x.squeeze(-1))
        x = F.relu(self.conv1(x, edge_index, edge_weight)) + x
        x = F.relu(self.conv2(x, edge_index, edge_weight)) + x
        x = F.relu(self.conv3(x, edge_index, edge_weight)) + x
        return x


class PolynomialDecayLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_updates,
        tot_updates,
        lr,
        end_lr,
        power,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False


class PerceiverRegressor(pl.LightningModule):
    def __init__(
        self,
        depth=8,
        latent_dim=80,
        latent_heads=8,
        latent_dim_head=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.perceiver = Perceiver(
            depth=depth,
            latent_dim=latent_dim,
            latent_heads=latent_heads,
            latent_dim_head=latent_dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            weight_tie_layers=weight_tie_layers,
        )

    def forward(self, x, **kwargs):
        return self.perceiver(x, **kwargs)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch).view(-1)
        loss = F.l1_loss(y_hat, batch.y)
        self.log("train/loss", loss, batch_size=batch.y.size(0))

        return loss

    def evaluate(self, batch, stage=None):
        y_hat = self(batch).view(-1)
        loss = F.l1_loss(y_hat, batch.y)
        self.log("val/loss", loss, batch_size=batch.y.size(0))

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4, weight_decay=0.01)

        scheduler = {
            "scheduler": PolynomialDecayLR(
                optimizer,
                warmup_updates=60000,
                tot_updates=1000000,
                lr=2e-4,
                end_lr=1e-9,
                power=1.0,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]


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
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
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

    def forward(self, x, context=None, mask=None, bias=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if bias is not None:
            bias = repeat(bias, "b n d -> (repeat b) n d", repeat=h)
            sim += bias

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


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        depth,
        latent_dim=80,
        latent_heads=8,
        latent_dim_head=64,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        final_head=True,
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

        # self.gnn = GCNZinc(64, 64)

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

        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {"_cache": should_cache}

            self.layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
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

        # self.embed_encodings = nn.Linear(8, 80)
        self.atom_encoder = nn.Embedding(64, latent_dim, padding_idx=0)
        self.edge_encoder = nn.Embedding(64, 1, padding_idx=0)
        self.embed_indegree = nn.Embedding(64, latent_dim, padding_idx=0)
        self.embed_outdegree = nn.Embedding(64, latent_dim, padding_idx=0)
        self.edge_dis_encoder = nn.Embedding(40, 1)
        self.spatial_pos_encoder = nn.Embedding(40, 1, padding_idx=0)

        self.multi_hop_max_dist = 20
        self.apply(lambda module: init_params(module, n_layers=depth))

    def forward(self, batch):

        spatial_pos_bias = self.spatial_pos_encoder(batch.spatial_pos)

        spatial_pos_ = batch.spatial_pos.clone()
        spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
        # set 1 to 1, x > 1 to x - 1
        spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
        if self.multi_hop_max_dist > 0:
            spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
            edge_input = batch.edge_input[:, :, :, : self.multi_hop_max_dist, :]
        # [n_graph, n_node, n_node, max_dist, n_head]
        edge_input = self.edge_encoder(edge_input).mean(-2)
        max_dist = edge_input.size(-2)
        edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, 1)
        edge_input_flat = torch.bmm(
            edge_input_flat,
            self.edge_dis_encoder.weight.reshape(-1, 1, 1)[:max_dist, :, :],
        )
        edge_input = edge_input_flat.reshape(max_dist, 64, 128, 128, 1).permute(
            1, 2, 3, 0, 4
        )
        edge_input = (
            edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
        ).permute(0, 3, 1, 2)

        bias = edge_input.squeeze(1) + spatial_pos_bias.squeeze(-1)
        data, mask = to_dense_batch(batch.x, batch.batch, max_num_nodes=128)
        # lap, _ = to_dense_batch(batch.lap, batch.batch, max_num_nodes=64)
        in_deg, _ = to_dense_batch(batch.indeg, batch.batch, max_num_nodes=128)
        out_deg, _ = to_dense_batch(batch.outdeg, batch.batch, max_num_nodes=128)

        # edge_attr = self.edge_encoder(
        #     to_dense_adj(
        #         batch.edge_index, batch.batch, batch.edge_attr, max_num_nodes=128
        #     )
        # ).squeeze(-1)
        # data = self.atom_encoder(data.squeeze(-1) + 1)
        # x = data + self.embed_encodings(lap).squeeze(-1)

        x = (
            self.atom_encoder(data.squeeze(-1))
            + self.embed_indegree(in_deg)
            + self.embed_outdegree(out_deg)
        )

        # layers

        for self_attns in self.layers:

            x = self_attns[0](x, mask=None, bias=bias) + x
            x = self_attns[1](x) + x

        # to logits
        return self.to_out(x)

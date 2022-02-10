# Code mainly taken from https://github.com/lucidrains/perceiver-pytorch

from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from .gnn import GNN_node_Virtualnode_no_batchnorm


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

    def forward(self, x, context=None, mask=None, is_query_mask=False):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max

            if not is_query_mask:
                mask = repeat(mask, "b j -> (b h) () j", h=h)
            else:
                mask = repeat(mask, "b j -> (b h) j ()", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


# main class


class PerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        depth,
        queries_dim,
        gnn_embed_dim,
        logits_dim,
        num_latents,
        latent_dim,
        cross_heads,
        latent_heads,
        cross_dim_head,
        latent_dim_head,
        weight_tie_layers,
        decoder_ff,
        attn_dropout,
        ff_dropout,
    ):
        super().__init__()

        self.lin1 = nn.Linear(1433, gnn_embed_dim)
        self.lin2 = nn.Linear(1433, gnn_embed_dim)
        lin_queries = nn.Linear(1433, gnn_embed_dim)
        self.gnn1 = GNN_node_Virtualnode_no_batchnorm(2, gnn_embed_dim, self.lin1, None)

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attend_block = nn.ModuleList(
            [
                PreNorm(
                    latent_dim,
                    Attention(
                        latent_dim,
                        queries_dim,
                        heads=cross_heads,
                        dim_head=cross_dim_head,
                        dropout=attn_dropout,
                    ),
                    # context_dim=queries_dim,
                ),
                PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout)),
            ]
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
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(
                queries_dim,
                latent_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
                dropout=attn_dropout,
            ),
            context_dim=latent_dim,
        )
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim, dropout=ff_dropout))
            if decoder_ff
            else None
        )

        self.to_logits = (
            nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()
        )

        self.pos_emb = nn.Embedding(2708, gnn_embed_dim)
        self.pos_emb_2 = nn.Embedding(2708, gnn_embed_dim)

    def forward(self, batch, mask=None, queries=None):

        emb = self.pos_emb(torch.arange(2708, device=batch.x.device))
        emb2 = self.pos_emb_2(torch.arange(2708, device=batch.x.device))
        # emb_latents = self.pos_emb_latents(torch.arange(8, device=batch.x.device))

        data = (self.gnn1(batch, training=self.training) + emb2).unsqueeze(0)

        if queries is None:
            queries = (self.lin2(batch.x) + emb).unsqueeze(0)

        b = data.shape[0]
        x = repeat(self.latents, "n d -> b n d", b=b)

        cross_attn, cross_ff = self.cross_attend_block

        # cross attention only happens once for Perceiver IO
        x = cross_attn(x, context=data) + x
        x = cross_ff(x) + x

        # layers

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # make sure queries contains batch dimension

        if queries.ndim == 2:
            queries = repeat(queries, "n d -> b n d", b=b)

        # cross attend from decoder queries to latents
        latents = self.decoder_cross_attn(queries, context=x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # final linear out
        logits = self.to_logits(latents)
        return logits


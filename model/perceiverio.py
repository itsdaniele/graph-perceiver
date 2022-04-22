# Code mainly taken from https://github.com/lucidrains/perceiver-pytorch

from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

import torch
import torch.nn.functional as F


import math

from .gnn import (
    GCNCORA,
    GCNPROTEINS,
    SAGEPROTEINS,
    GATPROTEINS,
    NNCONVPROTEINS,
    SAGEPROTEINSEMBED,
    GCNARXIV,
    GCNCIRCLES,
)


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
    def __init__(self, dim, fn, context_dim=None, do_norm_context=True):
        super().__init__()
        self.fn = fn
        self.do_norm_context = do_norm_context
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context) and self.do_norm_context is True:
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

        # return F.gelu(x)


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
        return self.to_out(out), rearrange(attn, "(b h) n d -> b h n d", h=h)


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


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
        gnn_encoder,
    ):
        super().__init__()

        if gnn_encoder is None:

            self.gnn = GCNPROTEINS(gnn_embed_dim)
            self.gnn = GCNCIRCLES(gnn_embed_dim)
            # self.gnn = SAGEPROTEINS(gnn_embed_dim)

        else:
            self.gnn1 = gnn_encoder
            self.out_gnn = nn.Linear(gnn_embed_dim, logits_dim)
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
                    context_dim=queries_dim,
                    do_norm_context=False,
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
            nn.Linear(queries_dim * 1, logits_dim)
            if exists(logits_dim)
            else nn.Identity()
        )

        self.apply(lambda module: init_params(module, n_layers=depth))
        # self.scale_out = nn.Parameter(torch.scalar_tensor(0.1))

        # self.input_linear = nn.Linear(8, gnn_embed_dim)

    def forward(self, batch, mask=None, queries=None):

        # this is done just for pretraining the gnn
        # data = self.out_gnn(F.relu(self.gnn1(batch, training=self.training))).unsqueeze(
        #     0
        # )
        # return data.squeeze(0)

        attns = []

        data = self.gnn(batch).unsqueeze(0)
        return data.squeeze(0), None

        if queries is None:
            queries = data

        b = data.shape[0]
        x = repeat(self.latents, "n d -> b n d", b=b)

        cross_attn, cross_ff = self.cross_attend_block

        # cross attention only happens once for Perceiver IO
        out, attn = cross_attn(x, context=data)
        attns.append(attn)
        x = out + x
        x = cross_ff(x) + x

        # layers

        for self_attn, self_ff in self.layers:
            x = self_attn(x)[0] + x
            x = self_ff(x) + x

        # make sure queries contains batch dimension

        if queries.ndim == 2:
            queries = repeat(queries, "n d -> b n d", b=b)

        # cross attend from decoder queries to latents
        latents, attn = self.decoder_cross_attn(queries, context=x)
        attns.append(attn)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # batch.x = latents.squeeze(0)
        # latents = self.gnn_final(batch).unsqueeze(0)
        # out = torch.cat([latents, queries], dim=-1)

        out = latents

        logits = self.to_logits(out)
        return logits.squeeze(0), attns


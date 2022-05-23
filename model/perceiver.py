# Code mainly taken from https://github.com/lucidrains/perceiver-pytorch

from functools import wraps
import math
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Reduce

import torch
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from .gnn import GNN_node_Virtualnode


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
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
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

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

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
        attn = self.dropout(attn)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


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
        num_classes,
        gnn,
        gnn_embed_dim=256,
        num_latents=512,
        latent_dim=512,
        queries_dim=256,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        attn_dropout=0.0,
        ff_dropout=0.0,
        weight_tie_layers=False,
        self_per_cross_attn=1,
        max_seq_len=None
    ):

        super().__init__()

        self.max_seq_len = max_seq_len
        self.gnn = gnn

        # for code
        # self.gnn = GNN_node_Virtualnode(
        #     4,
        #     gnn_embed_dim,
        #     node_encoder_cls(),
        #     edge_encoder_cls,
        #     gnn_type="gcn",
        #     drop_ratio=0.0,
        # )

        # if dataset_name == "nci":

        #     def edge_encoder_cls(_):
        #         def zero(_):
        #             return 0

        #         return zero

        #     self.gnn = GNN_node_Virtualnode(
        #         3,
        #         gnn_embed_dim,
        #         torch.nn.Linear(input_channels, gnn_embed_dim),
        #         edge_encoder_cls,
        #         gnn_type="gin",
        #         drop_ratio=0.5,
        #     )
        # else:
        #     self.gnn = GNN_node_Virtualnode(
        #         3, gnn_embed_dim, AtomEncoder(gnn_embed_dim), BondEncoder
        #     )

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.to_dim = nn.Linear(gnn_embed_dim, queries_dim)

        get_cross_attn = lambda: PreNorm(
            latent_dim,
            Attention(
                latent_dim,
                queries_dim,
                heads=cross_heads,
                dim_head=cross_dim_head,
                dropout=attn_dropout,
            ),
            context_dim=queries_dim,
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

        # for ogbg-code
        if max_seq_len is not None:
            self.graph_pred_linear_list = torch.nn.ModuleList()
            for i in range(max_seq_len):
                self.graph_pred_linear_list.append(
                    torch.nn.Linear(latent_dim, num_classes)
                )
        else:

            self.to_out = nn.Sequential(
                nn.LayerNorm(latent_dim), nn.Linear(latent_dim, num_classes),
            )

        self.apply(lambda module: init_params(module, n_layers=depth))

    def forward(self, batch):

        data = self.to_dim(self.gnn(batch))

        num_nodes = []
        for i in range(len(batch.batch)):
            mask = batch.batch.eq(i)
            num_node = mask.sum()
            num_nodes.append(num_node)

        ###>3000 for molecukes
        data, mask = to_dense_batch(data, batch.batch, max_num_nodes=max(num_nodes))

        b = data.shape[0]
        x = repeat(self.latents, "n d -> b n d", b=b)

        # layers

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        if self.max_seq_len is None:

            out = self.to_out(x).mean(-2)
            return out
        else:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.graph_pred_linear_list[i](x)[:, 0, :])

            return pred_list

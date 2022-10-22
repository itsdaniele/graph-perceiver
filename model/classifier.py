import torch
import torch.nn as nn
from .glab import GLAB
from .gnn import GNN_node_Virtualnode
from .modules import GCNConv

from functools import partial

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from .lightning import BaseLightning
from .glab import GLAB
from .perceiver import GLABLayer

from torch_geometric.utils import to_dense_batch
from functools import partial


class GLABClassifier(BaseLightning):
    def __init__(
        self,
        evaluator,
        num_classes,
        loss_fn,
        dataset,
        gnn_embed_dim,
        depth,
        num_latents,
        latent_dim,
        queries_dim,
        cross_heads,
        latent_heads,
        cross_dim_head,
        latent_dim_head,
        attn_dropout,
        ff_dropout,
        gnn_dropout,
        weight_tie_layers,
        self_per_cross_attn,
        gnn_type,
        num_gnn_layers,
        lr,
        weight_decay,
        scheduler,
        scheduler_params,
        arr_to_seq=None,
        max_seq_len=None,
    ):
        super().__init__(
            evaluator=evaluator,
            dataset=dataset,
            loss_fn=loss_fn,
            lr=lr,
            weight_decay=weight_decay,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            arr_to_seq=arr_to_seq,
        )

        self.save_hyperparameters()

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

        elif dataset == "zinc":
            if gnn_type == "gin-virtual":
                gnn = GNN_node_Virtualnode(
                    num_gnn_layers,
                    gnn_embed_dim,
                    AtomEncoder(gnn_embed_dim),
                    partial(torch.nn.Embedding, 4),
                    gnn_type="gin",
                    drop_ratio=gnn_dropout,
                )
            elif gnn_type == "gcn-virtual":
                gnn = GNN_node_Virtualnode(
                    num_gnn_layers,
                    gnn_embed_dim,
                    AtomEncoder(gnn_embed_dim),
                    partial(torch.nn.Embedding, 4),
                    gnn_type="gcn",
                    drop_ratio=gnn_dropout,
                )
        else:
            raise NotImplementedError()

        self.glab = GLAB(
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
        return self.glab(x, **kwargs)


class GLABClassifierV2(BaseLightning):
    def __init__(
        self,
        evaluator,
        num_classes,
        loss_fn,
        dataset,
        gnn_embed_dim,
        depth,
        num_latents,
        latent_dim,
        cross_heads,
        latent_heads,
        cross_dim_head,
        latent_dim_head,
        attn_dropout,
        ff_dropout,
        gnn_dropout,
        gnn_type,
        lr,
        weight_decay,
        scheduler,
        scheduler_params,
        arr_to_seq=None,
        **kwargs
    ):
        super().__init__(
            evaluator=evaluator,
            dataset=dataset,
            loss_fn=loss_fn,
            lr=lr,
            weight_decay=weight_decay,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            arr_to_seq=arr_to_seq,
        )

        if dataset in ["ogbg-molhiv", "ogbg-molpcba"]:
            if gnn_type == "gin-virtual":
                self.gnn = GNN_node_Virtualnode(
                    depth,
                    gnn_embed_dim,
                    AtomEncoder(gnn_embed_dim),
                    BondEncoder,
                    gnn_type="gin",
                    drop_ratio=gnn_dropout,
                )

        elif dataset == "zinc":
            if gnn_type == "gin-virtual":
                self.gnn = GNN_node_Virtualnode(
                    depth,
                    gnn_embed_dim,
                    AtomEncoder(gnn_embed_dim),
                    partial(torch.nn.Embedding, 4),
                    gnn_type="gin",
                    drop_ratio=gnn_dropout,
                )
            elif gnn_type == "gcn-virtual":
                self.gnn = GNN_node_Virtualnode(
                    depth,
                    gnn_embed_dim,
                    AtomEncoder(gnn_embed_dim),
                    partial(torch.nn.Embedding, 4),
                    gnn_type="gcn",
                    drop_ratio=gnn_dropout,
                )
        else:
            raise NotImplementedError()

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.layers = nn.ModuleList([])

        self.to_logits = nn.Linear(gnn_embed_dim, num_classes)
        self.to_gnn_dim = AtomEncoder(gnn_embed_dim)

        if dataset == "zinc":
            edge_encoder_cls = partial(nn.Embedding, 4)
        elif dataset == "ogbg-molpcba":
            edge_encoder_cls = BondEncoder
        else:
            raise NotImplementedError()
            
        for _ in range(depth):

            gnn_layer = GCNConv(gnn_embed_dim, edge_encoder_cls)
            glab_layer = GLABLayer(
                gnn_layer,
                gnn_embed_dim,
                latent_dim=latent_dim,
                cross_heads=cross_heads,
                latent_heads=latent_heads,
                cross_dim_head=cross_dim_head,
                latent_dim_head=latent_dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )

            self.layers.append(nn.ModuleList([gnn_layer, glab_layer]))

        self.save_hyperparameters()

    def forward(
        self, batch,
    ):

        local_out = self.gnn(batch)
        x = self.to_gnn_dim(batch.x)

        num_nodes = []
        for i in range(len(batch.batch)):
            mask = batch.batch.eq(i)
            num_node = mask.sum()
            num_nodes.append(num_node)
        max_num_nodes = max(num_nodes)

        local_out, _ = to_dense_batch(x, batch.batch, max_num_nodes=max_num_nodes)

        for i, (gnn_layer, glab_layer) in enumerate(self.layers):
            x = gnn_layer(x, batch.edge_index, batch.edge_attr)

            x_, mask = to_dense_batch(x, batch.batch, max_num_nodes=max_num_nodes)
            latents = glab_layer(self.latents if i == 0 else latents, x_)

        return self.to_logits(latents.mean(-2) + local_out.mean(-2))


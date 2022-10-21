import torch
from .perceiver import Perceiver
from .gnn import GNN_node_Virtualnode

from functools import partial

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from .lightning import BaseLightning


class PerceiverClassifier(BaseLightning):
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

        self.perceiver = Perceiver(
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
        return self.perceiver(x, **kwargs)


# Code mainly taken from https://github.com/lucidrains/perceiver-pytorch

from torch import nn

from einops import rearrange, repeat

import torch

from torch_geometric.utils import to_dense_batch

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from .perceiver import Perceiver


class GLAB(nn.Module):
    def __init__(
        self,
        gnn_embed_dim,
        depth,
        gnn,
        num_latents,
        latent_dim,
        queries_dim,
        cross_heads,
        latent_heads,
        cross_dim_head,
        latent_dim_head,
        attn_dropout,
        ff_dropout,
        weight_tie_layers,
        self_per_cross_attn,
        num_classes,
        max_seq_len=None,
    ):

        super().__init__()

        self.max_seq_len = max_seq_len
        self.gnn = gnn
        self.node_encoder = AtomEncoder(queries_dim)

        self.perceiver = Perceiver(
            num_freq_bands=2 * gnn_embed_dim + 1,
            max_freq=2 * gnn_embed_dim + 1 + 5,
            input_axis=1,
            depth=depth,
            input_channels=gnn_embed_dim,
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
            final_classifier_head=False,
        )

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

        self.to_dim = nn.Sequential(
            nn.LayerNorm(gnn_embed_dim,), nn.Linear(gnn_embed_dim, queries_dim)
        )

    def forward(self, batch):

        gnn_output = self.gnn(batch)  # [n_nodes, gnn_embed_dim]
        # transformer_input = self.node_encoder(
        #     batch.x
        # )  # [n_nodes, transformer_embed_dim]

        num_nodes = []
        for i in range(len(batch.batch)):
            mask = batch.batch.eq(i)
            num_node = mask.sum()
            num_nodes.append(num_node)
        max_num_nodes = max(num_nodes)

        ###>3000 for molecukes
        data, mask = to_dense_batch(
            gnn_output, batch.batch, max_num_nodes=max_num_nodes
        )
        data = self.to_dim(data)

        out = self.perceiver(data, mask)

        if self.max_seq_len is None:  # not code2

            out = out.mean(-2)
            out = self.to_out(out)
            return out
        else:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.graph_pred_linear_list[i](x)[:, 0, :])

            return pred_list


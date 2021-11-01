import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_emb=128):
        super(GCN, self).__init__()

        self.emb = torch.nn.Embedding(num_emb, 16)
        self.edge_emb = torch.nn.Embedding(16, 1)
        self.conv1 = GCNConv(16, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data, get_emb=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.emb(x.squeeze(-1))
        edge_weight = self.edge_emb(edge_weight).squeeze(-1)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x if get_emb is True else x.sum(dim=0)


import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_scatter import scatter_sum


class GCNZinc(torch.nn.Module):
    def __init__(self):
        super(GCNZinc, self).__init__()

        self.emb = torch.nn.Embedding(28, 64)
        self.edge_emb = torch.nn.Embedding(4, 16)
        self.conv1 = GCNConv(64, 64)
        self.conv2 = GCNConv(64, 64)
        self.lin = torch.nn.Linear(64, 1)

    def forward(self, data, get_emb=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.emb(x.squeeze(-1))
        # edge_weight = self.edge_emb(edge_weight)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        emb = self.conv2(x, edge_index)
        x = self.lin(emb)

        return emb if get_emb is True else scatter_sum(x, data.batch, dim=0).squeeze(-1)


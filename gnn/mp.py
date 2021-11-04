import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean


class GCNZinc(torch.nn.Module):
    def __init__(self):
        super(GCNZinc, self).__init__()

        self.emb = torch.nn.Embedding(28, 256)
        self.edge_emb = torch.nn.Embedding(32, 1)
        self.conv1 = GCNConv(256, 256)
        self.conv2 = GCNConv(256, 256)
        self.lin1 = torch.nn.Linear(256, 64)
        self.lin2 = torch.nn.Linear(64, 1)

    def forward(self, data, get_emb=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.emb(x.squeeze(-1))
        # edge_weight = self.edge_emb(edge_weight).squeeze(-1)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        emb = F.relu(self.conv2(x, edge_index))

        if hasattr(data, "batch"):
            x = F.relu(scatter_mean(emb, data.batch, dim=0))
        else:
            x = F.relu(torch.mean(emb, dim=0)).unsqueeze(0)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return emb if get_emb is True else x.squeeze(-1)


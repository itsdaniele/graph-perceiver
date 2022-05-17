import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv as GCNConvGeometric

### Virtual GNN to generate node embedding
class GCNPROTEINS(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.conv1 = GCNConvGeometric(8, emb_dim)
        self.conv2 = GCNConvGeometric(emb_dim, emb_dim)
        self.conv3 = GCNConvGeometric(emb_dim, emb_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.0)
        x = self.conv2(x, edge_index) + x
        x = F.relu(x)
        x = F.dropout(x, p=0.0)
        x = self.conv3(x, edge_index) + x
        return x

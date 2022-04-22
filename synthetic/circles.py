import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx


def save_graph(G, name="graph.png"):
    nx.draw(G)
    plt.show()
    plt.savefig(name, format="PNG")


def get_circles_dataset(
    n_classes=4, N=5, num_graphs=200, train_ratio=0.8, feats_dim=32
):

    Gs = []
    train_idx = []
    valid_idx = []

    for jj in range(num_graphs):

        i = np.random.choice(list(range(n_classes))) + 1
        n_nodes = N * i

        feats = np.random.randn(feats_dim)

        if jj <= int(train_ratio * num_graphs):

            if len(train_idx) == 0:
                train_idx += list(range(0, n_nodes))
            else:
                last = train_idx[-1] + 1
                train_idx += list(range(last, last + n_nodes))
        else:
            if len(valid_idx) == 0:
                last = train_idx[-1] + 1
                valid_idx += list(range(last, last + n_nodes))
            else:
                last = valid_idx[-1] + 1
                valid_idx += list(range(last, last + n_nodes))

        G = nx.cycle_graph(n_nodes)

        for j in range(n_nodes):
            G.nodes[j]["x"] = feats + np.random.normal(scale=0.01)
            G.nodes[j]["y"] = i - 1
        Gs.append(G)

    G = Gs[0]
    for i in range(1, len(Gs)):
        G = nx.disjoint_union(G, Gs[i])
    G = from_networkx(G)
    G.y = G.y.unsqueeze(-1)
    G.x = G.x.float()
    dataset = [G]

    return (
        dataset,
        torch.tensor(train_idx),
        torch.tensor(valid_idx),
        torch.tensor(valid_idx),
    )

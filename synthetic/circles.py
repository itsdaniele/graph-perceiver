import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.utils import from_networkx


# def save_graph(G, name="graph.png"):
#     nx.draw(G)
#     plt.show()
#     plt.savefig(name, format="PNG")


def get_circles_dataset(n_classes=4, N=5):

    Gs = []
    train_idx = []
    valid_idx = []

    for i in range(n_classes):
        n_nodes = N * (i + 1)

        feats = np.random.randn(32)
        train_idx += list(range(0, N * (i + 1)))
        G = nx.cycle_graph(n_nodes)

        for j in range(n_nodes):
            G.nodes[j]["x"] = feats + np.random.normal(scale=0.01)
            G.nodes[j]["y"] = i
        Gs.append(G)

    G = Gs[0]
    for i in range(1, len(Gs)):
        G = nx.disjoint_union(G, Gs[i])
    G = from_networkx(G)
    G.x = G.x.float()
    dataset = [G]

    num_valid_nodes = (N * (i + 1)) + (N * i)
    valid_idx = train_idx[-num_valid_nodes:]
    train_idx = train_idx[:-num_valid_nodes]
    return dataset, train_idx, valid_idx, valid_idx


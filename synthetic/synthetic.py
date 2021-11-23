import networkx as nx
import numpy as np

import random

from torch_geometric.utils.convert import from_networkx

n_graphs = 1000
n_nodes = 10
seed = 42


graphs = []
graphs_geometric = []
labels = []

val = [1.0, -1.0]


for _ in range(n_graphs):
    G1 = nx.erdos_renyi_graph(n_nodes, 0.5, seed=seed, directed=False)
    G2 = nx.erdos_renyi_graph(n_nodes, 0.5, seed=seed, directed=False)

    A = random.choice([0, 1])
    B = random.choice([0, 1])

    features_1 = np.random.normal(val[A], size=(G1.number_of_nodes(),))
    features_2 = np.random.normal(val[B], size=(G2.number_of_nodes(),))
    nx.set_node_attributes(G1, features_1, "x")
    nx.set_node_attributes(G2, features_2, "x")

    G = nx.disjoint_union(G1, G2)
    graphs.append(G)

    labels.append(int(bool(A) ^ bool(B)))


for graph in graphs:
    graphs_geometric.append(from_networkx(graph))


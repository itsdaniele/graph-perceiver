import pytorch_lightning as pl
from model_synthetic import PerceiverRegressor

from torch_geometric.data import DataLoader

from pytorch_lightning.loggers import WandbLogger

import networkx as nx
import numpy as np

import random

from torch_geometric.utils.convert import from_networkx

wandb_logger = WandbLogger(project="graph-perceiver")


def main():

    n_graphs = 15000
    n_nodes = 50

    graphs = []
    graphs_geometric = []
    labels = []

    val = [1.0, -1.0]

    for _ in range(n_graphs):
        G1 = nx.erdos_renyi_graph(n_nodes, 0.5, directed=False)
        G2 = nx.erdos_renyi_graph(n_nodes, 0.5, directed=False)

        A = random.choice([0, 1])
        B = random.choice([0, 1])

        attrs_1 = {}
        attrs_2 = {}

        for i in range(G1.number_of_nodes()):
            attrs_1[i] = np.random.normal(val[A], size=(10))
        for i in range(G2.number_of_nodes()):
            attrs_2[i] = np.random.normal(val[B], size=(10))
        nx.set_node_attributes(G1, attrs_1, "x")
        nx.set_node_attributes(G2, attrs_2, "x")

        G = nx.disjoint_union(G1, G2)
        graphs.append(G)

        labels.append(float(bool(A) ^ bool(B)))

    for i, graph in enumerate(graphs):
        graphs_geometric.append(from_networkx(graph))
        graphs_geometric[-1].x = graphs_geometric[-1].x.float()
        graphs_geometric[-1].y = labels[i]

    train_loader = DataLoader(
        graphs_geometric[:10000], batch_size=32, shuffle=True, num_workers=32
    )

    val_loader = DataLoader(
        graphs_geometric[10000:], batch_size=32, shuffle=False, num_workers=32
    )

    model = PerceiverRegressor(
        input_channels=256,
        depth=2,
        attn_dropout=0.0,
        ff_dropout=0.1,
        weight_tie_layers=True,
        encodings_dim=8,
    )
    trainer = pl.Trainer(
        gpus=0,
        log_gpu_memory=True,
        max_epochs=10000,
        progress_bar_refresh_rate=5,
        # logger=wandb_logger,
    )
    trainer.fit(model, train_loader, val_loader)
    # trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    main()

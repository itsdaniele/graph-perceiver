from torch_geometric.datasets import ZINC, GNNBenchmarkDataset, Planetoid
import torch

from torch_geometric.utils import to_dense_adj, get_laplacian, degree
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
import algos


def preprocess_item_lap(item, pos_enc_dim=8):

    item.lap = laplacian_positional_encoding(item, pos_enc_dim=pos_enc_dim)
    return item


def preprocess_item_deg(item, max_node=128, max_dist_=32):

    item.indeg = degree(item.edge_index[1], dtype=int) + 1  # +1 so 0 is for padding
    item.outdeg = degree(item.edge_index[0], dtype=int) + 1

    # N = item.x.size(0)

    # # node adj matrix [N, N] bool
    # adj = torch.zeros([N, N], dtype=torch.bool)
    # attn_edge_type = torch.zeros([N, N, 1], dtype=torch.long)
    # adj[item.edge_index[0, :], item.edge_index[1, :]] = True

    # shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    # max_dist = np.amax(shortest_path_result)
    # edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    # spatial_pos = pad_spatial_pos(
    #     torch.from_numpy((shortest_path_result)).long(), max_node
    # )
    # item.edge_input = pad_3d(torch.tensor(edge_input), max_node, max_node, max_dist_)
    # item.spatial_pos = spatial_pos
    return item


class MyGNNBenchmarkDataset(GNNBenchmarkDataset):
    def __init__(self, root, name="CLUSTER", split="train"):
        super().__init__(root, name=name, split=split)

    def download(self):
        super(GNNBenchmarkDataset, self).download()

    def process(self):
        super(GNNBenchmarkDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx

            return preprocess_item_deg(item, pos_enc_dim=32)
        else:
            return self.index_select(idx)


class MyZINCDataset(ZINC):
    def __init__(self, root, subset=False, split="train", encoding_type="lap"):
        super().__init__(root, subset=subset, split=split)
        self.encoding_type = encoding_type

    def download(self):
        super(MyZINCDataset, self).download()

    def process(self):
        super(MyZINCDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx

            item = (
                preprocess_item_lap(item)
                if self.encoding_type != "deg"
                else preprocess_item_deg(item)
            )
            return item
        else:
            return self.index_select(idx)


class MyCoraDataset(Planetoid):
    def __init__(self, root, name="Cora", encoding="deg"):
        super().__init__(root, name=name)
        self.encoding = encoding
        # self.data.lap = laplacian_positional_encoding(self.data, pos_enc_dim=32)

        self.data.indeg = degree(
            self.data.edge_index[1], dtype=int
        )  # add 1 so 0 is for padding
        self.data.outdeg = degree(self.data.edge_index[0], dtype=int)

    def download(self):
        super(MyCoraDataset, self).download()

    def process(self):
        super(MyCoraDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx

            return item
        else:
            return self.index_select(idx)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def laplacian_positional_encoding(data, pos_enc_dim=8):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    L = to_dense_adj(
        get_laplacian(data.edge_index, normalization="sym", dtype=torch.float)[0]
    ).squeeze(0)
    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.numpy())
    idx = EigVal.argsort()  # increasing order
    EigVec = np.real(EigVec[:, idx])
    return torch.from_numpy(EigVec[:, 1 : pos_enc_dim + 1]).float()


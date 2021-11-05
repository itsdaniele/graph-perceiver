from torch_geometric.datasets import ZINC, GNNBenchmarkDataset
import torch
import torch.nn.functional as F
import pickle
import scipy as sp

from scipy import sparse as sp


from torch_geometric.utils import to_dense_adj, get_laplacian
import numpy as np


def preprocess_item(item, pos_enc_dim=8):
    item.lap = laplacian_positional_encoding(item, pos_enc_dim=pos_enc_dim)
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

            return preprocess_item(item, pos_enc_dim=32)
        else:
            return self.index_select(idx)


class MyZINCDataset(ZINC):
    def __init__(self, root, subset=False, split="train", overwrite_x=None):
        super().__init__(root, subset=subset, split=split)

        self.overwrite_x = overwrite_x
        if overwrite_x is not None:
            with open(overwrite_x, "rb") as handle:
                self.new_x = pickle.load(handle)

    def download(self):
        super(MyZINCDataset, self).download()

    def process(self):
        super(MyZINCDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx

            if self.overwrite_x is not None:
                item.x = self.new_x[idx]
            return preprocess_item(item)
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


def laplacian_positional_encoding(data, pos_enc_dim=8):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    L = to_dense_adj(
        get_laplacian(data.edge_index, normalization="sym", dtype=torch.float)[0]
    ).squeeze(0)
    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L)
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    return torch.from_numpy(EigVec[:, 1 : pos_enc_dim + 1]).float()


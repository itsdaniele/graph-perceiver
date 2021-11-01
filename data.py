from torch_geometric.datasets import ZINC
import torch
import pickle


def preprocess_item(item):
    x, y = item.x, item.y
    return x, y


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


def collator(items, max_node_num=128):

    items = [item for item in items if item is not None]
    items = [(item[0], item[1]) for item in items]
    (xs, ys) = zip(*items)

    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])

    mask = torch.zeros_like(x).squeeze(-1).bool()

    for i in range(x.shape[0]):
        mask[i, : xs[i].shape[0]] = True

    return x, y, mask

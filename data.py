from torch_geometric.datasets import ZINC, GNNBenchmarkDataset, Planetoid


class MyZINCDataset(ZINC):
    def __init__(self, root, subset=False, split="train"):
        super().__init__(root, subset=subset, split=split)

    def download(self):
        super(MyZINCDataset, self).download()

    def process(self):
        super(MyZINCDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return item
        else:
            return self.index_select(idx)

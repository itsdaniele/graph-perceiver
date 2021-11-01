from mp import GCN
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import ZINC

from tqdm import trange

import pickle

from tqdm import tqdm

data_train = ZINC(root="./data", subset=True, split="train")
data_val = ZINC(root="./data", subset=True, split="val")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, data_train, dat_val = (
    GCN().to(device),
    data_train,
    data_val,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001,)


def train():

    model.train()
    for g in tqdm(data_train):

        optimizer.zero_grad()
        l = F.l1_loss(model(g), g.y)
        l.backward()
        optimizer.step()


@torch.no_grad()
def test():
    losses = []
    model.eval()
    for g in tqdm(data_val):
        l = F.l1_loss(model(g), g.y)
        losses.append(l.item())

    print(f"Eval loss: {np.mean(losses)}")


save_after = 0
for epoch in range(200):

    train()
    test()
    if epoch == save_after:
        print("Saving...")
        all_embs = []
        with torch.no_grad():
            for g in tqdm(data_train):
                embs = model(g, get_emb=True)
                all_embs.append(embs)
            with open("embs.pt", "wb") as handle:
                pickle.dump(all_embs, handle)
            break


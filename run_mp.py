from mp import GCNZinc
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
    GCNZinc().to(device),
    data_train,
    data_val,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)


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


save_after = 3
for epoch in range(200):

    train()
    test()
    if epoch == save_after:
        print("Saving...")
        all_embs_train = []
        all_embs_val = []
        with torch.no_grad():
            for g in tqdm(data_train):
                embs = model(g, get_emb=True)
                all_embs_train.append(embs)
            with open("embs_train.pt", "wb") as handle:
                pickle.dump(all_embs_train, handle)

            for g in tqdm(data_val):
                embs = model(g, get_emb=True)
                all_embs_val.append(embs)
            with open("embs_val.pt", "wb") as handle:
                pickle.dump(all_embs_val, handle)
            break


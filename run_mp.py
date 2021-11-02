from mp import GCNZinc
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader

from tqdm import trange

import pickle

from tqdm import tqdm

data_train = ZINC(root="./data", subset=True, split="train")
train_loader = DataLoader(data_train, batch_size=32, shuffle=True)


data_val = ZINC(root="./data", subset=True, split="val")
val_loader = DataLoader(data_val, batch_size=32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, data_train, dat_val = (
    GCNZinc().to(device),
    data_train,
    data_val,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)


def train():

    model.train()
    for batch in tqdm(train_loader):

        batch = batch.to(device)
        optimizer.zero_grad()
        l = F.l1_loss(model(batch), batch.y)
        l.backward()
        optimizer.step()


@torch.no_grad()
def test():
    losses = []
    model.eval()
    for batch in tqdm(val_loader):
        batch = batch.to(device)
        l = F.l1_loss(model(batch), batch.y)
        losses.append(l.item())

    print(f"Eval loss: {np.mean(losses)}")


save_after = 190
for epoch in range(200):

    train()
    test()
    if epoch == save_after:
        print("Saving...")
        all_embs_train = []
        all_embs_val = []
        with torch.no_grad():
            for g in tqdm(data_train):
                g = g.to(device)
                embs = model(g, get_emb=True)
                all_embs_train.append(embs.cpu())
            with open("embs_train_4.pt", "wb") as handle:
                pickle.dump(all_embs_train, handle)

            for g in tqdm(data_val):
                g = g.to(device)
                embs = model(g, get_emb=True)
                all_embs_val.append(embs.cpu())
            with open("embs_val_4.pt", "wb") as handle:
                pickle.dump(all_embs_val, handle)
            break


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd




class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_emb.weight.data.uniform_(-0.05, 0.05)
        self.item_emb.weight.data.uniform_(-0.05, 0.05)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.item_emb(v)
        b_u = self.user_bias(u).squeeze()
        b_v = self.item_bias(v).squeeze()

        return (U * V).sum(1) + b_u + b_v

    def get_recommendation(self, user, uninteracted_items, removed_movie):
        u = torch.LongTensor([user])
        b_u = self.user_bias(u).squeeze()
        U = self.user_emb(u)

        v = torch.LongTensor(uninteracted_items)
        b_v = self.item_bias(v).squeeze()
        V = self.item_emb(v)
        t = (U * V).sum(1) + b_u + b_v

        return t.detach().numpy()


def train_MF_model(model, df, epochs=10, lr=0.01, lmbd=0.001, wd=0.0, unsqueeze=False):
    Uoptimizer = torch.optim.Adam([model.user_emb.weight, model.user_bias.weight], lr=lr, weight_decay=wd)
    Ioptimizer = torch.optim.Adam([model.item_emb.weight, model.item_bias.weight], lr=lr, weight_decay=wd)
    users = torch.LongTensor(df.userId.values)  # .cuda()
    items = torch.LongTensor(df.movieId.values)  # .cuda()
    ratings = torch.FloatTensor(df.rating.values)  # .cuda()
    if unsqueeze:
        ratings = ratings.unsqueeze(1)  # convert from list to matrix [list,1]

    model.train()

    for i in range(epochs):
        cum_loss = 0

        y_hat = model(users, items)
        l2_norm = sum(p.pow(2.0).sum()
                      for p in [model.user_emb.weight, model.user_bias.weight])

        loss = F.mse_loss(y_hat, ratings)
        loss = loss + lmbd * l2_norm
        cum_loss += loss.item()
        loss.backward()
        Uoptimizer.step()
        Uoptimizer.zero_grad()

        y_hat = model(users, items)
        l2_norm = sum(p.pow(2.0).sum()
                      for p in [model.item_emb.weight, model.item_bias.weight])

        loss = F.mse_loss(y_hat, ratings)
        loss = loss + lmbd * l2_norm

        cum_loss += loss.item()
        loss.backward()
        Ioptimizer.step()
        Ioptimizer.zero_grad()

        if (i % 100 == 0): print("step:", i, cum_loss)
    users = torch.LongTensor(list(set(df.userId.values)))
    return (model.user_emb(users))



import torch
import torch.nn as nn



class Rope(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model

        #projection_layer
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

    def rope(self, x):

        seq_len, dim = x.shape()

        pos = torch.arange(seq_len, dtype=torch.flaot32).unsqueeze(1)

        freq = 1/(1000^(torch.arange(0, dim, 2).float() / dim))
        cos = torch.cos(freq)
        sin = torch.sin(freq)

        x_even = x[:, 0::2]
        x_odd = x[:, 1::2]

        #apply rotation
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_odd * sin + x_odd * cos

        x_out = torch.zeros_like(x)
        x_out[:, 0::2] = x_rot_even
        x_out[:, 1::2] = x_rot_odd

        return x_out

    def forward(self, x):

        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        Q_rot = self.rope(Q)
        K_rot = self.rope(K)

        #attention
        attn_scores = Q_rot @ K_rot.T / (self.d_model ** 0.5)
        weights = torch.softmax(attn_scores, dim=-1)

        out = weights @ V
        return out






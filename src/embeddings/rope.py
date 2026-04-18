
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()

        self.dim = dim

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # precompute
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        self.register_buffer("cos", torch.cos(freqs))
        self.register_buffer("sin", torch.sin(freqs))

    def forward(self, q, k, seq_len=None):
        # q, k: (batch, heads, seq_len, head_dim)

        if seq_len is None:
            seq_len = q.size(-2)

        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)

        q = self.apply_rotary(q, cos, sin)
        k = self.apply_rotary(k, cos, sin)

        return q, k

    def apply_rotary(self, x, cos, sin):
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        x_out = torch.zeros_like(x)
        x_out[..., 0::2] = x_rot_even
        x_out[..., 1::2] = x_rot_odd

        return x_out

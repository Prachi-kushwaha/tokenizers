import torch
import torch.nn as nn

class RMSnorm(nn.Module):
    def __init__(self, hidden_size, eps = 1e-16):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.tensor):
        rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight

def sinkhorn(logits, n_iters=20):
    """
    Project matrix onto doubly stochastic manifold.
    """
    M = torch.exp(logits)

    for _ in range(n_iters):
        M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
        M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)

    return M


class hyperconnection(nn.Module):
    def __init__(self, hc_mult:int, hidden_size:int, vocab_size):
        super().__init__()
        self.hc_mult = hc_mult
        self.scale = nn.Parameter(torch.ones(3) * 1e-2)
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rmsnorm = RMSnorm(hidden_size)
        self.weight_pre_proj = nn.Parameter(hc_mult*hidden_size, hc_mult)
        self.weight_post_proj = nn.Parameter(hc_mult*hidden_size, hc_mult)
        self.weight_res_proj = nn.Parameter(hc_mult*hidden_size, hc_mult*hc_mult)
        self.bias_pre = nn.Parameter(torch.zeros(1, hc_mult))
        self.bias_post = nn.Parameter(torch.zeros(hc_mult, 1))
        self.bias_res = nn.Parameter(torch.zeros(hc_mult, hc_mult))

    def forward(self, x:torch.tensor):
        x = self.embed(x)
        x = x.unsqueeze(2).expand(-1, -1, self.hc_mult, -1)
        B, T, N, D = x.shape

        # flatten residual manifold state
        x_flat = x.reshape(B, T, N * D)
        x_norm = self.rmsnorm(x_flat)
        alpha_pre, alpha_res, alpha_post = self.scale

        A_raw = alpha_pre * (x_norm * self.weight_pre_proj) + self.bias_pre
        B_raw = alpha_res * (x_norm * self.weight_res_proj) + self.bias_res
        C_raw = alpha_post * (x_norm * self.weight_post_proj) + self.bias_post

        A = torch.sigmoid(A_raw)
        C = 2.0 * torch.sigmoid(C_raw)
        B_view = B_raw.view(B, T, N, N)
        B = B_view + self.bias_res

        B_mat = sinkhorn(B_raw)

        return A, B_mat, C



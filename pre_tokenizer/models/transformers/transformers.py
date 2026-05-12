import torch
import torch.nn as nn

class PE(nn.Module):
    def __init__(self, hidden_size, seq_len):
        super().__init__()

        pe = torch.zeros(seq_len, hidden_size)

        position = torch.arange(0, seq_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, hidden_size, 2)
            * (-math.log(10000.0) / hidden_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, S, D)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x shape = (B, S, D)
        """
        S = x.size(1)

        return x + self.pe[:, :S, :]



class transformers(nn.Module):
    def __init__(self, vocab_size, seq_len, hidden_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.qkv_proj = nn.Linear(hidden_size, 3*hidden_size, bias = False)
        self.embedd = nn.Embedding(seq_len, hidden_size)

        self.pe = PE(hidden_size, seq_len)

        # qkv projection
        self.qkv_proj = nn.Linear(
            hidden_size,
            3 * hidden_size,
            bias=False
        )


    def forward(self, x):
        B, S = x.shape()
        token_emb = self.embedd(x)

        # apply positional encoding
        x = self.pe(token_emb)

        # qkv projection
        qkv = self.qkv_proj(x)

        # split q,k,v
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        return q, k, v




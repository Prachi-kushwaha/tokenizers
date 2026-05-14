import torch
import torch.nn as nn

import math

class PositionalEncoding(nn.Module):

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

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x, offset=0):

        S = x.size(1)

        return x + self.pe[:, offset:offset+S, :]

class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, num_heads):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None, past_kv=None):

        B, S, D = x.shape

        qkv = self.qkv_proj(x)

        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_key, past_value = past_kv
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present_kv = (k,v)

        attention = q @ k.transpose(-2, -1)

        attention = attention / math.sqrt(self.head_dim)

        if mask is not None:
           attention = attention.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)

        out = attention @ v

        out = out.transpose(1, 2).contiguous()

        out = out.view(B, S, D)

        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out, present_kv

class FeedForward(nn.Module):

    def __init__(self, hidden_size, mlp_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(0.1)
        )

    def forward(self, x):

        return self.net(x)


class TransformerEncoderBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, mlp_dim):
        super().__init__()

        self.attn = MultiHeadAttention(
            hidden_size,
            num_heads
        )

        self.norm1 = nn.LayerNorm(hidden_size)

        self.mlp = FeedForward(
            hidden_size,
            mlp_dim
        )

        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None, past_kv=None):

        # attention block
        attn_out, present_kv = self.attn(x, mask, past_kv)
        x = self.norm1(x + attn_out)

        # feedforward block
        x = self.norm2(x + self.mlp(x))

        return x, present_kv

class TransformerDecoderBlock(nn.Module):
    pass

class Transformer(nn.Module):

    def __init__(self, vocab_size, seq_len, hidden_size, num_heads, mlp_dim, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.pe = PositionalEncoding(hidden_size, seq_len)

        self.blocks = nn.ModuleList([ TransformerEncoderBlock(hidden_size, num_heads,mlp_dim)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.resid_dropout = nn.Dropout(0.1)

    def forward(self, x, past_kv=None):
        past_length = 0
        new_past_kv = []
        PAD_ID = 0
        if past_kv is None:
          past_kv = [None] * len(self.blocks)
          past_length = 0
          total_len = x.size(1)
          causal_mask = torch.tril(
          torch.ones(
            x.size(1),
            total_len,
            device=x.device
          )
        ).bool()
        else:
          past_length = past_kv[0][0].size(-2)

          total_len = past_length + x.size(1)
          causal_mask =torch.ones(
                  x.size(1),
                  total_len,
                  device=x.device
              ).bool()


        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        padding_mask = (x != PAD_ID).unsqueeze(1).unsqueeze(2)
        mask = causal_mask & padding_mask
        x = self.embedding(x)

        x = self.pe(x, offset=past_length)
        x = self.resid_dropout(x)


        for block, layer_past_kv in zip(self.blocks, past_kv):
            x, present_kv = block(x, mask, layer_past_kv)

            new_past_kv.append(present_kv)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits, new_past_kv
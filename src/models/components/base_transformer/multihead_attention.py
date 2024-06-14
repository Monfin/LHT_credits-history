import torch 
from torch import nn 

from src.models.components.utils.clone_modules import clone_modules

from typing import Tuple


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float, dropout: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = torch.sqrt(torch.tensor(temperature))
        self.softmax_dropout = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # (B, h, L, d)

        attn = torch.matmul(query, key) / self.temperature

        # (B, h, L, L) or (B, h, L, d_proj)

        if mask is not None:
            # mask (B, 1, L, 1) ~ same mask applied to all h heads and all L or d_k sequences
            attn.masked_fill_(~mask.unsqueeze(dim=1).unsqueeze(dim=-1), -1.0e9)
        
        attn = self.softmax_dropout(attn)

        context = torch.matmul(attn, value)

        # (B, h, L, d)

        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.1, seq_len: int = None, d_proj: int = None):
        super(MultiHeadAttention, self).__init__()

        self.linears = clone_modules(nn.Linear(d_model, d_model * n_heads), 3)

        self.seq_len = seq_len

        if self.seq_len is not None:
            if d_proj is None:
                d_proj = self.seq_len // 4

            self.E_projection = nn.Linear(self.seq_len, d_proj)
            self.F_projection = nn.Linear(self.seq_len, d_proj)

        self.attention = ScaledDotProductAttention(d_model, dropout)

        self.out_block = nn.Linear(d_model * n_heads, d_model)

        self.d_model = d_model
        self.n_heads = n_heads
        

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # (B, L, d)
        n_batches = query.size(0)

        query, key, value = [
            layer(x).view(n_batches, -1, self.n_heads, self.d_model).transpose(1, 2) \
                for layer, x in zip(self.linears, (query, key, value))
        ]

        # q, k, v ~ (B, h, L, d) # h = num_heads

        if self.seq_len is not None:
            key = self.E_projection(key.transpose(-2, -1))
            value = self.F_projection(value.transpose(-2, -1)).transpose(-2, -1)
            # q ~ (B, h, L, d)
            # k ~ (B, h, d, d_k) => P ~ (B, h, L, d_k)
            # v ~ (B, h, d_k, d) => matmul(P, v) ~ (B, h, L, d)
        else:
            key = key.transpose(-2, -1)

        context, _ = self.attention(query, key, value, mask=mask)

        # (B, h, L, d)

        # .contiguous() -> makes a copy of the tensor with its own layout ~ for view
        context = context.transpose(1, 2).contiguous().view(n_batches, -1, self.d_model * self.n_heads)

        # (B, L, d * h)

        context = self.out_block(context)

        # (B, L, d)

        return context
    

# LUNA
class LinearUnifiedNestedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.1, seq_len: int = None, d_proj: int = None):
        super(LinearUnifiedNestedAttention, self).__init__()

        self.pack_attention = MultiHeadAttention(d_model, n_heads, dropout, seq_len, d_proj)

        self.unpack_attention = MultiHeadAttention(d_model, n_heads, dropout, seq_len, d_proj)
        
        self.d_model = d_model
        self.n_heads = n_heads

    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, aggregates: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor]:

        packed_context = self.pack_attention(aggregates, key, value, mask)

        unpacked_context = self.unpack_attention(query, packed_context, packed_context)

        return unpacked_context, packed_context
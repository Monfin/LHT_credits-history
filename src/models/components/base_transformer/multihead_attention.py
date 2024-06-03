import torch 
from torch import nn 

from src.models.components.utils.clone_modules import clone_modules


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float, dropout: float = 0.1):
        super().__init__()

        self.temperature = torch.sqrt(torch.tensor(temperature))
        self.softmax_dropout = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # (B, h, L, d)

        attn = torch.matmul(query, key) / self.temperature

        # (B, h, L, L)

        if mask is not None:
            # mask (B, 1, L, 1) ~ same mask applied to all h heads and all L or d_k sequences
            attn.masked_fill_(~mask.unsqueeze(dim=1).unsqueeze(dim=-1), -1.0e9)
        
        attn = self.softmax_dropout(attn)

        x = torch.matmul(attn, value)

        # (B, h, L, d)

        return x, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, seq_len: int = None, d_k: int = 16, n_heads: int = 1, attn_dropout: float = 0.0, dropout: float = 0.1):
        super().__init__()

        self.linears = clone_modules(nn.Linear(d_model, d_model * n_heads), 3)

        self.seq_len = seq_len

        if self.seq_len is not None:
            self.E_projection = nn.Linear(self.seq_len, d_k)
            self.F_projection = nn.Linear(self.seq_len, d_k)

        self.attention = ScaledDotProductAttention(d_model, attn_dropout)
        self.dropout = nn.Dropout(dropout)

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

        x, _ = self.attention(query, key, value, mask=mask)

        # (B, h, L, d)

        # .contiguous() -> makes a copy of the tensor with its own layout ~ for view
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.d_model * self.n_heads)

        # (B, L, d * h)

        x = self.out_block(x)

        # (B, L, d)

        return x
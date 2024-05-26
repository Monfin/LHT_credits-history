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
        attn = torch.matmul(query, key.transpose(-2, -1)) / self.temperature

        if mask is not None:
            attn.masked_fill_(~mask, -1.0e9)
        
        attn = self.softmax_dropout(attn)

        x = torch.matmul(attn, value)

        return x, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 1, attn_dropout: float = 0.1):
        super().__init__()

        self.linears = clone_modules(nn.Linear(d_model, d_model * n_heads), 3)

        self.attention = ScaledDotProductAttention(d_model, attn_dropout)

        self.out_block = nn.Linear(d_model * n_heads, d_model)

        self.d_model = d_model
        self.n_heads = n_heads
        

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # (N, L, E)
        n_batches = query.size(0)

        query, key, value = [
            layer(x).view(n_batches, -1, self.n_heads, self.d_model) \
                for layer, x in zip(self.linears, (query, key, value))
        ]
        # q, k, v ~ (N, L, num_heads, E)

        mask = mask.view(n_batches, -1, 1, 1) # .unsqueeze(dim=-1).unsqueeze(dim=-1)

        x, _ = self.attention(query, key, value, mask=mask)
        # (N, L, num_heads, E)

        # .contiguous() -> makes a copy of the tensor with its own layout ~ for view
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.d_model * self.n_heads)
        # (N, L, E * num_heads)

        x = self.out_block(x)
        # (N, L, E)

        return x
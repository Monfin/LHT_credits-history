import torch
from torch import nn

from src.data.components.collate import ModelInput, SingleForwardState

from typing import Tuple


class EmbeddingLayer(nn.Module):
    def __init__(
            self,
            categorical_features: dict,
            embedding_dim: int = 32
        ) -> None:
        super(EmbeddingLayer, self).__init__()

        self.categorical_features = categorical_features

        self.embeddings = nn.ModuleList()

        for num_embs in self.categorical_features.values():
            embedding = nn.Embedding(num_embeddings=num_embs, embedding_dim=embedding_dim)

            nn.init.xavier_normal_(embedding.weight, gain=nn.init.calculate_gain('relu'))

            self.embeddings.append(embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = torch.concatenate(
            [embedding(x[..., idx]) for idx, embedding in enumerate(self.embeddings)], dim=-1
        ) # size = (batch_size, len(cat_features), embedding_dim)
        
        return x

    
class EncoderLayer(nn.Module):
    def __init__(
            self,
            categorical_features: dict,
            embedding_dim: int = 32,
            dropout_inputs: float = 0.5,
            non_linear: bool = False
        ) -> None:
        super(EncoderLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout_inputs)

        self.categorical_features = categorical_features

        self.embeddings = EmbeddingLayer(
            categorical_features=categorical_features,
            embedding_dim=embedding_dim
        )

        if non_linear:
            self.out_linear_block = nn.Sequential(
                nn.Linear(embedding_dim * len(self.categorical_features), embedding_dim),
                nn.ReLU(),

                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU()
            )
        else:
            self.out_linear_block = nn.Linear(embedding_dim * len(categorical_features), embedding_dim)

    def forward(self, inputs: ModelInput) -> SingleForwardState:
        embeddings = self.embeddings(inputs.categorical)
        x = self.dropout(embeddings)
        x = self.out_linear_block(x)

        return SingleForwardState(
            sequences=x, 
            lengths=inputs.lengths
        )
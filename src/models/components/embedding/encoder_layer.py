import torch
from torch import nn

from typing import List, Dict

from src.data.components.collate import ModelInput, SingleForwardState

from einops import rearrange


class EmbeddingLayer(nn.Module):
    def __init__(
            self,
            categorical_features: dict
        ) -> None:
        super(EmbeddingLayer, self).__init__()

        self.categorical_features = categorical_features

        self.embeddings = nn.ModuleList()

        for vocab_size, emb_dim in self.categorical_features.values():

            embedding = nn.Embedding(
                num_embeddings=vocab_size, 
                embedding_dim=emb_dim
            )

            nn.init.xavier_normal_(embedding.weight, gain=nn.init.calculate_gain('relu'))

            self.embeddings.append(embedding)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = torch.concatenate(
            [embedding(x[..., idx]) for idx, embedding in enumerate(self.embeddings)], dim=-1
        ) # size = (B, L, first_emb_dim)
        
        return x

    
class EncoderLayer(nn.Module):
    def __init__(
            self,
            numerical_features: List[str],
            categorical_features: Dict[str, List[int]],
            embedding_dim: int = 16,
            dropout_inputs: float = 0.5,
            non_linear: bool = False,
            num_batch_norm: bool = True
        ) -> None:
        super(EncoderLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout_inputs)

        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        _first_emb_dim = sum([shape[-1] for shape in self.categorical_features.values()])

        self.embeddings = EmbeddingLayer(
            categorical_features=categorical_features
        )

        if non_linear:
            self.out_linear_block = nn.Sequential(
                nn.Linear(
                    _first_emb_dim + len(self.numerical_features), 
                    embedding_dim
                ),
                nn.ReLU(),

                nn.Linear(embedding_dim, embedding_dim)
            )
        else:
            self.out_linear_block = nn.Linear(
                _first_emb_dim + len(self.numerical_features), 
                embedding_dim
            )

        if num_batch_norm:
            self.num_bn = nn.BatchNorm1d(len(self.numerical_features))
        else: 
            self.num_bn = None


    def forward(self, inputs: ModelInput) -> SingleForwardState:

        embeddings = self.embeddings(inputs.categorical)

        if self.num_bn is None:
            x_num = inputs.numerical
        else:
            x_num = rearrange(inputs.numerical, "N L H -> N H L")

            x_num = self.num_bn(x_num)

            x_num = rearrange(x_num, "N H L -> N L H")

        x = torch.concatenate((x_num, embeddings), dim=-1)

        x = self.dropout(x)
        x = self.out_linear_block(x)

        return SingleForwardState(
            sequences=x, 
            mask=inputs.mask
        )
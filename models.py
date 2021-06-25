import torch
from torch import Tensor
import torch.nn as nn


class GermEvalMLP(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 embedding_dim: int,
                 hidden_dim: int = 32,
                 dropout_rate: float = 0.,
                 hidden_activation: nn.Module = nn.ReLU()) -> None:
        super(GermEvalMLP, self).__init__()

        self.feature_dim = feature_dim

        self.embedding_reduction_layer = nn.Linear(embedding_dim, hidden_dim)

        self.linear_layer = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(feature_dim + hidden_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            hidden_activation,
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            hidden_activation,
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, features: Tensor) -> Tensor:
        document_embeddings = features[..., self.feature_dim:]

        reduced_embeddings = self.embedding_reduction_layer(document_embeddings)
        joint_features = torch.cat((features[..., :self.feature_dim], reduced_embeddings), dim=-1)

        return self.output_layer(self.mlp_layer(joint_features) + self.linear_layer(joint_features))

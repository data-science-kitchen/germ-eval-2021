from torch import Tensor
import torch.nn as nn


class GermEvalMLP(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 embedding_dim: int) -> None:
        super(GermEvalMLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.mlp(features)

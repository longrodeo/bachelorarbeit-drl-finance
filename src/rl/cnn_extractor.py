# -*- coding: utf-8 -*-
# src/rl/cnn_extractors.py
#
# Zwei FeatureExtractor für SB3:
#  - CNN1DExtractor: conv nur entlang der Feature-Achse (ordnunginvariant über Assets via Pooling)
#  - CNN2DExtractor: conv über Features×Assets (nutzt räumliche Nachbarschaft, reihenfolge-sensitiv)
#
# Erwartet Dict-Obs mit Keys: "X" (C,F,A), "g_scalars"(S,), "g_weights"(A,), "position"(A+1,)
# -> Das passt zu deinem Env-Print: X (2, 8, 8), g_scalars (2,), g_weights (8,), position (9,)

from __future__ import annotations
from typing import Dict, Any
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def _mlp(sizes, act=nn.ReLU, last_act=False):
    layers = []
    for i in range(len(sizes)-1):
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        if i < len(sizes)-2 or last_act:
            layers += [act()]
    return nn.Sequential(*layers)

class _GlobalHead(nn.Module):
    """Kombiniert globale Inputs (g_scalars, g_weights, position) in einen Vektor."""
    def __init__(self, dim_scalars: int, A: int, hidden: int = 64, out: int = 64):
        super().__init__()
        in_dim = dim_scalars + A + (A + 1)
        self.net = _mlp([in_dim, hidden, out])

    def forward(self, g_scalars: th.Tensor, g_weights: th.Tensor, position: th.Tensor):
        x = th.cat([g_scalars, g_weights, position], dim=1)
        return self.net(x)

class CNN1DExtractor(BaseFeaturesExtractor):
    """
    Conv NUR entlang der Feature-Achse (kern=(k,1)), damit keine künstliche
    Asset-Nachbarschaft entsteht. Danach ordnungsinvariante Aggregation über Assets.
    """
    def __init__(self, observation_space: gym.spaces.Dict, hidden: int = 64):
        super().__init__(observation_space, features_dim=1)  # Placeholder, setzen wir unten

        # Shapes aus dem Space lesen
        C, F, A = observation_space["X"].shape  # (channels, features, assets)
        self.C, self.F, self.A = C, F, A
        S = observation_space["g_scalars"].shape[0]

        # CNN: conv über Features (H) mit kernel (k,1)
        self.conv = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
        )
        # Feature-Pooling: erst Mittel/Max über Feature-Achse (H), dann über Assets (W)
        self.pool_features = nn.AdaptiveAvgPool2d((1, None))  # (B, 32, 1, A)
        # Zwei ordnungsinvariante Poolings über Assets: mean und max
        self.pool_assets_mean = nn.AdaptiveAvgPool2d((1, 1))  # (B, 32, 1, 1)
        self.pool_assets_max  = nn.AdaptiveMaxPool2d((1, 1))  # (B, 32, 1, 1)

        # Globale Inputs
        self.global_head = _GlobalHead(dim_scalars=S, A=A, hidden=hidden, out=hidden)

        # Finaler Merkmalskopf
        cnn_out = 32 * 2  # mean + max
        self.final = _mlp([cnn_out + hidden, 128, 128])
        self._features_dim = 128

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        # X: (B, C, F, A) für Conv2d
        X = obs["X"].float()               # (B, C, F, A) → MultiInputPolicy liefert schon B-First
        g_s = obs["g_scalars"].float()     # (B, S)
        g_w = obs["g_weights"].float()     # (B, A)
        pos = obs["position"].float()      # (B, A+1)

        h = self.conv(X)                   # (B, 32, F, A)
        h = self.pool_features(h)          # (B, 32, 1, A)
        meanA = self.pool_assets_mean(h)   # (B, 32, 1, 1)
        maxA  = self.pool_assets_max(h)    # (B, 32, 1, 1)
        h_cnn = th.cat([meanA, maxA], dim=1).squeeze(-1).squeeze(-1)  # (B, 64)

        h_glob = self.global_head(g_s, g_w, pos)  # (B, hidden)
        z = th.cat([h_cnn, h_glob], dim=1)        # (B, 64 + hidden)
        return self.final(z)                      # (B, 128)

class CNN2DExtractor(BaseFeaturesExtractor):
    """
    Conv über Features×Assets (kern (k_f, k_a)). Nutzt räumliche Muster,
    ist aber reihenfolge-sensitiv bzgl. Asset-Achse.
    """
    def __init__(self, observation_space: gym.spaces.Dict, hidden: int = 64):
        super().__init__(observation_space, features_dim=1)

        C, F, A = observation_space["X"].shape
        self.C, self.F, self.A = C, F, A
        S = observation_space["g_scalars"].shape[0]

        self.conv = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )
        # Global Average Pool über HxW
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # (B, 32, 1, 1)

        self.global_head = _GlobalHead(dim_scalars=S, A=A, hidden=hidden, out=hidden)

        self.final = _mlp([32 + hidden, 128, 128])
        self._features_dim = 128

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        X = obs["X"].float()
        g_s = obs["g_scalars"].float()
        g_w = obs["g_weights"].float()
        pos = obs["position"].float()

        h = self.conv(X)                   # (B, 32, F, A)
        h = self.gap(h).squeeze(-1).squeeze(-1)   # (B, 32)

        h_glob = self.global_head(g_s, g_w, pos)  # (B, hidden)
        z = th.cat([h, h_glob], dim=1)            # (B, 32 + hidden)
        return self.final(z)                      # (B, 128)

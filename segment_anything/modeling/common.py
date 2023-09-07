# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import Conv2d, Dropout
from typing import Optional, Tuple, Type
from operator import mul
from functools import reduce


class MultiScaleAdapterV3(nn.Module):
    def __init__(self, D_features, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = D_features // 3

        self.D_fc = nn.Linear(D_features, D_hidden_features)
        self.D_fc_final = nn.Linear(D_features, D_features)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(D_features, D_features, kernel_size=3, stride=2, padding=1)
        self.convtd = nn.ConvTranspose2d(D_features, D_features, kernel_size=2, stride=2)

    def forward(self, x):
        _, h, w, _ = x.shape
        target_size = (h, w)

        x1 = self.act(self.conv(x.permute(0, 3, 1, 2)))
        x1 = self.act(self.D_fc(x1.permute(0, 2, 3, 1)))
        x1 = F.interpolate(x1.permute(0, 3, 1, 2), size=target_size, mode='bilinear', align_corners=False)

        x2 = self.act(self.D_fc(x))

        x3 = self.act(self.convtd(x.permute(0, 3, 1, 2)))
        x3 = self.act(self.D_fc(x3.permute(0, 2, 3, 1)))
        x3 = F.interpolate(x3.permute(0, 3, 1, 2), size=target_size, mode='bilinear', align_corners=False)

        xc = torch.cat([x1.permute(0, 2, 3, 1), x2, x3.permute(0, 2, 3, 1)], dim=3)
        xc = self.D_fc_final(xc)

        if self.skip_connect:
            x = x + xc
        else:
            x = xc

        return x


class MultiScaleAdapterV2(nn.Module):
    def __init__(self, D_features, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = D_features // 2

        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_features, D_hidden_features)
        self.D_fc_final = nn.Linear(D_features, D_features)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(D_features, D_features, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        _, h, w, _ = x.shape
        target_size = (h, w)

        x1 = self.act(self.conv(x.permute(0, 3, 1, 2)))
        x1 = self.act(self.D_fc1(x1.permute(0, 2, 3, 1)))
        x1 = F.interpolate(x1.permute(0, 3, 1, 2), size=target_size, mode='bilinear', align_corners=False)

        x2 = self.act(self.D_fc2(x))

        xc = torch.cat([x1.permute(0, 2, 3, 1), x2], dim=3)
        xc = self.D_fc_final(xc)

        if self.skip_connect:
            x = x + xc
        else:
            x = xc
        return x


class MultiScaleAdapter(nn.Module):
    def __init__(self, D_features, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = D_features // 3

        self.D_fc = nn.Linear(D_features, D_hidden_features)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(D_hidden_features, D_hidden_features, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        _, h, w, _ = x.shape
        target_size = (h, w)

        x1 = self.act(self.D_fc(x))
        x1 = x1.permute(0, 3, 1, 2)

        x2 = self.act(self.conv(x1))
        x3 = self.act(self.conv(x2))

        x2 = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False)

        xc = torch.cat([x1, x2, x3], dim=1)
        xc = xc.permute(0, 2, 3, 1)

        if self.skip_connect:
            x = x + xc
        else:
            x = xc
        return x


class MyAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)

        self.adapter_feature = nn.Sequential(
            nn.BatchNorm2d(D_features),
            nn.GELU(),
            nn.ConvTranspose2d(D_features, D_hidden_features, kernel_size=2, stride=2),
            nn.BatchNorm2d(D_hidden_features),
            nn.GELU(),
            nn.Conv2d(D_hidden_features, D_features, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        xs = self.adapter_feature(x.permute(0, 3, 1, 2))
        xs = xs.permute(0, 2, 3, 1)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class VisualPrompt(nn.Module):
    def __init__(self, batch_size=2, num_prompt=8, prompt_config_drop=0, hidden_size=768, patch_size=14, scale=0.1):
        super().__init__()
        self.num_prompt = num_prompt
        self.hidden_size = hidden_size
        self.prompt_dropout = Dropout(prompt_config_drop)
        in_channel = batch_size * hidden_size
        out_channel = batch_size * num_prompt * num_prompt * hidden_size

        self.prompt_proj = nn.Linear(in_features=in_channel, out_features=out_channel)
        nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')

        self.prompt_embedding = nn.Parameter(torch.zeros(batch_size, 1, 1, hidden_size))
        # val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + hidden_size))  # noqa

        val = math.sqrt(6. / float(3 * patch_size * patch_size * hidden_size))
        nn.init.uniform_(self.prompt_embedding.data, -val, val)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape

        y = self.prompt_proj(self.prompt_embedding.view(-1))
        y = self.prompt_dropout(y.view(B, self.num_prompt, self.num_prompt, self.hidden_size))

        y = y.repeat(1, H // self.num_prompt, W // self.num_prompt, 1)

        contact = x + self.scale * y
        return contact


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class MLPBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            mlp_dim: int,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

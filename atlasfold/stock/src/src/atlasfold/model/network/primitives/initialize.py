"""Utility functions for initializing weights and biases.
Modified from OpenFold-3 initialize.py
"""

# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch


def _calculate_fan(linear_weight_shape: tuple[int, int], fan="fan_in") -> float:
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


@torch.no_grad()
def trunc_normal_init_(weights: torch.Tensor, scale=1.0, fan="fan_in"):
    """New truncated normal initialization consistent with
    pure PyTorch implementation.
    """
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)

    # Same to truncnorm.std with a=-2, b=2, loc=0, scale=1
    correction_factor = 0.87962566103423978

    std = math.sqrt(scale) / correction_factor

    torch.nn.init.trunc_normal_(
        weights,
        mean=0.0,
        std=std,
        a=-2.0 * std,
        b=2.0 * std,
    )


def lecun_normal_init_(weights: torch.Tensor):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights: torch.Tensor):
    trunc_normal_init_(weights, scale=2.0)


@torch.no_grad()
def zero_init_(weights: torch.Tensor):
    weights.fill_(0.0)


@torch.no_grad()
def bias_init_(bias: torch.Tensor, value: float = 0.0):
    bias.fill_(value)


@torch.no_grad()
def bias_zero_init_(bias: torch.Tensor):
    bias_init_(bias, value=0.0)

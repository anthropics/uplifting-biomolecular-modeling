# Copyright Generate Biomedicines, Inc.
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
# Modified by Richard Shuai (2025).

"""Layers for building Potts models.

This module contains layers for parameterizing Potts models from
graph embeddings.

Adapted from Chroma by Richard Shuai.
"""

from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

import caliby.data.const as const
from caliby.model.seq_denoiser.denoisers.seq_design.potts_state_space import (
    map_potts_indices_to_af3,
    normalize_potts_inputs,
)
from chroma.layers import graph


class GraphPotts(nn.Module):
    """Conditional Random Field (conditional Potts model) layer on a graph.

    Arguments:
        dim_nodes (int): Hidden dimension of node tensor.
        dim_edges (int): Hidden dimension of edge tensor.
        num_states (int): Size of the vocabulary.
        parameterization (str): Parameterization choice in
            `{'linear', 'factor', 'score', 'score_zsum', 'score_scale'}`, or
            any of those suffixed with `_beta`, which will add in a globally
            learnable temperature scaling parameter.
        symmetric_J (bool): If True enforce symmetry of Potts model i.e.
            `J_ij(s_i, s_j) = J_ji(s_j, s_i)`.
        init_scale (float): Scale factor for the weights and couplings at
            initialization.
        dropout (float): Probability of per-dimension dropout on `[0,1]`.
        num_factors (int): Number of factors to use for the `factor`
            parameterization mode.
        beta_init (float): Initial temperature scaling factor for parameterizations
            with the `_beta` suffix.

    Inputs:
        node_h (torch.Tensor): Node features with shape
            `(num_batch, num_nodes, dim_nodes)`.
        edge_h (torch.Tensor): Edge features with shape
            `(num_batch, num_nodes, num_neighbors, dim_edges)`.
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_nodes)`
        mask_ij (torch.Tensor): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`

    Outputs:
        h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
            `(num_batch, num_nodes, num_states)`.
        J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
            `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
    """

    def __init__(
        self,
        dim_nodes: int,
        dim_edges: int,
        num_states: int,
        parameterization: str = "score",
        symmetric_J: bool = True,
        init_scale: float = 0.1,
        dropout: float = 0.0,
        num_factors: Optional[int] = None,
        beta_init: float = 10.0,
    ):
        super(GraphPotts, self).__init__()
        self.dim_nodes = dim_nodes
        self.dim_edges = dim_edges
        self.num_states = num_states

        # Beta parameterization support temperature learning
        self.scale_beta = False
        if parameterization.endswith("_beta"):
            parameterization = parameterization.split("_beta")[0]
            self.scale_beta = True
            self.log_beta = nn.Parameter(np.log(beta_init) * torch.ones(1))

        self.init_scale = init_scale
        self.parameterization = parameterization
        self.symmetric_J = symmetric_J
        if self.parameterization == "linear":
            self.log_scale = nn.Parameter(np.log(init_scale) * torch.ones(1))
            self.W_h = nn.Linear(self.dim_nodes, self.num_states, bias=True)
            self.W_J = nn.Linear(self.dim_edges, self.num_states**2, bias=True)
        elif self.parameterization == "factor":
            self.log_scale = nn.Parameter(np.log(init_scale) * torch.ones(1))
            self.W_h = nn.Linear(self.dim_nodes, self.num_states, bias=True)
            self.W_J_left = nn.Linear(self.dim_edges, self.num_states**2, bias=True)
            self.W_J_right = nn.Linear(self.dim_edges, self.num_states**2, bias=True)
        elif self.parameterization == "score":
            if num_factors is None:
                num_factors = dim_edges
            self.num_factors = num_factors
            self.log_scale = nn.Parameter(np.log(init_scale) * torch.ones(1))
            self.W_h_bg = nn.Linear(self.dim_nodes, 1)
            self.W_J_bg = nn.Linear(self.dim_edges, 1)
            self.W_h = nn.Linear(self.dim_nodes, self.num_states, bias=True)
            self.W_J_left = nn.Linear(self.dim_edges, self.num_states * num_factors, bias=True)
            self.W_J_right = nn.Linear(self.dim_edges, self.num_states * num_factors, bias=True)
        elif self.parameterization == "score_zsum":
            if num_factors is None:
                num_factors = dim_edges
            self.num_factors = num_factors
            self.log_scale = nn.Parameter(np.log(init_scale) * torch.ones(1))
            self.W_h = nn.Linear(self.dim_nodes, self.num_states, bias=True)
            self.W_J_left = nn.Linear(self.dim_edges, self.num_states * num_factors, bias=True)
            self.W_J_right = nn.Linear(self.dim_edges, self.num_states * num_factors, bias=True)
        elif self.parameterization == "score_scale":
            if num_factors is None:
                num_factors = dim_edges
            self.num_factors = num_factors
            self.W_h_bg = nn.Linear(self.dim_nodes, 1)
            self.W_J_bg = nn.Linear(self.dim_edges, 1)
            self.W_h_log_scale = nn.Linear(self.dim_nodes, 1)
            self.W_J_log_scale = nn.Linear(self.dim_edges, 1)
            self.W_h = nn.Linear(self.dim_nodes, self.num_states)
            self.W_J_left = nn.Linear(self.dim_edges, self.num_states * num_factors)
            self.W_J_right = nn.Linear(self.dim_edges, self.num_states * num_factors)
        else:
            print(f"Unknown potts parameterization: {parameterization}")
            raise NotImplementedError
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_idx: torch.LongTensor,
        mask_i: torch.Tensor,
        mask_ij: torch.Tensor,
    ):
        mask_J = _mask_J(edge_idx, mask_i, mask_ij)

        if self.parameterization == "linear":
            # Compute site params (h) from node embeddings
            # Compute coupling params (J) from edge embeddings
            scale = torch.exp(self.log_scale)
            h = scale * mask_i.unsqueeze(-1) * self.W_h(node_h)
            J = scale * mask_J.unsqueeze(-1) * self.W_J(edge_h)
            J = J.view(list(edge_h.size())[:3] + ([self.num_states] * 2))
        elif self.parameterization == "factor":
            scale = torch.exp(self.log_scale)
            h = scale * mask_i.unsqueeze(-1) * self.W_h(node_h)
            mask_J = scale * mask_J.unsqueeze(-1)
            shape_J = list(edge_h.size())[:3] + ([self.num_states] * 2)
            J_left = (mask_J * self.W_J_left(edge_h)).view(shape_J)
            J_right = (mask_J * self.W_J_right(edge_h)).view(shape_J)
            J = torch.matmul(J_left, J_right)
            J = self.dropout(J)
            # Zero-sum gauge
            h = h - h.mean(-1, keepdim=True)
            J = J - J.mean(-1, keepdim=True) - J.mean(-2, keepdim=True) + J.mean(dim=[-1, -2], keepdim=True)
        elif self.parameterization == "score":
            node_h = self.dropout(node_h)
            edge_h = self.dropout(edge_h)

            scale = torch.exp(self.log_scale)
            mask_h = scale * mask_i.unsqueeze(-1)
            mask_J = scale * mask_J.unsqueeze(-1)
            h = mask_h * self.W_h(node_h)

            shape_J_prefix = list(edge_h.size())[:3]
            J_left = (mask_J * self.W_J_left(edge_h)).view(shape_J_prefix + [self.num_states, self.num_factors])
            J_right = (mask_J * self.W_J_right(edge_h)).view(shape_J_prefix + [self.num_factors, self.num_states])
            J = torch.matmul(J_left, J_right)

            # Zero-sum gauge
            h = h - h.mean(-1, keepdim=True)
            J = J - J.mean(-1, keepdim=True) - J.mean(-2, keepdim=True) + J.mean(dim=[-1, -2], keepdim=True)

            # Background components
            h = h + mask_h * self.W_h_bg(node_h)
            J = J + (mask_J * self.W_J_bg(edge_h)).unsqueeze(-1)
        elif self.parameterization == "score_zsum":
            node_h = self.dropout(node_h)
            edge_h = self.dropout(edge_h)

            scale = torch.exp(self.log_scale)
            mask_h_scale = scale * mask_i.unsqueeze(-1)
            mask_J_scale = scale * mask_J.unsqueeze(-1)
            h = mask_h_scale * self.W_h(node_h)

            shape_J_prefix = list(edge_h.size())[:3]
            J_left = (mask_J_scale * self.W_J_left(edge_h)).view(shape_J_prefix + [self.num_states, self.num_factors])
            J_right = (mask_J_scale * self.W_J_right(edge_h)).view(shape_J_prefix + [self.num_factors, self.num_states])
            J = torch.matmul(J_left, J_right)
            J = self.dropout(J)

            # Zero-sum gauge
            J = J - J.mean(-1, keepdim=True) - J.mean(-2, keepdim=True) + J.mean(dim=[-1, -2], keepdim=True)

            # Subtract off J background average
            mask_J = mask_J.view(list(mask_J.size()) + [1, 1])
            J_i_avg = J.sum(dim=[1, 2], keepdim=True) / mask_J.sum([1, 2], keepdim=True)
            J = mask_J * (J - J_i_avg)
        elif self.parameterization == "score_scale":
            node_h = self.dropout(node_h)
            edge_h = self.dropout(edge_h)

            mask_h = mask_i.unsqueeze(-1)
            mask_J = mask_J.unsqueeze(-1)
            h = mask_h * self.W_h(node_h)

            shape_J_prefix = list(edge_h.size())[:3]
            J_left = (mask_J * self.W_J_left(edge_h)).view(shape_J_prefix + [self.num_states, self.num_factors])
            J_right = (mask_J * self.W_J_right(edge_h)).view(shape_J_prefix + [self.num_factors, self.num_states])
            J = torch.matmul(J_left, J_right)

            # Zero-sum gauge
            h = h - h.mean(-1, keepdim=True)
            J = J - J.mean(-1, keepdim=True) - J.mean(-2, keepdim=True) + J.mean(dim=[-1, -2], keepdim=True)

            # Background components
            log_scale = np.log(self.init_scale)
            h_scale = torch.exp(self.W_h_log_scale(node_h) + log_scale)
            J_scale = torch.exp(self.W_J_log_scale(edge_h) + 2 * log_scale).unsqueeze(-1)
            h_bg = mask_h * self.W_h_bg(node_h)
            J_bg = (mask_J * self.W_J_bg(edge_h)).unsqueeze(-1)
            h = h_scale * (h + h_bg)
            J = J_scale * (J + J_bg)

        if self.symmetric_J:
            J = self._symmetrize_J(J, edge_idx, mask_ij)

        if self.scale_beta:
            beta = torch.exp(self.log_beta)
            h = beta * h
            J = beta * J
        return h, J

    def _symmetrize_J_serial(self, J, edge_idx, mask_ij):
        """Enforce symmetry of J matrices, serial version."""
        num_batch, num_residues, num_k, num_states, _ = list(J.size())

        # Symmetrization based on raw indexing - extremely slow; for debugging
        import time

        _start = time.time()
        J_symm = torch.zeros_like(J)
        for b in range(J.size(0)):
            for i in range(J.size(1)):
                for k_i in range(J.size(2)):
                    for k_j in range(J.size(2)):
                        j = edge_idx[b, i, k_i]
                        if edge_idx[b, j, k_j] == i:
                            J_symm[b, i, k_i, :, :] = (J[b, i, k_i, :, :] + J[b, j, k_j, :, :].transpose(-1, -2)) / 2.0
        speed = J.size(0) * J.size(1) / (time.time() - _start)
        print(f"symmetrized at {speed} residue/s")
        return J_symm

    def _symmetrize_J(self, J, edge_idx, mask_ij):
        """Enforce symmetry of J matrices via adding J_ij + J_ji^T"""
        num_batch, num_residues, num_k, num_states, _ = list(J.size())

        # Flatten and gather J_ji matrices using transpose indexing
        J_flat = J.view(num_batch, num_residues, num_k, -1)
        J_flat_transpose, mask_ji = graph.collect_edges_transpose(J_flat, edge_idx, mask_ij)
        J_transpose = J_flat_transpose.view(num_batch, num_residues, num_k, num_states, num_states)
        # Transpose J_ji matrices to symmetrize as (J_ij + J_ji^T)/2
        J_transpose = J_transpose.transpose(-2, -1)
        mask_ji = (0.5 * mask_ji).view(num_batch, num_residues, num_k, 1, 1)
        J_symm = mask_ji * (J + J_transpose)
        return J_symm

    def energy(
        self,
        S: torch.LongTensor,
        h: torch.Tensor,
        J: torch.Tensor,
        edge_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute Potts model energy from sequence.

        Inputs:
            S (torch.LongTensor): AF3-indexed sequence with shape `(num_batch, num_nodes)`.
            h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
                `(num_batch, num_nodes, num_states)`.
            J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
                `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
            edge_idx (torch.LongTensor): Edge indices with shape
                `(num_batch, num_nodes, num_neighbors)`.

        Outputs:
            U (torch.Tensor): Potts total energies with shape `(num_batch)`.
                Lower energies are more favorable.
        """
        U, _ = compute_potts_energy(S=S, h=h, J=J, edge_idx=edge_idx, mask_i=None, mask_ij=None)
        return U

    def pseudolikelihood(
        self,
        S: torch.LongTensor,
        h: torch.Tensor,
        J: torch.Tensor,
        edge_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute Potts pseudolikelihood from sequence

        Inputs:
            S (torch.LongTensor): AF3-indexed sequence with shape `(num_batch, num_nodes)`.
            h (torch.Tensor): Potts model fields :math:`h_i(s_i)` with shape
                `(num_batch, num_nodes, num_states)`.
            J (Tensor): Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
                `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
            edge_idx (torch.LongTensor): Edge indices with shape
                `(num_batch, num_nodes, num_neighbors)`.

        Outputs:
            log_probs (torch.Tensor): Potts log-pseudolihoods with shape
                `(num_batch, num_nodes, len(const.POTTS_TOKENS))`.
        """
        S_reduced, h, J, _ = normalize_potts_inputs(S=S, h=h, J=J, mask_i=None)

        # Gather J [Batch,i,j,A_i,A_j] => J_ij(:,A_j) [Batch,i,j,A_i]
        S_j = graph.collect_neighbors(S_reduced.unsqueeze(-1), edge_idx)
        S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, h.shape[-1], -1)
        J_ij = torch.gather(J, -1, S_j).squeeze(-1)

        # Sum out J contributions
        J_i = J_ij.sum(2)

        logits = h + J_i
        log_probs = F.log_softmax(-logits, dim=-1)
        return log_probs

    def sample(
        self,
        h: torch.Tensor,
        J: torch.Tensor,
        edge_idx: torch.LongTensor,
        mask_i: torch.Tensor,
        mask_ij: torch.Tensor,
        S: Optional[torch.LongTensor] = None,
        mask_sample: Optional[torch.Tensor] = None,
        num_sweeps: int = 100,
        temperature: float = 0.1,
        temperature_init: float = 1.0,
        penalty_func: Optional[Callable[[torch.LongTensor], torch.Tensor]] = None,
        differentiable_penalty: bool = True,
        rejection_step: bool = False,
        proposal: Literal["dlmc", "chromatic"] = "dlmc",
        verbose: bool = False,
        edge_idx_coloring: Optional[torch.LongTensor] = None,
        mask_ij_coloring: Optional[torch.Tensor] = None,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        """Sample from Potts model with Chromatic Gibbs sampling.

        Args:
            h: Potts model fields :math:`h_i(s_i)` with shape
                `(num_batch, num_nodes, num_states)`.
            J: Potts model couplings :math:`J_{ij}(s_i, s_j)` with shape
                `(num_batch, num_nodes, num_neighbors, num_states, num_states)`.
            edge_idx (torch.LongTensor): Edge indices with shape
                `(num_batch, num_nodes, num_neighbors)`.
            mask_i (torch.Tensor): Node mask with shape `(num_batch, num_nodes)`.
            mask_ij (torch.Tensor): Edge mask with shape
                `(num_batch, num_nodes, num_neighbors)`.
            S (torch.LongTensor, optional): Sequence for initialization with
                shape `(num_batch, num_nodes)`.
            mask_sample (torch.Tensor, optional): Binary sampling mask indicating
                positions which are free to change with shape
                `(num_batch, num_nodes)` or which tokens are acceptable at each position
                with shape `(num_batch, num_nodes, alphabet)`.
            num_sweeps (int): Number of sweeps of Chromatic Gibbs to perform,
                i.e. the depth of sampling as measured by the number of times
                every position has had an opportunity to update.
            temperature (float): Final sampling temperature.
            temperature_init (float): Initial sampling temperature, which will
                be linearly interpolated to `temperature` over the course of
                the burn in phase.
            penalty_func (Callable, optional): An optional penalty function which
                takes a sequence `S` and outputes a `(num_batch)` shaped tensor
                of energy adjustments, for example as regularization.
            differentiable_penalty (bool): If True, gradients of penalty function
                will be used to adjust the proposals.
            rejection_step (bool): If True, perform a Metropolis-Hastings
                rejection step.
            proposal (str): MCMC proposal for Potts sampling. Currently implemented
                proposals are `dlmc` for Discrete Langevin Monte Carlo [1] or `chromatic`
                for Gibbs sampling with graph coloring.
                [1] Sun et al. Discrete Langevin Sampler via Wasserstein Gradient Flow (2023).
            verbose (bool): If True print verbose output during sampling.
            edge_idx_coloring (torch.LongerTensor, optional): Alternative
                graph dependency structure that can be provided for the
                Chromatic Gibbs algorithm when it performs initial graph
                coloring. Has shape
                    `(num_batch, num_nodes, num_neighbors_coloring)`.
            mask_ij_coloring (torch.Tensor): Edge mask for the alternative dependency
                structure with shape `(num_batch, num_nodes, num_neighbors_coloring)`.

        Returns:
            S (torch.LongTensor): Sampled AF3-indexed sequences with
                shape `(num_batch, num_nodes)`.
            U (torch.Tensor): Sampled energies with shape `(num_batch)`. Lower
                is more favorable.
        """
        S_sample, U_sample = sample_potts(
            h,
            J,
            edge_idx,
            mask_i,
            mask_ij,
            S=S,
            mask_sample=mask_sample,
            num_sweeps=num_sweeps,
            temperature=temperature,
            temperature_init=temperature_init,
            penalty_func=penalty_func,
            differentiable_penalty=differentiable_penalty,
            rejection_step=rejection_step,
            proposal=proposal,
            verbose=verbose,
            edge_idx_coloring=edge_idx_coloring,
            mask_ij_coloring=mask_ij_coloring,
        )

        return S_sample, U_sample


def compute_potts_energy(
    *,
    S: torch.LongTensor,
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor | None,
    mask_ij: torch.Tensor | None,
    return_energy_components: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Potts energies for an AF3-indexed sequence.

    Args:
        S (torch.LongTensor): AF3-indexed sequence with shape `(num_batch, num_nodes)`.
        h (torch.Tensor): Field tensor with shape `(num_batch, num_nodes, C)`,
            where `C` is either `const.AF3_ENCODING.n_tokens` or
            `len(const.POTTS_TOKENS)`.
        J (Tensor): Coupling tensor with shape
            `(num_batch, num_nodes, num_neighbors, C, C)` using the same state
            dimension as `h`.
        mask_i (torch.Tensor): Optional node mask with shape `(num_batch, num_nodes)`.
        mask_ij (torch.Tensor): Optional edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, num_neighbors)`.
        return_energy_components (bool): If True, also return the field and
            coupling contributions to the total energy.

    Returns:
        U (torch.Tensor): Potts total energies with shape `(num_batch)`.
            Lower energies are more favorable.
        U_i (torch.Tensor): Potts local conditional energies in `const.POTTS_TOKENS`
            order with shape `(num_batch, num_nodes, len(const.POTTS_TOKENS))`.
        E_field (torch.Tensor, optional): Field energy contributions with shape `(num_batch)`.
            Only returned if `return_energy_components=True`.
        E_coupling (torch.Tensor, optional): Coupling energy contributions with shape `(num_batch)`.
            Only returned if `return_energy_components=True`.
    """
    S_reduced, h, J, _ = normalize_potts_inputs(S=S, h=h, J=J, mask_i=mask_i)
    return _compute_potts_energy_reduced(
        S=S_reduced,
        h=h,
        J=J,
        edge_idx=edge_idx,
        mask_i=mask_i,
        mask_ij=mask_ij,
        return_energy_components=return_energy_components,
    )


def _compute_potts_energy_reduced(
    *,
    S: torch.LongTensor,
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor | None,
    mask_ij: torch.Tensor | None,
    return_energy_components: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Potts energies for reduced-state inputs.

    Args:
        S: Sequence tensor with shape `(num_batch, num_nodes)` indexed by
            `const.POTTS_TOKENS`.
        h: Field tensor with shape `(num_batch, num_nodes, len(const.POTTS_TOKENS))`.
        J: Coupling tensor with shape
            `(num_batch, num_nodes, num_neighbors, len(const.POTTS_TOKENS), len(const.POTTS_TOKENS))`.
        edge_idx: Edge indices with shape `(num_batch, num_nodes, num_neighbors)`.
        mask_i: Optional node mask with shape `(num_batch, num_nodes)`.
        mask_ij: Optional edge mask with shape `(num_batch, num_nodes, num_neighbors)`.
        return_energy_components: If `True`, also return the field and coupling
            contributions to the total energy.
    """

    # Mask out nodes and edges that are not present.
    if mask_i is not None:
        h = h * mask_i[..., None]
    if mask_ij is not None:
        J = J * mask_ij[..., None, None]

    S_j = graph.collect_neighbors(S.unsqueeze(-1), edge_idx)
    S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, h.shape[-1], -1)
    J_ij = torch.gather(J, -1, S_j).squeeze(-1)

    # Sum out J contributions to yield local conditionals
    J_i = J_ij.sum(2)
    U_i = h + J_i

    # Correct for double counting in total energy
    U = (torch.gather(U_i, -1, S[..., None]) - 0.5 * torch.gather(J_i, -1, S[..., None])).sum((1, 2))

    if return_energy_components:
        # Field energy: contribution from fields h_i(s_i)
        U_field = torch.gather(h, -1, S[..., None]).sum((1, 2))
        # Coupling energy: contribution from couplings J_ij(s_i, s_j), with 0.5 factor to correct for double counting
        U_coupling = 0.5 * torch.gather(J_i, -1, S[..., None]).sum((1, 2))
        return U, U_i, U_field, U_coupling

    return U, U_i


@torch.no_grad()
def _color_graph(edge_idx, mask_ij, max_iter=100):
    """Stochastic graph coloring."""
    # Randomly assign initial colors
    B, N, K = edge_idx.shape
    # By Brooks we only need K + 1, but one extra color aids convergence
    num_colors = K + 2
    S = torch.randint(0, num_colors, (B, N), device=edge_idx.device)

    # Ignore self-attachement
    ix = torch.arange(edge_idx.shape[1], device=edge_idx.device)[None, ..., None]
    mask_ij = (mask_ij * torch.ne(edge_idx, ix).float())[..., None]

    # Iteratively replace clashing sites with an available color
    i = 0
    total_clashes = 1
    while total_clashes > 0 and i < max_iter:
        # Tabulate available colors in neighborhood
        O_i = F.one_hot(S, num_colors).float()
        N_i = (mask_ij * graph.collect_neighbors(O_i, edge_idx)).sum(2)
        clashes = (O_i * N_i).sum(-1)
        N_i = torch.where(N_i > 0, -float("inf") * torch.ones_like(N_i), N_i)

        # Resample from this distribution where clashing
        S_new = torch.distributions.categorical.Categorical(logits=N_i).sample()
        S = torch.where(clashes > 0, S_new, S)
        i += 1
        total_clashes = clashes.sum().item()
    return S


@torch.no_grad()
def sample_potts(
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor,
    mask_ij: torch.Tensor,
    S: Optional[torch.LongTensor] = None,
    mask_sample: Optional[torch.Tensor] = None,
    num_sweeps: int = 100,
    temperature: float = 1.0,
    temperature_init: float = 1.0,
    annealing_fraction: float = 0.8,
    penalty_func: Optional[Callable[[torch.LongTensor], torch.Tensor]] = None,
    differentiable_penalty: bool = True,
    rejection_step: bool = False,
    proposal: Literal["dlmc", "chromatic"] = "dlmc",
    verbose: bool = True,
    return_trajectory: bool = False,
    thin_sweeps: int = 3,
    edge_idx_coloring: Optional[torch.LongTensor] = None,
    mask_ij_coloring: Optional[torch.Tensor] = None,
) -> Union[
    tuple[torch.LongTensor, torch.Tensor],
    tuple[torch.LongTensor, torch.Tensor, list[torch.LongTensor], list[torch.Tensor]],
]:
    """Sample from a Potts model.

    Args:
        h (torch.Tensor): Field tensor with shape `(num_batch, num_nodes, C)`,
            where `C` is either `const.AF3_ENCODING.n_tokens` or
            `len(const.POTTS_TOKENS)`.
        J (Tensor): Coupling tensor with shape
            `(num_batch, num_nodes, num_neighbors, C, C)` using the same state
            dimension as `h`.
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_nodes)`.
        mask_ij (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
        S (torch.LongTensor, optional): AF3-indexed sequence for initialization
            with shape `(num_batch, num_nodes)`.
        mask_sample (torch.Tensor, optional): Binary sampling mask indicating
            positions which are free to change with shape
            `(num_batch, num_nodes)` or which tokens are acceptable at each position
            with shape `(num_batch, num_nodes, alphabet)`. A 3D mask may use the
            AF3 token axis or the reduced Potts token axis.
        num_sweeps (int): Number of sweeps of Chromatic Gibbs to perform,
            i.e. the depth of sampling as measured by the number of times
            every position has had an opportunity to update.
        temperature (float): Final sampling temperature.
        temperature_init (float): Initial sampling temperature, which will
            be linearly interpolated to `temperature` over the course of
            the burn in phase.
        annealing_fraction (float): Fraction of the total sampling run during
            which temperature annealing occurs.
        penalty_func (Callable, optional): An optional penalty function which
            takes a sequence `S` and outputes a `(num_batch)` shaped tensor
            of energy adjustments, for example as regularization.
        differentiable_penalty (bool): If True, gradients of penalty function
            will be used to adjust the proposals.
        rejection_step (bool): If True, perform a Metropolis-Hastings
            rejection step.
        proposal (str): MCMC proposal for Potts sampling. Currently implemented
                proposals are `dlmc` for Discrete Langevin Monte Carlo [1] or `chromatic`
                for Gibbs sampling with graph coloring.
                [1] Sun et al. Discrete Langevin Sampler via Wasserstein Gradient Flow (2023).
        verbose (bool): If True print verbose output during sampling.
        return_trajectory (bool): If True, also output the sampling trajectories
            of `S` and `U`.
        thin_sweeps (int): When returning trajectories, only save every `thin_sweeps`
            state to reduce memory usage.
        edge_idx_coloring (torch.LongerTensor, optional): Alternative
            graph dependency structure that can be provided for the
            Chromatic Gibbs algorithm when it performs initial graph
            coloring. Has shape
                `(num_batch, num_nodes, num_neighbors_coloring)`.
        mask_ij_coloring (torch.Tensor): Edge mask for the alternative dependency
            structure with shape `(num_batch, num_nodes, num_neighbors_coloring)`.

    Returns:
        S (torch.LongTensor): Sampled AF3-indexed sequences with
            shape `(num_batch, num_nodes)`.
        U (torch.Tensor): Sampled energies with shape `(num_batch)`. Lower is more
            favorable.
        S_trajectory (list[torch.LongTensor]): List of sampled sequences through
            time, each AF3-indexed with shape `(num_batch, num_nodes)`.
        U_trajectory (list[torch.Tensor]): List of sampled energies through time
            each with shape `(num_batch)`.
    """
    S_reduced, h, J, mask_sample_reduced = normalize_potts_inputs(
        S=S,
        h=h,
        J=J,
        mask_i=mask_i,
        mask_S=mask_sample,
    )

    # Initialize masked proposals and mask h
    mask_S, mask_mutatable, S_reduced = init_sampling_masks(-h, mask_sample_reduced, S_reduced)
    h_numerical_zero = h.max() + 1e3 * max(1.0, temperature)
    h = torch.where(mask_S > 0, h, h_numerical_zero * torch.ones_like(h))

    # Block update schedule
    if proposal == "chromatic":
        if edge_idx_coloring is None:
            edge_idx_coloring = edge_idx
        if mask_ij_coloring is None:
            mask_ij_coloring = mask_ij
        schedule = _color_graph(edge_idx_coloring, mask_ij_coloring)
        num_colors = schedule.max() + 1
        num_iterations = num_colors * num_sweeps
    else:
        num_iterations = num_sweeps

    num_iterations_annealing = int(annealing_fraction * num_iterations)
    temperatures = np.linspace(temperature_init, temperature, num_iterations_annealing).tolist() + [temperature] * (
        num_iterations - num_iterations_annealing
    )

    if proposal == "chromatic":
        _energy_proposal = lambda _S, _T: _potts_proposal_gibbs(
            _S,
            h,
            J,
            edge_idx,
            mask_i,
            mask_ij,
            T=_T,
            penalty_func=penalty_func,
            differentiable_penalty=differentiable_penalty,
        )
    elif proposal == "dlmc":
        _energy_proposal = lambda _S, _T: _potts_proposal_dlmc(
            _S,
            h,
            J,
            edge_idx,
            mask_i,
            mask_ij,
            T=_T,
            penalty_func=penalty_func,
            differentiable_penalty=differentiable_penalty,
        )
    else:
        raise NotImplementedError

    cumulative_sweeps = 0
    if return_trajectory:
        S_trajectory = []
        U_trajectory = []
    for i, T_i in enumerate(tqdm(temperatures, desc="Potts Sampling", leave=False)):
        # Cycle through Gibbs updates random sites to the update with fixed prob
        if proposal == "chromatic":
            mask_update = schedule.eq(i % num_colors)
        else:
            mask_update = torch.ones_like(S_reduced) > 0
        if mask_mutatable is not None:
            mask_update = mask_update * (mask_mutatable > 0)

        # Compute current energy and local conditionals
        U, logp = _energy_proposal(S_reduced, T_i)

        # Propose
        S_new = torch.distributions.categorical.Categorical(logits=logp).sample()
        S_new = torch.where(mask_update, S_new, S_reduced)

        # Metropolis-Hastings adjusment
        if rejection_step:

            def _flux(_U, _logp, _S):
                logp_transition = torch.gather(_logp, -1, _S[..., None])
                _logp_ij = (mask_update.float() * logp_transition[..., 0]).sum(1)
                flux = -_U / T_i + _logp_ij
                return flux

            U_new, logp_new = _energy_proposal(S_new, T_i)

            _flux_backward = _flux(U_new, logp_new, S_reduced)
            _flux_forward = _flux(U, logp, S_new)
            acc_ratio = torch.exp((_flux_backward - _flux_forward)).clamp(max=1.0)
            if verbose:  # and i % 100 == 0:
                print(
                    f"{(U_new - U).mean().item():0.2f}"
                    f"\t{(_flux_backward - _flux_forward).mean().item():0.2f}"
                    f"\t{acc_ratio.mean().item():0.2f}"
                )
            u = torch.bernoulli(acc_ratio)[..., None]
            S_reduced = torch.where(u > 0, S_new, S_reduced)
            cumulative_sweeps += (u * mask_update).sum(1).mean().item() / S_reduced.shape[1]
        else:
            S_reduced = S_new
            cumulative_sweeps += (mask_update).float().sum(1).mean().item() / S_reduced.shape[1]

        if return_trajectory and i % (thin_sweeps) == 0:
            S_trajectory.append(S_reduced)
            U_trajectory.append(U)

        U, _ = _compute_potts_energy_reduced(
            S=S_reduced, h=h, J=J, edge_idx=edge_idx, mask_i=mask_i, mask_ij=mask_ij
        )

    if verbose:
        print(f"Effective number of sweeps: {cumulative_sweeps}")
    S = map_potts_indices_to_af3(S_reduced)
    if return_trajectory:
        S_trajectory = [map_potts_indices_to_af3(S_i) for S_i in S_trajectory]
    if return_trajectory:
        return S, U, S_trajectory, U_trajectory
    else:
        return S, U


def init_sampling_masks(
    logits_init: torch.Tensor,
    mask_sample: Optional[torch.Tensor] = None,
    S: Optional[torch.LongTensor] = None,
    ban_S: Optional[list[int]] = None,
    pos_restrict_aatype: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Parse sampling masks and an initial sequence.

    Args:
        logits_init (torch.Tensor): Logits for sequence initialization with shape
            `(num_batch, num_nodes, alphabet)`.
        mask_sample (torch.Tensor, optional): Binary sampling mask indicating which
            positions are free to change with shape `(num_batch, num_nodes)` or which
            tokens are valid at each position with shape
            `(num_batch, num_nodes, alphabet)`. In the latter case, `mask_sample` will
            take priority over `S` except for positions in which `mask_sample` is
            all zero.
        S (torch.LongTensor optional): Initial sequence with shape
            `(num_batch, num_nodes)`.
        ban_S (list of int, optional): Optional list of alphabet indices to ban from
            all positions during sampling.
        pos_restrict_aatype (tuple of torch.Tensor, optional): Tuple of two tensors with shape
            `(num_batch, num_nodes)` and `(num_batch, num_nodes, alphabet)`, respectively,
            indicating which positions are restricted to certain aatypes and which aatypes
            are allowed at each position.

    Returns:
        mask_sample (torch.Tensor): Finalized position specific mask with shape
            `(num_batch, num_nodes, alphabet)`.
        S (torch.Tensor): Self-consistent initial `S` with shape
            `(num_batch, num_nodes)`.
    """

    if S is None and mask_sample is not None:
        raise Exception("To use masked sampling, please provide an initial S")

    if mask_sample is None:
        mask_S = torch.ones_like(logits_init)
    elif mask_sample.dim() == 2:
        # Position-restricted sampling
        mask_sample_expand = mask_sample[..., None].expand(logits_init.shape)
        O_init = F.one_hot(S, logits_init.shape[-1]).float()
        mask_S = mask_sample_expand + (1 - mask_sample_expand) * O_init
    elif mask_sample.dim() == 3:
        O_init = F.one_hot(S, logits_init.shape[-1]).float()
        # Mutation-restricted sampling
        mask_zero = (mask_sample.sum(-1, keepdim=True) == 0).float()
        mask_S = ((mask_zero * O_init + mask_sample) > 0).float()
    else:
        raise NotImplementedError

    # Handle aatype restrictions.
    if ban_S is not None:
        # Ban certain aatypes
        mask_S[:, :, ban_S] = 0.0

    if pos_restrict_aatype is not None:
        # Restrict to certain aatypes at certain positions.
        restrict_pos_mask, allowed_aatype_mask = pos_restrict_aatype  # (B, N), (B, N, K)
        mask_S[restrict_pos_mask.bool()] = allowed_aatype_mask[restrict_pos_mask.bool()]

    mask_S_1D = (mask_S.sum(-1) > 1).float()  # check where we can sample

    logits_init_masked = 1000 * mask_S + logits_init
    S_init = torch.distributions.categorical.Categorical(logits=logits_init_masked).sample()
    S = torch.where(mask_S_1D.bool(), S_init, S)  # where we can sample, set S to S_init
    S = torch.where(
        mask_S.sum(-1) == 1, mask_S.argmax(-1), S
    )  # where there is only one possible aatype, set S to the aatype
    return mask_S, mask_S_1D, S


def _potts_proposal_gibbs(S, h, J, edge_idx, mask_i, mask_ij, T=1.0, penalty_func=None, differentiable_penalty=True):
    U, U_i = _compute_potts_energy_reduced(S=S, h=h, J=J, edge_idx=edge_idx, mask_i=mask_i, mask_ij=mask_ij)

    if penalty_func is not None:
        if differentiable_penalty:
            with torch.enable_grad():
                S_onehot = F.one_hot(S, h.shape[0 - 1]).float()
                S_onehot.requires_grad = True
                U_penalty = penalty_func(S_onehot)
                U_i_adjustment = torch.autograd.grad(U_penalty.sum(), [S_onehot])[0].detach()
                U_penalty = U_penalty.detach()
            U_i = U_i + 0.5 * U_i_adjustment
        else:
            U_penalty = penalty_func(S_onehot)
        U = U + U_penalty

    logp_i = F.log_softmax(-U_i / T, dim=-1)
    return U, logp_i


def _potts_proposal_dlmc(
    S,
    h,
    J,
    edge_idx,
    mask_i,
    mask_ij,
    T=1.0,
    penalty_func=None,
    differentiable_penalty=True,
    dt=0.1,
    autoscale=True,
    balancing_func="sigmoid",
):
    # Compute energy gap
    U, U_i = _compute_potts_energy_reduced(S=S, h=h, J=J, edge_idx=edge_idx, mask_i=mask_i, mask_ij=mask_ij)
    # print(U)
    U_i = U_i
    if penalty_func is not None:
        O = F.one_hot(S, h.shape[0 - 1]).float()
        if differentiable_penalty:
            with torch.enable_grad():
                O.requires_grad = True
                U_penalty = penalty_func(O)
                U_i_adjustment = torch.autograd.grad(U_penalty.sum(), [O])[0].detach()
                U_penalty = U_penalty.detach()
                U_i_adjustment = U_i_adjustment - torch.gather(U_i_adjustment, -1, S[..., None])

            U_i_mutate = U_i - torch.gather(U_i, -1, S[..., None])
            U_i = U_i + U_i_adjustment
        else:
            U_penalty = penalty_func(O)
        U = U + U_penalty

    # Compute local equilibrium distribution
    logP_j = F.log_softmax(-U_i / T, dim=-1)

    # Compute transition log probabilities
    O = F.one_hot(S, h.shape[0 - 1]).float()
    logP_i = torch.gather(logP_j, -1, S[..., None])
    if balancing_func == "sqrt":
        log_Q_ij = 0.5 * (logP_j - logP_i)
    elif balancing_func == "sigmoid":
        log_Q_ij = F.logsigmoid(logP_j - logP_i)
    else:
        raise NotImplementedError

    rate = torch.exp(log_Q_ij - logP_j)

    # Compute transition probability
    logP_ij = logP_j + (-(-dt * rate).expm1()).log()
    p_flip = ((1.0 - O) * logP_ij.exp()).sum(-1, keepdim=True)

    # DEBUG:
    # flux = ((1. - O) * torch.exp(log_Q_ij)).mean([1,2], keepdim=True)
    # print(f" ->Flux is {flux.item():0.2f}, FlipProb is {p_flip.mean():0.2f}")

    logP_ii = (1.0 - p_flip).clamp(1e-5).log()
    logP_ij = (1.0 - O) * logP_ij + O * logP_ii
    return U, logP_ij


def _mask_J(edge_idx, mask_i, mask_ij):
    # Remove self edges
    device = edge_idx.device
    ii = torch.arange(edge_idx.shape[1]).view((1, -1, 1)).to(device)
    not_self = torch.ne(edge_idx, ii).type(torch.float32)

    # Remove missing edges
    self_present = mask_i.unsqueeze(-1)
    neighbor_present = graph.collect_neighbors(self_present, edge_idx)
    neighbor_present = neighbor_present.squeeze(-1)

    mask_J = not_self * self_present * neighbor_present
    if mask_ij is not None:
        mask_J = mask_ij * mask_J
    return mask_J


def log_composite_likelihood(
    S: torch.LongTensor,
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor,
    mask_ij: torch.Tensor,
    smoothing_alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Potts pairwise composite likelihoods from an AF3-indexed sequence.

    Inputs:
        S (torch.LongTensor): AF3-indexed sequence with shape `(num_batch, num_nodes)`.
        h (torch.Tensor): Field tensor with shape `(num_batch, num_nodes, C)`,
            where `C` is either `const.AF3_ENCODING.n_tokens` or
            `len(const.POTTS_TOKENS)`.
        J (Tensor): Coupling tensor with shape
            `(num_batch, num_nodes, num_neighbors, C, C)` using the same state
            dimension as `h`.
        edge_idx (torch.LongTensor): Edge indices with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_nodes)`
        mask_ij (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
        smoothing_alpha (float): Label smoothing probability on `(0,1)`.

    Outputs:
        logp_ij (torch.Tensor): Potts pairwise composite likelihoods evaluated
            for the current sequence with shape
            `(num_batch, num_nodes, num_neighbors)`.
        mask_p_ij (torch.Tensor): Edge mask with shape
            `(num_batch, num_nodes, num_neighbors)`.
    """
    S_reduced, h, J, _ = normalize_potts_inputs(S=S, h=h, J=J, mask_i=mask_i)
    return _log_composite_likelihood_reduced(
        S=S_reduced,
        h=h,
        J=J,
        edge_idx=edge_idx,
        mask_i=mask_i,
        mask_ij=mask_ij,
        smoothing_alpha=smoothing_alpha,
    )


def _log_composite_likelihood_reduced(
    *,
    S: torch.LongTensor,
    h: torch.Tensor,
    J: torch.Tensor,
    edge_idx: torch.LongTensor,
    mask_i: torch.Tensor,
    mask_ij: torch.Tensor,
    smoothing_alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute composite likelihood for reduced-state inputs.

    Args:
        S: Sequence tensor with shape `(num_batch, num_nodes)` indexed by
            `const.POTTS_TOKENS`.
        h: Field tensor with shape `(num_batch, num_nodes, len(const.POTTS_TOKENS))`.
        J: Coupling tensor with shape
            `(num_batch, num_nodes, num_neighbors, len(const.POTTS_TOKENS), len(const.POTTS_TOKENS))`.
        edge_idx: Edge indices with shape `(num_batch, num_nodes, num_neighbors)`.
        mask_i: Node mask with shape `(num_batch, num_nodes)`.
        mask_ij: Edge mask with shape `(num_batch, num_nodes, num_neighbors)`.
        smoothing_alpha: Label smoothing probability.
    """
    num_batch, num_residues, num_k, num_states, _ = list(J.size())

    # Gather J clamped at j
    # [Batch,i,j,A_i,A_j] => J_ij(:,A_j) [Batch,i,j,A_i]
    S_j = graph.collect_neighbors(S.unsqueeze(-1), edge_idx)
    S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, num_states, -1)
    # (B,i,j,A_i)
    J_clamp_j = torch.gather(J, -1, S_j).squeeze(-1)

    # Gather J clamped at i
    S_i = S.view(num_batch, num_residues, 1, 1, 1)
    S_i = S_i.expand(-1, -1, num_k, num_states, num_states)
    # (B,i,j,1,A_j)
    J_clamp_i = torch.gather(J, -2, S_i)

    # Compute background per-site contributions that sum out J
    # (B,i,j,A_i) => (B,i,A_i)
    r_i = h + J_clamp_j.sum(2)
    r_j = graph.collect_neighbors(r_i, edge_idx)

    # Remove J_ij from the i contributions
    # (B,i,A_i) => (B,i,:,A_i,:)
    r_i = r_i.view([num_batch, num_residues, 1, num_states, 1])
    r_i_minus_ij = r_i - J_clamp_j.unsqueeze(-1)

    # Remove J_ji from the j contributions
    # (B,j,A_j) => (B,:,j,:,A_j)
    r_j = r_j.view([num_batch, num_residues, num_k, 1, num_states])
    r_j_minus_ji = r_j - J_clamp_i

    # Composite likelihood (B,i,j,A_i,A_j)
    logits_ij = r_i_minus_ij + r_j_minus_ji + J
    logits_ij = logits_ij.view([num_batch, num_residues, num_k, -1])
    logp = F.log_softmax(-logits_ij, dim=-1)
    logp = logp.view([num_batch, num_residues, num_k, num_states, num_states])

    # Score the current sequence under
    # (B,i,j,A_i,A_j) => (B,i,j,A_i) => (B,i,j)
    logp_j = torch.gather(logp, -1, S_j).squeeze(-1)
    S_i = S.view(num_batch, num_residues, 1, 1).expand(-1, -1, num_k, -1)
    logp_ij = torch.gather(logp_j, -1, S_i).squeeze(-1)

    # Optional label smoothing (scaled assuming per-token smoothing )
    if smoothing_alpha > 0.0:
        # Foreground probability
        num_bins = num_states**2
        prob_no_smooth = (1.0 - smoothing_alpha) ** 2
        prob_background = (1.0 - prob_no_smooth) / float(num_bins - 1)
        # The second term corrects for double counting in background sum
        p_foreground = prob_no_smooth - prob_background
        logp_ij = p_foreground * logp_ij + prob_background * logp.sum([-2, -1])

    mask_p_ij = _mask_J(edge_idx, mask_i, mask_ij)
    logp_ij = mask_p_ij * logp_ij
    return logp_ij, mask_p_ij

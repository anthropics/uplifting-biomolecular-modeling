# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


from typing import Optional

import torch
import torch.nn as nn

from xfold.nn.primitives import Transition, OuterProductMean
from xfold.nn.triangle_multiplication import TriangleMultiplication
from xfold.nn.attention import GridSelfAttention, MSAAttention
from xfold.nn.diffusion_transformer import SelfAttention

from xfold import fastnn


class PairformerBlock(nn.Module):
    """Implements Algorithm 17 [Line2-Line8] in AF3
    Ref to: openfold/model/evoformer.py and protenix/model/modules/pairformer.py
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_pair: int = 128,
        c_single: int = 384,
        c_hidden_mul: int = 128,
        n_heads_pair: int = 4,
        num_intermediate_factor: int = 4,
        with_single: bool = True,
    ) -> None:
        """
        Args:
            n_heads (int, optional): number of head [for SelfAttention]. Defaults to 16.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_hidden_mul (int, optional): hidden dim [for TriangleMultiplicationOutgoing].
                Defaults to 128.
            n_heads_pair (int, optional): number of head [for TriangleAttention]. Defaults to 4.
        """
        super(PairformerBlock, self).__init__()
        self.n_heads = n_heads
        self.with_single = with_single
        self.num_intermediate_factor = num_intermediate_factor

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            c_pair=c_pair, _outgoing=True
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            c_pair=c_pair, _outgoing=False)
        self.pair_attention1 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=False
        )
        self.pair_attention2 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=True
        )
        self.pair_transition = Transition(
            c_x=c_pair, num_intermediate_factor=self.num_intermediate_factor)
        self.c_single = c_single
        if self.with_single is True:
            self.single_pair_logits_norm = fastnn.LayerNorm(c_pair)
            self.single_pair_logits_projection = nn.Linear(
                c_pair, n_heads, bias=False)
            self.single_attention_ = SelfAttention(
                c_x=c_single, num_head=n_heads, use_single_cond=False)
            self.single_transition = Transition(c_x=self.c_single)

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        single: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the PairformerBlock.

        Args:
            pair (torch.Tensor): [..., N_token, N_token, c_pair]
            pair_mask (torch.Tensor): [..., N_token, N_token]
            single (torch.Tensor, optional): [..., N_token, c_single]
            seq_mask (torch.Tensor, optional): [..., N_token]

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: pair, single
        """
        pair += self.triangle_multiplication_outgoing(pair, mask=pair_mask)
        pair += self.triangle_multiplication_incoming(pair, mask=pair_mask)
        pair += self.pair_attention1(pair, mask=pair_mask)
        pair += self.pair_attention2(pair, mask=pair_mask)
        pair += self.pair_transition(pair)

        if self.with_single is True:
            pair_logits = self.single_pair_logits_projection(
                self.single_pair_logits_norm(pair))

            pair_logits = pair_logits.permute(2, 0, 1)

            single += self.single_attention_(single,
                                             seq_mask,
                                             pair_logits=pair_logits)

            single += self.single_transition(single)
            return pair, single
        return pair


class EvoformerBlock(nn.Module):
    def __init__(self, c_msa: int = 64, c_pair: int = 128, n_heads_pair: int = 4) -> None:
        super(EvoformerBlock, self).__init__()

        self.outer_product_mean = OuterProductMean(
            c_msa=c_msa, num_output_channel=c_pair)
        self.msa_attention1 = MSAAttention(c_msa=c_msa, c_pair=c_pair)
        self.msa_transition = Transition(c_x=c_msa)

        self.triangle_multiplication_outgoing = TriangleMultiplication(
            c_pair=c_pair, _outgoing=True)
        self.triangle_multiplication_incoming = TriangleMultiplication(
            c_pair=c_pair, _outgoing=False)
        self.pair_attention1 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=False
        )
        self.pair_attention2 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=True
        )
        self.pair_transition = Transition(c_x=c_pair)

    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair += self.outer_product_mean(msa, msa_mask)

        msa += self.msa_attention1(msa, msa_mask, pair)
        msa += self.msa_transition(msa)

        pair += self.triangle_multiplication_outgoing(pair, mask=pair_mask)
        pair += self.triangle_multiplication_incoming(pair, mask=pair_mask)
        pair += self.pair_attention1(pair, mask=pair_mask)
        pair += self.pair_attention2(pair, mask=pair_mask)
        pair += self.pair_transition(pair)

        return msa, pair

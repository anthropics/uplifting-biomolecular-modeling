# Uses a simple pattern described here (https://github.com/nboyd/joltz/blob/main/src/joltz/__init__.py) to translate ProteinMPNN from torch to jax
# Would probably be a lot more readable if we didn't follow the original ProteinMPNN implementation so closely
#
# Derived from ProteinMPNN (https://github.com/dauparas/ProteinMPNN),
# Copyright (c) 2022 Justas Dauparas, MIT License. See ./NOTICE.

import equinox as eqx
import jax
import numpy as np
import torch
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int
from joltz.backend import (
    AbstractFromTorch,
    LayerNorm,
    Linear,
    register_from_torch,
    Embedding,
    from_torch,
)

from pathlib import Path

from . import torch_mpnn
import importlib


MPNN_ALPHABET = list("ACDEFGHIKLMNPQRSTVWYX")


@register_from_torch(torch_mpnn.PositionalEncodings)
class PositionalEncodings(AbstractFromTorch):
    num_embeddings: int
    max_relative_feature: int
    linear: eqx.nn.Linear

    def __call__(self, offset, mask):
        d = jnp.clip(
            offset + self.max_relative_feature,
            0,
            2 * self.max_relative_feature,
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = jax.nn.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        return self.linear(d_onehot)


# Not clear how important it is to exactly match pytorch gelu
def gelu(x, approximate: bool = False) -> Array:
    if approximate:
        sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x.dtype)
        cdf = 0.5 * (1.0 + jnp.tanh(sqrt_2_over_pi * (x + 0.044715 * (x**3))))
        return x * cdf
    else:
        sqrt_2 = np.sqrt(2).astype(x.dtype)
        return jnp.array(x * (jax.lax.erf(x / sqrt_2) + 1) / 2, dtype=x.dtype)


@register_from_torch(torch.nn.modules.activation.GELU)
class GELU(AbstractFromTorch):
    def __call__(self, x):
        return gelu(x)


@register_from_torch(torch.nn.modules.dropout.Dropout)
class Dropout(AbstractFromTorch):
    p: float
    inplace: bool
    training: bool

    def __call__(self, x, *, key=None):
        assert not self.inplace
        if not self.training:
            return x
        return eqx.nn.Dropout(p=self.p)(x, key=key)


@register_from_torch(torch_mpnn.PositionWiseFeedForward)
class PositionWiseFeedForward(AbstractFromTorch):
    W_in: Linear
    W_out: Linear
    act: GELU

    def __call__(self, h_V):
        return self.W_out(self.act(self.W_in(h_V)))


def gather_nodes(nodes, neighbor_idx):
    batch_size, num_nodes, num_channels = nodes.shape
    _, _, num_neighbors = neighbor_idx.shape

    # Flatten and expand indices
    neighbors_flat = neighbor_idx.reshape((batch_size, -1))

    # Create batch indices
    batch_idx = jnp.arange(batch_size)[:, None, None]
    batch_idx = jnp.broadcast_to(batch_idx, (batch_size, num_nodes, num_neighbors))
    batch_idx = batch_idx.reshape((batch_size, -1))

    # Gather features
    neighbor_features = nodes[batch_idx, neighbors_flat]
    neighbor_features = neighbor_features.reshape(
        batch_size, num_nodes, num_neighbors, num_channels
    )

    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    batch_size, num_nodes, num_channels = nodes.shape
    _, num_neighbors = neighbor_idx.shape

    # Create batch indices
    batch_idx = jnp.arange(batch_size)[:, None]
    batch_idx = jnp.broadcast_to(batch_idx, (batch_size, num_neighbors))

    # Gather features
    neighbor_features = nodes[batch_idx, neighbor_idx]

    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes_gathered = gather_nodes(h_nodes, E_idx)
    h_nn = jnp.concatenate([h_neighbors, h_nodes_gathered], axis=-1)
    return h_nn


@register_from_torch(torch_mpnn.EncLayer)
class EncLayer(AbstractFromTorch):
    num_hidden: int
    num_in: int
    scale: float
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    dropout3: eqx.nn.Dropout
    norm1: LayerNorm
    norm2: LayerNorm
    norm3: LayerNorm
    W1: Linear
    W2: Linear
    W3: Linear
    W11: Linear
    W12: Linear
    W13: Linear
    act: GELU
    dense: PositionWiseFeedForward

    def __call__(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)

        # Expand h_V to match h_EV's size
        _, _, num_neighbors, _ = h_EV.shape
        # h_V_expand = jnp.expand_dims(h_V, axis=-2)
        h_V_expand = h_V[:, :, None, :]
        h_V_expand = jnp.broadcast_to(
            h_V_expand,
            (h_EV.shape[0], h_EV.shape[1], h_EV.shape[2], h_V_expand.shape[-1]),
        )

        # Concatenate and process through MLP
        h_EV = jnp.concatenate([h_V_expand, h_EV], axis=-1)
        h_message = h_EV
        h_message = self.act(self.W1(h_message))
        h_message = self.act(self.W2(h_message))
        h_message = self.W3(h_message)

        # Apply attention mask if provided
        if mask_attend is not None:
            mask_attend = jnp.expand_dims(mask_attend, axis=-1)
            h_message = mask_attend * h_message

        # Aggregate messages and apply normalization
        dh = jnp.sum(h_message, axis=-2) / self.scale
        dh = self.dropout1(dh)
        h_V = self.norm1(h_V + dh)

        # Dense layer update
        dh = self.dense(h_V)
        dh = self.dropout2(dh)
        h_V = self.norm2(h_V + dh)

        # Apply node mask if provided
        if mask_V is not None:
            mask_V = jnp.expand_dims(mask_V, axis=-1)
            h_V = mask_V * h_V

        # Second branch - edge update
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = jnp.expand_dims(h_V, axis=-2)
        h_V_expand = jnp.broadcast_to(
            h_V_expand,
            (h_EV.shape[0], h_EV.shape[1], h_EV.shape[2], h_V_expand.shape[-1]),
        )
        h_EV = jnp.concatenate([h_V_expand, h_EV], axis=-1)

        # Process through second MLP
        h_message = h_EV
        h_message = self.act(self.W11(h_message))
        h_message = self.act(self.W12(h_message))
        h_message = self.W13(h_message)

        # Final edge update with normalization
        h_message = self.dropout3(h_message)  # Training mode
        h_E = self.norm3(h_E + h_message)

        return h_V, h_E


@register_from_torch(torch_mpnn.DecLayer)
class DecLayer(AbstractFromTorch):
    num_hidden: int
    num_in: int
    scale: float
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    norm1: LayerNorm
    norm2: LayerNorm
    W1: Linear
    W2: Linear
    W3: Linear
    act: GELU
    dense: PositionWiseFeedForward

    def __call__(self, h_V, h_E, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""
        # Expand h_V to match h_E's neighbor dimension
        _, _, num_neighbors, _ = h_E.shape
        h_V_expand = jnp.expand_dims(h_V, axis=-2)
        h_V_expand = jnp.broadcast_to(
            h_V_expand, (*h_V.shape[:-1], num_neighbors, h_V.shape[-1])
        )

        # Concatenate and process through MLP
        h_EV = jnp.concatenate([h_V_expand, h_E], axis=-1)
        h_message = h_EV
        h_message = self.act(self.W1(h_message))
        h_message = self.act(self.W2(h_message))
        h_message = self.W3(h_message)

        # Apply attention mask if provided
        if mask_attend is not None:
            mask_attend = jnp.expand_dims(mask_attend, axis=-1)
            h_message = mask_attend * h_message

        # Aggregate messages and apply normalization
        dh = jnp.sum(h_message, axis=-2) / self.scale
        dh = self.dropout1(dh)
        h_V = self.norm1(h_V + dh)

        # Position-wise feedforward
        dh = self.dense(h_V)
        dh = self.dropout2(dh)
        h_V = self.norm2(h_V + dh)

        # Apply node mask if provided
        if mask_V is not None:
            mask_V = jnp.expand_dims(mask_V, axis=-1)
            h_V = mask_V * h_V

        return h_V


@register_from_torch(torch_mpnn.ProteinFeatures)
class ProteinFeatures(AbstractFromTorch):
    edge_features: int
    node_features: int
    top_k: int
    augment_eps: float
    num_rbf: int
    num_positional_embeddings: int

    embeddings: PositionalEncodings
    edge_embedding: Linear
    norm_edges: LayerNorm

    def _get_edge_idx(self, X, mask, eps=1e-6):
        """get edge index
        input: mask.shape = (...,L), X.shape = (...,L,3)
        return: (...,L,k)
        """
        mask_2D = mask[..., None, :] * mask[..., :, None]
        dX = X[..., None, :, :] - X[..., :, None, :]
        D = jnp.sqrt(jnp.square(dX).sum(-1) + eps)
        D_masked = jnp.where(mask_2D, D, D.max(-1, keepdims=True))
        k = min(self.top_k, X.shape[-2])
        return jax.vmap(lambda v: jax.lax.top_k(-v, k))(D_masked)[1]

    def _rbf(self, D):
        """radial basis function (RBF)
        input: (...,L,k)
        output: (...,L,k,?)
        """
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = jnp.linspace(D_min, D_max, D_count)
        D_sigma = (D_max - D_min) / D_count
        return jnp.exp(-(((D[..., None] - D_mu) / D_sigma) ** 2))

    def _get_rbf(self, A, B, E_idx):
        D = jnp.sqrt(jnp.square(A[..., :, None, :] - B[..., None, :, :]).sum(-1) + 1e-6)
        D_neighbors = jnp.take_along_axis(D, E_idx, 1)
        return self._rbf(D_neighbors)

    def _call_single(
        self,
        X: Float[Array, "L 4 3"],
        residue_idx: Int[Array, " L"],
        chain_idx: Int[Array, " L"],
        mask: Bool[Array, " L"],
        *,
        key: jax.random.PRNGKey,
    ):
        if self.augment_eps > 0:
            key, use_key = jax.random.split(key)
            X = X + self.augment_eps * jax.random.normal(use_key, X.shape)

        ##########################
        # get atoms
        ##########################
        # N,Ca,C,O,Cb
        Y = X.swapaxes(0, 1)  # (length, atoms, 3) -> (atoms, length, 3)
        assert Y.shape[0] == 4
        # add Cb
        b, c = (Y[1] - Y[0]), (Y[2] - Y[1])
        Cb = -0.58273431 * jnp.cross(b, c) + 0.56802827 * b - 0.54067466 * c + Y[1]
        Y = jnp.concatenate([Y, Cb[None]], 0)

        ##########################
        # gather edge features
        ##########################
        # get edge indices (based on ca-ca distances)
        E_idx = self._get_edge_idx(Y[1], mask)

        # rbf encode distances between atoms
        # This is pretty ugly.
        edges = jnp.array(
            [
                [1, 1],
                [0, 0],
                [2, 2],
                [3, 3],
                [4, 4],
                [1, 0],
                [1, 2],
                [1, 3],
                [1, 4],
                [0, 2],
                [0, 3],
                [0, 4],
                [4, 2],
                [4, 3],
                [3, 2],
                [0, 1],
                [2, 1],
                [3, 1],
                [4, 1],
                [2, 0],
                [3, 0],
                [4, 0],
                [2, 4],
                [3, 4],
                [2, 3],
            ]
        )
        RBF_all = jax.vmap(lambda x: self._get_rbf(Y[x[0]], Y[x[1]], E_idx))(edges)
        RBF_all = RBF_all.transpose((1, 2, 0, 3))
        RBF_all = RBF_all.reshape(RBF_all.shape[:-2] + (-1,))

        # residue index offset
        offset = jnp.take_along_axis(
            residue_idx[:, None] - residue_idx[None, :], E_idx, 1
        )

        # chain index offset
        E_chains = (chain_idx[:, None] == chain_idx[None, :]).astype(int)
        E_chains = jnp.take_along_axis(E_chains, E_idx, 1)
        E_positional = self.embeddings(offset, E_chains)

        E = jnp.concatenate((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx

    def __call__(
        self,
        X: Float[Array, "B L 5 3"],
        mask: Bool[Array, "B L"],
        residue_idx: Int[Array, "B L"],
        chain_idx: Int[Array, "B L"],
        *,
        key: jax.random.PRNGKey,
    ):
        assert X.ndim == 4, X.shape
        assert residue_idx.ndim == 2
        B = X.shape[0]
        return jax.vmap(
            lambda X, r_idx, c_idx, m, k: self._call_single(X, r_idx, c_idx, m, key=k)
        )(X, residue_idx, chain_idx, mask, jax.random.split(key, B))


@register_from_torch(torch_mpnn.ProteinMPNN)
class ProteinMPNN(AbstractFromTorch):
    node_features: int
    edge_features: int
    hidden_dim: int

    features: ProteinFeatures
    W_e: Linear
    W_s: Embedding

    encoder_layers: list[EncLayer]
    decoder_layers: list[DecLayer]

    W_out: Linear

    def encode(
        self,
        *,
        X: Float[Array, "N 4 3"],
        mask: Bool[Array, " N"],
        residue_idx: Int[Array, " N"],
        chain_encoding_all: Int[Array, " N"],
        key
    ):
        # add batch dimension :/
        (X, mask, residue_idx, chain_encoding_all) = jax.tree.map(
            lambda x: x[None], (X, mask, residue_idx, chain_encoding_all)
        )
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all, key = key)
        h_V = jnp.zeros((E.shape[0], E.shape[1], E.shape[-1]))
        h_E = self.W_e(E)
        mask_attend = gather_nodes(mask[..., None], E_idx)[..., 0]
        mask_attend = mask[..., None] * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        return h_V, h_E, E_idx

    def decode(
        self,
        *, 
        S: Float[Array, "N 23"],
        h_V,
        h_E,
        E_idx,
        decoding_order: Float[Array, " N"],
        mask: Bool[Array, " N"],
    ):
        # add batch dim to S, decoding_order, mask
        S, decoding_order, mask  = jax.tree.map(lambda x: x[None], (S, decoding_order, mask))
        # Concatenate sequence embeddings for autoregressive decoder
        h_S = S @ self.W_s.weight
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        # generate a random decoding order
        decoding_order = jnp.argsort(
            decoding_order,
            axis=-1,
        )
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = jax.nn.one_hot(
            decoding_order, num_classes=mask_size
        )
        # turn the decoding order into an autoregressive mask
        order_mask_backward = jnp.einsum(
            "ij, biq, bjp->bqp",
            (1 - jnp.triu(jnp.ones((mask_size, mask_size)))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = jnp.take_along_axis(order_mask_backward, E_idx, axis=2)[..., None]
        mask_1D = mask.reshape((mask.shape[0], mask.shape[1], 1, 1))
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        return jax.nn.log_softmax(logits, axis=-1)

    def __call__(
        self,
        X: Float[Array, "N 4 3"],
        S: Float[Array, "N 23"],
        mask: Bool[Array, " N"],
        residue_idx: Int[Array, " N"],
        chain_encoding_all: Int[Array, " N"],
        decoding_order: Float[Array, " N"],
        *,
        key=None,
    ):
        """
        Computes log-probabilities of each amino acid at each position in the sequence.

        Args:

            X: Float[Array, "N 4 3"] - Coordinates of the atoms in the protein in the order N, C-alpha, C, O
            S: Float[Array, "N 23"] - Sequence as one-hot matrix
            mask: Bool[Array, " N"] - Mask of valid positions
            residue_idx: Int[Array, " N"] - Residue index *WITH* gaps of at least 100 between chains
            chain_encoding_all: Int[Array, " N"] - Chain index as int, e.g. [0 0 0 1 1 1 1 2 2 2]
            decoding_order: Float[Array, " N"] - Autoregressive decoding order

        Returns:

            Float[Array, "N 23"] - Log-probabilities of each amino acid at each position in the sequence

        """
        # add batch dimension :/
        (X, S, mask, residue_idx, chain_encoding_all, decoding_order) = jax.tree.map(
            lambda x: x[None],
            (X, S, mask, residue_idx, chain_encoding_all, decoding_order),
        )

        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all, key = key)

        h_V = jnp.zeros((E.shape[0], E.shape[1], E.shape[-1]))
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask[..., None], E_idx)[..., 0]
        mask_attend = mask[..., None] * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = S @ self.W_s.weight  # self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        # generate a random decoding order
        decoding_order = jnp.argsort(
            decoding_order,
            axis=-1,
        )
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = jax.nn.one_hot(
            decoding_order, num_classes=mask_size
        )
        # turn the decoding order into an autoregressive mask
        order_mask_backward = jnp.einsum(
            "ij, biq, bjp->bqp",
            (1 - jnp.triu(jnp.ones((mask_size, mask_size)))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        mask_attend = jnp.take_along_axis(order_mask_backward, E_idx, axis=2)[..., None]
        mask_1D = mask.reshape((mask.shape[0], mask.shape[1], 1, 1))
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        return jax.nn.log_softmax(logits, axis=-1)

    @staticmethod
    def from_pretrained(
        checkpoint_path: Path = importlib.resources.files(__package__) / "weights/v_48_020.pt",
        backbone_noise=0.00,
    ):
        checkpoint = torch.load(checkpoint_path)
        hidden_dim = 128
        num_layers = 3
        model = torch_mpnn.ProteinMPNN(
            num_letters=21,
            node_features=hidden_dim,
            edge_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            augment_eps=backbone_noise,
            k_neighbors=checkpoint["num_edges"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return from_torch(model)


def load_mpnn_sol(backbone_noise=0.0):
    return ProteinMPNN.from_pretrained(
        importlib.resources.files(__package__) / "weights/soluble_v_48_020.pt",
        backbone_noise=backbone_noise,
    )

def load_mpnn(backbone_noise=0.0):
    return ProteinMPNN.from_pretrained(
        importlib.resources.files(__package__) / "weights/v_48_020.pt",
        backbone_noise=backbone_noise,
    )

def load_abmpnn(backbone_noise=0.0):
    return ProteinMPNN.from_pretrained(
        importlib.resources.files(__package__) / "weights/abmpnn.pt",
        backbone_noise=backbone_noise,
    )
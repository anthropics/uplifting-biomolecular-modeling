# Log-likelihood losses for proteinMPNN
# 1. BoltzProteinMPNNLoss: Average log-likelihood of soft binder sequence given Boltz-predicted complex structure
# 2. FixedChainInverseFoldingLL: Average log-likelihood of fixed monomer sequence given fixed monomer structure

import gemmi
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from ..common import TOKENS, LossTerm
from ..proteinmpnn.mpnn import MPNN_ALPHABET, ProteinMPNN
from .structure_prediction import StructureModelOutput


def boltz_to_mpnn_matrix():
    """Converts from standard tokenization to ProteinMPNN tokenization"""
    T = np.zeros((len(TOKENS), len(MPNN_ALPHABET)))
    for i, tok in enumerate(TOKENS):
        mpnn_idx = MPNN_ALPHABET.index(tok)
        T[i, mpnn_idx] = 1
    return T


def _per_chain_residue_idx(asym_id, residue_idx):
    """Mosaic per-chain residue indexing with a 100-residue gap.

    Adjusts global residue indices to be per-chain, then offsets each chain
    by 100 to match ProteinMPNN's training convention. Hardcodes 16 max chains.
    """
    chain_lengths = (asym_id[:, None] == jnp.arange(16)[None]).sum(-2)
    res_idx_adjustment = jnp.cumsum(chain_lengths, -1) - chain_lengths
    return (
        residue_idx
        + (asym_id[:, None] == jnp.arange(16)[None]) @ res_idx_adjustment
    ) + 100 * asym_id


def load_chain(chain: gemmi.Chain) -> tuple[str, Float[Array, "N 4 3"]]:
    coords = np.zeros((len(chain), 4, 3))

    def _set_coords(idx: int, atom_idx: int, atom_name: str):
        try:
            atom = chain[idx].sole_atom(atom_name)
            pos = atom.pos
            coords[idx, atom_idx, 0] = pos.x
            coords[idx, atom_idx, 1] = pos.y
            coords[idx, atom_idx, 2] = pos.z
        except Exception:
            print(f"Failed to get {atom_name} for residue {chain[idx].name}")
            coords[idx, atom_idx] = np.nan

    for idx in range(len(chain)):
        _set_coords(idx, 0, "N")
        _set_coords(idx, 1, "CA")
        _set_coords(idx, 2, "C")
        _set_coords(idx, 3, "O")

    return gemmi.one_letter_code([r.name for r in chain]), coords


class FixedStructureInverseFoldingLL(LossTerm):
    sequence_boltz: Float[Array, "N 20"]
    mpnn: ProteinMPNN
    encoded_state: tuple
    name: str
    stop_grad: bool = False

    def __call__(
        self,
        binder_sequence: Float[Array, "N 20"],
        *,
        key,
    ):
        binder_length = binder_sequence.shape[0]
        complex_length = self.sequence_boltz.shape[0]
        # assert self.coords.shape[0] == self.encoded_state.shape[1], "Sequence length mismatch"

        # replace binder sequence
        sequence = self.sequence_boltz.at[:binder_length].set(binder_sequence)

        sequence_mpnn = sequence @ boltz_to_mpnn_matrix()
        mpnn_mask = jnp.ones(complex_length, dtype=jnp.int32)

        # generate a decoding order that ends with binder
        decoding_order = jax.random.uniform(key, shape=(complex_length,))
        decoding_order = decoding_order.at[:binder_length].add(2.0)
        logits = self.mpnn.decode(
            S=sequence_mpnn,
            h_V=self.encoded_state[0],
            h_E=self.encoded_state[1],
            E_idx=self.encoded_state[2],
            mask=mpnn_mask,
            decoding_order=decoding_order,
        )[0]
        if self.stop_grad:
            logits = jax.lax.stop_gradient(logits)

        ll = (logits * sequence_mpnn).sum(-1)[:binder_length].mean()

        return -ll, {f"{self.name}_ll": ll}

    @staticmethod
    def from_structure(
        st: gemmi.Structure,
        mpnn: ProteinMPNN,
        stop_grad: bool = False,
    ):
        st = st.clone()
        st.remove_ligands_and_waters()
        st.remove_alternative_conformations()
        st.remove_empty_chains()
        model = st[0]

        sequences_and_coords = [load_chain(c) for c in model]

        residue_idx = np.concatenate(
            [
                np.arange(len(s)) + chain_idx * 100
                for (chain_idx, (s, _)) in enumerate(sequences_and_coords)
            ]
        )

        chain_encoding = np.concatenate(
            [
                np.ones(len(s)) * chain_idx
                for (chain_idx, (s, _)) in enumerate(sequences_and_coords)
            ]
        )
        coords = np.concatenate([c for (_, c) in sequences_and_coords])
        # encode the structure
        h_V, h_E, E_idx = mpnn.encode(
            X=coords,
            mask=jnp.ones(coords.shape[0], dtype=jnp.int32),
            residue_idx=residue_idx,  # jnp.arange(len(chain)),
            chain_encoding_all=chain_encoding,  # jnp.zeros(len(chain), dtype=jnp.int32),
            key=jax.random.key(np.random.randint(1000000)),
        )
        # one hot sequence
        full_sequence = "".join(s for (s, _) in sequences_and_coords)

        return FixedStructureInverseFoldingLL(
            sequence_boltz=jax.nn.one_hot(
                [TOKENS.index(AA) if AA in TOKENS else 0 for AA in full_sequence], 20
            ),
            mpnn=mpnn,
            encoded_state=(h_V, h_E, E_idx),
            name=st.name,
            stop_grad=stop_grad,
        )


class ProteinMPNNLoss(LossTerm):
    """Average log-likelihood of binder sequence given predicted complex structure

    Args:

        mpnn: ProteinMPNN
        num_samples: int
        stop_grad: bool = True : Whether to stop gradient through the structure module output

    """

    mpnn: ProteinMPNN
    num_samples: int
    stop_grad: bool = True

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        # Get the atoms required for proteinMPNN:
        # In order these are N, C-alpha, C, O
        coords = output.backbone_coordinates
        if self.stop_grad:
            coords = jax.lax.stop_gradient(coords)

        binder_length = sequence.shape[0]

        # NOTE: this will completely fail if any tokens are non-protein!
        # all_atom_coords = structure_output.sample_atom_coords
        # coords = jnp.stack([all_atom_coords[first_atom_idx + i] for i in range(4)], -2)
        full_sequence = output.full_sequence.at[:binder_length].set(sequence)
        total_length = full_sequence.shape[0]

        sequence_mpnn = full_sequence @ boltz_to_mpnn_matrix()
        mpnn_mask = jnp.ones(total_length, dtype=jnp.int32)
        residue_idx = _per_chain_residue_idx(output.asym_id, output.residue_idx)

        h_V, h_E, E_idx = self.mpnn.encode(
            X=coords,
            mask=mpnn_mask,
            residue_idx=residue_idx,
            chain_encoding_all=output.asym_id,
            key=key,
        )

        def decoder_LL(key):
            # MPNN is cheap, let's call the decoder a few times to average over random decoding order
            # generate a decoding order
            # this should be random but end with the binder
            decoding_order = (
                jax.random.uniform(key, shape=(total_length,))
                .at[:binder_length]
                .add(2.0)
            )

            logits = self.mpnn.decode(
                S=sequence_mpnn,
                h_V=h_V,
                h_E=h_E,
                E_idx=E_idx,
                mask=mpnn_mask,
                decoding_order=decoding_order,
            )[0]

            return (
                (logits[:binder_length] * (sequence @ boltz_to_mpnn_matrix()))
                .sum(-1)
                .mean()
            )

        binder_ll = (
            jax.vmap(decoder_LL)(jax.random.split(key, self.num_samples))
        ).mean()

        return -binder_ll, {"protein_mpnn_ll": binder_ll}

# TODO: implement autoregressive sampling
# for now though the jacobi method converges quickly enough
def inverse_fold(
    mpnn: ProteinMPNN,
    binder_length: int,
    output: StructureModelOutput,
    temp: float,
    key,
    jacobi_iterations: int = 10,
    bias: Float[Array, "N 20"] | None = None,
):
    coords = output.backbone_coordinates

    total_length = output.full_sequence.shape[0]

    mpnn_mask = jnp.ones(total_length, dtype=jnp.int32)
    residue_idx = _per_chain_residue_idx(output.asym_id, output.residue_idx)

    h_V, h_E, E_idx = mpnn.encode(
        X=coords,
        mask=mpnn_mask,
        residue_idx=residue_idx,
        chain_encoding_all=output.asym_id,
        key=key,
    )

    decoding_order = (
        jax.random.uniform(key, shape=(total_length,)).at[:binder_length].add(2.0)
    )

    gumbel = jax.random.gumbel(key, (binder_length, 20))

    def seq_to_logits(sequence: Int[Array, " N"]):
        full_sequence = output.full_sequence.at[:binder_length].set(
            jax.nn.one_hot(sequence, 20, dtype=jnp.int32)
        )

        sequence_mpnn = full_sequence @ boltz_to_mpnn_matrix()

        logits = mpnn.decode(
            S=sequence_mpnn,
            h_V=h_V,
            h_E=h_E,
            E_idx=E_idx,
            mask=mpnn_mask,
            decoding_order=decoding_order,
        )[0]

        return logits[:binder_length] @ boltz_to_mpnn_matrix().T

    sequence = jax.random.randint(key = key, minval=0, maxval=20, shape=binder_length)

    def step(sequence, _):
        logits = seq_to_logits(sequence)
        if bias is not None:
            logits += bias
        sequence = (logits + temp * gumbel).argmax(-1)
        return sequence, None

    sequence, _ = jax.lax.scan(step, sequence, length=jacobi_iterations)

    return sequence


class InverseFoldingSequenceRecovery(LossTerm):
    """
        Inner product of binder sequence and average sequence from ProteinMPNN
        Bit of an odd loss; essentially moves the binder sequence towards the average sequence predicted by ProteinMPNN for the current structure.
        Can be thought of as a continuous version of AF2Cycler.

    Args:
        mpnn: ProteinMPNN instance
        temp: temperature for sampling MPNN
        num_samples: number of samples to average over

    """

    mpnn: ProteinMPNN
    temp: Float
    num_samples: int = 16
    jacobi_iterations: int = 10
    bias: Float[Array, "N 20"]  = None

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        sequences = jax.vmap(
            lambda k: jax.nn.one_hot(
                inverse_fold(
                    self.mpnn,
                    binder_length=sequence.shape[0],
                    output=output,
                    temp=self.temp,
                    key=k,
                    jacobi_iterations=self.jacobi_iterations,
                    bias = self.bias,
                ),
                20,
            )
        )(jax.random.split(key, self.num_samples))
        average_sequence = sequences.mean(0)
        average_sequence = jax.lax.stop_gradient(average_sequence)
        ip = (average_sequence * sequence).sum(-1).mean()
        return -ip, {"sequence_recovery": ip}


class AllResiduePLLLoss(LossTerm):
    """Negative mean per-residue MPNN pseudo-log-likelihood over the binder.

    For each binder position `i`, the decoder is run with `i` placed last in
    the autoregressive order so it conditions on every other residue (target
    chain + all other binder positions). The resulting log-probability
    distribution at `i` is mapped back to the 20-letter mosaic alphabet and
    dotted with `sequence[i]` — the PSSM-weighted conditional log-likelihood.
    The loss is the negative mean over binder positions.

    The encoder runs once. The decoder runs `binder_length` times via
    `jax.lax.map(..., batch_size=chunk_size)` with `jax.checkpoint`, so peak
    memory is bounded at one chunk's worth of activations.

    Decoding-order trick: a uniform base order in `[0, 1)` is offset by
    `+2.0` on binder positions (floats them past the target) and then the
    target position is set to `4.0` so it decodes strictly last.

    `chunk_size` controls how many binder positions' decoder calls are
    vmapped together inside `jax.lax.map`. Larger = faster, more memory.
    """

    mpnn: ProteinMPNN
    chunk_size: int = 10
    name: str = "pll"

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_length = sequence.shape[0]
        total_length = output.full_sequence.shape[0]

        b2m = boltz_to_mpnn_matrix()  # [20, 21] numpy
        mpnn_mask = jnp.ones(total_length, dtype=jnp.int32)
        residue_idx = _per_chain_residue_idx(output.asym_id, output.residue_idx)

        # Full sequence in MPNN tokens: target from output, binder from current PSSM.
        full_seq = output.full_sequence.at[:binder_length].set(sequence)
        sequence_mpnn = full_seq @ b2m

        encode_key, order_key = jax.random.split(key)
        h_V, h_E, E_idx = self.mpnn.encode(
            X=output.backbone_coordinates,
            mask=mpnn_mask,
            residue_idx=residue_idx,
            chain_encoding_all=output.asym_id,
            key=encode_key,
        )

        base_order = (
            jax.random.uniform(order_key, (total_length,))
            .at[:binder_length].add(2.0)
        )

        @jax.checkpoint
        def per_position_pll(i):
            decoding_order = base_order.at[i].set(
                jnp.asarray(4.0, dtype=base_order.dtype)
            )
            log_p = self.mpnn.decode(
                S=sequence_mpnn, h_V=h_V, h_E=h_E, E_idx=E_idx,
                mask=mpnn_mask, decoding_order=decoding_order,
            )[0, i]                              # [21] log-probs at i
            return jnp.dot(log_p @ b2m.T, sequence[i])

        plls = jax.lax.map(
            per_position_pll, jnp.arange(binder_length), batch_size=self.chunk_size,
        )
        mean_pll = plls.mean()
        return -mean_pll, {"pll": mean_pll, "pseudo_perplexity": jnp.exp(-mean_pll)}

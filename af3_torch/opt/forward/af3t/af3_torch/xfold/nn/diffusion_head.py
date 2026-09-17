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


import torch
import torch.nn as nn

from xfold import feat_batch
from xfold.nn import featurization, utils
from xfold.nn.diffusion_transformer import DiffusionTransformer, DiffusionTransition
from xfold.nn.atom_cross_attention import AtomCrossAttEncoder, AtomCrossAttDecoder

from xfold import fastnn
from xfold import of3
from xfold.nn import fourier_constants

# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0


class FourierEmbeddings(nn.Module):
    """cos(2*pi*(x*w+b)).  AF3 layout: w,b are the fixed constants of noise_level_embeddings.py;
    OF3 layout: w,b are parameters (diffusion_head/fourier_embedding_weight|bias) loaded by the converter."""
    def __init__(self, dim: int):
        super(FourierEmbeddings, self).__init__()
        self.dim = dim
        self.register_buffer("weight", torch.tensor(fourier_constants.WEIGHT, dtype=torch.float32))
        self.register_buffer("bias", torch.tensor(fourier_constants.BIAS, dtype=torch.float32))
        assert self.weight.shape[0] == dim and self.bias.shape[0] == dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(2 * torch.pi * (x[..., None] * self.weight + self.bias))


def noise_schedule(t, smin=0.0004, smax=160.0, p=7):
    return (
        SIGMA_DATA
        * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p
    )


def random_rotation(device, dtype):
    # Create a random rotation (Gram-Schmidt orthogonalization of two
    # random normal vectors)
    v0, v1 = torch.randn(size=(2, 3), dtype=dtype, device=device)
    e0 = v0 / torch.clamp(torch.linalg.norm(v0), min=1e-10)
    v1 = v1 - e0 * torch.dot(v1, e0)
    e1 = v1 / torch.clamp(torch.linalg.norm(v1), min=1e-10)
    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2])


def random_augmentation(
    positions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply random rigid augmentation.

    Args:
      positions: atom positions of shape (<common_axes>, 3)
      mask: per-atom mask of shape (<common_axes>,)

    Returns:
      Transformed positions with the same shape as input positions.
    """

    center = utils.mask_mean(
        mask[..., None], positions, dim=(-2, -3), keepdim=True, eps=1e-6
    )
    rot = random_rotation(device=positions.device, dtype=positions.dtype)
    translation = torch.randn(
        size=(3,), dtype=positions.dtype, device=positions.device)

    # The rigid transform acts on ABSOLUTE coordinates (tens of Å): it runs in fp32 whatever autocast policy the caller holds.
    # torch.einsum is an autocast-lowered op — under the runner's bf16 autocast it would round every coordinate to bfloat16
    # (0.125 Å steps at 16-32 Å from the centre) at each of the sampler's steps; JAX AlphaFold 3 applies this rotation in fp32.
    with torch.autocast(device_type=positions.device.type, enabled=False):
        augmented_positions = (
            torch.einsum(
                '...i,ij->...j',
                (positions - center).to(torch.float32),
                rot.to(torch.float32),
            )
            + translation
        )
    return augmented_positions * mask[..., None]


def augment_and_noise_batched(positions: torch.Tensor, mask: torch.Tensor, noise_scale: torch.Tensor, plan: "DrawPlan",
                              step_idx: int, s0: int, n: int) -> torch.Tensor:
    """Lever 'prologue': the sample-batched step's per-sample prologue — random_augmentation(positions[j], mask) + noise_scale * the step
    noise, for the b samples of `positions` [b, N, A, 3] — with every random DRAW made exactly as the serial statements make it (per sample,
    after plan.seek(s0 + j, step): random_rotation's randn((2, 3)), random_augmentation's randn((3,)), the noise randn((rows, A, 3)) —
    the same calls at the same Philox positions, so the same bits, and the generator left where they leave it) and the ARITHMETIC restated
    once over the sample axis (Gram-Schmidt, centring, the fp32 rigid transform, the mask, the noise lay-up and add: 5 x ~12 small launches
    -> ~12). Numerics: the batched dot / norm / einsum may round differently from the per-sample statements in the last bit (the tolerance
    class, augmentation only; the noise term is bit-identical)."""
    b = int(positions.shape[0])
    dev, dt = positions.device, positions.dtype
    vs, ts, zs = [], [], []
    for j in range(b):
        plan.seek(s0 + j, step_idx)                                                   # the generator where stock's loop stands before this (sample, step)
        vs.append(torch.randn(size=(2, 3), dtype=dt, device=dev))                   # random_rotation's draw
        ts.append(torch.randn(size=(3,), dtype=dt, device=dev))                     # random_augmentation's translation draw
        zs.append(torch.randn(size=(plan.rows,) + tuple(positions.shape[2:]), device=noise_scale.device))   # the step noise draw
    V = torch.stack(vs); T = torch.stack(ts); Z = torch.stack(zs)                    # [b, 2, 3], [b, 3], [b, rows, A, 3]
    v0, v1 = V[:, 0], V[:, 1]                                                        # random_rotation, per sample along the leading axis
    e0 = v0 / torch.clamp(torch.linalg.norm(v0, dim=-1, keepdim=True), min=1e-10)
    v1 = v1 - e0 * (v1 * e0).sum(dim=-1, keepdim=True)
    e1 = v1 / torch.clamp(torch.linalg.norm(v1, dim=-1, keepdim=True), min=1e-10)
    e2 = torch.cross(e0, e1, dim=-1)
    rot = torch.stack([e0, e1, e2], dim=-2)                                           # [b, 3, 3]: rows e0, e1, e2 (torch.stack([e0, e1, e2]) per sample)
    m = mask.reshape((1,) * (positions.dim() - 1 - mask.dim()) + tuple(mask.shape))[..., None]   # [1, N, A, 1]: the atom mask at positions' rank
    center = utils.mask_mean(m, positions, dim=(-2, -3), keepdim=True, eps=1e-6)      # [b, 1, 1, 3] per sample (random_augmentation's centre)
    with torch.autocast(device_type=dev.type, enabled=False):                        # the rigid transform in fp32 (random_augmentation's own rule)
        aug = torch.einsum('b...i,bij->b...j', (positions - center).to(torch.float32), rot.to(torch.float32)) + T.reshape((b,) + (1,) * (positions.dim() - 2) + (3,))
    aug = aug * m
    return aug + noise_scale * lay_rows(Z, n, 1)


def lay_rows(draw: torch.Tensor, n: int, dim: int) -> torch.Tensor:
    """`draw` (C rows on its token axis `dim`) laid into `n` token rows: itself when n == C, else zero rows appended (padding tokens get no
    draw and are masked out of every output). The sample-batched sampler's draws are made at AlphaFold3._draw_rows rows (stock: n)."""
    C = int(draw.shape[dim])
    if n == C:
        return draw
    pad = list(draw.shape); pad[dim] = n - C
    return torch.cat([draw, torch.zeros(pad, dtype=draw.dtype, device=draw.device)], dim=dim)


class DrawPlan:
    """Where stock's serial sampler makes each random draw — so the sample-batched sampler (lever 'sbatch') makes the SAME draws.

    Stock's stream order (AlphaFold3._sample_diffusion / _apply_denoising_step): ONE initial draw ``randn((S, rows, *atoms, 3))``, then for
    sample s, for step t, the unit: ``random_rotation``'s ``randn((2, 3))``, ``random_augmentation``'s ``randn((3,))``, the step noise
    ``randn((rows, *atoms, 3))``. The CUDA generator is Philox — counter-based: a call's values depend only on (seed, offset, the call), and
    each call advances the offset by an amount that depends only on the call — so unit (s, t) begins at ``base + (s*T + t) * unit`` wherever
    it falls in program order. ``initial()`` issues the initial draw and MEASURES ``unit`` (two units issued on the live generator, required
    equal, then rewound: the measuring draws never happened); ``seek(s, t)`` puts the generator at the unit's start and the caller issues
    stock's own three calls (stock's bits); ``finish()`` leaves it where stock's loop ends (``base + S*T*unit``). A generator whose offset
    cannot be set or whose advance is not constant is served by ``kind = 'state_walk'``: the state before every unit recorded once in stock's
    order (values discarded) and restored per unit — the same guarantee at the price of S*T throw-away units. Nothing here pre-draws the
    trajectory's noise: memory is stock's."""

    def __init__(self, device, num_samples: int, steps: int, rows: int, atom_shape, dtype=torch.float32):
        self.device = torch.device(device); self.S = int(num_samples); self.T = int(steps); self.rows = int(rows); self.dtype = dtype
        self.noise_shape = (self.rows,) + tuple(int(a) for a in atom_shape) + (3,)
        self.gen = None
        if self.device.type == "cuda":
            idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
            self.gen = torch.cuda.default_generators[idx]
        self.kind = "philox_offsets" if (self.gen is not None and hasattr(self.gen, "get_offset") and hasattr(self.gen, "set_offset")) else "state_walk"
        self.base = self.unit = None; self.states = None; self.final_state = None

    # -- stock's calls (values discarded: only the generator's advance matters) --
    def _unit_draws(self):
        torch.randn(size=(2, 3), dtype=self.dtype, device=self.device)      # random_rotation
        torch.randn(size=(3,), dtype=self.dtype, device=self.device)        # random_augmentation's translation
        torch.randn(size=self.noise_shape, device=self.device)              # the step noise

    def _get_state(self):
        return self.gen.get_state() if self.gen is not None else torch.get_rng_state()

    def _set_state(self, state):
        if self.gen is not None:
            self.gen.set_state(state)
        else:
            torch.set_rng_state(state)

    def initial(self) -> torch.Tensor:
        """Stock's one initial draw for all samples, ``[S, rows, *atoms, 3]``; the plan is measured (or walked) right after it and the generator
        rewound to just after it."""
        if self.kind == "philox_offsets":
            g = self.gen
            x = torch.randn((self.S,) + self.noise_shape, device=self.device)
            self.base = int(g.get_offset())
            self._unit_draws(); u1 = int(g.get_offset()) - self.base
            self._unit_draws(); u2 = int(g.get_offset()) - self.base - u1
            if u1 == u2 and u1 > 0 and u1 % 4 == 0 and self.base % 4 == 0:
                self.unit = u1
                g.set_offset(self.base)                                      # rewound: the two measuring units never happened
                return x
            self.kind = "state_walk"                                         # not a constant advance on this stack: walk the states instead (re-issue from the start)
            g.set_offset(self.base)
            return self._walk(x)
        x = torch.randn((self.S,) + self.noise_shape, device=self.device)
        return self._walk(x)

    def _walk(self, x):
        self.states = []
        for _s in range(self.S):
            row = []
            for _t in range(self.T):
                row.append(self._get_state()); self._unit_draws()
            self.states.append(row)
        self.final_state = self._get_state()
        if self.S and self.T:
            self._set_state(self.states[0][0])
        return x

    def seek(self, s: int, t: int) -> None:
        """Put the generator where stock's loop stands before unit (sample s, step t)'s three draws."""
        if self.kind == "philox_offsets":
            self.gen.set_offset(self.base + (int(s) * self.T + int(t)) * self.unit)
        else:
            self._set_state(self.states[s][t])

    def finish(self) -> None:
        """Leave the generator where stock's serial loop leaves it (after sample S-1's step T-1)."""
        if self.kind == "philox_offsets":
            self.gen.set_offset(self.base + self.S * self.T * self.unit)
        else:
            self._set_state(self.final_state)


GRAPH_CAPTURE_RETRIES = 2      # failed whole-step captures retried (each after a synchronize + empty_cache and one more eager step, on the step after that) before the step-graph lever steps aside for the process


def _is_oom(e: BaseException) -> bool:
    """A CUDA out-of-memory error, by type (torch.OutOfMemoryError / torch.cuda.OutOfMemoryError) or the caching allocator's message."""
    types_ = tuple(t for t in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)) if isinstance(t, type))
    return (bool(types_) and isinstance(e, types_)) or "CUDA out of memory" in str(e)


def _end_failed_capture(graph, caller_stream) -> None:
    """After a failed whole-step capture, put the device back in eager order: end a capture the failure left open on the current stream,
    return to the caller's stream (torch.cuda.graph leaves its capture stream current when ending the capture raised), drop the partial
    graph and its pool, clear the pending error. Each step is best effort; the eager step that follows reports anything still wrong."""
    for step in (lambda: torch.cuda.is_current_stream_capturing() and graph is not None and graph.capture_end(),
                 lambda: torch.cuda.set_stream(caller_stream),
                 lambda: graph is not None and graph.reset(),
                 torch.cuda.synchronize,
                 torch.cuda.empty_cache):
        try:
            step()
        except Exception:
            pass


class DiffusionHead(nn.Module):
    def __init__(self):
        super(DiffusionHead, self).__init__()

        self.c_act = 768
        self.pair_channel = 128
        self.seq_channel = 384

        self.c_pair_cond_initial = 267
        self.pair_cond_initial_norm = fastnn.LayerNorm(
            self.c_pair_cond_initial, bias=False)
        self.pair_cond_initial_projection = nn.Linear(
            self.c_pair_cond_initial, self.pair_channel, bias=False)

        self.pair_transition_0 = DiffusionTransition(
            self.pair_channel, c_single_cond=None)
        self.pair_transition_1 = DiffusionTransition(
            self.pair_channel, c_single_cond=None)

        # OF3's single conditioning LayerNorm was trained on 833 channels (AF3: 831); the JAX fork re-inserts two zero columns.
        self.c_single_cond_initial = 833 if of3.OF3 else 831
        self.single_cond_initial_norm = fastnn.LayerNorm(
            self.c_single_cond_initial, bias=False)
        self.single_cond_initial_projection = nn.Linear(
            self.c_single_cond_initial, self.seq_channel, bias=False)

        self.c_noise_embedding = 256
        self.noise_embedding_initial_norm = fastnn.LayerNorm(
            self.c_noise_embedding, bias=False)
        self.noise_embedding_initial_projection = nn.Linear(
            self.c_noise_embedding, self.seq_channel, bias=False)

        self.single_transition_0 = DiffusionTransition(
            self.seq_channel, c_single_cond=None)
        self.single_transition_1 = DiffusionTransition(
            self.seq_channel, c_single_cond=None)

        self.atom_cross_att_encoder = AtomCrossAttEncoder(per_token_channels=self.c_act,
                                                          with_token_atoms_act=True,
                                                          with_trunk_pair_cond=True,
                                                          with_trunk_single_cond=True)

        self.single_cond_embedding_norm = fastnn.LayerNorm(
            self.seq_channel, bias=False)
        self.single_cond_embedding_projection = nn.Linear(
            self.seq_channel, self.c_act, bias=False)

        self.transformer = DiffusionTransformer()

        self.output_norm = fastnn.LayerNorm(self.c_act, bias=False)

        self.atom_cross_att_decoder = AtomCrossAttDecoder()

        self.fourier_embeddings = FourierEmbeddings(dim=256)

    def _pair_conditioning(self, batch, embeddings, use_conditioning: bool) -> torch.Tensor:
        """Step-invariant part of the conditioning (depends only on embeddings['pair'] and token features)."""
        rows = int(getattr(self, "pair_cond_rows", 0) or 0)          # big lever 'paircond_chunk' (af3_torch_opt/big.py): the same statements per block of
        if rows and int(embeddings['pair'].shape[0]) > rows:          # `rows` rows of the token grid into one preallocated result (xfold/nn/paircond_rows.py)
            from xfold.nn import paircond_rows as _paircond_rows
            return _paircond_rows.pair_conditioning_rows(self, batch, embeddings, use_conditioning, rows)
        pair_embedding = use_conditioning * embeddings['pair']
        rel_features = featurization.create_relative_encoding(
            batch.token_features, max_relative_idx=32, max_relative_chain=2
        ).to(dtype=pair_embedding.dtype)
        features_2d = torch.concatenate([pair_embedding, rel_features], dim=-1)
        pair_cond = self.pair_cond_initial_projection(
            self.pair_cond_initial_norm(features_2d)
        )
        pair_cond += self.pair_transition_0(pair_cond)
        pair_cond += self.pair_transition_1(pair_cond)
        return pair_cond

    def _single_conditioning(self, batch, embeddings, noise_level, use_conditioning: bool) -> torch.Tensor:
        single_embedding = use_conditioning * embeddings['single']
        target_feat = embeddings['target_feat']
        features_1d = torch.concatenate(
            [single_embedding, target_feat], dim=-1)
        if of3.OF3:
            _n = features_1d.shape[0]
            _z = torch.zeros((_n, 1), dtype=features_1d.dtype, device=features_1d.device)
            features_1d = torch.concatenate([features_1d[:, :415], _z, features_1d[:, 415:446], _z, features_1d[:, 446:]], dim=-1)
        single_cond = self.single_cond_initial_projection(
            self.single_cond_initial_norm(features_1d))
        noise_embedding = self.fourier_embeddings(
            (1 / 4) * torch.log(noise_level / SIGMA_DATA)
        )
        single_cond += self.noise_embedding_initial_projection(
            self.noise_embedding_initial_norm(noise_embedding)
        )
        single_cond += self.single_transition_0(single_cond)
        single_cond += self.single_transition_1(single_cond)
        return single_cond

    def _conditioning(
        self,
        batch,
        embeddings: dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        use_conditioning: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        st = getattr(self, "_static", None)
        if st is not None and st.get("primed"):
            pair_cond = st["pair_cond"]          # HOIST: computed once per trajectory by prime_static() (identical arithmetic)
        else:
            pair_cond = self._pair_conditioning(batch, embeddings, use_conditioning)
        single_cond = self._single_conditioning(batch, embeddings, noise_level, use_conditioning)
        return single_cond, pair_cond

    # ------------------------------------------------------------------ HOIST + whole-step CUDA graph (inference levers)
    @torch.no_grad()
    def prime_static(self, batch, embeddings, use_conditioning: bool = True):
        """Compute the step-invariant conditioning ONCE for this (batch, embeddings) and store it in PERSISTENT buffers
        (same storage across trajectories, so a captured CUDA graph keeps reading valid memory):
        pair_cond [N,N,128] and the diffusion transformer's per-block pair logits [24,16,N,N]."""
        st = getattr(self, "_static", None) or {}
        new = {}
        new["pair_cond"] = self._pair_conditioning(batch, embeddings, use_conditioning)
        new["pair_logits"] = self.transformer.precompute_pair_logits(new["pair_cond"])
        if getattr(self, "fold_key_mask_into_pair_logits", False):   # the fused token transformer's attention row reads the key mask from the bias (forward.swap_in_dtk sets
            km = batch.token_features.mask                           # this): the masked keys' logits := -1e9 here, once per trajectory -- every consumer of the hoisted logits
            pl = new["pair_logits"]                                  # masks the same keys itself (stock: masked_fill(-1e9); the kit's kernels: + -1e9), so their outputs stand
            pl.masked_fill_((km == 0).reshape((1,) * (pl.dim() - 1) + (-1,)), -1e9)
        if getattr(self, "hoist_level", 2) >= 2:
            enc = self.atom_cross_att_encoder.compute_static(batch, trunk_single_cond=embeddings['single'], trunk_pair_cond=new["pair_cond"])
            for k, v in enc.items():
                new["enc." + k] = v
            new["dec_pair_logits"] = self.atom_cross_att_decoder.atom_transformer_decoder.compute_pair_logits(enc["pair_cond"])
            if getattr(self, "use_atom_window", False):              # lever 'atom_window': both atom transformers' step-invariant kernel operands, hoisted with
                for tag, tf in (("awE.", self.atom_cross_att_encoder.atom_transformer_encoder),      # the rest (DiffusionCrossAttTransformer.window_operands)
                                ("awD.", self.atom_cross_att_decoder.atom_transformer_decoder)):
                    for k, v in tf.window_operands(enc["queries_single_cond"], enc["queries_mask"]).items():
                        new[tag + k] = v
                new["aw.rows_host"] = int(new["awE.rows"].item())   # the rows handed to the kernels (32 * ceil(real atoms / 32)): ONE read-back per trajectory, here,
                nb = max(1, new["aw.rows_host"] // 32)                                                    # outside any capture
                for b in range(int(new["enc.enc_pair_logits"].shape[0])):                                 # the window kernel's bias operand per block (the real query blocks' logits,
                    new["awE.bias%d" % b] = new["enc.enc_pair_logits"][b][:nb].to(torch.float32).contiguous()   # fp32, keys contiguous): laid once per trajectory here instead of
                    new["awD.bias%d" % b] = new["dec_pair_logits"][b][:nb].to(torch.float32).contiguous()       # converted at every denoiser call (same values: the step's bits stand)
        # persistent buffers: copy into existing storage when shapes match (keeps captured CUDA graphs valid), else replace + drop graph     # outside any capture
        same = all(k in st and torch.is_tensor(st[k]) and st[k].shape == v.shape and st[k].dtype == v.dtype for k, v in new.items() if torch.is_tensor(v))
        if same:
            for k, v in new.items():
                if torch.is_tensor(v): st[k].copy_(v)
                else: st[k] = v                                      # plain values (the window path's row count) are the new trajectory's
        else:
            for k, v in new.items():
                st[k] = v.clone() if torch.is_tensor(v) else v
            st.pop("graph", None)
        st["primed"] = True
        self._static = st
        return st

    def _enc_static(self):
        st = getattr(self, "_static", None)
        if st is None or not st.get("primed") or "enc.pair_cond" not in st:
            return None, None
        enc = {k[4:]: v for k, v in st.items() if isinstance(k, str) and k.startswith("enc.")}
        if "aw.rows_host" in st:                                      # lever 'atom_window': the hoisted window operands ride with the encoder statics / beside the decoder logits
            enc["enc_window"] = dict({k[4:]: v for k, v in st.items() if isinstance(k, str) and k.startswith("awE.")}, rows_host=st["aw.rows_host"])
            enc["dec_window"] = dict({k[4:]: v for k, v in st.items() if isinstance(k, str) and k.startswith("awD.")}, rows_host=st["aw.rows_host"])
        return enc, st.get("dec_pair_logits")

    def clear_static(self):
        self._static = None

    def forward_graphed(self, positions_noisy, noise_level, batch, embeddings, use_conditioning: bool = True):
        """Whole denoiser step (conditioning + atom encoder + 24-block transformer + atom decoder) as ONE CUDA graph per
        (bucket, batch object). Call prime_static(batch, embeddings) once per trajectory first. Inputs are copied into static
        buffers; the returned tensor is a static output buffer (consume or clone before the next call)."""
        st = self._static
        assert st is not None and st.get("primed"), "call prime_static(batch, embeddings) first"
        key = (tuple(positions_noisy.shape), positions_noisy.dtype, id(batch))
        g = st.get("graph")
        if (g is None or g["key"] != key) and st.get("eager_first") != key:
            st["eager_first"] = key                      # the first step at a new key runs ONCE eagerly on the caller's stream before any capture: the compiled
            return self.forward(positions_noisy, noise_level, batch, embeddings, use_conditioning)   # glue meets the shapes (recompiles, autotuning,
        if g is None or g["key"] != key:                 # lazy module loads) outside the warm-up / capture; the next call captures (same statements: bitwise)
            caller_stream = torch.cuda.current_stream(); graph = None
            try:
                import time as _time
                t0 = _time.perf_counter()
                x_s = positions_noisy.clone(); t_s = noise_level.clone().reshape(())
                e_s = {k: v.clone() for k, v in embeddings.items()}
                side = getattr(self, "_capture_stream", None)                       # ONE warm-up stream per module for the life of the process: a fresh
                if side is None:                                                    # stream per capture leaves a cuBLAS workspace behind per item (the
                    side = self._capture_stream = torch.cuda.Stream()               # allocated-memory growth per item under stepgraph); same statements
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        self.forward(x_s, t_s, batch, e_s, use_conditioning)
                torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):   # only THIS thread's CUDA calls belong to the capture: another
                    y_s = self.forward(x_s, t_s, batch, e_s, use_conditioning)      # thread's allocator / event / query call (torch's default
                torch.cuda.synchronize()                                            # 'global' mode counts those) cannot invalidate it
                gs = getattr(self, "graph_stats", None) or {"captures": 0, "capture_s": 0.0, "shapes": []}
                gs["captures"] += 1; gs["capture_s"] = round(gs["capture_s"] + _time.perf_counter() - t0, 4)
                if list(key[0]) not in gs["shapes"]: gs["shapes"].append(list(key[0]))
                self.graph_stats = gs
            except Exception as e:                       # the whole-step graph cannot be captured on this stack (a capture-unsafe op, a launch failure):
                if _is_oom(e):                           # the lever steps aside BY NAME for the rest of the process (graph_failures, read by the
                    _end_failed_capture(graph, caller_stream)   # caller's census) and the same statements run eagerly, hoist kept; an out-of-memory is
                    raise                                # capacity, never a step-aside — it propagates as it did, the device back in eager order
                import sys, traceback                    # the innermost frames of the failed capture, on stderr: which statement the capture refused
                word = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160] if str(e) else "")
                frames = [l.strip() for l in traceback.format_exc().splitlines() if l.strip().startswith("File ")][-6:]
                retries = list(getattr(self, "graph_retries", ()))
                if len(retries) < GRAPH_CAPTURE_RETRIES:  # a failed capture is retried on the NEXT step (this one runs eagerly below: same statements, bitwise):
                    self.graph_retries = retries + [word]   # a first-launch effect inside the capture (a kernel's lazy module load, a late recompile) is gone by
                    print("[diffusion_head] step graph capture failed (%s), retrying on the next step (%d/%d); innermost frames: %s" % (word, len(retries) + 1, GRAPH_CAPTURE_RETRIES, " <- ".join(reversed(frames))), file=sys.stderr, flush=True)
                else:                                    # then; only a capture that keeps failing steps the lever aside for the process (graph_failures)
                    self.use_step_graph = False
                    self.graph_failures = list(getattr(self, "graph_failures", ())) + [word]
                    print("[diffusion_head] step graph capture failed (%s) -> the hoisted step runs eagerly for the rest of the process; innermost frames: %s" % (word, " <- ".join(reversed(frames))), file=sys.stderr, flush=True)
                st.pop("graph", None)
                _end_failed_capture(graph, caller_stream)
                if self.use_step_graph:                  # a retry is pending: settle the device, hand the allocator's cached blocks back, and let the next
                    try:                                 # step run once eagerly again (a second warm step outside any capture) before the re-capture on the
                        torch.cuda.synchronize(); torch.cuda.empty_cache()   # step after it — same statements either way (bitwise)
                    except Exception:
                        pass
                    st.pop("eager_first", None)
                return self.forward(positions_noisy, noise_level, batch, embeddings, use_conditioning)
            g = dict(key=key, graph=graph, x=x_s, t=t_s, e=e_s, y=y_s); st["graph"] = g
        g["x"].copy_(positions_noisy); g["t"].copy_(noise_level.reshape(()))
        for k, v in embeddings.items():
            if g["e"][k].data_ptr() != v.data_ptr(): g["e"][k].copy_(v)
        g["graph"].replay()
        return g["y"]

    def forward(
        self,
        positions_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        use_conditioning: bool
    ) -> torch.Tensor:
        # Get conditioning
        trunk_single_cond, trunk_pair_cond = self._conditioning(
            batch=batch,
            embeddings=embeddings,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
        )

        # Extract features
        sequence_mask = batch.token_features.mask
        atom_mask = batch.predicted_structure_info.atom_mask

        # Position features
        act = positions_noisy * atom_mask[..., None]
        act = act / torch.sqrt(noise_level**2 + SIGMA_DATA**2)

        _enc_st, _dec_pl = self._enc_static()
        enc = self.atom_cross_att_encoder(
            batch=batch,
            token_atoms_act=act,
            trunk_single_cond=embeddings['single'],
            trunk_pair_cond=trunk_pair_cond,
            static=_enc_st,
        )
        act = enc.token_act

        act += self.single_cond_embedding_projection(
            self.single_cond_embedding_norm(trunk_single_cond)
        )

        _st = getattr(self, "_static", None)
        act = self.transformer(
            act=act,
            single_cond=trunk_single_cond,
            mask=sequence_mask,
            pair_cond=trunk_pair_cond,
            pair_logits=(_st["pair_logits"] if (_st is not None and _st.get("primed")) else None),
        )
        act = self.output_norm(act)

        # (Possibly) atom-granularity decoder
        position_update = self.atom_cross_att_decoder(
            batch=batch,
            token_act=act,
            enc=enc,
            pair_logits=_dec_pl,
            window=(_enc_st or {}).get("dec_window"),
        )

        skip_scaling = SIGMA_DATA**2 / (noise_level**2 + SIGMA_DATA**2)
        out_scaling = (
            noise_level * SIGMA_DATA /
            torch.sqrt(noise_level**2 + SIGMA_DATA**2)
        )

        # the denoised coordinates are formed in fp32: the decoder's bf16 update is widened BEFORE it is scaled and added to the
        # fp32 noisy coordinates (a 0-dim fp32 scale times a bf16 tensor is a bf16 product under torch's promotion rules)
        return (
            skip_scaling * positions_noisy + out_scaling * position_update.to(torch.float32)
        ) * atom_mask[..., None]

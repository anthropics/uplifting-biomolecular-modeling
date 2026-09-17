"""Lever sampler_hoist (class exact) — the diffusion roll-out's STEP-INVARIANT work done once per DiffusionHead.sample call instead of once per
step (200 steps by default), by the SAME statements on the SAME operands in the SAME dtype / autocast state, the resulting tensors
handed back by reference afterwards: a hoist, not a fusion — every hoisted quantity is the tensor stock recomputes, bit for bit
(outputs byte-identical to stock under `--det 1`; the kit's CPU unit test compares every hoisted tensor with the stock one by torch.equal).
The stock loop (diffusion_head.py DiffusionHead.sample L283-373, its RNG draws, inference_step, the EDM update) is NOT re-stated: the hoists sit
at the module forwards the loop calls into, scoped to ONE roll-out by the state object the sample wrapper opens and closes (thread-local; every
held tensor is dropped when sample() returns).

Hoists (AFO_SAMPLER_HOISTS = all | a comma list; "-name" removes one from all; dependencies close: atom_win needs atom_c + atom_pair, atom_cond
and atom_bias need atom_win; AFO_SAMPLER_HOIST=0 keeps the lever installed and every roll-out stock, counted `disabled`):
  dit_bias   attention.py L94-98 (and dit_sdpa's copy of that statement): `attn_bias(mask) + pair_bias[i].to(q.dtype)`, the [B,1,16,L,L] additive
             bias of each of the 12 DiT blocks that stock re-materialises per block per step -> folded ONCE per roll-out into the block's own slice of
             the pair-bias tensor PairConditioning produced (in place at the first denoiser call: `+` is commutative bit for bit and the sum has
             the slice's shape and dtype, so the hoist holds NO extra memory) and returned by reference at every later call.  Inside a roll-out
             PairConditioning.forward hands sample() a stand-in (HoistedPairBias) that answers exactly what the stock path does to pair_bias —
             `.unsqueeze(2)` (diffusion_head.py L170), `[i]` (diffusion_transformer.py L241), `.to(q.dtype)` then `bias + it` (attention.py L97) —
             and raises by name on anything else; the module structure that fixes those statements (12 conditioned SelfAttention blocks with a
             pair bias, use_high_precision off, Attention.forward = stock or dit_sdpa's) is checked when the roll-out opens, else PairConditioning
             returns the stock tensor (counted `dit_bias:<reason>`).  A pair bias whose dtype is not q's takes the out-of-place sum (`dit_bias_cast`).
  atom_c     atom_attention.py L152-158 (AtomEncoder.forward): c = embedding_atoms(aatype), the fp32 atom mask, c * mask — batch-only quantities.
  atom_pair  atom_attention.py L62-89 (AtomAttentionStack.forward, encoder AND decoder stack): LocalAttentionIndex(res_idx, asym_id, seq_mask), the
             windowed one-hot `batch["atom_rel_pos"].unsqueeze(1)` (under relpos_lazy a LazyFeat that REPLAYS its producer at every call: 2 x steps
             producer runs per roll-out become 2) and p = linear_pair_bais(rel_pos) with its rearrange().contiguous() -> once per stack per roll-out.
  atom_win   diffusion_transformer.py L399-424 (AtomTransformerStack.forward before its block loop): the atom-level attention mask (repeat, to_k,
             &=), the windowed conditioning single_cond_q / single_cond_k, the pair-bias view -> once per stack per roll-out; the block loop
             (L426-439) runs every step as stock, on the held tensors.
  atom_cond  AdaLN.forward (normalization.py L74-76) of the 4 atom blocks' three AdaLNs (attention.adaln_a_q / adaln_a_k, transition.adaln) and
             the two conditioning gates (CrossAttention.forward L293 linear_ada_out, ConditionedTransitionBlock.forward L57 linear_g): their `cond`
             is the windowed conditioning this roll-out holds (single_cond_q / _k), so sigmoid(linear_g(layernorm_cond(cond))), linear_bias(...)
             and the two gate linears are evaluated ONCE per roll-out and held (the leaves only: 8 small tensors per block); per step the AdaLN
             runs `layernorm(a)` and `sig * a + bias` — the stock statement on the held operands.  Installed per INSTANCE (a forward on those
             sub-modules only, identity-keyed on tensors the roll-out holds, so a key can never alias a freed tensor; outside a roll-out or on
             any other input it is the class forward).  The two functional torch.sigmoid calls on the held gates stay per step.
  atom_bias  attention.py L94-98 for the 4 atom blocks: `attn_bias(mask) + pair_bias_i.to(q.dtype)` [B,1,W,2,56,168] -> once per block per roll-out
             (the dit_bias stand-in protocol, out of place: 4 small tensors held, p itself untouched).
  scond      diffusion_transformer.py L105 (SingleConditioning.forward): proj_single_cond(cat([s, aatype])) — an fp32 LayerNorm + Linear over
             [B,L,c_s+21] that depends on s and the batch only -> once per roll-out; the Fourier / transition part stays per step.
Not taken here (each needs the loop body itself re-stated): random_augmentation's two mask.any() host syncs and the per-step
torch.tensor(c_noise) upload (sampler_hostsync re-states the loop for exactly those), the ~mask views, inference_step's zeros_like round trip.

Composition.  denoiser_graph: every held tensor is created at the FIRST denoiser call of the roll-out — the graph's eager warm-up call, before the
capture — and read by reference by the captured graph exactly like pair_bias and the batch tensors (its call key pins the stand-in by id and
holds it; this lever holds every tensor until DiffusionHead.sample returns, i.e. past the graph's last replay).  relpos_lazy: composes — the
LazyFeat producer replay happens inside the atom_pair hoist once per stack per roll-out; nothing resident is added (rel_pos is dropped as soon as
p exists, as in stock).  dit_sdpa: its folded statement computes the same `attn_bias + pair_bias.to(q.dtype)` and receives the same held sum.  atom_sdpa (fast):
its DiT-rank calls delegate to that statement (dit_bias composes); its atom-rank calls re-state the bias sum into its own aligned buffer, so
atom_bias steps aside by name under it (`off=atom_bias:atom_sdpa`) — the other atom hoists are untouched.  atom_tf32 (fast): wraps
AtomAttentionStack.forward with a TF32 window and is installed AFTER this lever (row order), so its window encloses the hoisted prologue: p is
built once inside the same window it would run in per call.
diffusion_bf16: every hoist runs where its stock statement runs, under the same ambient autocast, and the stand-in learns q.dtype from the
statement itself (the mask bias arrives already cast): the bf16 sampler folds bf16 into bf16, the fp32 one fp32 into fp32.  ln_bf16 and any
class-level LayerNorm/Linear patch compose (the memo calls the class attribute current at call time).
Guards (stock by name, counted): AFO_SAMPLER_HOIST=0 (`disabled`); a head in training mode or a roll-out under grad (`train`); a score model
whose module structure is not the one mirrored here (`<hoist>:<reason>`, that hoist off for the roll-out; only the declared step-asides — atom_bias:atom_sdpa — are expected words); a stock function whose source digest is
not one the mirrored statements were written against (that hoist off at install, LEVER line `off=<hoist>:source`; all off -> not applied).
Device-agnostic (CPU and CUDA alike: the kit's CPU unit test drives the installed lever)."""
from __future__ import annotations

import os
import threading
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from opt_core.counters import Ledger

from . import Installed, rebind

NAME = "LOCAL.atlasfold.sampler_hoist"
IMPL = "hoist"
ENV_SWITCH = "AFO_SAMPLER_HOIST"            # =0: installed, every roll-out stock (counted `disabled`)
ENV_HOISTS = "AFO_SAMPLER_HOISTS"           # all | comma list | -name
HOISTS: Tuple[str, ...] = ("dit_bias", "atom_c", "atom_pair", "atom_win", "atom_cond", "atom_bias", "scond")
NEEDS = {"atom_win": ("atom_c", "atom_pair"), "atom_cond": ("atom_win",), "atom_bias": ("atom_win",)}
STEP_ASIDE = ("atom_bias:atom_sdpa",)              # hoists that step aside BY NAME under a named sibling lever of the row (declared composition, not a refusal)
EXPECTED = ("disabled",) + STEP_ASIDE
T_HEAD = "atlasfold.model.network.diffusion_head"
T_ATOM = "atlasfold.model.network.atom_attention"
T_DT = "atlasfold.model.network.diffusion_transformer"
T_ATT = "atlasfold.model.network.attention"
T_NORM = "atlasfold.model.network.primitives.normalization"
# sha256 (opt_core.diffusion_loop.source_guard.source_sha256) of every stock text the mirrored statements were written against (atlasfold v1.0.0 as
# carried in stock/src).  A hoist whose function reads differently is off by name at install; the table takes a second digest when an upstream
# edit is read and found immaterial.
SOURCE_SHA256 = {
    "AtomEncoder.forward": ("276c9fb81beddb273287dddb6d0d5024776c9678745a14bd678bb6e9094676e9",),
    "AtomAttentionStack.forward": ("4d57144e2c5b6d066e73908f0e10b6024afc91103b0b4c1b39114367a768e595",),
    "AtomTransformerStack.forward": ("5e04b18ba55902bc26c94104ec4106bb7d2242a1e9367676d4f1c44ed97b2900",),
    "SingleConditioning.forward": ("b002f1ccd8abf3f492de8a51a10fb79ae0f0f25bd505edc7debce594823107b2",),
    "Attention.forward": ("ce542cab4d11c7d851d4743597a8c0ae27a11ea1b2b27ad7ee88c59b3e1880bd",),
    "DiffusionTransformerStack.forward": ("646a69ab4b97e866ce9cae33d6fc54bbb862d34ced3d3d2744b189e68287b9aa",),
    "DiffusionModule.forward": ("811ca9604d03fd70c2defbb5eb2e8b0605fae200e9224804303226973f59c199",),
    "AdaLN.forward": ("2365a0ab8c97565094c686043c4e7d46c14b3b9effc10c554c695efbd4b8f680",),
}
MIRRORS = {                                  # hoist -> the stock functions whose statements it mirrors or whose protocol it relies on
    "dit_bias": ("Attention.forward", "DiffusionTransformerStack.forward", "DiffusionModule.forward"),
    "atom_c": ("AtomEncoder.forward",),
    "atom_pair": ("AtomAttentionStack.forward",),
    "atom_win": ("AtomTransformerStack.forward",),
    "atom_cond": ("AdaLN.forward",),
    "atom_bias": ("Attention.forward",),
    "scond": ("SingleConditioning.forward",),
}
DIT_BIAS_WRAPPERS = ("Attention.forward[atlasfold_opt:dit_sdpa]", "Attention.forward[atlasfold_opt:atom_sdpa]", "Attention.forward[atlasfold_opt:dit_apb]")   # Attention.forward wrappers under which the DiT (rank-4) call keeps the `bias + pair_bias.to(q.dtype)` statement
ATOM_BIAS_WRAPPERS = ("Attention.forward[atlasfold_opt:dit_sdpa]", "Attention.forward[atlasfold_opt:dit_apb]")             # ... and under which the ATOM (rank-5) call keeps it (atom_sdpa re-states the atom statement into its own aligned buffer: atom_bias steps aside by name under it)
KNOWN_ATTENTION_WRAPPERS = DIT_BIAS_WRAPPERS
_CUR = threading.local()
_MEMO_FLAG = "_atlasfold_opt_sampler_hoist_memo"


class HoistError(RuntimeError):
    """An operation outside the stock protocol reached a hoisted stand-in (cannot happen with the module structure checked at roll-out open)."""


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENV_SWITCH, "1")).strip() not in ("0", "off", "false", "no")


def parse_hoists(text: Optional[str]) -> Tuple[str, ...]:
    """AFO_SAMPLER_HOISTS grammar: '' / 'all' = every hoist; 'a,b' = those; '-a,-b' = all but those. Unknown names raise. Dependencies close
    (a hoist whose prerequisite is absent is dropped)."""
    text = (text or "").strip()
    if not text or text == "all":
        chosen = list(HOISTS)
    else:
        items = [t.strip() for t in text.split(",") if t.strip()]
        neg = [t[1:] for t in items if t.startswith("-")]
        pos = [t for t in items if not t.startswith("-")]
        for t in neg + pos:
            if t not in HOISTS and t != "all":
                raise ValueError(f"{ENV_HOISTS}: unknown hoist {t!r}; known: {', '.join(HOISTS)}")
        chosen = [h for h in HOISTS if h not in neg] if (neg and (not pos or pos == ["all"])) else [h for h in HOISTS if h in pos and h not in neg]
    return close_deps(chosen)


def close_deps(chosen: Sequence[str]) -> Tuple[str, ...]:
    s = [h for h in HOISTS if h in chosen]
    changed = True
    while changed:
        changed = False
        for h in list(s):
            if any(d not in s for d in NEEDS.get(h, ())):
                s.remove(h); changed = True
    return tuple(s)


def _cur():
    return getattr(_CUR, "ro", None)


# ----------------------------------------------------------------------------------------------------------------- the hoisted additive attention bias
def _is_add(func) -> bool:
    return getattr(func, "__name__", "") in ("add", "__add__", "__radd__")


class BlockBias:
    """Stands in for ONE block's pair bias at `attn_bias = attn_bias + pair_bias.to(q.dtype)` (attention.py L96-97; dit_sdpa's folded statement):
    the first time the statement runs in a roll-out the sum is computed by torch from the statement's own operands (in place into the operand
    slice when `inplace` and the dtypes agree, else out of place) and kept; every later run with a mask bias of the same shape / dtype / device
    gets that tensor.  `.to(dtype)` records the dtype the statement asks for.  Anything else is a HoistError by name.  Holds no reference to
    its container (no reference cycle: the held storage is freed by refcount the moment the roll-out closes, not at the next GC pass)."""
    __slots__ = ("index", "operand", "inplace", "req_dtype", "full", "key", "name", "hits", "folds_inplace", "folds_oop")

    def __init__(self, index: int, operand: torch.Tensor, inplace: bool, name: str):
        self.index, self.operand, self.inplace, self.name = index, operand, inplace, name
        self.req_dtype = None
        self.full = None
        self.key = None
        self.hits = self.folds_inplace = self.folds_oop = 0

    # -- the protocol
    def to(self, *args, **kwargs):
        dtype = kwargs.get("dtype")
        for a in args:
            if isinstance(a, torch.dtype):
                dtype = a
            elif isinstance(a, torch.Tensor):
                dtype = a.dtype
        dev = self.operand.device if self.operand is not None else (self.full.device if self.full is not None else None)
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, (torch.device, str)) and not isinstance(a, torch.dtype) and dev is not None and torch.device(a) != dev:
                raise HoistError(f"sampler_hoist.{self.name}: pair bias asked to move to {a} (built on {dev})")
        if dtype is not None:
            self.req_dtype = dtype
        return self

    def fold(self, other: torch.Tensor) -> torch.Tensor:
        if not isinstance(other, torch.Tensor):
            raise HoistError(f"sampler_hoist.{self.name}: `+` with a {type(other).__name__}")
        key = (tuple(other.shape), other.dtype, other.device)
        if self.full is not None:
            if key != self.key:
                raise HoistError(f"sampler_hoist.{self.name}: mask bias {key} unlike the first call's {self.key}")
            self.hits += 1
            return self.full
        op = self.operand
        if op is None:
            raise HoistError(f"sampler_hoist.{self.name}: the roll-out that built this bias has closed")
        if other.dim() != op.dim() or other.shape[-3] != 1 or torch.broadcast_shapes(tuple(other.shape), tuple(op.shape)) != tuple(op.shape):
            raise HoistError(f"sampler_hoist.{self.name}: `+` operand {tuple(other.shape)} is not a mask bias for a pair bias {tuple(op.shape)}")
        want = self.req_dtype if self.req_dtype is not None else op.dtype
        if self.inplace and op.dtype == want == other.dtype and op.device == other.device:
            full = op.add_(other)                                            # bias + pb == pb += bias bit for bit (one rounding of the same two operands)
            self.folds_inplace += 1
        else:
            full = other + op.to(want)                                       # the stock statement verbatim, once
            self.folds_oop += 1
        self.operand = None                                                  # the slice is reachable through `full` only from here on
        self.full, self.key = full, key
        return full

    def release(self) -> None:
        """End of the roll-out: drop the held storage now (refcount), not at a GC pass."""
        self.operand = None
        self.full = None

    def __radd__(self, other):
        return self.fold(other)

    __add__ = __radd__

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        alpha = kwargs.pop("alpha", 1)
        if _is_add(func) and len(args) == 2 and not kwargs and alpha == 1:
            a, b = args
            if isinstance(b, BlockBias) and isinstance(a, torch.Tensor):
                return b.fold(a)
            if isinstance(a, BlockBias) and isinstance(b, torch.Tensor):
                return a.fold(b)
        who = next((x for x in list(args) + list(kwargs.values()) if isinstance(x, BlockBias)), None)
        raise HoistError(f"sampler_hoist.{getattr(who, 'name', 'bias')}: {getattr(func, '__name__', func)} is outside the hoisted pair-bias protocol (.to(dtype), bias + it)")

    def __repr__(self):
        return f"BlockBias({self.name}[{self.index}], built={self.full is not None})"


class HoistedPairBias:
    """Stands in for the DiT pair-bias tensor [Nblock, B, H, L, L] between PairConditioning.forward and the attention statement: `.unsqueeze(2)`
    (diffusion_head.py L170) returns it, `[i]` (diffusion_transformer.py L241) returns block i's BlockBias over the [B,1,H,L,L] slice.  The raw
    tensor is reachable only through this object, which is what makes the in-place fold safe; `release()` drops it when the roll-out closes."""
    __slots__ = ("raw6", "blocks", "n", "name")

    def __init__(self, raw: torch.Tensor, name: str = "dit_bias"):
        self.raw6 = raw.unsqueeze(2)                                         # [n, B, 1, H, L, L] — the view stock takes at diffusion_head.py L170
        self.n = int(raw.shape[0])
        self.blocks: Dict[int, BlockBias] = {}
        self.name = name

    def unsqueeze(self, dim):
        if dim not in (2, -4):
            raise HoistError(f"sampler_hoist.{self.name}: unsqueeze({dim}) — stock takes unsqueeze(2)")
        return self

    def __getitem__(self, i):
        if not isinstance(i, int):
            raise HoistError(f"sampler_hoist.{self.name}: index {i!r} — stock takes pair_bias[i] per block")
        if i < 0:
            i += self.n
        if not 0 <= i < self.n:
            raise IndexError(i)
        b = self.blocks.get(i)
        if b is None:
            if self.raw6 is None:
                raise HoistError(f"sampler_hoist.{self.name}: the roll-out that built this pair bias has closed")
            b = self.blocks[i] = BlockBias(i, self.raw6[i], inplace=True, name=self.name)
        return b

    def __len__(self):
        return self.n

    @property
    def hits(self):
        return sum(b.hits for b in self.blocks.values())

    @property
    def folds_inplace(self):
        return sum(b.folds_inplace for b in self.blocks.values())

    @property
    def folds_oop(self):
        return sum(b.folds_oop for b in self.blocks.values())

    @property
    def shape(self):
        return self.raw6.shape

    @property
    def dtype(self):
        return self.raw6.dtype

    @property
    def device(self):
        return self.raw6.device

    def release(self) -> None:
        for b in self.blocks.values():
            b.release()
        self.blocks.clear()
        self.raw6 = None

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        raise HoistError(f"sampler_hoist.dit_bias: {getattr(func, '__name__', func)} on the hoisted pair-bias stack — the stock path only takes .unsqueeze(2) and [i]")

    def __repr__(self):
        return f"HoistedPairBias(n={self.n}, built={len([b for b in self.blocks.values() if b.full is not None])})"


class AtomBiasSet:
    """The 4 atom blocks' biases (out of place: p stays untouched)."""
    __slots__ = ("blocks", "name")

    def __init__(self, pb: torch.Tensor, name: str = "atom_bias"):
        self.name = name
        self.blocks = [BlockBias(i, pb[i], inplace=False, name=name) for i in range(int(pb.shape[0]))]

    @property
    def hits(self):
        return sum(b.hits for b in self.blocks)

    @property
    def folds(self):
        return sum(b.folds_oop + b.folds_inplace for b in self.blocks)

    def release(self) -> None:
        for b in self.blocks:
            b.release()
        self.blocks = []


# ----------------------------------------------------------------------------------------------------------------- one roll-out
class Roll:
    """State of ONE DiffusionHead.sample call: which hoists engage on this head, the tensors held for the roll-out (dropped at close), the memo."""

    def __init__(self, head, hoists: Sequence[str], ledger: Ledger):
        self.head, self.ledger = head, ledger
        self.hoists = set(hoists)
        self.off: Dict[str, str] = {}                                        # hoist -> reason it is off for THIS roll-out (config)
        self.mods = set()
        self.held: List[object] = []
        self.held_ids = set()
        self.inv_ids = set()
        self.enc: Dict[int, tuple] = {}
        self.astack: Dict[int, SimpleNamespace] = {}
        self.tstack: Dict[int, SimpleNamespace] = {}
        self.memo: Dict[tuple, torch.Tensor] = {}
        self.memo_hits = self.memo_miss = 0
        self.scond = None
        self.scond_hits = 0
        self.pair: Optional[HoistedPairBias] = None
        self.atom_sets: List[AtomBiasSet] = []
        self.relpos_mats = 0
        self.pc_id = self.sc_id = None
        self.closed = False
        self._configure()

    # -- structure checks: a hoist engages only on the module structure its statements mirror
    def _configure(self) -> None:
        import importlib
        A = importlib.import_module(T_ATOM); DT = importlib.import_module(T_DT); ATT = importlib.import_module(T_ATT); H = importlib.import_module(T_HEAD)
        head = self.head
        sm = getattr(head, "score_model", None)
        if not isinstance(sm, H.DiffusionModule):
            for h in list(self.hoists):
                self._off(h, "score_model")
            return
        self.mods = {id(m) for m in sm.modules()}
        pc, sc = getattr(head, "pair_conditioning", None), getattr(head, "single_conditioning", None)
        if isinstance(pc, DT.PairConditioning):
            self.pc_id = id(pc); self.mods.add(id(pc))
        else:
            self._off("dit_bias", "pair_conditioning")
        if isinstance(sc, DT.SingleConditioning) and isinstance(getattr(sc, "proj_single_cond", None), torch.nn.Module):
            self.sc_id = id(sc); self.mods.add(id(sc))
        else:
            self._off("scond", "single_conditioning")
        att_ok = _attention_forward_known(ATT.Attention, DIT_BIAS_WRAPPERS)
        atom_att_ok = _attention_forward_known(ATT.Attention, ATOM_BIAS_WRAPPERS)
        # DiT: 12 conditioned SelfAttention blocks with a pair bias through the SDPA statement
        dts = getattr(sm, "diffusion_transformer", None)
        ok = isinstance(dts, DT.DiffusionTransformerStack) and len(getattr(dts, "blocks", ())) > 0
        if ok:
            for blk in dts.blocks:
                at = getattr(blk, "attention", None)
                ok = ok and isinstance(at, ATT.SelfAttention) and bool(getattr(at, "use_pair_bias", False)) and isinstance(getattr(at, "attn", None), ATT.Attention) \
                    and not bool(getattr(at.attn, "use_high_precision", True))
        if not ok:
            self._off("dit_bias", "dit_blocks")
        elif att_ok is not None:
            self._off("dit_bias", att_ok)
        # atom side: encoder / decoder stacks of AtomTransformerBlocks (CrossAttention with pair bias, not high precision, conditioned)
        enc, dec = getattr(sm, "atom_encoder", None), getattr(sm, "atom_decoder", None)
        stacks = []
        a_ok = isinstance(enc, A.AtomEncoder) and isinstance(getattr(enc, "stack", None), A.AtomAttentionStack) and isinstance(dec, A.AtomDecoder) \
            and isinstance(getattr(dec, "stack", None), A.AtomAttentionStack)
        if a_ok:
            for st in (enc.stack, dec.stack):
                ts = getattr(st, "stack", None)
                a_ok = a_ok and isinstance(ts, DT.AtomTransformerStack) and len(getattr(ts, "blocks", ())) == st.num_blocks
                stacks.append(ts)
        if not a_ok:
            for h in ("atom_c", "atom_pair", "atom_win", "atom_cond", "atom_bias"):
                self._off(h, "atom_modules")
        else:
            blocks = [b for ts in stacks for b in ts.blocks]
            NORM = importlib.import_module(T_NORM)
            c_ok = all(isinstance(getattr(b, "attention", None), ATT.CrossAttention) and b.attention.use_conditioning and b.use_conditioning
                       and isinstance(getattr(b, "transition", None), DT.ConditionedTransitionBlock)
                       and all(isinstance(getattr(o, k, None), NORM.AdaLN) for o, k in ((b.attention, "adaln_a_q"), (b.attention, "adaln_a_k"), (b.transition, "adaln")))
                       for b in blocks)
            if not c_ok:
                self._off("atom_cond", "atom_blocks"); self._off("atom_bias", "atom_blocks")
            b_ok = all(isinstance(getattr(b, "attention", None), ATT.CrossAttention) and b.attention.use_pair_bias
                       and not bool(getattr(b.attention.attn, "use_high_precision", True)) for b in blocks)
            if not b_ok:
                self._off("atom_bias", "atom_attention")
            elif atom_att_ok is not None:
                self._off("atom_bias", atom_att_ok)                              # e.g. off=atom_bias:atom_sdpa in the fast row (that lever re-states the atom statement itself)
            if "atom_cond" in self.hoists and not getattr(head, _MEMO_FLAG, False):
                _install_memos(blocks)
                try:
                    object.__setattr__(head, _MEMO_FLAG, True)
                except Exception:  # noqa: BLE001
                    pass
        self.hoists = set(close_deps(list(self.hoists)))

    def _off(self, hoist: str, why: str) -> None:
        """Hoist `hoist` does not engage for this roll-out: counted `<hoist>:<why>` (expected only for the declared step-asides, e.g.
        atom_bias:atom_sdpa; any other word is an unexpected fallback the gate refuses by name) and printed as off=<hoist>:<why>."""
        if hoist in self.hoists:
            self.hoists.discard(hoist)
            self.off[hoist] = why
            self.ledger.fallback(f"{hoist}:{why}")

    # -- held tensors
    def hold(self, *objs) -> None:
        for o in objs:
            if o is not None and id(o) not in self.held_ids:
                self.held.append(o); self.held_ids.add(id(o))

    def holds(self, *objs) -> bool:
        return all(o is None or id(o) in self.held_ids for o in objs)

    def invariant(self, *objs) -> None:
        self.hold(*objs)
        for o in objs:
            if o is not None:
                self.inv_ids.add(id(o))

    def engaged(self) -> bool:
        return bool(self.pair is not None or self.enc or self.astack or self.tstack or self.scond is not None or self.memo)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        led = self.ledger
        led.count("rollouts")
        if self.engaged():
            led.serve(self._shape())
        elif self.hoists:
            led.fallback("no_calls_in_rollout")
        if self.pair is not None:
            led.count("dit_folds", self.pair.folds_inplace); led.count("dit_folds_cast", self.pair.folds_oop); led.count("dit_hits", self.pair.hits)
        for s in self.atom_sets:
            led.count("atom_bias_folds", s.folds); led.count("atom_bias_hits", s.hits)
        led.count("atom_builds", len(self.astack) + len(self.tstack) + len(self.enc))
        led.count("relpos_mats", self.relpos_mats)
        led.count("memo_miss", self.memo_miss); led.count("memo_hits", self.memo_hits)
        led.count("scond_hits", self.scond_hits)
        for h, why in self.off.items():
            led.set(f"off_{h}", why)
        # drop every reference: the held tensors' lifetime is the roll-out (released by refcount here, no cycles: nothing waits for a GC pass)
        if self.pair is not None:
            self.pair.release()
        for s in self.atom_sets:
            s.release()
        self.held.clear(); self.held_ids.clear(); self.inv_ids.clear(); self.enc.clear(); self.astack.clear(); self.tstack.clear()
        self.memo.clear(); self.scond = None; self.pair = None; self.atom_sets.clear(); self.head = None

    def _shape(self) -> str:
        try:
            for st in self.astack.values():
                m = st.atom_mask
                return f"L{int(m.shape[-2])}xB{int(m.shape[0])}"
            if self.pair is not None:
                return f"L{int(self.pair.raw6.shape[-1])}xB{int(self.pair.raw6.shape[1])}"
        except Exception:  # noqa: BLE001
            pass
        return "rollout"


def _attention_forward_known(attention_cls, known=None) -> Optional[str]:
    """None when Attention.forward is the stock text (digest) possibly under wrappers listed in `known`; else the reason word (the first unknown
    wrapper's lever name, or attention_source)."""
    known = DIT_BIAS_WRAPPERS if known is None else known
    f = attention_cls.forward
    depth = 0
    while hasattr(f, "__wrapped_stock__") and depth < 8:
        q = getattr(f, "__qualname__", "")
        if q not in known:
            return q[q.find("atlasfold_opt:") + len("atlasfold_opt:"):-1] if "atlasfold_opt:" in q else "attention_wrapper"
        f = f.__wrapped_stock__; depth += 1
    try:
        from opt_core.diffusion_loop.source_guard import source_sha256
        if source_sha256(f) not in SOURCE_SHA256["Attention.forward"]:
            return "attention_source"
    except Exception:  # noqa: BLE001
        return "attention_source"
    return None


# ----------------------------------------------------------------------------------------------------------------- atom_cond: per-instance memo on held invariants
_MEMO_TAG = "atlasfold_opt:sampler_hoist.atom_cond"


def _memo_forward(mod):
    """Leaf memo for a one-input module (the two gate Linears): held invariant input -> the module's output, once per roll-out."""
    def forward(x, *args, **kwargs):
        cls_forward = type(mod).forward                                      # the class attribute current NOW (ln_bf16 and friends compose)
        ro = _cur()
        if ro is None or args or kwargs or "atom_cond" not in ro.hoists or not isinstance(x, torch.Tensor) or id(x) not in ro.inv_ids:
            return cls_forward(mod, x, *args, **kwargs)
        key = (id(mod), id(x))
        out = ro.memo.get(key)
        if out is None:
            out = cls_forward(mod, x)
            ro.memo[key] = out
            ro.hold(out)
            ro.memo_miss += 1
        else:
            ro.memo_hits += 1
        return out
    forward.__qualname__ = f"{type(mod).__name__}.forward[{_MEMO_TAG}]"
    return forward


def _ada_forward(mod):
    """AdaLN.forward (normalization.py L74-76) with its cond branch memoised: `cond` a held invariant -> (sigmoid(linear_g(layernorm_cond(cond))),
    linear_bias(layernorm_cond(cond))) once per roll-out; per call `layernorm(a)` and `sig * a + bias` exactly as the stock statement orders them."""
    def forward(a, cond, *args, **kwargs):
        ro = _cur()
        if ro is None or args or kwargs or "atom_cond" not in ro.hoists or not isinstance(cond, torch.Tensor) or id(cond) not in ro.inv_ids:
            return type(mod).forward(mod, a, cond, *args, **kwargs)
        key = (id(mod), id(cond))
        sb = ro.memo.get(key)
        # Line 1
        a = mod.layernorm(a)
        if sb is None:
            # Line 2
            c = mod.layernorm_cond(cond)
            # Line 3 (the cond-only factors)
            sb = (mod.sigmoid(mod.linear_g(c)), mod.linear_bias(c))
            ro.memo[key] = sb
            ro.hold(*sb)
            ro.memo_miss += 1
        else:
            ro.memo_hits += 1
        sig, bias = sb
        # Line 3
        a = sig * a + bias
        return a
    forward.__qualname__ = f"AdaLN.forward[{_MEMO_TAG}]"
    return forward


def _install_memos(blocks) -> int:
    """Instance-level forwards on the conditioning sub-modules of every atom block: the three AdaLNs (attention.adaln_a_q / adaln_a_k,
    transition.adaln) and the two gate Linears (attention.linear_ada_out, transition.linear_g).  Outside a roll-out, or on an input the roll-out
    does not hold as invariant, each is the class forward."""
    n = 0
    for b in blocks:
        for ada in (getattr(b.attention, "adaln_a_q", None), getattr(b.attention, "adaln_a_k", None), getattr(b.transition, "adaln", None)):
            if isinstance(ada, torch.nn.Module) and "forward" not in ada.__dict__ and all(hasattr(ada, k) for k in ("layernorm", "layernorm_cond", "linear_g", "sigmoid", "linear_bias")):
                ada.forward = _ada_forward(ada); n += 1
        for m in (getattr(b.attention, "linear_ada_out", None), getattr(b.transition, "linear_g", None)):
            if isinstance(m, torch.nn.Module) and "forward" not in m.__dict__:
                m.forward = _memo_forward(m); n += 1
    return n


def remove_memos(module) -> int:
    """Undo _install_memos under `module` (tests / restore)."""
    n = 0
    for m in module.modules():
        f = m.__dict__.get("forward")
        if f is not None and _MEMO_TAG in getattr(f, "__qualname__", ""):
            del m.__dict__["forward"]; n += 1
    for m in module.modules():
        if getattr(m, _MEMO_FLAG, False):
            try:
                object.__delattr__(m, _MEMO_FLAG)
            except Exception:  # noqa: BLE001
                pass
    return n


# ----------------------------------------------------------------------------------------------------------------- the wrappers (one per stock forward)
def make_sample(inner, hoists: Sequence[str], ledger: Ledger):
    def sample(self, *args, **kwargs):
        if not enabled():
            ledger.fallback("disabled"); ledger.count("rollouts")
            return inner(self, *args, **kwargs)
        if getattr(self, "training", False) or torch.is_grad_enabled():
            ledger.fallback("train"); ledger.count("rollouts")
            return inner(self, *args, **kwargs)
        ro = Roll(self, hoists, ledger)
        prev = _cur()
        _CUR.ro = ro
        try:
            return inner(self, *args, **kwargs)
        finally:
            _CUR.ro = prev
            ro.close()
    sample.__qualname__ = "DiffusionHead.sample[atlasfold_opt:sampler_hoist]"
    return sample


def make_pc_forward(prev):
    def forward(self, batch, z):
        out = prev(self, batch, z)
        ro = _cur()
        if ro is None or "dit_bias" not in ro.hoists or id(self) != ro.pc_id or ro.pair is not None:
            return out
        n = len(ro.head.score_model.diffusion_transformer.blocks)
        if not isinstance(out, torch.Tensor) or out.dim() != 5 or int(out.shape[0]) != n:
            ro._off("dit_bias", "pair_bias_shape")
            return out
        ro.pair = HoistedPairBias(out)
        return ro.pair
    forward.__qualname__ = "PairConditioning.forward[atlasfold_opt:sampler_hoist.dit_bias]"
    return forward


def make_sc_forward(prev):
    def forward(self, batch, s, c_noise):
        ro = _cur()
        if ro is None or "scond" not in ro.hoists or id(self) != ro.sc_id:
            return prev(self, batch, s, c_noise)
        aatype = batch["aatype"]
        key = (id(s), id(aatype))
        st = ro.scond
        if st is not None and st[0] == key:
            sp = st[1]; ro.scond_hits += 1
        else:                                                                # diffusion_transformer.py L105, once per (s, aatype) of the roll-out
            sp = self.proj_single_cond(torch.cat([s, aatype], dim=-1))
            ro.scond = (key, sp, s, aatype)                                  # s / aatype held so their ids stay theirs
        # L107-114 per step, verbatim
        fourier_embed = self.fourier_embed(c_noise)  # [B, N, d_fourier]
        fourier_embed = self.linear_fourier(self.layernorm_fourier(fourier_embed))
        s = sp[:, None, :, :] + fourier_embed[:, :, None, :]  # [B, N, L, c_s]
        for transition in self.transitions:
            s = s + transition(s)
        return s
    forward.__qualname__ = "SingleConditioning.forward[atlasfold_opt:sampler_hoist.scond]"
    return forward


def make_enc_forward(prev):
    def forward(self, batch, r_noisy):
        ro = _cur()
        if ro is None or "atom_c" not in ro.hoists or id(self) not in ro.mods:
            return prev(self, batch, r_noisy)
        st = ro.enc.get(id(self))
        if st is None:                                                       # atom_attention.py L152-158: the batch-only statements, once
            aatype = batch["aatype"]  # (B, L, 21)
            c = self.embedding_atoms(aatype).unflatten(-1, (14, -1))  # (B, L, 14, C_a)
            c = c.unsqueeze(-4)  # (B, 1, L, 14, C_a)
            mask = batch["atom14_mask"].unsqueeze(1).to(c.dtype)  # (B, 1, L, 14)
            cm = c * mask[..., None]  # (B, 1, L, 14, C_a)
            st = ro.enc[id(self)] = (c, mask, cm)
            ro.invariant(c, mask, cm)
        c, mask, cm = st
        q = c
        q = q + self.linear_in(r_noisy)  # (B, N, L, 14, C_a)
        q = q * mask[..., None]  # (B, N, L, 14, C_a)
        q = self.stack(batch, q, cm)  # (B, N, L, 14, C_a)
        q = q * mask[..., None]
        return q, cm
    forward.__qualname__ = "AtomEncoder.forward[atlasfold_opt:sampler_hoist.atom_c]"
    return forward


def make_astack_forward(prev):
    def forward(self, batch, q, c):
        ro = _cur()
        if ro is None or "atom_pair" not in ro.hoists or id(self) not in ro.mods:
            return prev(self, batch, q, c)
        st = ro.astack.get(id(self))
        if st is None:                                                       # atom_attention.py L62-89 up to the stack call, once per stack per roll-out
            from atlasfold.model.network.misc import LocalAttentionIndex
            import einops
            res_idx = batch["res_idx"].unsqueeze(1)  # (B, 1, L)
            asym_id = batch["asym_id"].unsqueeze(1)  # (B, 1, L)
            seq_mask = batch["seq_mask"].unsqueeze(1)  # (B, 1, L)
            rel_pos = batch["atom_rel_pos"].unsqueeze(1)  # (B, 1, W, Lq, 14, Lk, 14, bins) — a LazyFeat under relpos_lazy: its producer replays here
            if not isinstance(batch["atom_rel_pos"], torch.Tensor):
                ro.relpos_mats += 1
            atom_mask = batch["atom14_mask"].unsqueeze(1)  # (B, 1, L, 14)
            local_attn_idx = LocalAttentionIndex(res_idx, asym_id, seq_mask, window_size=4, max_r=4)
            W, Lq, Lk = local_attn_idx.W, local_attn_idx.Lq, local_attn_idx.Lk
            assert rel_pos.shape[2:7] == (W, Lq, 14, Lk, 14), (
                f"Expected rel_pos shape (B, 1, {W}, {Lq}, 14, {Lk}, 14), "
                f"but got {rel_pos.shape}."
            )
            p = self.linear_pair_bais(rel_pos)
            del rel_pos
            p = einops.rearrange(p, "... q a1 k a2 (n h) -> n ... h q a1 k a2", a1=14, a2=14, n=self.num_blocks, h=self.num_heads).contiguous()
            st = ro.astack[id(self)] = SimpleNamespace(lai=local_attn_idx, p=p, atom_mask=atom_mask)
            ro.invariant(local_attn_idx, p, atom_mask)
        q = self.stack(
            q,  # [B, N, L, 14, C_a]
            mask=st.atom_mask,  # [B, 1, L, 14]
            local_attn_index=st.lai,
            single_cond=c,
            pair_bias=st.p,  # [Nblocks, B, 1, W, Nheads, Lq, 14, Lk, 14]
        )
        return q
    forward.__qualname__ = "AtomAttentionStack.forward[atlasfold_opt:sampler_hoist.atom_pair]"
    return forward


def make_tstack_forward(prev):
    def forward(self, a, mask, local_attn_index, single_cond, pair_bias):
        ro = _cur()
        if ro is None or "atom_win" not in ro.hoists or id(self) not in ro.mods or not ro.holds(mask, local_attn_index, single_cond, pair_bias) \
                or local_attn_index is None:
            return prev(self, a, mask, local_attn_index, single_cond, pair_bias)
        import einops
        key = (id(self), id(mask), id(local_attn_index), id(single_cond), id(pair_bias))
        st = ro.tstack.get(id(self))
        if st is not None and st.key != key:
            return prev(self, a, mask, local_attn_index, single_cond, pair_bias)
        if st is None:                                                       # diffusion_transformer.py L399-424, once per stack per roll-out
            attn_mask = local_attn_index.attn_mask  # [*, W, Lq, Lk]
            attn_mask = einops.repeat(attn_mask, "... q k -> ... (q 14) (k 14)")
            mask_k = local_attn_index.to_k(mask, dim=-2)  # [*, W, Lk, 14]
            attn_mask &= einops.rearrange(mask_k, "... w k a -> ... w 1 (k a)", a=14)
            if single_cond is not None:
                single_cond_q, single_cond_k = local_attn_index(single_cond, dim=-3)
                single_cond_q, single_cond_k = map(
                    lambda x: einops.rearrange(x, "... w l a c -> ... w (l a) c", a=14),
                    (single_cond_q, single_cond_k),
                )
            else:
                single_cond_q, single_cond_k = None, None
            pb = einops.rearrange(pair_bias, "n ... h q a1 k a2 -> n ... h (q a1) (k a2)") if pair_bias is not None else None
            biases = None
            if pb is not None and "atom_bias" in ro.hoists and int(pb.shape[0]) >= len(self.blocks):
                aset = AtomBiasSet(pb); ro.atom_sets.append(aset); biases = aset.blocks
            st = ro.tstack[id(self)] = SimpleNamespace(key=key, attn_mask=attn_mask, cq=single_cond_q, ck=single_cond_k, pb=pb, biases=biases)
            ro.invariant(attn_mask, single_cond_q, single_cond_k, pb)
        W, Lq = local_attn_index.W, local_attn_index.Lq
        for i, block in enumerate(self.blocks):                             # L426-439 per step, verbatim on the held tensors
            a_q, a_k = local_attn_index(a, dim=-3)
            a_q, a_k = map(
                lambda x: einops.rearrange(x, "... w l a c -> ... w (l a) c", a=14),
                (a_q, a_k),
            )
            if st.pb is None:
                pair_bias_i = None
            elif st.biases is not None:
                pair_bias_i = st.biases[i]
            else:
                pair_bias_i = st.pb[i]
            a_q = block(a_q, a_k, st.attn_mask, st.cq, st.ck, pair_bias_i)
            a = einops.rearrange(a_q, "... w (q a) c -> ... (w q) a c", q=Lq, w=W, a=14)
        return a
    forward.__qualname__ = "AtomTransformerStack.forward[atlasfold_opt:sampler_hoist.atom_win]"
    return forward


# ----------------------------------------------------------------------------------------------------------------- install
def source_check(hoists: Sequence[str]) -> Tuple[Tuple[str, ...], Dict[str, str], Dict[str, str]]:
    """(hoists whose mirrored stock texts match, {hoist: 'source:<fn>'} for the others, {fn: observed digest})."""
    import importlib
    from opt_core.diffusion_loop.source_guard import source_sha256
    where = {"AtomEncoder.forward": (T_ATOM, "AtomEncoder"), "AtomAttentionStack.forward": (T_ATOM, "AtomAttentionStack"),
             "AtomTransformerStack.forward": (T_DT, "AtomTransformerStack"), "SingleConditioning.forward": (T_DT, "SingleConditioning"),
             "Attention.forward": (T_ATT, "Attention"), "DiffusionTransformerStack.forward": (T_DT, "DiffusionTransformerStack"),
             "DiffusionModule.forward": (T_HEAD, "DiffusionModule"), "AdaLN.forward": (T_NORM, "AdaLN")}
    seen: Dict[str, str] = {}
    bad = set()
    for fn, (modname, clsname) in where.items():
        f = getattr(importlib.import_module(modname), clsname).forward
        while hasattr(f, "__wrapped_stock__"):                                # another lever's wrapper: read the stock text beneath it
            f = f.__wrapped_stock__
        try:
            d = source_sha256(f)
        except Exception as e:  # noqa: BLE001
            d = f"unreadable:{type(e).__name__}"
        seen[fn] = d
        if d not in SOURCE_SHA256[fn]:
            bad.add(fn)
    off = {h: "source:" + "+".join(sorted(fn for fn in MIRRORS[h] if fn in bad)) for h in hoists if any(fn in bad for fn in MIRRORS[h])}
    ok = close_deps([h for h in hoists if h not in off])
    for h in hoists:
        if h not in ok and h not in off:
            off[h] = "needs:" + "+".join(NEEDS.get(h, ()))
    return ok, off, seen


def _line(ledger: Ledger, tag: str, hoists: Sequence[str], off: Dict[str, str]):
    """The LEVER line: the ledger's own pairs (served / fallback_by / shapes + the counters as facts) plus `off=<hoist>:<why>,...` for hoists that
    did not engage (install-time source digests, roll-out-time config) and the switch word under AFO_SAMPLER_HOIST=0."""
    def line():
        ev = {}
        offs = dict(off)
        offs.update({k[4:]: v for k, v in ledger.facts().items() if k.startswith("off_")})
        if offs:
            ev["off"] = ",".join(f"{h}:{w}" for h, w in sorted(offs.items()))
        if not enabled():
            ev["switch"] = f"{ENV_SWITCH}=0"
        return ledger.line(tag, **ev)
    return line


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    try:
        H = importlib.import_module(T_HEAD); A = importlib.import_module(T_ATOM); DT = importlib.import_module(T_DT)
    except Exception as e:  # noqa: BLE001
        return Installed("sampler_hoist", False, reason=f"import:{type(e).__name__}")
    try:
        wanted = parse_hoists(os.environ.get(ENV_HOISTS))
    except ValueError as e:
        return Installed("sampler_hoist", False, reason=f"env:{e}")
    hoists, off, seen = source_check(wanted)
    if not hoists:
        return Installed("sampler_hoist", False, reason="no_hoist:" + ",".join(f"{h}={w}" for h, w in sorted(off.items())), facts={"digests": seen})
    ledger = Ledger(NAME, impl=IMPL, origin="kit", expected=EXPECTED)
    for k in ("rollouts", "dit_folds", "dit_hits", "atom_builds", "relpos_mats", "memo_hits", "memo_miss", "scond_hits"):
        ledger.set(k, 0)
    ledger.set("hoists", "+".join(hoists))
    head_cls, pc_cls, sc_cls = H.DiffusionHead, DT.PairConditioning, DT.SingleConditioning
    enc_cls, ast_cls, tst_cls = A.AtomEncoder, A.AtomAttentionStack, DT.AtomTransformerStack
    stock = {"sample": head_cls.sample, "pc": pc_cls.forward, "sc": sc_cls.forward, "enc": enc_cls.forward, "ast": ast_cls.forward, "tst": tst_cls.forward}
    rebind(head_cls, "sample", make_sample(stock["sample"], hoists, ledger), stock["sample"])
    if "dit_bias" in hoists:
        rebind(pc_cls, "forward", make_pc_forward(stock["pc"]), stock["pc"])
    if "scond" in hoists:
        rebind(sc_cls, "forward", make_sc_forward(stock["sc"]), stock["sc"])
    if "atom_c" in hoists:
        rebind(enc_cls, "forward", make_enc_forward(stock["enc"]), stock["enc"])
    if "atom_pair" in hoists:
        rebind(ast_cls, "forward", make_astack_forward(stock["ast"]), stock["ast"])
    if "atom_win" in hoists:
        rebind(tst_cls, "forward", make_tstack_forward(stock["tst"]), stock["tst"])

    def restore() -> None:
        for cls_, attr, key in ((head_cls, "sample", "sample"), (pc_cls, "forward", "pc"), (sc_cls, "forward", "sc"), (enc_cls, "forward", "enc"),
                                (ast_cls, "forward", "ast"), (tst_cls, "forward", "tst")):
            cur = getattr(cls_, attr)
            if getattr(cur, "__wrapped_stock__", None) is stock[key]:
                setattr(cls_, attr, stock[key])

    def gate():
        return ledger.gate(require_served=False) if not enabled() else ledger.gate()
    return Installed("sampler_hoist", True, lines=[_line(ledger, tag, hoists, off)], gates=[gate],
                     facts={"impl": IMPL, "hoists": hoists, "off": off, "digests": seen, "ledger": ledger, "restore": restore})

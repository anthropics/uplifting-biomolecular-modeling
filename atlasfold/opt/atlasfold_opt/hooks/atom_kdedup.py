"""Lever atom_kdedup — the windowed atom attention's KEY side evaluated once per atom row instead of once per key window.

Stock: AtomTransformerStack.forward (diffusion_transformer.py L275-287) hands every atom block `a_q, a_k = local_attn_index(a, dim=-3)`:
a_q is the [B, N, W, 4*14, c] query-window view of the atom tensor `a`, a_k the [B, N, W, 12*14, c] KEY-window view — an overlapping
`unfold` (window 12 residues, step 4) of a zero-padded copy of the same rows (misc.py LocalAttentionIndex.to_k L121-138): every atom row
sits in three key windows (plus the 2 x 4-residue halo per sample).  CrossAttention.forward (attention.py L266-294) then runs the key
side ROW-WISE on that view: `a_k = self.adaln_a_k(a_k, single_cond_k)` (normalization.py L71-76: LayerNorm(a_k) — whose first act is a
contiguous copy of the overlapping view —, sigmoid(linear_g(layernorm_cond(cond))) * a + linear_bias(cond)) and, inside Attention.forward
(L69), `k, v = self.linear_k(a_k), self.linear_v(a_k)`: LayerNorm, the AdaLN combine and two c x c projections over W*168 rows per sample,
i.e. 3x the atom rows, at every block of both stacks, every denoiser call.  The key-side LayerNorm, the AdaLN elementwise and
linear_k / linear_v are the four largest module costs of the atom encoder + decoder.

This lever (site CrossAttention.forward): every one of those operations is a ROW-WISE map (LayerNorm over c, an elementwise combine with a
conditioning row that is itself the same unfold of per-atom conditioning rows, a Linear), and a_k's strides PROVE it is `rows.unfold`
(stride pattern (.., .., step*c, c, 1) over one storage: a_k[b, n, w, j, :] == rows[b, n, w*step + j, :] by address arithmetic, whoever built
it) — so f(a_k) == unfold(f(rows)) element for element.  The lever reads the row matrix R [B, N, (W-1)*step + Lk, c] behind a_k and R_c
behind single_cond_k as VIEWS (as_strided over the same storage and offset, bounds-checked), evaluates normalization.py L71-76 and the two
key/value linears ON THE ROWS (the same modules, statements, dtypes and autocast state, ~1/3 of the rows), and hands Attention.forward
the results as the SAME overlapping window views k / v [B, N, W, Lk, c] (as_strided with a_k's own stride pattern): the attention statement
below — upstream's MATH SDPA in exact / off, atom_sdpa's fused kernel in fast (both split heads by a view of the last dim and read k / v by
stride) — is untouched and receives element-identical operands.  How k / v reach it BY NAME: for the duration of the one `self.attn(a_q,
a_k, mask, pair_bias)` call (L285, verbatim) the block's `attn.linear_k` / `attn.linear_v` INSTANCES answer `linear_k(x)` with the held window
view when `x is a_k` (identity on the very object handed down; any other input is the class forward current at call time) — the stock
`k, v = self.linear_k(a_k), self.linear_v(a_k)` of Attention.forward and of the known wrappers over it (KNOWN_ATTENTION_WRAPPERS) read them unchanged; nothing
else of those functions is relied on.  The query side (`adaln_a_q`, linear_q, the gate, linear_o, linear_ada_out) is the stock call sequence.
Per element the arithmetic is the stock arithmetic in the stock order; what changes is the GEMM / LayerNorm ROW COUNT (M), which a BLAS
is free to answer with another kernel, so byte-identity with `--mode off --det 1` is a property of the BLAS on the card, not of the
arithmetic: in the exact row a per-card row floor applies (EXACT_MIN_ROWS below); the fast / big rows (tolerance class) have none.

Halo rows: the 4-residue zero padding of to_k is part of R (it is the padded copy): stock evaluates AdaLN / linear_k on those zero rows inside
every window, the lever evaluates the same statements on the same zero rows once — identical operands either way (and the keys are masked).
Conditioning leaves: `sigmoid(linear_g(layernorm_cond(R_c)))` and `linear_bias(layernorm_cond(R_c))` depend on the roll-out-constant atom
conditioning only.  Inside a sampler_hoist roll-out that holds single_cond_k as an invariant (its atom_win hoist) they are evaluated ONCE per
roll-out per block and join that roll-out's memo (held by it, freed when DiffusionHead.sample returns — the atom_cond hoist's own lifetime
rule; counted cond_memo_hit / cond_memo_miss); outside one (sampler_hoist absent, ablated or stock for the roll-out) they are evaluated per call
on the rows (cond_per_call) — 1/3 of the stock work either way, nothing held by this lever past a call.  Memory: the row-shaped LayerNorm
output, AdaLN output, k and v REPLACE stock's window-shaped ones (3x larger): peak can only fall.
Composition: Attention.forward wrappers known by qualname at install (KNOWN_ATTENTION_WRAPPERS: they read k / v through the two linears as
upstream does), atom_tf32 (this body runs inside its window: TF32 products as the stock body would), sampler_hoist (adaln_a_q / the gates keep
their instance memos — they are CALLED, not re-stated; adaln_a_k's memo is not consulted: its leaves are window-shaped), denoiser_graph (Python
at warm-up + capture, kernels replayed; memo leaves are created at the eager warm-up call and read by reference by the graph), diffusion_bf16
(the atom islands are autocast-off: fp32, as stock), relpos_lazy / ln_bf16 (class-level Linear / LayerNorm forwards are what the rows call).
Guards, by name: AFO_ATOM_KDEDUP=0 -> installed, every call the statement below (`disabled`); an a_k / single_cond_k that is not an unfold
view of one row matrix (`kv_layout` / `cond_layout`), a rank other than 5 (`rank`) -> the statement below, counted (none occurs in this
model: unexpected -> the gate refuses); at install: CrossAttention.forward already wrapped (`wrapped:<qualname>`), a re-stated / relied-on
stock text whose digest changed (`source:<fn>`), an Attention.forward wrapper not known here (`attention_wrapper:<qualname>`) -> not applied."""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from . import Installed, rebind, size_gated

NAME = "LOCAL.atlasfold.atom_kdedup"
ENV = "AFO_ATOM_KDEDUP"
TARGET = "atlasfold.model.network.attention"                                   # CrossAttention.forward (+ the Attention modules' linear_k / linear_v instances)
T_NORM = "atlasfold.model.network.primitives.normalization"
IMPL = "rows(adaln_k,linear_k,linear_v)+unfold_view"
EXPECTED = ("disabled", "below_exact_min_rows")
KNOWN_ATTENTION_WRAPPERS = ("Attention.forward[atlasfold_opt:dit_sdpa]", "Attention.forward[atlasfold_opt:atom_sdpa]")   # wrappers that read k / v through self.linear_k(a_k) / self.linear_v(a_k) as upstream does
SOURCE_SHA256 = {                                                              # opt_core.diffusion_loop.source_guard.source_sha256 of the stock texts re-stated (CrossAttention, AdaLN) or relied on (Attention: the two linear calls)
    "CrossAttention.forward": ("e090db6b55a3b9b53556bbf3da2089f7589195d6b61b89751984abfcffeb58c5",),
    "AdaLN.forward": ("2365a0ab8c97565094c686043c4e7d46c14b3b9effc10c554c695efbd4b8f680",),
    "Attention.forward": ("ce542cab4d11c7d851d4743597a8c0ae27a11ea1b2b27ad7ee88c59b3e1880bd",),
}
STASH = "_atlasfold_opt_atom_kdedup_kv"                                       # attribute on the Attention instance for the duration of one served call
KV_TAG = "atlasfold_opt:atom_kdedup.kv"
_STATE = {"override": None, "kv_contiguous": False, "exact_min_rows": 0}   # kv_contiguous: in-process knob (not a kit switch) — hand k / v over as materialised windows instead of views; exact_min_rows: the exact row's per-card floor in force (set at install)
# The EXACT row's floor per card, in atom rows behind a_k (the `r<rows>` of the LEVER line's shapes): the row count is the only thing that changes,
# and a BLAS may answer a smaller row count with another kernel — on cc 8.0 the row form is byte-identical to `--mode off --det 1` from this floor up,
# so in the exact row on that card every call with fewer rows runs the stock statement, BY NAME (`below_exact_min_rows`); cc 9.0 needs no floor (0).
# A card not listed takes the conservative floor (EXACT_MIN_ROWS_DEFAULT).  The fast / big rows (tolerance class) have no floor on any card.
EXACT_MIN_ROWS = {(9, 0): 0, (8, 0): 9072}
EXACT_MIN_ROWS_DEFAULT = 9072


def exact_min_rows(mode: str, cc) -> int:
    """The row floor this activation enforces: the table's value for `cc` in the exact row (unlisted card: EXACT_MIN_ROWS_DEFAULT), 0 elsewhere."""
    if mode != "exact" or cc is None:                                       # tolerance-class rows: no floor; no CUDA device (CPU hosts): nothing to key on
        return 0
    return int(EXACT_MIN_ROWS.get(tuple(cc), EXACT_MIN_ROWS_DEFAULT))


def _cc():
    try:
        return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def enabled() -> bool:
    if _STATE["override"] is not None:
        return bool(_STATE["override"])
    return os.environ.get(ENV, "1") != "0"


def bench_arm(label: str) -> None:
    """In-process override of AFO_ATOM_KDEDUP (not a kit switch): label 'A' = the statement below (lever inert), any other label = the row form."""
    _STATE["override"] = (label != "A")


bench_arm.state = lambda: {"atom_kdedup": enabled()}   # type: ignore[attr-defined]


# ----------------------------------------------------------------------------------------------------------------- geometry: is `t` rows.unfold ?
def unfold_rows(t) -> Optional[SimpleNamespace]:
    """For a rank-5 [B, N, W, Lk, c] tensor whose strides say it is an overlapping unfold (window Lk rows, step `step` rows) of ONE row matrix per
    (b, n): the geometry + that row matrix as a VIEW [B, N, Rn, c] (Rn = (W-1)*step + Lk) over the same storage.  None when the strides do not
    have that form or the storage is too small for it (nothing is assumed about who built `t`)."""
    if not isinstance(t, torch.Tensor) or t.dim() != 5:
        return None
    B, N, W, Lk, c = (int(x) for x in t.shape)
    sB, sN, sW, sL, sc = (int(x) for x in t.stride())
    if c < 1 or Lk < 1 or W < 1 or sc != 1 or sL != c or sW <= 0 or sW % c != 0:
        return None
    step = sW // c
    if step > Lk:                                                              # gaps between windows: not the local-attention unfold (and rows would be undefined)
        return None
    if (B > 1 and sB < 0) or (N > 1 and sN < 0):
        return None
    Rn = (W - 1) * step + Lk
    off = int(t.storage_offset())
    need = off + (B - 1) * (sB if B > 1 else 0) + (N - 1) * (sN if N > 1 else 0) + Rn * c
    try:
        have = t.untyped_storage().nbytes() // t.element_size()
    except Exception:  # noqa: BLE001
        return None
    if need > have:
        return None
    rows = t.as_strided((B, N, Rn, c), (sB if B > 1 else Rn * c, sN if N > 1 else Rn * c, c, 1), off)
    return SimpleNamespace(B=B, N=N, W=W, Lk=Lk, c=c, step=step, Rn=Rn, rows=rows)


def fold_windows(rows: torch.Tensor, g: SimpleNamespace) -> torch.Tensor:
    """rows [B, N, Rn, c'] (contiguous) -> the overlapping window view [B, N, W, Lk, c'] with the stride pattern of the tensor `g` was read from."""
    B, N, Rn, c = (int(x) for x in rows.shape)
    assert (B, N, Rn) == (g.B, g.N, g.Rn), ((B, N, Rn), (g.B, g.N, g.Rn))
    rows = rows.contiguous()
    win = rows.as_strided((B, N, g.W, g.Lk, c), (N * Rn * c, Rn * c, g.step * c, c, 1), int(rows.storage_offset()))
    return win.contiguous() if _STATE.get("kv_contiguous") else win


# ----------------------------------------------------------------------------------------------------------------- k / v hand-over by identity
def _kv_forward(mod, owner, which: str):
    """Instance forward of `owner.linear_k` / `owner.linear_v`: while `owner` carries a stash for the object `x`, answer the held window view;
    any other input (or no stash) is the class forward current at call time (relpos_lazy and friends compose)."""
    def forward(x, *args, **kwargs):
        st = getattr(owner, STASH, None)
        if st is not None and not args and not kwargs and x is st.key:
            st.hits[which] += 1
            return st.k if which == "k" else st.v
        return type(mod).forward(mod, x, *args, **kwargs)
    forward.__qualname__ = f"{type(mod).__name__}.forward[{KV_TAG}.{which}]"
    return forward


def _arm_instances(attn) -> Optional[str]:
    """Install the two instance forwards on `attn.linear_k` / `attn.linear_v` once; None when in place, else the reason word."""
    for which, name in (("k", "linear_k"), ("v", "linear_v")):
        m = getattr(attn, name, None)
        if not isinstance(m, torch.nn.Module):
            return f"no_{name}"
        f = m.__dict__.get("forward")
        if f is None:
            m.forward = _kv_forward(m, attn, which)
        elif KV_TAG not in getattr(f, "__qualname__", ""):
            return f"{name}_instance_forward"
    return None


def remove_instances(module) -> int:
    """Undo _arm_instances under `module` (tests / restore)."""
    n = 0
    for m in module.modules():
        f = m.__dict__.get("forward")
        if f is not None and KV_TAG in getattr(f, "__qualname__", ""):
            del m.__dict__["forward"]; n += 1
        if STASH in m.__dict__:
            try:
                delattr(m, STASH)
            except Exception:  # noqa: BLE001
                pass
    return n


# ----------------------------------------------------------------------------------------------------------------- the conditioning leaves (roll-out constant)
def _rollout():
    """sampler_hoist's open roll-out state (its Roll) when that lever is installed and a roll-out is open in this thread, else None."""
    try:
        from . import sampler_hoist as SH
        return SH._cur()
    except Exception:  # noqa: BLE001
        return None


def _cond_leaves(ada, Rc: torch.Tensor, cond_key, ledger) -> Tuple[torch.Tensor, torch.Tensor]:
    """normalization.py L74 + the cond-only factors of L76 on the conditioning ROWS: (sigmoid(linear_g(layernorm_cond(Rc))), linear_bias(...)).
    Memoised for the roll-out when sampler_hoist holds `cond_key` (the single_cond_k object) as a roll-out invariant; per call otherwise."""
    ro = _rollout()
    if ro is not None and "atom_cond" in getattr(ro, "hoists", ()) and id(cond_key) in getattr(ro, "inv_ids", ()):
        key = ("atom_kdedup.rows", id(ada), id(cond_key))
        sb = ro.memo.get(key)
        if sb is None:
            c = ada.layernorm_cond(Rc)
            sb = (ada.sigmoid(ada.linear_g(c)), ada.linear_bias(c))
            ro.memo[key] = sb
            ro.hold(*sb)
            ledger.count("cond_memo_miss")
        else:
            ledger.count("cond_memo_hit")
        return sb
    c = ada.layernorm_cond(Rc)
    ledger.count("cond_per_call")
    return ada.sigmoid(ada.linear_g(c)), ada.linear_bias(c)


# ----------------------------------------------------------------------------------------------------------------- install
def _attention_chain(att_cls) -> Tuple[Optional[str], object]:
    """(None, innermost) when Attention.forward is upstream's text possibly beneath KNOWN wrappers; else (reason, innermost-so-far)."""
    f = att_cls.forward
    depth = 0
    while hasattr(f, "__wrapped_stock__") and depth < 8:
        q = getattr(f, "__qualname__", "")
        if q not in KNOWN_ATTENTION_WRAPPERS:
            return f"attention_wrapper:{q or type(f).__name__}", f
        f = f.__wrapped_stock__; depth += 1
    return None, f


def source_check() -> Tuple[Optional[str], dict]:
    import importlib
    from opt_core.diffusion_loop.source_guard import source_sha256
    att = importlib.import_module(TARGET); norm = importlib.import_module(T_NORM)
    seen = {}
    cur = att.CrossAttention.forward
    if hasattr(cur, "__wrapped_stock__"):
        return f"wrapped:{getattr(cur, '__qualname__', '?')}", seen
    why, inner_att = _attention_chain(att.Attention)
    if why is not None:
        return why, seen
    ada_f = norm.AdaLN.forward
    while hasattr(ada_f, "__wrapped_stock__"):
        ada_f = ada_f.__wrapped_stock__
    for fn, f in (("CrossAttention.forward", cur), ("Attention.forward", inner_att), ("AdaLN.forward", ada_f)):
        try:
            d = source_sha256(f)
        except Exception as e:  # noqa: BLE001
            d = f"unreadable:{type(e).__name__}"
        seen[fn] = d
        if d not in SOURCE_SHA256[fn]:
            return f"source:{fn}", seen
    return None, seen


def make_forward(stock, ledger):
    def forward(self, a_q, a_k, mask, pair_bias=None, single_cond_q=None, single_cond_k=None):
        if not enabled():
            ledger.fallback("disabled")
            return stock(self, a_q, a_k, mask, pair_bias, single_cond_q, single_cond_k)
        if not isinstance(a_q, torch.Tensor) or a_q.dim() != 5:
            ledger.fallback("rank")
            return stock(self, a_q, a_k, mask, pair_bias, single_cond_q, single_cond_k)
        g = unfold_rows(a_k)
        if g is None or int(a_q.shape[-1]) != g.c or tuple(int(x) for x in a_q.shape[:3]) != (g.B, g.N, g.W):
            ledger.fallback("kv_layout")
            return stock(self, a_q, a_k, mask, pair_bias, single_cond_q, single_cond_k)
        if g.Rn < _STATE["exact_min_rows"]:                                       # the exact row's per-card floor (EXACT_MIN_ROWS): the stock statement, by name
            ledger.fallback("below_exact_min_rows")
            return stock(self, a_q, a_k, mask, pair_bias, single_cond_q, single_cond_k)
        gc = None
        if self.use_conditioning:
            gc = unfold_rows(single_cond_k)
            if gc is None or (gc.W, gc.Lk, gc.step, gc.Rn) != (g.W, g.Lk, g.step, g.Rn) or gc.B not in (1, g.B) or gc.N not in (1, g.N):
                ledger.fallback("cond_layout")
                return stock(self, a_q, a_k, mask, pair_bias, single_cond_q, single_cond_k)
        attn = self.attn
        why = _arm_instances(attn)
        if why is not None:
            ledger.fallback(why)
            return stock(self, a_q, a_k, mask, pair_bias, single_cond_q, single_cond_k)
        # attention.py L266-273 — the query side verbatim; the key side = normalization.py L71-76 on the ROWS behind a_k / single_cond_k
        if self.use_conditioning:
            assert single_cond_q is not None and single_cond_k is not None
            a_q = self.adaln_a_q(a_q, single_cond_q)
            ada = self.adaln_a_k
            # Line 1
            a_rows = ada.layernorm(g.rows)
            # Line 2 + the cond-only factors of Line 3 (roll-out constant)
            sig, bias = _cond_leaves(ada, gc.rows, single_cond_k, ledger)
            # Line 3
            a_rows = sig * a_rows + bias
        else:
            assert single_cond_q is None and single_cond_k is None
            a_q = self.layernorm_a_q(a_q)
            a_rows = self.layernorm_a_k(g.rows)

        if self.use_pair_bias:
            assert pair_bias is not None
        else:
            assert pair_bias is None

        if mask.ndim == a_q.ndim - 1:
            mask = mask.unsqueeze(-2)  # [*, Lk] -> [*, 1, Lk]

        # attention.py L69 `k, v = self.linear_k(a_k), self.linear_v(a_k)` on the rows, folded back into a_k's own window layout (views)
        k_w = fold_windows(attn.linear_k(a_rows), g)
        v_w = fold_windows(attn.linear_v(a_rows), g)
        del a_rows
        st = SimpleNamespace(key=a_k, k=k_w, v=v_w, hits={"k": 0, "v": 0})
        setattr(attn, STASH, st)
        try:
            # Prepare attention pair bias input
            # [*, W, Lq/k, c] -> [*, W, H, Lq/k, c_h]
            out = self.attn(a_q, a_k, mask, pair_bias)  # [*, Lq, c]
        finally:
            try:
                delattr(attn, STASH)
            except AttributeError:
                pass
        if st.hits["k"] != 1 or st.hits["v"] != 1:                            # the statement below projected a_k itself (correct, not deduplicated): say so
            ledger.count("kv_unread")
        del st, k_w, v_w

        # Gate output
        g_ = torch.sigmoid(self.linear_g(a_q))  # [*, Lq, c]
        a_q = self.linear_o(g_ * out)  # [*, Lq, c]

        if self.use_conditioning:
            assert single_cond_q is not None
            a_q = torch.sigmoid(self.linear_ada_out(single_cond_q)) * a_q
        ledger.serve(f"W{g.W}xN{g.N}x{int(a_q.shape[-2])}x{g.Lk}r{g.Rn}")
        return a_q
    forward.__qualname__ = "CrossAttention.forward[atlasfold_opt:atom_kdedup]"
    return forward


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    from opt_core.counters import Ledger
    try:
        att = importlib.import_module(TARGET)
        importlib.import_module(T_NORM)
    except Exception as e:  # noqa: BLE001
        return Installed("atom_kdedup", False, reason=f"import:{type(e).__name__}")
    why, seen = source_check()
    if why is not None:
        return Installed("atom_kdedup", False, reason=why, facts={"digests": seen})
    ledger = Ledger(NAME, impl=IMPL, origin="kit", expected=EXPECTED)
    for k in ("cond_memo_hit", "cond_memo_miss", "cond_per_call", "kv_unread"):
        ledger.set(k, 0)
    _STATE["exact_min_rows"] = exact_min_rows(mode, _cc())
    cls = att.CrossAttention
    stock = cls.forward
    rebind(cls, "forward", make_forward(stock, ledger), stock)

    def restore() -> None:
        if getattr(cls.forward, "__wrapped_stock__", None) is stock:
            setattr(cls, "forward", stock)

    def line():
        ev = {"exact_min_rows": _STATE["exact_min_rows"]}
        if not enabled():
            ev["switch"] = f"{ENV}=0"
        return ledger.line(tag, **ev)
    return Installed("atom_kdedup", True, lines=[line], gates=[size_gated(ledger)],
                     facts={"impl": IMPL, "env": os.environ.get(ENV, "1"), "digests": seen, "ledger": ledger, "restore": restore, "exact_min_rows": _STATE["exact_min_rows"]})

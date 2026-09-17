"""stepgraph — the tree's own lever on the diffusion sampler: ONE denoiser step captured into a CUDA graph per `sample_diffusion()` call and
replayed for the remaining steps (the core's capture discipline, `opt_core.capture.graphs.GraphCache`).

What it does. Inside one `sample_diffusion()` call (opendde/model/generator.py, Alg. 18) the 200 denoiser calls
`denoise_net(x_noisy=…, t_hat_noise_level=…, <roll-out constants>)` have ONE static signature. The lever runs the first call eagerly (the kit's
`dit_hoist` records its step-invariant buffers there: they are then constants of this sampler call, read by address), hands the second call to
`GraphCache.run` — eager-first answer, one warm-up on static copies of (x_noisy, t_hat), capture into a private pool, one replay held BIT-EXACT
to the eager answer (the core's capture discipline) — holds the first replay on NEW inputs (call 3) to an eager re-run (bit-exact under the
deterministic recipe; within the eager step's own run-to-run spread otherwise, see below), and replays it
for calls 3..200: per step one copy-in of x_noisy and t_hat, one graph launch, one clone of the denoised coordinates. The graph, its pool and the
static copies live for ONE sampler call and are dropped in its `finally` (`GraphCache.reset`: synchronize, drop, gc, empty_cache) — the next item
captures afresh on its own shapes and its own hoist buffers (`recaptures` counts captures after the process's first), so no stale roll-out constant
or hoist slot can ever be replayed (the DITFAST hoist reallocates its slots per sampler call). A once-per-process poison probe replays the armed graph on NaN coordinates and requires NaN out — the static inputs are really read.

What it does not touch. The sampler loop is STOCK: every random draw (initial noise, the per-step augmentation and noise) is made by the stock
code, eagerly, from the stock generators in stock order (`rng="forbid"`: a denoiser that drew from the CUDA generator would be refused by name).
Numerics: a replay launches the kernels the eager call launches, in the same order on the same data. Under `--det 1` (torch deterministic
algorithms) the eager denoiser is reproducible and the replay is HELD bit-exact to it (calls 2 and 3) or the lever disables itself by name for
the process; without `--det` the eager denoiser is not run-to-run reproducible (float atomics in the atom->token aggregation), so the first
replay on new inputs is held to an eager re-run within max(`ODDE_STEPGRAPH_RTOL`, SPREAD_K x the eager step's own run-to-run spread, measured
there by running it twice: the float atomics of the atom->token sums make the stock step itself non-reproducible at ~1e-3 of max|x| in bf16)
+ finiteness instead, and the LEVER row says which check ran (`held=` / `tol_checks= tol_rel= spread=`). Non-finite coordinates after a replayed call disable the lever and re-run the stock sampler for that
call (same rollout seed) — never handed back.

The atom->token sum. torch 2.7's deterministic `scatter_add_` on CUDA (the sort-based index_put path) is NOT capturable (it synchronises the
stream), and its default CUDA path is float atomics (not reproducible run to run): the one such op in the step is the atom->token mean of
`AtomAttentionEncoder` (`opendde.utils.scatter_utils.scatter_sum` through `aggregate_atom_to_token`). Inside the graphed region of a sampler
call (both numerics modes) the lever serves that call with `segment_sum`:
the per-token sequential fp32-accumulated sum over the token's atoms in ascending atom position (the stable-sort order torch's kernel uses; a
gather table built from `atom_to_token_idx` at admission, outside the capture), cast once at the end — the arithmetic torch's deterministic kernel performs, so it is BITWISE the eager result (every capture is
then held bit-exact to the eager step by the cache, which is the standing proof; tests/test_stepgraph.py holds the sum bitwise on CPU).
The head call and everything outside a sampler context run the stock op.

Requires the LayerNorms to be recorded by a stream capture: upstream's fused LayerNorm launches on the legacy stream (lever `lnstream` puts it
on the current stream). Before the first capture of a process a functional canary (one LayerNorm of the class the model is built from,
captured, replayed on NEW input, compared with eager) refuses by name (`layernorm_not_capture_safe`) when it is not.

Composition (cli.pred, like chunk_lift's ceiling): `plan()` reads the query's residue tokens; the lever is composed INTO the line only when
every item counts within [`ODDE_STEPGRAPH_MIN_TOKENS`, `ODDE_STEPGRAPH_MAX_TOKENS`] (defaults GATE_MIN / GATE_MAX below;
no query read = composed in, the call's own admission decides) and the line does not run the offload unit (`ODDE_OFFLOAD` exported: big above its offload gate streams the pair tensors the
step reads) — else composed out by name (LEVER row `state=off reason=above_gate:<t>/<g> | below_gate:<t>/<g> | offload_line`); above a card's
row (CARD_ROWS: sm_80 caps it at 1,024 tokens) `card_gate:<word>_above:<cap>`; above the line's sampler-word row (WORD_ROWS: the big word caps
it at 400 tokens — the resident big line carries the lever below its offload gate; above that row the replayed fp32 step is slower than the
eager step under that word and the private pool grows with N^2) `word_gate:<word>_above:<cap>`. `--n_gpu P>1` (line BIG_TP) never carries it
(modes.BIG_TP_DROP). Refused by name at the first denoiser call of a sampler call (the call then runs the
stock eager sampler; counted in `refused`, the run PARTIAL by the kit's rule when no call of the process replayed): positional call, autograd
on, training mode, `inplace_safe=False` (upstream then mutates the cached p_lm), Fold-CP / atom-window arguments, guidance (TFG) enabled, a
distributed world, `DIT_HOIST_CROSSCHECK` set (a host-side torch.equal inside the step), the LayerNorm canary, the memory admission
(`ODDE_STEPGRAPH_MEMFRAC` of free device memory for the estimated step arena), a structural-token count over HARD_MAX_STRUCT_TOKENS.

Installed from `stack._apply` for the lines that carry the lever through the core's per-site patch of the module-global name the model
resolves at call time (`opendde.model.opendde.sample_diffusion`; the DITFAST hoist wraps the same name at runner creation — either order composes)
plus a class-level route of `DiffusionModule.__call__` while a sampler call of this lever is live. Env knobs (registry.KNOBS; exported by no
line): ODDE_STEPGRAPH_MAX_TOKENS, _MIN_TOKENS, _HEAD (eager calls before capture, 1), _RTOL, _MEMFRAC (0.5), _VERBOSE (0).

"""
from __future__ import annotations

import json
import re
import os
import sys
import time
from typing import Any, Callable, Dict, Optional

TAG = "opendde-opt"
LEVER = "stepgraph"
TARGET = "opendde.model.opendde"                      # the module whose global `sample_diffusion` the model resolves at call time
ATTR = "sample_diffusion"
DM_MODULE, DM_CLASS = "opendde.model.modules.diffusion", "DiffusionModule"
SCATTER_MODULE, SCATTER_ATTR = "opendde.utils.scatter_utils", "scatter_sum"   # the atom->token aggregation's sum (scatter_mean calls it by module-global name)
GATE_MAX_ENV, GATE_MIN_ENV = "ODDE_STEPGRAPH_MAX_TOKENS", "ODDE_STEPGRAPH_MIN_TOKENS"
GATE_MAX, GATE_MIN = 99999, 0                        # residue tokens (big.count_tokens): the composed size gate's defaults (open; the card / word rows and the call's own admission bound it)
HARD_MAX_STRUCT_TOKENS = 6000                        # structural (diffusion) tokens: a hard refusal inside the call whatever the plan said (~1.93 x residues)
RTOL, SPREAD_K = 1e-4, 16.0                          # without --det: the first replay on new inputs within max(RTOL, SPREAD_K x the eager step's own run-to-run spread) of eager
ABOVE_GATE, BELOW_GATE, OFFLOAD_LINE, CARD_GATE, WORD_GATE = "above_gate", "below_gate", "offload_line", "card_gate", "word_gate"
CARD_ROWS = {"80": (1024, "sm80_replay_not_bit_exact")}    # per compute capability (stack.gpu_info()'s "sm"): (the largest residue-token count at which the per-step graph
#   is composed in on that card, the word). sm_80 (A100): the captured denoiser step holds bit-exact to the eager step up to 1,024 tokens, but its
#   capture-time consistency replay differs from the eager step above that (seen at 1,400 tokens) -> composed OUT by name above 1,024 tokens
#   there: LEVER state=off reason=card_gate:sm80_replay_not_bit_exact_above:1024.
#   sm_90 (H100) and cards without a row: no cap (the general gate only). A capture whose replay differs from the eager step on any card / size
#   the rows do not anticipate is the call's own named step-aside to the eager sampler (MISMATCH_ASIDE below), never a partial run.
SAMPLER_WORD_SWITCH = "ODDE_DIT_ATTN"                  # the switch that carries a line's sampler tier word (modes.SAMPLER_SWITCHES: exact on S1, fast on LSTAR2A, big on the big lines)
WORD_ROWS: Dict[str, tuple] = {"big": (400, "big_graph_not_ahead")}   # per sampler tier word: (the largest residue-token count at which the per-step graph is
#   composed in under that word, the word) — a per-word row like CARD_ROWS, asked after the card's. big: with the plan-for-capture window
#   (`planning()`, below) the resident big line's captured fp32 unfused step holds to the eager step at every size on H100, but it only PAYS up to
#   400 tokens: above that bucket the big word's fp32 DiT cells name fpf_apb:fp16 in the eager column and fp32 fpf_apb in the graph column, so the
#   replayed step's kernels are slower than the eager step's, and the private pool grows with N^2 -> composed OUT by name above 400 tokens under
#   that word: LEVER state=off reason=word_gate:big_graph_not_ahead_above:400 (the eager sampler serves there). Without the window, bindings
#   inside the step that choose by the live capture state alone (the pair-bias-attention binding's timing column; ln_core's step-aside to
#   upstream's LayerNorm while capturing) would make the eager oracle and the captured step run different kernels. exact / fast: no row (fast's
#   fp16 DiT cells name one row in both columns at every size).
MISMATCH_ASIDE = "verify_mismatch"                   # the capture's consistency replay (core GraphCache stages verify1 / verify2) or this lever's first-replay hold
#   differed from the eager step BEFORE any graph output was consumed (the cache answered that call eagerly): the graph is off for the process
#   BY NAME and the stock eager sampler serves every step -- a step-aside (ASIDE_WORDS), the run complete; refused-word form
#   verify_mismatch_<stage>_maxabs_<diff> / verify_mismatch_hold_rel_<rel>_bar_<bar>.
CAPTURE_CHECK_STAGES = ("verify1", "verify2")          # the core cache's consistency stages at capture (its refusal kind is their common stem): a mismatch there is the step-aside
CAPTURE_CHECK_KIND = CAPTURE_CHECK_STAGES[0][:-1]     # the refusal kind word of those stages in the cache's `refused (<kind>): …` disable reason
__version__ = "0.1.0"

STATS: Dict[str, Any] = {"installed": False, "armed": False, "routed": False, "disabled": None, "ln_canary": None, "why_last": None,
                         "sampler_calls": 0, "engaged_calls": 0, "planned_calls": 0, "denoiser_calls": 0, "eager_head": 0, "eager_other": 0, "captures": 0,
                         "recaptures": 0, "replays": 0, "held": 0, "tol_checks": 0, "tol_max_rel": 0.0, "eager_spread_max": 0.0, "poison_probe": None,
                         "time": {"eager_calls": 0, "eager_host_s": 0.0, "replay_calls": 0, "replay_host_s": 0.0, "replay_gpu_s": 0.0, "sampler_s": 0.0, "sampler_calls_timed": 0, "steps_timed": 0},
                         "refused": {}, "failures": 0, "verify_aside": None, "capture_s": 0.0, "pool_mib_max": 0, "nonfinite_reruns": 0, "det": None,
                         "segsum_calls": 0, "segsum_passthrough": 0, "keys": [], "core": {}, "patch": None}
_STATS0 = json.loads(json.dumps({k: v for k, v in STATS.items() if k != "patch"}))
_CUR = [None]                                        # the live sampler context (one sampler call at a time; nested calls stand aside)


def planning() -> bool:
    """The plan-for-capture window: True while a sampler call this
    lever has ADMITTED runs its denoiser steps — the eager head call, the capture cache's eager reference / warm-up / capture / consistency
    replays, the held first replay's eager re-runs, the replays — i.e. from the first denoiser call's admission to the sampler call's end, unless
    the call refused or stood aside meanwhile. A binding inside the step whose choice depends on the capture state reads it and makes, for EVERY
    call of the window, the choice it makes under capture: levers/SAMPLER `odde_apb_bind.selection` asks the provider's graph-timed column under a
    tier word (fast | big; an exact word keeps following the literal capture state — its admissible rows equal the statement bitwise), and
    `lncore` steps aside to upstream's LayerNorm (what the graph records) — so the eager oracle the capture is checked and held against IS the
    captured step's kernels. False outside a sampler call, in a refused / stood-aside call (its steps are the stock eager sampler's: the eager
    column, the provider LayerNorm — as without the lever), with the lever off. A host-side flag; no device work."""
    c = _CUR[0]
    return c is not None and bool(getattr(c, "admitted", False)) and c.refused is None and STATS.get("disabled") is None

_ORIG: Dict[str, Any] = {}
_PLAN = {"plan": None}


class ActivationError(RuntimeError):
    """`opendde.model.opendde` has no `sample_diffusion` to wrap: the kit's activation fails by name."""


def _is_oom(e) -> bool:
    """The core's one out-of-memory classifier (imported at call, not at module load: modes.resolve imports this module on trees without the core)."""
    from opt_core.oom import is_oom as _core_is_oom
    return _core_is_oom(e)


def _env(k: str, default: str) -> str:
    return os.environ.get("ODDE_STEPGRAPH_" + k, default)


def _log(msg: str) -> None:
    sys.stderr.write(f"[{TAG}] {LEVER}: {msg}\n")
    sys.stderr.flush()


def _vlog(msg: str) -> None:
    if _env("VERBOSE", "0") not in ("", "0"):
        _log(msg)


def _refuse(reason: str) -> str:
    n = STATS["refused"].get(reason, 0)
    STATS["refused"][reason] = n + 1
    STATS["why_last"] = reason
    if n == 0:
        _log(f"REFUSED reason={reason} -> the stock eager sampler serves this call")
    return reason


# ---------------------------------------------------------------------------------------------------------------- deterministic, capturable atom->token sum
def _device_refusal(x) -> Optional[str]:
    """The graph path needs CUDA tensors (the unit tests replace this and `_make_cache` with a CPU stand-in)."""
    return None if x.is_cuda else "not_cuda"


def segment_table(atom_to_token_idx, n_token: int):
    """The gather table of `segment_sum` from a 1-D atom->token index (read on the host: call this OUTSIDE a capture): (table [n_token, M] of each
    token's atom positions in ascending position order — the order torch's deterministic kernel adds them in (a stable sort of the index) —,
    mask [n_token, M], M) on the index's device; None for an empty index or one outside [0, n_token)."""
    import torch
    idx = atom_to_token_idx.detach().reshape(-1).to("cpu", torch.long)
    if idx.numel() == 0 or int(idx.min()) < 0 or int(idx.max()) >= n_token:
        return None
    order = torch.argsort(idx, stable=True)                                # atom positions grouped by token, ascending inside a token
    counts = torch.bincount(idx, minlength=n_token)
    m = int(counts.max())
    starts = torch.cumsum(counts, 0) - counts
    j = torch.arange(m)
    mask = j[None, :] < counts[:, None]
    tab = order[torch.where(mask, starts[:, None] + j[None, :], torch.zeros(1, dtype=torch.long))]
    tab = torch.where(mask, tab, torch.zeros(1, dtype=torch.long))
    dev = atom_to_token_idx.device
    return tab.to(dev), mask.to(dev), m


def segment_sum(src, tab, mask, m: int, acc_dtype=None):
    """sum over dim -2 of `src` [..., N_atom, d] into [..., n_token, d]: per token, its atoms added one at a time in atom order into an fp32
    (opmath) accumulator that starts at zero, cast to src.dtype once — torch's deterministic scatter_add_ arithmetic, in capturable ops."""
    import torch
    lead, d = src.shape[:-2], src.shape[-1]
    n_token = tab.shape[0]
    acc_dtype = acc_dtype or (torch.float32 if src.dtype in (torch.float16, torch.bfloat16, torch.float32) else src.dtype)
    g = src.index_select(-2, tab.reshape(-1)).reshape(*lead, n_token, m, d)
    acc = torch.zeros(*lead, n_token, d, dtype=acc_dtype, device=src.device)
    for jj in range(m):
        acc = torch.where(mask[:, jj, None], acc + g[..., :, jj, :].to(acc_dtype), acc)
    return acc.to(src.dtype)


def _scatter_sum_routed(src, index, dim=-1, out=None, dim_size=None):
    """Installed as `opendde.utils.scatter_utils.scatter_sum`: the stock function, except inside the GRAPHED region of a live sampler context
    (the capture cache's eager-first call, warm-up, capture and replays, and the lever's own eager re-runs) whose admission built the segment
    table for THIS index — then `segment_sum`: bitwise the stock's deterministic result, capturable, and reproducible run to run (the stock
    CUDA sum is float atomics without the deterministic recipe), so a replay is held BIT-EXACT to the eager step in either numerics mode. The
    sampler's head call and anything outside a context run the stock function; a call the table does not fit passes through (counted)."""
    import torch
    orig = _ORIG["scatter_sum"]
    ctx = _CUR[0]
    seg = getattr(ctx, "seg", None) if ctx is not None else None
    if seg is None or out is not None or not ctx.graphed_region:
        return orig(src, index, dim, out, dim_size)
    tab, mask, m, n_atom, n_token = seg
    d_ = dim % src.dim()
    ok = (torch.is_tensor(index) and index.dim() == 1 and int(index.shape[0]) == n_atom and (dim_size is None or int(dim_size) == n_token)
          and index.device == tab.device and ((src.dim() >= 2 and d_ == src.dim() - 2) or (src.dim() == 1 and d_ == 0)) and int(src.shape[d_]) == n_atom
          and (index is ctx.seg_index or index.data_ptr() == ctx.seg_index.data_ptr()))
    if not ok:
        STATS["segsum_passthrough"] += 1
        return orig(src, index, dim, out, dim_size)
    STATS["segsum_calls"] += 1
    if src.dim() == 1:
        return segment_sum(src[:, None], tab, mask, m)[:, 0]
    return segment_sum(src, tab, mask, m)


def _route_scatter() -> None:
    if "scatter_sum" in _ORIG:
        return
    mod = sys.modules.get(SCATTER_MODULE) or __import__(SCATTER_MODULE, fromlist=["_"])
    _ORIG["scatter_sum"] = getattr(mod, SCATTER_ATTR)
    _ORIG["scatter_mod"] = mod
    setattr(mod, SCATTER_ATTR, _scatter_sum_routed)


# ---------------------------------------------------------------------------------------------------------------- the size gate (cli.pred plan)
def gate() -> tuple:
    """(min_tokens, max_tokens) in residue tokens: the knobs when set, else the measured defaults."""
    def rd(env, dflt):
        v = (os.environ.get(env) or "").strip()
        try:
            return int(v) if v else dflt
        except ValueError:
            return dflt
    return rd(GATE_MIN_ENV, GATE_MIN), rd(GATE_MAX_ENV, GATE_MAX)


_CARD = {"sm": None, "read": False}


def _card_sm():
    """This process's compute capability word ('90', '80', …) as stack.gpu_info() reads it (nvidia-smi / torch), once; None when no GPU is visible."""
    if not _CARD["read"]:
        _CARD["read"] = True
        try:
            from . import stack
            sm = (stack.gpu_info() or {}).get("sm")
            _CARD["sm"] = str(sm).replace(".", "") if sm else None
        except Exception:  # noqa: BLE001
            _CARD["sm"] = None
    return _CARD["sm"]


def card_row():
    """(cap_tokens, word, sm) = this card's CARD_ROWS row, or None (no row: the general gate only)."""
    sm = _card_sm()
    row = CARD_ROWS.get(sm) if sm else None
    return (int(row[0]), str(row[1]), sm) if row else None


def word_row(line=None):
    """(cap_tokens, word, sampler_word) = the WORD_ROWS row of ``line``'s sampler tier word (its ``ODDE_DIT_ATTN`` export), or None (no row: the
    general gate and the card's row only)."""
    sw = (getattr(line, "exports", None) or {}).get(SAMPLER_WORD_SWITCH) if line is not None else None
    row = WORD_ROWS.get(sw) if sw else None
    return (int(row[0]), str(row[1]), sw) if row else None


def policy(tokens, wr=None) -> dict:
    """The gate decision over ``tokens`` = ``big.count_tokens(jobs)`` ({item: {"residue_tokens", …}}) or None (no query read): ``within`` only
    when every item counts inside [min, max], at or under this card's row (CARD_ROWS) if it has one, and at or under the line's sampler-word
    row ``wr`` = :func:`word_row` (WORD_ROWS) if it has one."""
    lo, hi = gate()
    cr = card_row()
    card = {"sm": cr[2], "cap": cr[0], "word": cr[1]} if cr else None
    wrow = {"sampler_word": wr[2], "cap": wr[0], "word": wr[1]} if wr else None
    if tokens is None:                                                             # no query read (the env route, a dry run): composed in — the call's own admission guards the step
        return {"gate": (lo, hi), "card": card, "word_row": wrow, "max_tokens": None, "min_tokens": None, "n_items": 0, "within": True, "case": "unknown",
                "reason": "no query read before activation: stepgraph composed in (the sampler call's own admission decides)"}
    counts = [int(t["residue_tokens"]) for t in tokens.values()]
    mx, mn, n = max(counts, default=0), min(counts, default=0), len(counts)
    if n and mn >= lo and mx <= hi:
        if cr and mx > cr[0]:                                                      # within the general gate but above this card's row: composed out by the card's name
            return {"gate": (lo, hi), "card": card, "word_row": wrow, "max_tokens": mx, "min_tokens": mn, "n_items": n, "within": False, "case": "card_above",
                    "reason": f"an item above this card's row ({mx} residue tokens > {cr[0]} on sm_{cr[2]}: {cr[1]}): stepgraph composed out"}
        if wr and mx > wr[0]:                                                      # within the gate and the card's row but above the line's sampler-word row: composed out by that word's name
            return {"gate": (lo, hi), "card": card, "word_row": wrow, "max_tokens": mx, "min_tokens": mn, "n_items": n, "within": False, "case": "word_above",
                    "reason": f"an item above the {wr[2]} sampler word's row ({mx} residue tokens > {wr[0]}: {wr[1]}): stepgraph composed out"}
        return {"gate": (lo, hi), "card": card, "word_row": wrow, "max_tokens": mx, "min_tokens": mn, "n_items": n, "within": True, "case": "within",
                "reason": f"every item within the gate ({mn}..{mx} residue tokens in [{lo}, {hi}]" + (f", sm_{cr[2]} row <= {cr[0]}" if cr else "")
                          + (f", {wr[2]} word row <= {wr[0]}" if wr else "") + "): stepgraph composed in"}
    case = "above" if mx > hi else "below"
    return {"gate": (lo, hi), "card": card, "word_row": wrow, "max_tokens": mx, "min_tokens": mn, "n_items": n, "within": False, "case": case,
            "reason": f"an item outside the gate ({mn}..{mx} residue tokens vs [{lo}, {hi}]): stepgraph composed out"}


def plan(res, query_path) -> dict:
    """The pre-activation hook (``cli.pred``): the gate decision for a line that carries the lever, from the query's residue tokens; kept for
    :func:`compose`. ``{}`` for ``off``."""
    line = getattr(res, "line", None)
    if line is None:
        _PLAN["plan"] = None
        return {}
    tokens = None
    if query_path:
        from . import inputs as _inputs
        from .big import count_tokens
        tokens = count_tokens(_inputs.load_query(query_path))
    pol = policy(tokens, word_row(line))                                           # the line's sampler-word row (WORD_ROWS): the mode's line as resolved, before the size gates compose it
    pol["tokens"] = tokens
    _PLAN["plan"] = pol
    return pol


def planned():
    return dict(_PLAN["plan"]) if _PLAN["plan"] is not None else None


def word(line=None):
    """The reason the lever is composed out (``above_gate:<t>/<g>``, ``below_gate:<t>/<g>``, ``card_gate:<word>_above:<cap>``,
    ``word_gate:<word>_above:<cap>``, ``offload_line``), None when composed in."""
    if line is not None and (line.exports or {}).get("ODDE_OFFLOAD"):
        return OFFLOAD_LINE
    pol = _PLAN["plan"]
    lo, hi = gate()
    if pol is None or pol.get("within"):                                           # no plan (env route / dry run) or every item within: composed in
        return None
    if pol.get("case") == "below":
        return f"{BELOW_GATE}:{pol.get('min_tokens')}/{lo}"
    if pol.get("case") == "card_above":                                             # this card's row (CARD_ROWS): card_gate:<word>_above:<cap>
        c = pol.get("card") or {}
        return f"{CARD_GATE}:{c.get('word')}_above:{c.get('cap')}"
    if pol.get("case") == "word_above":                                             # the line's sampler-word row (WORD_ROWS): word_gate:<word>_above:<cap>
        c = pol.get("word_row") or {}
        return f"{WORD_GATE}:{c.get('word')}_above:{c.get('cap')}"
    return f"{ABOVE_GATE}:{pol.get('max_tokens')}/{hi}"


def compose(line):
    """``line`` as the plan composes it: itself when every item is within the gate and the line runs no offload stage, else without the lever."""
    if line is None or LEVER not in line.levers or word(line) is None:
        return line
    from . import modes
    return modes.line_without(line, (LEVER,))


def gated_off(line) -> dict:
    """``{"stepgraph": <word>}`` when the plan composed the lever out of a line whose row carries it — the static row, or for a single-GPU big
    line below its offload gate the resident row (``modes.big_line(name, ())``: fast's resident set, which carries the lever); {}
    otherwise (composed in; ``off``; the offload line, whose static row never carried it: its LEVER row reads ``state=off`` as before)."""
    from . import modes
    if line is None or LEVER in line.levers:
        return {}
    static = modes.LINES.get(line.name)
    if static is None:
        return {}
    w = word(line)
    if LEVER not in static.levers:                                                  # a big line: the static row is the offload line (no graph over streamed pair tensors)
        if w in (None, OFFLOAD_LINE) or line.name not in getattr(modes, "BIG_MEM_LEVERS", {}) or LEVER not in modes.big_line(line.name, ()).levers:
            return {}
    return {LEVER: w} if w is not None else {}


# ---------------------------------------------------------------------------------------------------------------- the capture cache
def _make_cache(det: bool):
    kw = dict(max_entries=2, warmup=1, min_sightings=1, rng="forbid", strict=False, on_fail="disable",          # the core's default hold: the capture's replay
              clone_outputs=True, log=_log, verbose=_env("VERBOSE", "0") not in ("", "0"), capture_error_mode="thread_local")   # bit-exact to the eager-first answer
    from opt_core.capture import graphs as G                                   # the core's ONE capture discipline
    return G.GraphCache(f"opendde.{LEVER}", **kw)


# ---------------------------------------------------------------------------------------------------------------- LayerNorm capture canary
def ln_capture_canary(device) -> Dict[str, Any]:
    """Is the LayerNorm the model is built from RECORDED by a stream capture? Capture LN(x) into a tiny graph, replay it on NEW input, compare with
    eager (a legacy-stream launch runs at capture time and is absent from the graph: the replay leaves the old output -> mismatch). Once per process."""
    import torch
    if STATS["ln_canary"] is not None:
        return STATS["ln_canary"]
    out: Dict[str, Any] = {"ok": False, "impl": None, "why": None}
    try:
        from opendde.model.modules.primitives import LayerNorm                # the class the sampler's modules are built from
        ln = LayerNorm(128).to(device=device, dtype=torch.float32).eval()
        out["impl"] = type(ln).__module__ + "." + type(ln).__name__
        g = torch.Generator(device="cpu").manual_seed(1234)
        x1 = torch.randn(64, 128, generator=g).to(device)
        x2 = torch.randn(64, 128, generator=g).to(device) * 3.0 + 1.0
        with torch.no_grad():
            ref2 = ln(x2).clone()
            xs = x1.clone()
            torch.cuda.synchronize(device)
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                ln(xs)                                                          # warm-up (the extension's lazy load happens outside the capture)
            torch.cuda.current_stream(device).wait_stream(side)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                ys = ln(xs)
            xs.copy_(x2)
            graph.replay()
            torch.cuda.synchronize(device)
            out["ok"] = bool(torch.equal(ys, ref2))
            out["max_abs_diff"] = float((ys - ref2).abs().max())
            del graph, ys
        if not out["ok"]:
            out["why"] = "the bound LayerNorm is not recorded by a stream capture (legacy-default-stream launch): lever lnstream must be serving"
    except Exception as e:  # noqa: BLE001
        if _is_oom(e):
            raise
        out["why"] = f"canary raised {e!r}"
    try:
        from . import lnstream as _ls
        out["lnstream"] = _ls.STATS.get("state")
    except Exception:  # noqa: BLE001
        out["lnstream"] = None
    STATS["ln_canary"] = out
    _vlog(f"LayerNorm capture canary: {json.dumps(out)}")
    return out


# ---------------------------------------------------------------------------------------------------------------- one sampler call
class _Ctx:
    """State of ONE sample_diffusion() call: the denoiser instance, its constant keyword arguments, the capture cache."""

    def __init__(self, dm, det: bool):
        self.dm, self.det = dm, det
        self.n = 0
        self.head = max(1, int(_env("HEAD", "1")))
        self.cache = None
        self.fn: Optional[Callable] = None
        self.rest_ids: Optional[Dict[str, int]] = None
        self.refused: Optional[str] = None
        self.admitted = False                                                    # admission passed at the first denoiser call: the plan-for-capture window is open (planning())
        self.busy = False
        self.check_left = 1                                                      # hold the FIRST replay on new inputs to an eager re-run (bit-exact under --det)
        self.replays_seen = 0
        self.probe_due = STATS["poison_probe"] is None                          # the poison probe: once per process
        self.seg = None                                                          # (tab, mask, M, n_atom, n_token): segment_sum's gather table, built at admission
        self.seg_index = None
        self.graphed_region = False                                              # True around the capture cache's calls and the lever's eager re-runs (scatter route)
        self.events = []                                                         # (start, end) CUDA events around replayed calls, read once at close (no per-step sync)
        self.t_enter = time.perf_counter()

    def _admit(self, kwargs) -> Optional[str]:
        import torch
        x, s_inputs = kwargs.get("x_noisy"), kwargs.get("s_inputs")
        if not (torch.is_tensor(x) and torch.is_tensor(s_inputs)):
            return "no_tensor_inputs"
        why = _device_refusal(x)
        if why is not None:
            return why
        if torch.is_grad_enabled():
            return "autograd_on"
        if self.dm.training:
            return "training_mode"
        if not kwargs.get("inplace_safe", False) and kwargs.get("p_lm") is not None:
            return "inplace_safe_false_mutates_cached_p_lm"
        for k in ("pair_z_spec", "atom_window_spec", "foldcp_attention_bias", "foldcp_group"):
            if kwargs.get(k) is not None:
                return "foldcp_argument_" + k
        if os.environ.get("ODDE_OFFLOAD", "").strip():                          # big's offload unit streams the pair tensors the step reads: composed out by plan; belt and braces
            return "offload_unit_active"
        if os.environ.get("DIT_HOIST_CROSSCHECK", "").strip():                  # dit_hoist would run torch.equal (a host sync) inside a captured step
            return "dit_hoist_crosscheck_set"
        n_tok = int(s_inputs.shape[-2])
        if n_tok > HARD_MAX_STRUCT_TOKENS:
            return f"structural_tokens_{n_tok}_over_{HARD_MAX_STRUCT_TOKENS}"
        if True:                                                                 # the atom->token sum is served by segment_sum inside the graphed region (both numerics modes)
            feats = kwargs.get("input_feature_dict") or {}
            a2t = feats.get("atom_to_token_idx") if isinstance(feats, dict) else None
            if not torch.is_tensor(a2t) or a2t.dim() != 1 or int(a2t.shape[0]) != int(x.shape[-2]):
                return "atom_to_token_idx_unusable"
            try:
                _route_scatter()
            except Exception as e:  # noqa: BLE001
                return f"scatter_route_failed:{type(e).__name__}"
            seg = segment_table(a2t, n_tok)
            if seg is None:
                return "atom_to_token_idx_out_of_range"
            self.seg = (seg[0], seg[1], seg[2], int(a2t.shape[0]), n_tok)
            self.seg_index = a2t
        if x.is_cuda:
            can = ln_capture_canary(x.device)
            if not can["ok"]:
                return "layernorm_not_capture_safe"
            n_samp = int(x.shape[-3]) if x.dim() >= 3 else 1
            n_atom = int(x.shape[-2])
            est = 3 * n_samp * 16 * n_tok * n_tok * 4 + 64 * n_samp * n_atom * 128 * 4 + (1 << 30)   # generous step-arena bound: 3 live [S,16,N,N] fp32 score/bias tensors + atom activations + 1 GiB
            free_now = torch.cuda.mem_get_info(x.device)[0] + (torch.cuda.memory_reserved(x.device) - torch.cuda.memory_allocated(x.device))
            frac = float(_env("MEMFRAC", "0.5"))
            if est > frac * free_now:
                return f"memory_est_{est >> 20}MiB_over_{frac}_of_free_{free_now >> 20}MiB"
        return None

    def step(self, args, kwargs):
        i = self.n
        self.n += 1
        STATS["denoiser_calls"] += 1
        call = _ORIG["dm_call"]
        if self.refused is not None:
            STATS["eager_other"] += 1
            return call(self.dm, *args, **kwargs)
        if args:
            self.refused = _refuse("positional_denoiser_call")
            STATS["eager_other"] += 1
            return call(self.dm, *args, **kwargs)
        rest = {k: v for k, v in kwargs.items() if k not in ("x_noisy", "t_hat_noise_level")}
        if i == 0:
            why = self._admit(kwargs)
            if why is not None:
                self.refused = _refuse(why)
                STATS["eager_other"] += 1
                return call(self.dm, **kwargs)
            self.admitted = True                                                 # the window opens BEFORE the eager head call: its rows are the capture's rows
            STATS["planned_calls"] += 1
            self.rest_ids = {k: id(v) for k, v in rest.items()}
            dm = self.dm

            def fn(x, t, _rest=rest, _dm=dm, _call=call):
                return _call(_dm, x_noisy=x, t_hat_noise_level=t, **_rest)
            self.fn = fn
            self.cache = _make_cache(self.det)
            STATS["engaged_calls"] += 1
        elif self.rest_ids != {k: id(v) for k, v in rest.items()}:
            self.refused = _refuse("rollout_constants_rebound_mid_call")
            self._drop()
            STATS["eager_other"] += 1
            return call(self.dm, **kwargs)
        tm = STATS["time"]
        if i < self.head:
            STATS["eager_head"] += 1
            t0 = time.perf_counter()
            out = call(self.dm, **kwargs)
            tm["eager_calls"] += 1; tm["eager_host_s"] += time.perf_counter() - t0     # host wall of an eager denoiser call (enqueue-bound or GPU-bound, whichever binds)
            return out
        x = kwargs["x_noisy"]
        seen = self.replays_seen
        ev = None
        if x.is_cuda and self.cache is not None and self._armed():                # a replay is due: CUDA events around it (elapsed read at close, after the sampler's own sync)
            import torch
            ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            ev[0].record()
        t0 = time.perf_counter()
        out = self._graphed(x, kwargs["t_hat_noise_level"])
        dt = time.perf_counter() - t0
        if ev is not None:
            ev[1].record()
        if self.replays_seen == seen + 1 and self.replays_seen > 1 + (0 if self.det else 1):   # a plain replay (not the capture call, not the held first replay)
            tm["replay_calls"] += 1; tm["replay_host_s"] += dt
            if ev is not None:
                self.events.append(ev)
        return out

    def _graphed(self, x, t):
        import torch
        cache, fn = self.cache, self.fn
        before = cache.stats_["replays"]
        self.busy = True
        self.graphed_region = True
        try:
            out = cache.run(fn, x, t)
        except Exception as e:  # noqa: BLE001
            if not _is_oom(e):
                raise
            self.refused = _refuse("oom_at_capture")                            # a capture-time OOM never kills the fold: free the arena, go eager for this call
            self._drop()
            torch.cuda.empty_cache()
            out = fn(x, t)
            STATS["eager_other"] += 1
            return out
        finally:
            self.busy = False
            self.graphed_region = False
        for pk in (cache.stats().get("per_key") or []) if not self.replays_seen else ():        # the graph's private pool, read while the entry is live
            STATS["pool_mib_max"] = max(STATS["pool_mib_max"], int(pk.get("pool_bytes") or 0) >> 20)
        if cache.stats_["replays"] > before:
            self.replays_seen += 1
            if self.check_left and self.replays_seen == 1:                      # hold the FIRST replay (new inputs, reused arena) to an eager re-run: bit-exact under --det, within the bar otherwise
                self.check_left -= 1
                self.busy = self.graphed_region = True
                try:
                    ref = fn(x, t)
                    ref2 = fn(x, t)                                             # the eager step twice: its OWN run-to-run spread is the bar without the
                finally:                                                        # deterministic recipe (zero when every kernel of the step is reproducible)
                    self.busy = self.graphed_region = False
                scale = max(float(ref.abs().max()), 1e-30)
                diff = float((out - ref).abs().max()) if bool(torch.isfinite(out).all()) else float("inf")
                spread = float((ref2 - ref).abs().max()) / scale
                rel = diff / scale
                bar = 0.0 if self.det else max(float(_env("RTOL", str(RTOL))), SPREAD_K * spread)   # --det: bit-exact; otherwise within K x the eager step's own spread
                STATS["tol_checks"] += 1
                STATS["tol_max_rel"] = max(STATS["tol_max_rel"], rel if rel == rel else float("inf"))
                STATS["eager_spread_max"] = max(STATS["eager_spread_max"], spread)
                if not (rel <= bar):                                            # the held replay differs from the eager step: its output is dropped (the eager answer
                    cache.disable(f"tolerance check failed: rel {rel:.3e} > bar {bar:.3e} (eager spread {spread:.3e})")   # returns) and the graph is off for the process BY NAME --
                    word = f"{MISMATCH_ASIDE}_hold_rel_{rel:.3e}_bar_{bar:.3e}"    # a step-aside to the eager sampler (nothing replayed was consumed), the run complete
                    STATS["verify_aside"] = STATS.get("verify_aside") or word
                    self.refused = _refuse(word)
                    return ref
            if self.probe_due and self._armed():                                # once per process: the static inputs are really read by the replay
                self.probe_due = False
                self._poison_probe(x, t)
        elif getattr(cache, "disabled", None) is not None and self.refused is None:
            self._cache_disabled(str(cache.disabled))                          # capture-check mismatch -> named step-aside (eager serves, complete); anything else -> failure by name
        return out

    def _cache_disabled(self, why: str):
        """The capture cache disabled itself during this call (it answered eagerly). A consistency mismatch at capture (core stages verify1 / verify2:
        the replay differed from the eager step before any graph output was consumed) is a named step-aside to the eager sampler for the process;
        every other reason (a capture that raised, …) is a failure by name as before."""
        word = note_cache_disabled(why)
        self.refused = STATS["why_last"]
        return word

    def _armed(self) -> bool:
        ents = getattr(self.cache, "_entries", None) or {}
        for ent in ents.values():
            if isinstance(ent, tuple) or getattr(ent, "armed", False):
                return True
        return False

    def _poison_probe(self, x, t):
        """NaN into the graph's static coordinate input, one replay, NaN required in the output; the static inputs restored by the next real copy-in
        (GraphCache copies the live arguments in before every replay). Counts as one replay in the census."""
        import torch
        self.busy = True
        try:
            if hasattr(self.cache, "poison"):                                   # a cache with its own probe (the unit tests' CPU stand-in)
                moved = self.cache.poison()
            else:
                probe = self.cache.run(self.fn, torch.full_like(x, float("nan")), t)
                moved = bool(torch.isnan(probe).any())
            STATS["poison_probe"] = "pass" if moved else "fail"
            if not moved:
                self.cache.disable("poison probe: the replay ignored a NaN static input (stale read)")
                STATS["disabled"] = "poison_probe_failed"
                self.refused = _refuse("poison_probe_failed")
                STATS["failures"] += 1
        except Exception as e:  # noqa: BLE001
            if _is_oom(e):
                raise
            STATS["poison_probe"] = f"error:{type(e).__name__}"
        finally:
            self.busy = False

    def _drop(self):
        if self.cache is not None:
            try:
                self._harvest()
                self.cache.reset("item")
            except Exception as e:  # noqa: BLE001
                if _is_oom(e):
                    raise
                _log(f"cache reset raised {e!r}")
            self.cache = None
        self.fn = None

    def _harvest(self):
        st = self.cache.stats()
        caps = int(st.get("captures", 0))
        if caps:
            STATS["recaptures"] += caps - (1 if STATS["captures"] == 0 else 0)
        STATS["captures"] += caps
        STATS["replays"] += int(st.get("replays", 0))
        STATS["held"] += int(st.get("verify_pass", 0))                    # the core's count of bit-exact capture holds that passed
        STATS["capture_s"] = round(STATS["capture_s"] + float(st.get("capture_s", 0.0)), 4)
        STATS["pool_mib_max"] = max([STATS["pool_mib_max"]] + [int(pk.get("pool_bytes") or 0) >> 20 for pk in (st.get("per_key") or [])])   # the graph's private pool (per entry)
        for k, v in st.items():                                                 # the core's own buckets: refused_<kind>, eager_<kind>, failures
            if (k.startswith("refused_") or k.startswith("eager_") or k == "failures") and isinstance(v, int) and v:
                STATS["core"][k] = STATS["core"].get(k, 0) + v
        if st.get("disabled"):
            STATS["core"]["disabled"] = str(st["disabled"])[:200]
        for pk in (st.get("per_key") or [])[:2]:
            if len(STATS["keys"]) < 8 and isinstance(pk, dict):
                STATS["keys"].append({k: pk.get(k) for k in ("replays", "capture_s", "pool_bytes", "eager_s", "replay_s")})

    def close(self):
        tm = STATS["time"]
        if self.events:
            try:
                import torch
                torch.cuda.synchronize()
                tm["replay_gpu_s"] += sum(a.elapsed_time(b) for a, b in self.events) / 1000.0
                tm["steps_timed"] += len(self.events)
            except Exception:  # noqa: BLE001
                pass
            self.events = []
        if self.n:
            tm["sampler_s"] += time.perf_counter() - self.t_enter; tm["sampler_calls_timed"] += 1
        self._drop()


# ---------------------------------------------------------------------------------------------------------------- the two patches
def _dm_call(self, *args, **kwargs):
    ctx = _CUR[0]
    if ctx is None or ctx.dm is not self or ctx.busy:
        return _ORIG["dm_call"](self, *args, **kwargs)
    return ctx.step(args, kwargs)


def _route_dm_call() -> bool:
    """Route `DiffusionModule.__call__` through the live context (class level; the DITFAST hoist wraps the instance's `forward` underneath)."""
    if STATS["routed"]:
        return True
    import torch
    mod = sys.modules.get(DM_MODULE) or __import__(DM_MODULE, fromlist=["_"])
    cls = getattr(mod, DM_CLASS)
    _ORIG["dm_class"] = cls
    _ORIG["dm_call"] = torch.nn.Module.__call__
    _ORIG["dm_had_call"] = "__call__" in cls.__dict__
    cls.__call__ = _dm_call
    STATS["routed"] = True
    return True


def _guidance_on(kw) -> bool:
    g = kw.get("guidance_configs")
    if g is None:
        return False
    try:
        from opendde.tfg import parse_tfg_config
        return bool(parse_tfg_config(g).enable)
    except Exception:  # noqa: BLE001
        return True                                                             # cannot tell -> stand aside


def make_wrapper(orig):
    """The function installed as `opendde.model.opendde.sample_diffusion`: the stock sampler with one denoiser-step graph per call."""
    def stepgraph_sample_diffusion(*args, **kw):
        import torch
        STATS["sampler_calls"] += 1
        dm = kw.get("denoise_net")
        why = None
        if STATS["disabled"] is not None:
            why = "disabled_for_process:" + str(STATS["disabled"])[:80]
        elif args or dm is None:
            why = "positional_sampler_call"
        elif _CUR[0] is not None:
            why = "nested_sampler_call"
        elif _guidance_on(kw):
            why = "guidance_enabled"
        else:
            try:
                _route_dm_call()
                if not isinstance(dm, _ORIG["dm_class"]):
                    why = "denoise_net_is_not_DiffusionModule"
            except Exception as e:  # noqa: BLE001
                if _is_oom(e):
                    raise
                why = f"route_failed:{type(e).__name__}"
            if why is None:
                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                        why = "distributed_world_size_gt_1"
                except Exception:  # noqa: BLE001
                    pass
        if why is not None:
            _refuse(why)
            return orig(*args, **kw)
        det = bool(torch.are_deterministic_algorithms_enabled())
        STATS["det"] = det
        ctx = _Ctx(dm, det=det)
        _CUR[0] = ctx
        try:
            out = orig(*args, **kw)
        finally:
            _CUR[0] = None
            replayed = ctx.replays_seen
            ctx.close()
        if replayed and torch.is_tensor(out) and not bool(torch.isfinite(out).all()):   # never hand back non-finite coordinates from a replayed call
            STATS["disabled"] = "nonfinite_coordinates_after_replay"
            STATS["failures"] += 1
            if kw.get("rollout_seed") is None:
                raise RuntimeError(f"[{TAG}] {LEVER}: non-finite coordinates after graph replay and no rollout_seed to re-run the stock sampler identically; "
                                   f"lever disabled for the process (MODEL_OPT_LEVERS_OFF={LEVER} leaves it out)")
            _log("non-finite coordinates after graph replay -> lever disabled for the process; re-running the STOCK sampler for this call (same rollout_seed)")
            STATS["nonfinite_reruns"] += 1
            out = orig(*args, **kw)
        return out
    stepgraph_sample_diffusion._orig = orig
    stepgraph_sample_diffusion._stepgraph = True
    return stepgraph_sample_diffusion


# ---------------------------------------------------------------------------------------------------------------- install / census
def _sync() -> None:
    p = STATS.get("patch")
    if p is not None:
        STATS["installed"], STATS["armed"] = p.state == "installed", p.state == "armed"


def install() -> None:
    """Patch the sampler name now when `opendde.model.opendde` is imported, else at its import (the core's per-site patch). Idempotent."""
    _sync()
    if STATS["installed"] or STATS["armed"]:
        return
    from opt_core import autoload
    try:
        STATS["patch"] = autoload.patch_attr_at_import(TARGET, ATTR, make_wrapper, tag=TAG, name=LEVER)
    except autoload.PatchError as e:
        raise ActivationError(str(e)) from None
    _sync()


def remove() -> None:
    """Undo the class-level route (tests); the module patch is the core's (autoload)."""
    cls = _ORIG.get("dm_class")
    if cls is not None and cls.__dict__.get("__call__") is _dm_call and not _ORIG.get("dm_had_call"):
        del cls.__call__
    mod = _ORIG.pop("scatter_mod", None)
    orig = _ORIG.pop("scatter_sum", None)
    if mod is not None and orig is not None and getattr(mod, SCATTER_ATTR, None) is _scatter_sum_routed:
        setattr(mod, SCATTER_ATTR, orig)
    STATS["routed"] = False


def _reset() -> None:
    """Test support: process state back to import time (the patch object dropped; the class route undone)."""
    remove()
    STATS.clear(); STATS.update(json.loads(json.dumps(_STATS0))); STATS["patch"] = None
    _PLAN["plan"] = None
    _CARD.update(sm=None, read=False)
    _CUR[0] = None


def replays() -> int:
    return int(STATS["replays"])


ASIDE_WORDS = ("positional_sampler_call", "nested_sampler_call", "guidance_enabled", "distributed_world_size_gt_1", "no_tensor_inputs",
               "autograd_on", "training_mode", "inplace_safe_false", "offload_unit_active", "dit_hoist_crosscheck_set", "structural_tokens_",
               "atom_to_token_idx_", "memory_est_", "positional_denoiser_call", "rollout_constants_rebound_mid_call", "oom_at_capture",
               "not_cuda", "disabled_for_process", MISMATCH_ASIDE)
# ^ the by-design step-asides: a sampler call outside the lever's domain (a feature stock accepts that the graph does not serve, another rank,
#   the memory admission, the structural-token ceiling) runs the STOCK sampler, counted by word — the run is complete, the LEVER row says
#   `state=off reason=aside:<word>` when no call of the process engaged (stack.refresh: inert by design) or `aside=<word>x<n>` beside the
#   census when some did. Every other word is a failure (a capture that failed or was not bit-exact, a failed hold or poison probe, a route
#   that could not be installed): a fallback BY NAME, the run PARTIAL.


def _is_aside(why: str) -> bool:
    return any(why.startswith(w) for w in ASIDE_WORDS)


_DETAIL_RE = re.compile(r"'max_abs_diff': ([^,}\s]+).*?'stage': '([A-Za-z0-9_]+)'", re.S)


def note_cache_disabled(why: str) -> str:
    """Account for the capture cache disabling itself (it answered the call eagerly): a consistency refusal at capture (``refused (<kind>): replay
    differs from eager …`` -- core GraphCache stages verify1 / verify2, before any graph output was consumed) is the named step-aside
    ``verify_mismatch_<stage>_maxabs_<diff>`` (counted in ``refused`` under an ASIDE word, ``STATS["verify_aside"]``; the eager sampler serves the
    process, the run complete); any other reason is the failure ``capture_failed:<reason>`` (``STATS["disabled"]``, a fallback by name, PARTIAL).
    Returns the refused word."""
    why = str(why)
    if why.startswith(f"refused ({CAPTURE_CHECK_KIND})"):
        m = _DETAIL_RE.search(why)
        diff, stage = (m.group(1), m.group(2)) if m else ("unknown", "capture")
        try:
            diff = f"{float(diff):.3e}"
        except ValueError:
            pass
        word = f"{MISMATCH_ASIDE}_{stage}_maxabs_{diff}"
        STATS["verify_aside"] = STATS.get("verify_aside") or word
        return _refuse(word)
    STATS["disabled"] = "capture_failed:" + why[:160]
    STATS["failures"] += 1
    return _refuse("capture_failed_or_not_bit_exact")


def asides() -> dict:
    """{word: n} of this process's by-design step-asides (sampler calls that ran the stock sampler by the lever's own domain rule)."""
    return {w: n for w, n in sorted((STATS.get("refused") or {}).items()) if _is_aside(w) and not w.startswith("disabled_for_process")}


def aside_reason():
    """`aside:<word>[+<k> more]` when sampler calls of this process stepped aside by design and none engaged the graph; None otherwise."""
    a = asides()
    if not a or STATS["replays"] > 0 or STATS["disabled"] or fallbacks((LEVER,)):
        return None
    if STATS.get("verify_aside") and STATS.get("verify_aside") in a:               # the graph went off for the process by a capture-check mismatch and no step of the process
        return f"aside:{STATS['verify_aside']}"                                    # replayed: that is THE reason (the eager sampler served), whatever else stood aside
    top = max(a.items(), key=lambda kv: kv[1])[0]
    return f"aside:{top}" + (f"+{len(a) - 1}_more" if len(a) > 1 else "")


def fallbacks(planned) -> list:
    """The lever's FAILURES at exit, by name: disabled for the process (a capture that failed or was not bit-exact, a failed hold or poison
    probe, non-finite coordinates) or a refusal word that is not a by-design step-aside (ASIDE_WORDS). A process whose every sampler call
    replayed or stepped aside by design has no event; a process that never reached the sampler has none either (ran-or-refuse names it)."""
    if LEVER not in planned:
        return []
    out = []
    if STATS["disabled"]:
        out.append(f"{LEVER}:disabled:{STATS['disabled']}")
    for why, n in sorted((STATS.get("refused") or {}).items()):
        if _is_aside(why):
            continue
        out.append(f"{LEVER}:refused:{why}(x{n})")
    return out


def kit_stats() -> dict:
    _sync()
    d = {k: v for k, v in STATS.items() if k != "patch"}
    tm = STATS.get("time") or {}
    if tm.get("replay_calls"):                                                   # per denoiser call, ms: eager (head calls) vs replay host / GPU; the loop's other work per step
        d["eager_ms"] = round(1000.0 * tm["eager_host_s"] / max(1, tm["eager_calls"]), 2)
        d["replay_host_ms"] = round(1000.0 * tm["replay_host_s"] / tm["replay_calls"], 2)
        d["replay_gpu_ms"] = round(1000.0 * tm["replay_gpu_s"] / tm["steps_timed"], 2) if tm.get("steps_timed") else None
        calls = STATS["eager_head"] + STATS["eager_other"] + STATS["captures"] + STATS["replays"]
        d["step_ms"] = round(1000.0 * tm["sampler_s"] / max(1, calls), 2)      # the sampler call's wall per denoiser call (everything: loop, RNG, Euler, denoiser)

    d["gate"] = gate()
    d["plan"] = planned()
    d["aside"] = asides()
    d["refused"] = {w: n for w, n in (STATS.get("refused") or {}).items() if not _is_aside(w)}
    return json.loads(json.dumps(d, default=str))

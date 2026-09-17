"""Lever triattn_core — the attention CORE of the fused triangle-attention block (`triatt_block`, and `pair_block_residual` through the same call)
bound to the core's triangle-attention provider `opt_core.kernels.triattn` BY THE MODE'S TIER WORD: the block runs
``opt_core.attn.pair_fused.tri_attn_block(..., core="tier:<word>")`` with <word> = ``fast`` under --mode fast and ``big`` under --mode big,
and the provider serves the measured cell's row for that word at each call class (cc, dtype, head_dim, heads, keys, mask form) — its own
table decides the row per cell and card; a row that cannot serve a call class steps aside inside the
provider BY NAME to the block's flash core (``cores=tier:<word>=flash_triattn(refused:<kind>)``).  No kit preference table, no kit row floor:
this lever carries no row name.  It is a lever OF `triatt_block`'s served path, not a patch site of its own: `hooks/pair_cells.core_pick` asks it
which core each SERVED block call runs (below the block's floor the block itself steps aside, so this lever sees only served calls).
Steps aside BY NAME per call: `index_2p31:<B>x<N>` — a shape at which a tensor of the served path reaches 2**31 elements: that call runs the
provider default core instead of the word, counted, never a refusal.
Individually switchable: MODEL_OPT_LEVERS_OFF=triattn_core (the block then runs the provider's own default core, `hooks/pair_cells.CORE`);
AFO_TRIATTN_WORD=<row or tier word> is a developer override of the word (e.g. ``cuda_sm90a``, ``k2b``) — the provider refuses an unknown
word by name at install.
LEVER line: name=LOCAL.atlasfold.triattn_core impl=opt_core.kernels.triattn@<core>:<word> origin=core served=<calls> fallback=<n>
fallback_by=<word:n> min_tokens=<the block's floor> shapes=<BxN:start|end:n> core=tier:<word> triattn_native=<state> cuda_sm90a=<state>
form=mask_bias index_limit=2147483648 plan=<N:row,…> refused=<kind->flash_triattn:n|none> rows=<row:n|none> word=<word>
(cuda_*=<state>: whether that provider row could serve the block's geometry on this card + stack — informational: ``ok`` | the provider's
refusal word | ``absent``)."""
import os
from typing import Dict, Optional

from . import Installed
from . import pair_cells as PC

LEVER = "triattn_core"
NAME = "LOCAL.atlasfold.triattn_core"
OVER = "triatt_block"                                   # the lever whose served path this core belongs to (row order: installed before this one)
TARGET = "atlasfold.model.network.primitives.triangle_update"
WORD_ENV = "AFO_TRIATTN_WORD"
TIER_OF_MODE = {"fast": "fast", "big": "big", "exact": "exact"}   # the provider's tier word IS the mode word
INDEX_LIMIT = 2 ** 31
OWN_WORDS = ("index_2p31", "no_calls")
FORM = "mask_bias"                                      # the block's call form: per-row key mask + pair bias (kernels.triattn's `forms` cell key)


def word(mode: str = "fast") -> str:
    """AFO_TRIATTN_WORD (a developer override: a provider row or tier word) else the MODE's tier word."""
    w = (os.environ.get(WORD_ENV) or "").strip()
    return w or TIER_OF_MODE.get(mode, "fast")


def index_refusal(B: int, N: int, H: int = PC.TRUNK_H, D: int = PC.TRUNK_D) -> Optional[str]:
    """`index_2p31:<B>x<N>` when a tensor of the served path reaches 2**31 elements (an element offset could then pass int32), else None.
    Tensors: q|k|v|o [B,N,H,N,D]; g|u [B,N,N,H*D] (same count); the bias plane [B,1,H,N,ceil16(N)] fp32; the key mask [B,N,1,1,N]."""
    B, N, H, D = int(B), int(N), int(H), int(D)
    skp = -(-N // 16) * 16
    top = max(B * N * H * N * D, B * H * N * skp, B * N * N)
    return f"index_2p31:{B}x{N}" if top >= INDEX_LIMIT else None


def cuda_row_state(T, cc, stack_key) -> str:
    """`ok` | `no_prebuilt:<stack>` | `arch` | `absent` — whether the provider's cuda_sm90a row can serve on this card + torch stack (informational)."""
    if "cuda_sm90a" not in getattr(T, "ROW_NAMES", ()):
        return "absent"
    if tuple(cc) != (9, 0):
        return "arch"
    try:
        ok, why = T.admits("cuda_sm90a", cc, "bf16", PC.TRUNK_D, PC.TRUNK_H, 896, "fwd", stack=stack_key)
    except Exception as e:  # noqa: BLE001
        return f"error:{type(e).__name__}"
    return "ok" if ok else str(why).replace(" ", "_")[:72]


def native_row_state(T, cc, stack_key) -> str:
    """`ok` | `<refusal word>` | `absent` — whether the provider row this LEVER field names can serve on this card + stack (informational)."""
    if "triattn_native" not in getattr(T, "ROW_NAMES", ()):
        return "absent"
    try:
        ok, why = T.admits("triattn_native", cc, "bf16", PC.TRUNK_D, PC.TRUNK_H, 896, "fwd", stack=stack_key)
    except Exception as e:  # noqa: BLE001
        return f"error:{type(e).__name__}"
    return "ok" if ok else str(why).replace(" ", "_")[:72]


def served_cores(PF, w: str):
    """(rows, refused) read off attn.pair_fused's SERVED_CORES for the tier core ``tier:<w>``: rows = {provider row: calls},
    refused = {'<kind>->flash_triattn': calls} (the provider's named step-aside to the block's flash core)."""
    rows: Dict[str, int] = {}; refused: Dict[str, int] = {}
    prefix = f"{PF.TIER_CORE_PREFIX}{w}="
    for key, n in dict(getattr(PF, "SERVED_CORES", {}) or {}).items():
        if not str(key).startswith(prefix):
            continue                                   # another core's census (the provider default's own `default=…` entries)
        served = str(key)[len(prefix):]
        if served.startswith("flash_triattn(refused:") and served.endswith(")"):
            kind = served[len("flash_triattn(refused:"):-1]
            refused[f"{kind}->flash_triattn"] = refused.get(f"{kind}->flash_triattn", 0) + int(n)
            rows["flash_triattn"] = rows.get("flash_triattn", 0) + int(n)
        else:
            rows[served] = rows.get(served, 0) + int(n)
    return rows, refused


class Picker:
    """hooks/pair_cells' core picker protocol: pick(B, N, ending) -> the core word for THIS served block call (``tier:<word>``), or None
    (= run the stock statement, counted here by name); result(core, outcome, word, shape) <- what happened."""

    def __init__(self, PF, ledger, core_word: str):
        self.PF, self.ledger, self.core = PF, ledger, core_word
        self.plan: Dict[int, str] = {}
        self.sizes = set()

    def min_tokens(self) -> int:
        return PC.MIN_TOKENS

    def pick(self, module, z4, ending: bool):
        B, N = int(z4.shape[0]), int(z4.shape[-2])
        why = index_refusal(B, N, int(getattr(module, "num_heads", PC.TRUNK_H)), int(getattr(module, "channel_hidden", PC.TRUNK_D)))
        if why is not None:
            self.ledger.fallback(why); return PC.CORE                          # by name: the provider default core serves that shape (counted here), never a refusal
        return self.core

    def result(self, core, outcome: str, word=None, shape=None):
        if outcome == "served":
            self.ledger.serve(shape)
            try:
                self.sizes.add(int(str(shape).split(":")[0].split("x")[-1]))
            except (TypeError, ValueError):
                pass
        elif outcome == "error":
            self.ledger.error(str(word or "error")[:40])
        else:
            self.ledger.fallback(str(word or "declined").replace(" ", "_")[:60])

    def plan_line(self, cc) -> str:
        """`<N>:<row>` per served size — the provider's own resolution of this word at the block's call class (informational)."""
        out = []
        for n in sorted(self.sizes):
            if n not in self.plan:
                try:
                    self.plan[n] = self.PF.resolve_tier_core(self.core[len(self.PF.TIER_CORE_PREFIX):], cc, "bf16", PC.TRUNK_D, PC.TRUNK_H, n, form=FORM).row
                except Exception as e:  # noqa: BLE001
                    self.plan[n] = "refused:" + str(getattr(e, "kind", type(e).__name__)).replace(" ", "_")[:40]
            out.append(f"{n}:{self.plan[n]}")
        return ",".join(out) or "none"


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    try:
        tu = importlib.import_module(TARGET)
        import torch
        import opt_core
        from opt_core.counters import Ledger
        from opt_core.attn import pair_fused as PF
    except Exception as e:  # noqa: BLE001
        return Installed(LEVER, False, reason=f"import:{type(e).__name__}:{str(e)[:80]}".replace(" ", "_"))
    try:
        from opt_core.kernels import triattn as T
    except Exception as e:  # noqa: BLE001 — a core without the provider: named
        return Installed(LEVER, False, reason=f"core:triattn_not_in_opt_core@{opt_core.__version__}:{type(e).__name__}")
    for cls_name in ("TriangleAttentionStartingNode", "TriangleAttentionEndingNode"):
        if OVER not in str(getattr(getattr(tu, cls_name, None), "forward", None).__qualname__ if getattr(tu, cls_name, None) is not None else ""):
            return Installed(LEVER, False, reason=f"requires:{OVER}")                     # a lever of the block's served path: the block must be installed (row order)
    w = word(mode)
    base = w.split("@")[0]
    if base not in tuple(T.ROW_NAMES) + tuple(T.TIER_WORDS):
        return Installed(LEVER, False, reason=f"unknown_word:{w[:40]}".replace(" ", "_"))
    if not PF._core_ok(PF.TIER_CORE_PREFIX + w):                                        # pragma: no cover — the face accepts every provider word behind the prefix
        return Installed(LEVER, False, reason=f"core_word_refused:{w[:40]}")
    cc = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else (0, 0)
    try:
        from opt_core.kernels.triattn import cuda_sm90a as CS
        stack = CS.stack_key()
    except Exception:  # noqa: BLE001
        stack = None
    from ..registry import LEVERS
    words = tuple(LEVERS[LEVER]["expected"])
    core_word = PF.TIER_CORE_PREFIX + w
    ledger = Ledger(NAME, impl=f"opt_core.kernels.triattn@{opt_core.__version__}:{w}", origin="core", min_tokens=PC.MIN_TOKENS, expected=words, max_shapes=16)
    ledger.set("word", w); ledger.set("core", core_word); ledger.set("form", FORM); ledger.set("index_limit", INDEX_LIMIT)
    ledger.set("cuda_sm90a", cuda_row_state(T, cc, stack)); ledger.set("triattn_native", native_row_state(T, cc, stack))
    picker = Picker(PF, ledger, core_word)
    PC.set_core_picker(picker)

    def line():
        rows, refused = served_cores(PF, w)
        ledger.set("rows", ",".join(f"{r}:{n}" for r, n in sorted(rows.items())) or "none")
        ledger.set("refused", ",".join(f"{k}:{n}" for k, n in sorted(refused.items())) or "none")
        ledger.set("plan", picker.plan_line(cc))
        return ledger.line(tag)
    return Installed(LEVER, True, lines=[line], gates=[PC.gate_for(ledger, words)],
                     facts={"impl": ledger.impl, "min_tokens": PC.MIN_TOKENS, "ledger": ledger, "word": w, "core": core_word, "picker": picker})

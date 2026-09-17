"""The core's pair_fused cells at AtlasFold's pair geometry — the ONE place the two fused pair levers (hooks/triatt_block.py `triatt_block`,
hooks/pair_transition_fused.py `pair_transition`) read their cell verdict, route their carried kernels and judge their census.

Geometry: AtlasFold's pair track is c_z = 128, 4 triangle-attention heads x 32, transition factor 4 (block.py L152-209: PairStack / PairBlock of the
LM stack, the 48-block Pairformer and the confidence heads) — the keys of the core table's rows (opt_core/attn/pair_fused_cells.json: fpf prologue /
epilogue (128, 4, 32), fpf transition (128, 512); rows per compute capability).  The verdict is read ONCE at install
(opt_core.attn.pair_fused.preflight — nothing launches): a table row serves (LEVER fields `cells=<cc>|<triton or *> cell_rows=<row ids>`), or the
core's SAFE settings serve a capability without rows (`cells=safe settings=safe:no_cell:<shape>`, the core prints its ONE line at the first call) —
either way the lever is engaged; a capability NOTHING serves (a key the table lists as measured off, or no safe settings admit the shape) installs
the lever all the same and its calls run the stock statement per call, counted with the core's `no-cell:<impl>:<piece>:<key>:<cc>|<triton>` word
(`cells=none`; exit 0).  Both levers serve N >= MIN_TOKENS only: below 512 tokens the fused surround is slower than the stock statements it replaces
-> `below_min_tokens`, counted."""
from typing import Dict, List, Optional, Sequence, Tuple

TRUNK_C, TRUNK_H, TRUNK_D, TRUNK_N = 128, 4, 32, 4                 # c_z, tri-attention heads x head dim, transition factor (hidden = TRUNK_N * c_z)
MIN_TOKENS = 512                                                    # the size floor of both levers (reason word `below_min_tokens`)
IMPL, CORE, LN = "fpf", "default", "fused"                          # opt_core.attn.pair_fused impl / attention core WITHOUT the core lever / LayerNorm placement the levers run.  With the
# triattn_core lever in the row (fast, big) every served block call runs core=`tier:<mode word>` (hooks/triattn_core: the provider's measured row per
# call class, its named step-aside to the flash core inside the provider); `default` is what the block passes when that lever is ablated
# (MODEL_OPT_LEVERS_OFF=triattn_core) or for a shape past its int32 index bound = the provider's OWN per-call-class choice
# (attn.pair_fused.DEFAULT_CORE_TABLE), so either way no row name lives in this kit.

PIECES: Dict[str, List[Tuple[str, str, Tuple[int, ...]]]] = {         # lever -> the core cell pieces it needs at the trunk geometry
    "triatt_block": [(IMPL, "prologue", (TRUNK_C, TRUNK_H, TRUNK_D)), (IMPL, "epilogue", (TRUNK_C, TRUNK_H, TRUNK_D))],
    "pair_transition": [(IMPL, "transition", (TRUNK_C, TRUNK_N * TRUNK_C))],
    "triatt_block_exact": [(IMPL, "prologue", (TRUNK_C, TRUNK_H, TRUNK_D)), (IMPL, "epilogue", (TRUNK_C, TRUNK_H, TRUNK_D))],   # the exact-class construction (ln='stock') reads the same table rows
}
KERNELS: Dict[str, Tuple[str, ...]] = {                             # lever -> the carried kernel names its served path imports (routed to the core copies at install)
    "triatt_block": ("fpf_triatt_pro", "fpf_triatt_epi", "flash_triattn"),
    "pair_transition": ("fpf_transition",),
    "triatt_block_exact": ("fpf_triatt_pro", "fpf_triatt_epi"),
}
_CORE_PICKER: List = [None]                                         # the triattn_core lever's per-call attention-core picker (hooks/triattn_core.py: the MODE's tier word `tier:<word>`); None = the provider default CORE on every call


def set_core_picker(picker) -> None:
    """Install (or clear, None) the per-call attention-core picker of triatt_block's served path: an object with
    ``pick(module, z4, ending) -> core word | None`` (None = the stock statement serves this call, counted by the picker by name) and
    ``result(core, outcome, word, shape)`` (outcome 'served')."""
    _CORE_PICKER[0] = picker


def core_pick(module, z4, ending: bool) -> str:
    """The attention core word this served triangle-attention call runs: the picker's (the triattn_core lever: ``tier:<mode word>``, or None =
    the stock statement for this call), the provider default CORE without one."""
    p = _CORE_PICKER[0]
    if p is None:
        return CORE
    try:
        return p.pick(module, z4, ending)
    except Exception:  # noqa: BLE001 — a picker that raises never costs the call: the pinned core serves it
        return CORE


def core_result(core: str, outcome: str, word: Optional[str], shape: str) -> None:
    """Tell the picker what became of a block call served with its core word (its census)."""
    p = _CORE_PICKER[0]
    if p is not None and core != CORE:
        try:
            p.result(core, outcome, word, shape)
        except Exception:  # noqa: BLE001 — a picker's bookkeeping never costs the call
            pass
SHAPE_WORD = {"triatt_block": f"c_z={TRUNK_C},H={TRUNK_H}", "pair_transition": f"c_z={TRUNK_C},n={TRUNK_N}", "triatt_block_exact": f"c_z={TRUNK_C},H={TRUNK_H}"}   # the kit's naming of the key on a `settings=safe:no_cell:<shape>` word


def route(lever: str) -> None:
    """Route the lever's carried kernel names to the core copies before first import (as hooks/triatt.py does for flash_triattn); a name this
    core does not carry stays whatever sys.path resolves."""
    try:
        from opt_core import kernels as _K
    except Exception:  # noqa: BLE001
        return
    for name in KERNELS[lever]:
        try:
            if name in _K.names():
                _K.route(name)
        except Exception:  # noqa: BLE001
            pass


def verdict(lever: str, device=None, stack: Optional[Tuple[str, str]] = None) -> dict:
    """What serves the lever's pieces on the running device (or on a named ``stack`` = (cc 'M.m', triton 'M.m'); CPU tests name one):
    {"served": True|False|None, "cells": "<cc>|<triton or *>[+…]" | "safe" | None, "cell_rows": "<id>+<id>" | None, "settings": <the core's
    safe word> | None, "reason": "" | <the core's no-cell word of the first unserved piece>}.  served None = no CUDA device and no stack named
    (nothing to decide; the per-call gate answers `cpu`)."""
    from opt_core.attn import pair_fused as PF
    if stack is None:
        import torch
        if not torch.cuda.is_available():
            return {"served": None, "cells": None, "cell_rows": None, "settings": None, "reason": "no_cuda_device"}
        device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    ds = PF.preflight(PIECES[lever], device, stack=stack)
    refused = [d for d in ds if not d.served]
    if refused:
        return {"served": False, "cells": None, "cell_rows": None, "settings": None, "reason": refused[0].reason}
    safe = [d for d in ds if d.safe]
    if safe:
        return {"served": True, "cells": "safe", "cell_rows": None, "settings": safe[0].word(SHAPE_WORD[lever]), "reason": ""}
    keys = "+".join(sorted({d.served_by for d in ds}))
    rows = "+".join(str(d.row.get("id", "?")) for d in ds)
    return {"served": True, "cells": keys, "cell_rows": rows, "settings": None, "reason": ""}


def epilogue_word(PF=None) -> str:
    """Which implementation served the fused block's EPILOGUE piece in this process, by the provider's own per-call plan (the core's
    pair_fused cell decides, never the kit): ``fpf`` — the fused epilogue kernel (its cell serves on this card), or
    ``cell:off(<why>)>lnl:<n>`` — the cell for (c_z 128, H 4, D 32) on this capability is named OFF by measurement, so the provider ran the
    gate-transpose + GEMM statements piece for those <n> calls, BY NAME (``opt_core.attn.pair_fused.SERVED_PIECES``)."""
    if PF is None:
        from opt_core.attn import pair_fused as PF
    served = getattr(PF, "SERVED_PIECES", None) or {}
    n = 0; whys = []
    for tag, count in served.items():
        if str(tag).startswith("epilogue=fpf>lnl(off:"):
            n += int(count)
            why = str(tag)[len("epilogue=fpf>lnl(off:"):].split(")", 1)[0]
            if why and why not in whys:
                whys.append(why)
    if n == 0:
        return "fpf"
    return "cell:off(%s)>lnl:%d" % ("+".join(whys) or "not-measured", n)


def set_piece_words(ledger, PF=None) -> None:
    """Put the per-piece census on a fused-block lever's LEVER line (read at print time): ``epilogue=<epilogue_word()>``."""
    try:
        ledger.set("epilogue", epilogue_word(PF))
    except Exception as e:  # noqa: BLE001 — a census word never costs the line
        ledger.set("epilogue", f"unknown:{type(e).__name__}")


def record(ledger, v: dict) -> None:
    """Put the verdict on the lever's LEVER line as ledger facts: `cells=` always (`none` when nothing serves or no device), `cell_rows=` when
    table rows serve, `settings=` when the core's safe settings serve."""
    ledger.set("cells", v["cells"] if v.get("served") else "none")
    if v.get("cell_rows"):
        ledger.set("cell_rows", v["cell_rows"])
    if v.get("settings"):
        ledger.set("settings", v["settings"])


def expected(reason: str, words: Sequence[str]) -> bool:
    """A counted fallback reason is EXPECTED when it equals or starts with a listed word (registry.LEVERS[<lever>]["expected"]): the levers'
    words are prefixes of parametric reasons — `dtype:<t>`, `c:<n>`, `factor:<n>`, the core's `no-cell:<impl>:<piece>:<key>:<cc>|<triton>`
    (the rule of opt_core.trimul.Lever)."""
    return any(reason == w or reason.startswith(w) for w in words)


def gate_for(ledger, words: Sequence[str]):
    """The lever's exit gate (hooks.size_gated semantics: a run whose calls ALL fell back for expected reasons — every input below MIN_TOKENS,
    an fp32-only run — is a legitimate stock run, ok) judged with :func:`expected`'s prefix rule; kernel errors and any other reason refuse."""
    def gate():
        from opt_core.gates import Gate
        f = ledger.fields()
        if f["errors"]:
            return ledger.gate(require_served=False)                    # the core's refusal sentence names the error types
        unexp = {r: n for r, n in f["fallback_by"].items() if not expected(r, words)}
        if unexp:
            from opt_core.report import kv
            return Gate(name=ledger.name, ok=False, reason="unexpected fallback: " + kv(fallback_by=dict(sorted(unexp.items()))), details=f, words=tuple(f["words"]))
        return Gate(name=ledger.name, ok=True, details=f, words=tuple(f["words"]))
    return gate

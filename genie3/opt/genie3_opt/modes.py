"""The mode table — the one place in this tree where a mode names its levers and the switches that select them.

A mode is a list of lever ids (registry.LEVERS); its switches are the kit driver's own flags. `resolve(mode)` turns a mode into the
concrete line: the driver, its flags, the attach point. Nothing else in the package holds a lever value, and no config file does. One
value is a run parameter, not a lever value: the batch size B — upstream's own request key `generation.dataset.batch_size` (loader.py
default 1; on a kit line the value after `--batch-size`, `with_batch`: `design --batch_size B`, else the request's key, else upstream's
default 1 — every arm computes the same request; the batched throughput configuration is `--batch_size 8` on any mode).

Modes (`MODES`):
  off    stock: `genie3 generate <the caller's own arguments>` — upstream's console script, its own flags, cwd = the checkout; the kit
         adds one banner line and its census lines, nothing else.
  exact  the batched capture line — `<genie3_opt>/g3batch.py --config <experiment.yaml> --batch-size B --cuda-graphs --hoist
         --reuse-graphs 16 --lean-pair --wide-capture --alloc expandable`: L1 persistent process + L2 sync-free step and forward patches + L4 CUDA-graph replay of the denoiser core +
         L8 the pair featuriser's step-invariant terms hoisted out of the step loop + L9 in-request graph reuse (one capture per distinct
         batch shape per process) + L11 upstream's own batch semantics at B (the request's designs B per denoiser call in dataset order,
         noised by one draw over the batch tensor per step, written per batch — what `genie3 generate` computes with
         `generation.dataset.batch_size: B`) + L17 the lean pair stack + L18 wide capture + L19 the allocator setting (expandable
         segments). Byte-identical PDB files per design name vs eager deterministic stock AT THE SAME BATCH SIZE
         (`generation.compile: false`, the same `experiment.seed`); not vs compiled stock, whose fused kernels move stock's own bytes.
  fast   the exact line + L12 the pair transition chunked per design (`--pt-chunk design`, class 2) + L13 TF32 tensor-core matmuls (`--tf32`, class 3)
         + L7 the shared core's fused TriangleMultiplication kernel (`--trimul fpf`, class 3; every card with a served row in
         fpf_cells.json — H100 / H200, A100) + L16 the compiled denoiser core inside the graph (`--compile`, class 3): the tolerance tier — different-but-equivalent samples, matched distributionally against
         stock's own seed envelope, never bit for bit. The default (DEFAULT_MODE). A request whose every pair extent is under the kernel's
         floor (101 tokens) declines L7 by name on the KERNELS census line and runs the module's own forward (trimul.word
         `declined:below_min_tokens`).

The default mode is `DEFAULT_MODE` (`fast`), a literal value; `--mode` / GENIE3_OPT override it (`--mode exact` selects the bit-exact
line), and an unset GENIE3_OPT leaves the env route inert at the stock command line (off). Every lever of the registry (registry.LEVERS) is in at least
one mode of this table, and the mode set is the whole surface: no per-lever switch composes a line (the one run parameter is the batch size).
"""
from dataclasses import dataclass, replace as _replace
from typing import Dict, List, Optional, Tuple

from . import registry
from .registry import G3BATCH, PACKAGE

MODE_NAMES: Tuple[str, ...] = ("off", "exact", "fast")
DEFAULT_MODE = "fast"     # the package default: a literal value, not a computed rule — `fast` wherever a fast mode ships; `exact`, the bit-exact line, is selected by name

BATCH_FLAG = "--batch-size"
B_DEFAULT = "1"           # a kit line's batch when neither `design --batch_size` nor the request sets generation.dataset.batch_size: upstream's own default (loader.py setdefault 1) — the kit changes no workload key
BATCH_KEY = ("generation", "dataset", "batch_size")                                        # upstream's request key (src/genie3/config/loader.py: setdefault 1)
BATCHED_CHAIN: Tuple[Tuple[str, str], ...] = ((PACKAGE, G3BATCH),)                          # python <genie3_opt>/g3batch.py …
EXACT_LEVERS: Tuple[str, ...] = ("L1", "L2", "L4", "L8", "L9", "L11", "L17", "L18", "L19")
EXACT_FLAGS: Tuple[str, ...] = tuple(registry.flags(("L11", "L4", "L8", "L9", "L17", "L18", "L19")))   # --batch-size 8 --cuda-graphs --hoist --reuse-graphs 16 --lean-pair --wide-capture --alloc expandable
FAST_LEVERS: Tuple[str, ...] = EXACT_LEVERS + ("L12", "L13", "L7", "L16")
FAST_FLAGS: Tuple[str, ...] = EXACT_FLAGS + tuple(registry.flags(("L12", "L13", "L7", "L16")))    # … --pt-chunk design --tf32 --trimul fpf --compile
TF32_FLAGS: Tuple[str, ...] = tuple(registry.flags(("L13",)))                             # ("--tf32",)
TRIMUL_LEVER = "L7"
TRIMUL_FLAG, TRIMUL_ON = registry.LEVERS[TRIMUL_LEVER].switch[1], registry.LEVERS[TRIMUL_LEVER].switch[2]   # "--trimul", "fpf": the DRIVER's flag pair on fast's line (registry L7's switch)


@dataclass(frozen=True)
class Mode:
    name: str
    tier: Optional[int]                        # 1 byte-identical claim | 2 tolerance | None stock
    levers: Tuple[str, ...]                    # registry ids, in the kit's own order
    attach: str                                # "stock-cli" | "driver"
    driver: Tuple[Tuple[str, str], ...]        # ((kit, relpath),): python <driver> --config … <flags>
    flags: Tuple[str, ...] = ()                # driver flags after the request
    note: str = ""


MODES: Dict[str, Mode] = {
    "off": Mode("off", None, (), "stock-cli", (), note="genie3 generate <the caller's arguments, verbatim> (upstream's console script, cwd = the checkout)"),
    "exact": Mode("exact", 1, EXACT_LEVERS, "driver", BATCHED_CHAIN, EXACT_FLAGS,
                  note="the batched capture line (g3batch.py over the two carried drivers): persistent process, sync-free step + patches, CUDA-graph replay, "
                       "the pair featuriser's step-invariant terms hoisted and its per-step tail inside the graph, the pair stack's intermediates released as consumed, expandable allocator segments, "
                       "in-request graph reuse, upstream's own batch semantics at B (design --batch_size, else generation.dataset.batch_size, else upstream's 1); "
                       "byte-identical PDBs vs eager deterministic stock at the same batch size (not vs compiled stock)"),
    "fast": Mode("fast", 2, FAST_LEVERS, "driver", BATCHED_CHAIN, FAST_FLAGS,
                 note="the exact line + the pair transition chunked per design (--pt-chunk design, class 2) + TF32 tensor-core matmuls (--tf32, class 3) + the shared core's fused "
                      "TriangleMultiplication kernel (--trimul fpf, class 3) + the inductor-compiled denoiser core inside the CUDA graph (--compile, class 3): the tolerance tier, "
                      "matched distributionally; the default mode"),
}


class ModeError(ValueError):
    """An unknown mode, or a lever switch the mode cannot take; the message is the named fact."""


@dataclass(frozen=True)
class Resolution:
    mode: str
    tier: Optional[int]
    levers: Tuple[str, ...]
    attach: str
    driver: Tuple[Tuple[str, str], ...]
    flags: Tuple[str, ...]
    note: str

    @property
    def line(self) -> str:
        """The lever line as the kit writes it (driver + flags; the request as a placeholder), for the activation line and the manifest."""
        if not self.driver:
            return "stock CLI"
        chain = " ".join(f"<{kit}>/{rel}" for kit, rel in self.driver)
        return " ".join((chain, "--config <experiment.yaml>") + self.flags)


def resolve(mode: Optional[str]) -> Resolution:
    """The concrete line for `mode`; no mode = DEFAULT_MODE. An unknown mode is refused by name, before anything runs."""
    if mode is None or not str(mode).strip():
        mode = DEFAULT_MODE
    m = str(mode).strip().lower()
    if m not in MODE_NAMES:
        raise ModeError(f"unknown mode {mode!r} (expected {'|'.join(MODE_NAMES)})")
    md = MODES[m]
    return Resolution(md.name, md.tier, md.levers, md.attach, md.driver, md.flags, md.note)


def batch_size(res: Resolution) -> Optional[int]:
    """The batch size the line passes (the value after --batch-size), or None on the stock line."""
    if BATCH_FLAG in res.flags:
        return int(res.flags[res.flags.index(BATCH_FLAG) + 1])
    return None


def request_batch(request: dict) -> Optional[int]:
    """The request's own `generation.dataset.batch_size`, or None when the file leaves it unset."""
    cur = request
    for k in BATCH_KEY:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return None if cur is None else int(cur)


def with_batch(res: Resolution, batch: Optional[int]) -> Resolution:
    """The same line at batch size `batch` (the request's generation.dataset.batch_size): the value after --batch-size replaced. None keeps
    the table's default; a batch below 1 is refused by name (upstream's loader raises on it too). The stock line carries no driver flag
    (upstream reads the key itself) and is returned unchanged."""
    if batch is None or res.attach != "driver":
        return res
    if int(batch) < 1:
        raise ModeError(f"generation.dataset.batch_size {batch}: the batch size is an integer >= 1")
    flags = list(res.flags)
    flags[flags.index(BATCH_FLAG) + 1] = str(int(batch))
    return _replace(res, flags=tuple(flags))


def trimul_of(res: Resolution) -> str:
    """The TriangleMultiplication provider a line runs: `fpf` when it carries L7's flag pair (fast), else `stock` (the stock and exact lines)."""
    return TRIMUL_ON if TRIMUL_FLAG in res.flags and res.flags[res.flags.index(TRIMUL_FLAG) + 1] == TRIMUL_ON else "stock"


def effective(mode: Optional[str], batch: Optional[int] = None) -> Resolution:
    """The line as a pass runs it: the mode's line at the request's batch size (None keeps the mode's own). The activation report (its
    ACTIVE line: levers, flags, line) and the pass both read this."""
    return with_batch(resolve(mode), batch)


def precision_of(res: Resolution) -> str:
    """The matmul precision a line runs at: `tf32` when it carries L13's flag, else `fp32` (every stock and exact line)."""
    return "tf32" if TF32_FLAGS and all(f in res.flags for f in TF32_FLAGS) else "fp32"


def table() -> List[dict]:
    """The whole table as rows (as `check` reports it): every mode with its levers and switches."""
    rows = []
    for m in MODE_NAMES:
        md = MODES[m]
        rows.append({"mode": m, "tier": md.tier, "levers": list(md.levers), "flags": list(md.flags), "attach": md.attach,
                     "driver": [f"{k}/{r}" for k, r in md.driver], "note": md.note})
    return rows

"""The kit's ONE binding of its kernel levers to the shared core's JAX-family provider `opt_core.kernels.pallas`:
the levers `trimul`, `triatt`, `ln`, `opm_fold` and `transition` keep their adapters here (the haiku class rebinding, the parameter trees, the by-name
gates, the census and the LEVER line) and take THE KERNEL from the provider row a WORD names (rows these adapters bind: `cd_trimul`,
`cd_triatt`, `cd_ln`, `cd_opm`, `cd_transition` under `opt_core/kernels/pallas/` and `opt_core/kernels/pallas_triatt/`),
selected per (jax line, compute capability, dtype, op family, tokens, direction) by `opt_core.kernels.pallas.select`.

Words (the provider's grammar): a ROW word (`cd_trimul`, …) serves exactly that row at the row's own defaults — what the kit's private module
ran, byte for byte; a TIER word (`fast` | `exact` | `big`, narrowed by `prefer=` to the rows this adapter can bind) serves the provider's
MEASURED winner of the cell and names the stock statement (`xla`) outside every measured cell. Each lever binds ONE word (`BINDINGS[<lever>].word`).
Lever `trimul` binds the TIER word (`TIER_WORD` = `fast`, `prefer=(cd_trimul,)`): per (jax line, card, family, size bucket, direction) the provider
serves its measured winner among the rows this adapter binds — `cd_trimul` in every cell the design model's triangle multiplications run on the
measured cards of this kit's jax line (pair c=128 and template c=64, outgoing and incoming, differentiated) — and a cell the provider has not
measured, or measured slower than the stock multiplication, runs the stock method BY NAME for that call (`fallback_by=cell_xla:<n>`), never
silently. Every lever also censuses what the tier word names for each cell it served (`tier=`), printed beside the rows served.

Every lever's LEVER line gains, from `facts(<lever>)`:
    provider=opt_core.kernels.pallas@<core version> word=<bound word> row=<arm served>[:<traced calls>,…] tier=<arm the tier word `fast` names>:<cells>,…
    [uncovered=<n cells>:<family>/<direction>,…]      (the cells of this run the provider's table has no measured number for — named, never swallowed)
(the provider's own cell census also prints at exit: `[opt_core] CELLS …` + one `OPT_IN:pallas|…` / `UNCOVERED_CELL:…`
token per decided call class — the bound word's selections are recorded there as they are made; the tier look-up beside them is not a decision
and is read without recording), and its `impl=` names the row module that ran with the first 8 hex digits of its source's sha256 (`origin=core`: the implementation lives in
the shared core). A provider refusal (an unknown word, MODEL_OPT_LEVERS_OFF naming the provider or the row, a row that cannot serve the model's
main cell) is the lever's refusal BY NAME at install (levers.install: `state=skipped reason=cannot_run detail=…`); a per-call selection that
names a row this adapter does not bind (a tier word's `xla`) runs the stock method for that call, counted `fallback_by=cell_<row>:<n>`.
Pure Python at import (no jax): the registry protocol and the dry-run routes import the lever modules on GPU-less hosts.
"""
from __future__ import annotations

import collections
import hashlib
import importlib
import os
from typing import Any, Dict, Optional, Tuple

PROVIDER = "opt_core.kernels.pallas"
TIER_WORD = "fast"                                  # the tier word whose resolution is censused beside the bound word (the tier this kit's levers belong to)
MAIN_TOKENS = 512                                   # the admission cell at install: the model's main family at a representative size
CELL_FALLBACK = "cell_"                             # fallback reason prefix when the bound word names a row this adapter does not bind for a call (`cell_xla`)


class ProviderRefusal(RuntimeError):
    """The provider refused the lever's word BY NAME (kind = the provider's Refusal.kind, or `import:<module>`)."""

    def __init__(self, text: str, kind: Optional[str] = None):
        super().__init__(text)
        self.kind = kind


class Binding:
    """One lever's binding: the provider op, the rows this adapter can bind -> the row module each imports, the word bound, the model's main
    family (admission), and the per-process census of what was resolved."""

    def __init__(self, lever: str, op: str, word: str, rows: Dict[str, str], main: Dict[str, Any], direction: str = "fwdbwd", dtype: str = "bf16"):
        self.lever, self.op, self.word, self.rows, self.main, self.direction, self.dtype = lever, op, word, dict(rows), dict(main), direction, dtype
        self.reset()

    def reset(self) -> None:
        self.admitted: Optional[Any] = None                                   # the Selection of the main cell at install
        self.stack: Optional[Tuple[str, Optional[str]]] = None                # (jax line, cc | None) the selections were made for
        self.cells: Dict[Tuple, Tuple[str, Dict[str, Any], Optional[str]]] = {}   # (family, dtype, bucket, direction) -> (arm, config, cell_key) under the bound word
        self.tier: Dict[Tuple, Tuple[str, Optional[str]]] = {}               # … -> (arm, cell_key) under TIER_WORD
        self.served: collections.Counter = collections.Counter()             # arm -> traced calls served
        self.modules: Dict[str, Any] = {}                                     # row -> imported module
        self.notes: Dict[str, str] = {}                                       # the provider's notes / refusal texts worth printing once (report, not the line)
        self._noted: Dict[str, int] = {}                                      # note_cells: served-shape key -> calls already censused


# The five levers' bindings. `main` = the model's principal family for the op (AlphaFold-Multimer design model: pair channel 128, 4 heads x 32
# in triangle attention, MSA channel 256 with 2 sequences, outer-product channel 32 -> 128).
BINDINGS: Dict[str, Binding] = {
    "trimul": Binding("trimul", "trimul", TIER_WORD, {"cd_trimul": "opt_core.kernels.pallas.cd_trimul.trimul_pallas"},
                      dict(form="af2", unit="pair", c=128, c_hidden=128, equation="outgoing")),
    "triatt": Binding("triatt", "attn", TIER_WORD, {"cd_triatt": "opt_core.kernels.pallas_triatt.triatt_attn"},      # the tier word: the provider's measured row per cell (rows this adapter binds: cd_triatt)
                      dict(kind="tri", heads=4, head_dim=32)),
    "ln": Binding("ln", "ln", TIER_WORD, {"cd_ln": "opt_core.kernels.pallas.cd_layers.layers_ln"},
                  dict(unit="pair", c=128)),
    "opm_fold": Binding("opm_fold", "opm", TIER_WORD, {"cd_opm": "opt_core.kernels.pallas.cd_layers.layers_opm"},
                        dict(form="af2", c_m=256, c=32, f=128, n_seq=2)),
    "transition": Binding("transition", "transition", TIER_WORD, {"cd_transition": "opt_core.kernels.pallas.cd_layers.layers_transition"},
                          dict(form="af2", activation="relu", c=128, factor=4)),     # the tier word: the provider's measured row per cell (pair transition c128 x4 = the main cell; c64 x2 template / c64 x4 extra-MSA / c256 x4 MSA cells per call); `xla` cells run the stock class by name
}


def provider():
    """The provider package (pure Python at import: the cell table and `select`)."""
    return importlib.import_module(PROVIDER)


def core_version() -> str:
    try:
        return str(importlib.import_module("opt_core").__version__)
    except Exception:                                                         # noqa: BLE001
        return "unknown"


def provider_word() -> str:
    return f"{PROVIDER}@{core_version()}"


def stack() -> Tuple[str, Optional[str]]:
    """(jax line 'X.Y', compute capability 'M.m' | None) of this process — what the provider's cells are keyed by. cc is None off a GPU
    backend (the provider then selects for its default card, OPT_CORE_PALLAS_CC or 9.0: a row word serves the same row on any card)."""
    try:
        import jax  # noqa: WPS433
        line = ".".join(str(jax.__version__).split(".")[:2])
    except Exception:                                                         # noqa: BLE001 - no jax / a stand-in: the provider names the line unknown by itself
        return "unknown", None
    try:                                                                      # the default backend's device, as the provider's faces read it ('9.0', '8.0')
        cc = str(getattr(jax.devices()[0], "compute_capability", None) or "")
    except Exception:                                                         # noqa: BLE001
        cc = ""
    return line, (cc or None)


def module_sha8(mod) -> str:
    """`<module basename>@<first 8 hex digits of the source file's sha256>` — what ran, byte for byte."""
    name = mod.__name__.rsplit(".", 1)[-1]
    try:
        with open(mod.__file__, "rb") as fh:
            return f"{name}@{hashlib.sha256(fh.read()).hexdigest()[:8]}"
    except (OSError, AttributeError, TypeError):
        return f"{name}@unknown"


def row_file(lever: str, row: Optional[str] = None) -> Optional[str]:
    """The row module's source file, located WITHOUT importing it (importlib's finder imports the parent packages only, which are pure Python):
    the off / skipped lines name the implementation on hosts where the kernels cannot import."""
    import importlib.util  # noqa: WPS433
    b = BINDINGS[lever]
    row = row or bound_row(lever)
    try:
        spec = importlib.util.find_spec(b.rows[row])
    except Exception:                                                         # noqa: BLE001
        return None
    return getattr(spec, "origin", None) if spec is not None else None


def bound_row(lever: str) -> str:
    """The row the lever serves: the admitted selection's row once installed, else the row its word names (a row word), else its first bindable row."""
    b = BINDINGS[lever]
    if b.admitted is not None and getattr(b.admitted, "row", None) in b.rows:
        return b.admitted.row
    return b.word if b.word in b.rows else next(iter(b.rows))


def kernel_word(lever: str, row: Optional[str] = None) -> str:
    """`<row module name>@<first 8 hex digits of its source's sha256>` — the LEVER line's impl= (what ran / would run, byte for byte)."""
    b = BINDINGS[lever]
    row = row or bound_row(lever)
    name = b.rows[row].rsplit(".", 1)[-1]
    f = row_file(lever, row)
    if not f:
        return f"{name}@unknown"
    try:
        with open(f, "rb") as fh:
            return f"{name}@{hashlib.sha256(fh.read()).hexdigest()[:8]}"
    except OSError:
        return f"{name}@unknown"


def row_module(b: Binding, row: str):
    """The row's module, imported once (ProviderRefusal `import:<path>` when it does not import here)."""
    if row not in b.rows:
        raise ProviderRefusal(f"lever {b.lever}: row {row!r} is not one this adapter binds ({'|'.join(b.rows)})", kind="row_unbindable")
    if row not in b.modules:
        path = b.rows[row]
        try:
            b.modules[row] = importlib.import_module(path)
        except Exception as e:                                                # noqa: BLE001
            raise ProviderRefusal(f"lever {b.lever}: row module {path} does not import here ({type(e).__name__}: {e})", kind=f"import:{path}") from None
    return b.modules[row]


def admit(lever: str):
    """At install: resolve the lever's word for the model's MAIN cell through the provider and import the row module it names. Returns the
    module. Raises ProviderRefusal (the provider's refusal by name, or a word that names a row this adapter cannot bind for the main cell)."""
    b = BINDINGS[lever]
    P = provider()
    line, cc = stack()
    b.stack = (line, cc)
    try:                                                                      # the core's cell census names the engine in its exit lines / dump (observation only)
        from opt_core import cell_census  # noqa: WPS433
        cell_census.set_context(engine="colabdesign")
    except Exception:                                                         # noqa: BLE001 - an older core has no census: nothing to name
        pass
    fam = P.family(b.op, **b.main)
    try:
        sel = P.select(line, cc, b.dtype, b.op, fam, MAIN_TOKENS, b.direction, word=b.word, prefer=tuple(b.rows))
    except (P.Refusal, ValueError) as e:                                      # the provider's refusal by name | a word / card it cannot parse
        raise ProviderRefusal(f"lever {lever}: the provider refused word {b.word!r} for {fam} on jax {line} / cc {cc}: {e}", kind=getattr(e, "kind", None) or type(e).__name__) from None
    if sel.row not in b.rows:
        raise ProviderRefusal(f"lever {lever}: word {b.word!r} names row {sel.row!r} for the model's main cell ({fam}, {MAIN_TOKENS} tokens, {b.direction}) "
                              f"— not a row this adapter binds ({'|'.join(b.rows)}); {sel.note or ''}".strip(), kind="row_unbindable")
    b.admitted = sel
    if sel.note:
        b.notes["admit"] = str(sel.note)
    return row_module(b, sel.row)


def _bucket(n: int) -> int:
    """Selections are memoised per size bucket the provider's cells use (N<=200 / 400 / 500 / 800 / 1200 / above): one select per bucket."""
    for edge in (200, 400, 500, 800, 1200):
        if n <= edge:
            return edge
    return 1 << 20


def _nearest_depth(P, op: str, prefix: str, s: int) -> int:
    """The provider's measured sequence depth at or above s for families '<prefix><S>' (else the deepest measured; s when none)."""
    depths = []
    for f in P.families(op):
        if f.startswith(prefix):
            tail = f[len(prefix):]
            try:
                depths.append(int(tail.split("_")[0]))
            except ValueError:
                continue
    if not depths:
        return s
    above = sorted(d for d in depths if d >= s)
    return above[0] if above else max(depths)


def family_word(lever: str, **shape) -> str:
    """The provider family word of one call of the lever (shape words per op: see `resolve`)."""
    b = BINDINGS[lever]
    P = provider()
    if b.op == "opm":                                                         # nearest measured MSA depth, as the provider's own face picks it
        s = int(shape.get("n_seq", 1))
        shape = dict(shape, n_seq=_nearest_depth(P, "opm", "%s_opm_cm%d_c%d_f%d_S" % (shape.get("form", "af2"), int(shape["c_m"]), int(shape["c"]), int(shape["f"])), s))
    elif b.op == "attn" and shape.get("kind") == "msarow":
        s = int(shape.get("n_seq", 1))
        shape = dict(shape, n_seq=_nearest_depth(P, "attn", "msarow_s", s))
    elif b.op == "ln" and shape.get("unit") == "msa":
        s = int(shape.get("n_seq", 1))
        shape = dict(shape, n_seq=_nearest_depth(P, "ln", "msa_c%d_s" % int(shape["c"]), s))
    return P.family(b.op, **shape)


def _cell(lever: str, n_tokens: int, dtype: str, direction: Optional[str], count: int, **shape) -> Tuple[str, Dict[str, Any]]:
    """(arm, config) of the call's cell under the bound word, memoised per (family, dtype, size bucket, direction); the tier word's arm for the
    same cell is censused beside it; `count` traced calls are added to the arm's served count."""
    b = BINDINGS[lever]
    P = provider()
    if b.stack is None:
        b.stack = stack()
    line, cc = b.stack
    d = direction or b.direction
    try:
        fam = family_word(lever, **shape)
    except Exception:                                                         # noqa: BLE001 - a shape the provider's family grammar does not name: censused under its raw words
        fam = "_".join(f"{k}{v}" for k, v in sorted(shape.items()))
    key = (fam, dtype, _bucket(int(n_tokens)), d)
    if key not in b.cells:
        n = min(int(n_tokens), key[2])
        try:
            sel = P.select(line, cc, dtype, b.op, fam, n, d, word=b.word, prefer=tuple(b.rows))
            b.cells[key] = (sel.arm, dict(sel.config or {}), sel.cell_key)
        except Exception as e:                                                # noqa: BLE001 - the bound word refused for this cell (Refusal, by name) or the selection failed: the stock method, by that name
            b.cells[key] = (f"refused_{getattr(e, 'kind', None) or type(e).__name__}", {}, None)
            b.notes.setdefault(f"refused:{fam}/{d}", str(e)[:300])
        try:                                                                  # the tier word's arm for the same cell, LOOKED UP not decided: the provider's pure selection
            look = getattr(P, "_select", P.select)                            # (the core records every select() in opt_core.cell_census; a look-up must not count as a served decision)
            t = look(line, cc, dtype, b.op, fam, n, d, word=TIER_WORD, prefer=tuple(b.rows))
            b.tier[key] = (t.arm, t.cell_key)
        except Exception as e:                                                # noqa: BLE001
            b.tier[key] = (f"refused_{getattr(e, 'kind', None) or type(e).__name__}", None)
    arm, cfg, _cell_key = b.cells[key]
    b.served[arm] += int(count)
    return arm, dict(cfg)


def resolve(lever: str, n_tokens: int, dtype: str = "bf16", direction: Optional[str] = None, **shape):
    """Per traced call: (module | None, config) for the call's cell under the bound word — None when the word names a row this adapter does
    not bind (the caller runs the stock method, `fallback_by=cell_<row>`). The tier word's resolution of the same cell is censused beside it.
    Memoised per (family, dtype, size bucket, direction)."""
    b = BINDINGS[lever]
    arm, cfg = _cell(lever, n_tokens, dtype, direction, 1, **shape)
    row = arm.split("@", 1)[0].split(":", 1)[0]
    if row in b.rows:
        return row_module(b, row), cfg
    return None, {}


def note_cells(lever: str, shapes: Dict[str, int], dtype: str = "bf16") -> None:
    """Census after the fact for a lever whose op serves every call through ONE function (triatt: the serve layer's contract): the served-shape
    keys of its Ledger (`B<b>xH<h>xS<s>xD<d>[+…]: n`) resolved cell by cell exactly as `resolve` would have, counts included. Idempotent per key."""
    import re  # noqa: WPS433
    b = BINDINGS[lever]
    seen: Dict[str, int] = getattr(b, "_noted", {})
    for key, n in sorted(shapes.items()):
        done = int(seen.get(key, 0))
        if int(n) <= done:
            continue
        m = re.match(r"B(\d+)xH(\d+)xS(\d+)xD(\d+)", str(key))
        if not m:
            continue
        B, H, S, D = (int(x) for x in m.groups())
        if D <= 8:
            shape = dict(kind="extramsa_slab512", heads=H, head_dim=D, n_seq=B)
        elif D == 16:
            shape = dict(kind="tmpl", heads=H, head_dim=D)
        elif B != S:
            shape = dict(kind="msarow", heads=H, head_dim=D, n_seq=B)
        else:
            shape = dict(kind="tri", heads=H, head_dim=D)
        _cell(lever, S, dtype, None, int(n) - done, **shape)                   # the Ledger's count = the traced calls of that shape
        seen[key] = int(n)
    b._noted = seen


def fallback_word(lever: str, n_tokens: int, dtype: str = "bf16", direction: Optional[str] = None, **shape) -> str:
    """The fallback reason word of a call `resolve` handed back to the stock method: `cell_<arm>`."""
    b = BINDINGS[lever]
    try:
        fam = family_word(lever, **shape)
    except Exception:                                                         # noqa: BLE001
        fam = "_".join(f"{k}{v}" for k, v in sorted(shape.items()))
    arm = b.cells.get((fam, dtype, _bucket(int(n_tokens)), direction or b.direction), ("unknown", {}, None))[0]
    return f"{CELL_FALLBACK}{arm}"


def _census(counter) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(counter.items())) or "none"


def facts(lever: str) -> Dict[str, str]:
    """The binding's printed facts: provider, word, row (arms served : traced calls), tier (the tier word's arm : cells), uncovered (cells
    without a measured number under the tier word, by family/direction) — blank-free tokens for the LEVER line."""
    b = BINDINGS[lever]
    tier = collections.Counter(arm for arm, _ in b.tier.values())
    unc = sorted({f"{k[0]}/{k[3]}" for k, (arm, cell) in b.tier.items() if cell is None})
    out = {"provider": provider_word(), "word": b.word, "row": _census(b.served) if b.served else (b.admitted.row if b.admitted is not None else "none"),
           "tier": _census(tier)}
    if unc:
        out["uncovered"] = f"{len(unc)}:" + ",".join(unc)
    if "census_error" in b.notes:
        out["census_error"] = b.notes["census_error"].split(":", 1)[0]
    return out


def report(lever: str) -> Dict[str, Any]:
    """Everything resolved for the lever in this process (evidence(); the cells by key)."""
    b = BINDINGS[lever]
    return {"provider": provider_word(), "word": b.word, "stack": b.stack, "admitted": (b.admitted.as_dict() if hasattr(b.admitted, "as_dict") else None),
            "cells": {"|".join(str(x) for x in k): {"arm": v[0], "config": v[1], "cell": v[2]} for k, v in b.cells.items()},
            "tier": {"|".join(str(x) for x in k): {"arm": v[0], "cell": v[1]} for k, v in b.tier.items()},
            "served": dict(b.served), "notes": dict(b.notes)}


def reset_for_tests() -> None:
    for b in BINDINGS.values():
        b.reset()

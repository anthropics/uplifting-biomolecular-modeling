"""The `trimul_form` lever (exact class; the `exact` line; rides on `trimul_exact`): the triangle-multiplication call classes whose statement is
the MODULE's own inference path -- `TriangleMultiplicativeUpdate._inference_forward`, what `PairBlock.tri_mul_out_in` executes when
`use_cueq_triangle_kernels` is False at the call: the template pair stack (c_z 64, c_hidden 64; `TemplatePairBlock` never passes the flag) on
every input, and every pair block when the runner configuration switches the cuEquivariance kernels off -- are served by the core's ONE
triangle-multiplication provider asked for the EXACT TIER UNDER THIS ENGINE'S FORM KEY -- `select(word="exact", form="of3_module")`, spelt
`exact+of3_module` through `opt_core.trimul.by_word`: the provider's row `of3_form` (that statement issued whole-tensor
-- LayerNorms on the core's exactln row, the projections and the gate as the module's cuBLAS GEMMs, the contraction in upstream's 256-column
blocks; `MODULE_EXACT_ROWS`: the exact tier's value only under its form key) where a form cell of `TRIMUL_CELLS.json` proves it bitwise to
this module, refused by name elsewhere (the module serves).  `trimul_exact` keeps the bare word `exact` (no form: the stock library op's
exact tier) for the classes whose statement is the library (the flag on at the call: the trunk, MSA-module and confidence pair stacks under
the stock configuration).  A module-statement class is served only after the first call of each
(class, token count) proved `torch.equal` against the module's statement on that call's operands -- cuBLAS chooses its kernels per problem
shape, so the proof is held per shape, not per bucket; a shape whose bits differ refuses its class for the process, named, and the module
serves it.  Calls the row does not state (use_triton_triangle_kernels, the module's out-of-place path) are the module's, counted by reason.

Switch: OPENFOLD3_OB0_OPT_TRIMUL_EXACT_FORM=1, read by `trimul_exact` at install (this module holds the record and the words; it patches nothing
itself).  Left off (MODEL_OPT_LEVERS_OFF=trimul_form) the module-statement classes are the line's, counted `form_off` on trimul_exact's census.
Evidence: `[openfold3_ob0-opt/trimul_form] armed ...` and the exit line
`[openfold3_ob0-opt/trimul_form] LEVER name=trimul_form state=on word=exact+of3_module row=of3_form served=<calls> proven=<classes|none> shapes=<n>
refused=<reason>:<n>,...|none line=<reason>:<n>,...|none cells=<form cell>|...|none` (state=off when not requested; state=refused
reason=core_form_absent:... when the pinned core does not name the form)."""
import atexit
import os
import sys
import threading

ENV = "OPENFOLD3_OB0_OPT_TRIMUL_EXACT_FORM"
PREFIX = "[openfold3_ob0-opt/trimul_form]"
VALUES = ("1",)
FORM = "of3_module"                                # this engine's form key (opt_core.kernels.trimul FORMS): the module statement the exact tier answers under it
WORD = "exact+" + FORM                             # the tier spelling the binding passes (opt_core.trimul.by_word: word exact, form of3_module)
ROW = "of3_form"                                   # the row that form resolves to (FORMS[FORM]; ROW_NAMES / EXACT_ROWS / MODULE_EXACT_ROWS) -- the census names it
MODULE_TAG = "module"                              # the class-key field trimul_exact appends for a call whose statement is the module's (`<cell key>|module`)
CORE_FLOOR = "0.5.67.0"                            # the first core release keying the exact tier by form (the kit's [tool.opt_core] pin is the gate; this word is for the census)
STATE = {"installed": False, "state": "off", "reason": None, "served": 0, "proven": [], "shapes": [], "refused": {}, "line": {}, "cells": {}}
_LOCK = threading.Lock()


def requested(environ=None) -> bool:
    v = ((os.environ if environ is None else environ).get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    return True


def serving() -> bool:
    return bool(STATE.get("installed")) and STATE.get("state") == "on"


def bump(key, word, n=1):
    """Count `n` under STATE[key][word] (refused / line) or STATE['served'] (key == 'served')."""
    with _LOCK:
        if key == "served":
            STATE["served"] = int(STATE["served"]) + n
        else:
            d = STATE[key]
            d[word] = d.get(word, 0) + n


def note_proven(class_key, unit):
    """The first call of (class, token count) `unit` proved torch.equal against the module: the class is listed once, the shape counted once."""
    with _LOCK:
        if class_key not in STATE["proven"]:
            STATE["proven"].append(class_key)
        if unit not in STATE["shapes"]:
            STATE["shapes"].append(unit)


def statement_word(use_cueq_triangle_kernels: bool) -> str:
    """The statement a `PairBlock.tri_mul_out_in` call executes, by the flag at the call: `library` (the cuEquivariance op through the module's
    wrapper) or `module` (TriangleMultiplicativeUpdate._inference_forward).  The word decides the provider word: `exact` for library classes (no
    form: the table's exact tier against the library op), `exact+of3_module` for module classes (the exact tier under this engine's form)."""
    return "library" if use_cueq_triangle_kernels else MODULE_TAG


def provider_word(statement: str) -> str:
    return WORD if statement == MODULE_TAG else "exact"


def class_key(cell_key: str, statement: str) -> str:
    """The census / memo key of a call class: the core's cell key, plus `|module` when the statement is the module's (the same cell keys both)."""
    return f"{cell_key}|{MODULE_TAG}" if statement == MODULE_TAG else str(cell_key)


def proof_unit(key: str, statement: str, n_tokens: int) -> str:
    """What one first-call proof covers: the class for a library-statement class (kernel rows, shape-generic -- trimul_exact's rule),
    the (class, token count) pair for a module-statement class (cuBLAS heuristics choose per problem shape)."""
    return f"{key}@{int(n_tokens)}" if statement == MODULE_TAG else key


def fields() -> str:
    j = lambda d: ",".join(f"{k}:{v}" for k, v in sorted(d.items())) or "none"
    return (f"word={WORD} row={ROW} served={STATE['served']} proven={','.join(STATE['proven']) or 'none'} shapes={len(STATE['shapes'])} "
            f"refused={j(STATE['refused'])} line={j(STATE['line'])} cells={'|'.join(sorted(STATE['cells'])) or 'none'}")


def census_line() -> str:
    st = STATE.get("state", "off")
    body = fields() if st == "on" else f"reason={STATE.get('reason') or 'not_requested'}"
    return f"{PREFIX} LEVER name=trimul_form state={st} {body}"


def arm(KT, environ=None) -> bool:
    """Called by trimul_exact.install() with the core's pure trimul face `KT` (opt_core.kernels.trimul): True when the switch is set AND the core
    keys the exact tier by this engine's form (FORMS[FORM] == ROW, a module-exact row) -- trimul_exact then asks `exact+of3_module` for
    module-statement classes.  Idempotent; registers the exit line."""
    if STATE.get("installed"):
        return STATE.get("state") == "on"
    if not requested(environ):
        STATE.update(installed=True, state="off", reason="not_requested")
        return False
    rows, forms = tuple(getattr(KT, "MODULE_EXACT_ROWS", ()) or ()), dict(getattr(KT, "FORMS", {}) or {})
    if forms.get(FORM) != ROW or ROW not in rows or ROW not in tuple(getattr(KT, "ROW_NAMES", ()) or ()):
        STATE.update(installed=True, state="refused", reason=f"core_form_absent:{FORM}(opt_core>={CORE_FLOOR})")
        sys.stderr.write(f"{PREFIX} not armed: the core's trimul provider keys no exact form {FORM} -> {ROW} (FORMS={sorted(forms)}, MODULE_EXACT_ROWS={rows}); "
                         f"the module-statement classes stay the line's\n")
        atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
        return False
    STATE.update(installed=True, state="on", reason=None)
    sys.stderr.write(f"{PREFIX} armed: module-statement TriMul classes (use_cueq_triangle_kernels False at the call: the template pair stack; every pair block "
                     f"when the library is off) -> opt_core.kernels.trimul word {WORD} (row {ROW} where a form cell proves it), proven torch.equal per (class, "
                     f"token count) against the module in this process\n")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return True

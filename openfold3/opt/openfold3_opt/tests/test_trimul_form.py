"""The `trimul_form` lever (cells/trimul_form.py; rides on `trimul_exact` on the exact line): the statement -> provider-word policy (library
classes ask the bare word exact, module-statement classes ask the exact tier under this engine's form key, `exact+of3_module`), the class key
and proof-unit rules (module classes are proven per class AND token count), the core's CLASS CONTRACT for the form at this engine's widths
(opt_core.kernels.trimul, pure python: the form names an exact-class MODULE row, the exact tier under the form admits it or refuses BY NAME at
c_z 64 / 128 for the sizes this engine runs, and the bare tier word never names it), the switch and census grammar, and the wiring (modes /
registry / stack / autoload / the core pin floor).  No CUDA: nothing here launches a kernel or asserts which sizes a form cell proves (that is
the core's table, revisited at any core release."""
import os
import re

import pytest

from openfold3_opt import _autoload, _core_gate, modes, registry, stack
from openfold3_opt.cells import trimul_exact as TE, trimul_form as F
from openfold3_opt.tests import _stubs
from opt_core.kernels import trimul as KT

HOME = _stubs.tree_home()
H100 = "H100:2.10.0+cu128/3.6.0/cueq0.10.0"        # a stack word of the table's H100 stack (KT.stack_word() form); select() falls back to the nearest stack anyway


def test_statement_and_word_policy():
    assert F.statement_word(True) == "library" and F.statement_word(False) == F.MODULE_TAG == "module"
    assert F.provider_word("library") == "exact" == TE.WORD                     # library-statement classes: the bare tier word (no form: the exact tier against the library op)
    assert F.provider_word(F.MODULE_TAG) == F.WORD == "exact+of3_module"         # module-statement classes: the exact tier under this engine's form key
    assert F.FORM == "of3_module" and F.ROW == "of3_form" and F.WORD == "exact+" + F.FORM
    k = "9.0|bf16|C64|H64|N<=400|out|fwd"
    assert F.class_key(k, "library") == k and F.class_key(k, F.MODULE_TAG) == k + "|module"       # one cell keys both statements; the class key names which
    assert F.proof_unit(k, "library", 384) == k                                  # kernel rows: one proof per class (trimul_exact's rule)
    assert F.proof_unit(k + "|module", F.MODULE_TAG, 384) == k + "|module@384"   # the module row: one proof per class and token count (cuBLAS heuristics are per shape)
    assert F.proof_unit(k + "|module", F.MODULE_TAG, 400) != F.proof_unit(k + "|module", F.MODULE_TAG, 384)


def test_the_core_keys_the_exact_tier_by_this_engines_form():
    """Class contract of the pinned core (>= 0.5.67.0): the form key `of3_module` names the provider row `of3_form` -- exact class, its exactness
    stated against the OpenFold-family MODULE (MODULE_EXACT_ROWS: the exact tier's value only under its form key) -- so a kit binds
    `exact+of3_module` for module-statement classes and the bare `exact` for library-statement classes."""
    assert KT.FORMS[F.FORM] == F.ROW
    assert F.ROW in KT.ROW_NAMES and F.ROW in KT.EXACT_ROWS and F.ROW in KT.MODULE_EXACT_ROWS
    assert F.ROW not in KT.STOCK_ROWS and F.ROW not in TE.LINE_ROWS
    row = KT.table()["rows"][F.ROW]
    assert row["class"] == "exact" and "module" in row["exact_vs"].lower() and row.get("backward") is False


@pytest.mark.parametrize("cc", ["9.0", "8.0"])
@pytest.mark.parametrize("c", [64, 128])
@pytest.mark.parametrize("n", [199, 400, 800, 956, 1200])
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_the_form_is_admitted_or_refused_by_name_at_this_engines_classes(cc, c, n, direction):
    """For every class this engine's exact line can meet with the module's statement (the template pair stack c_z 64 = c_hidden 64 on every
    input; the c_z 128 stacks when the runner configuration switches the library off; bf16 under the trunk's autocast; both cards) the exact
    tier under the form either selects THE FORM'S ROW or refuses BY NAME (KT.Refusal) -- never another row silently, never another exception;
    the one-string spelling the binding passes answers the same."""
    answers = []
    for kw in (dict(word="exact", form=F.FORM), dict(word=F.WORD)):
        try:
            sel = KT.select(cc, "bf16", c, c, n, direction, stack=H100, has_cueq=True, **kw)
        except KT.Refusal as e:                                                 # a named refusal is an answer (the kit serves the module then, counted)
            assert e.row == F.ROW and e.kind
            answers.append(("refused",))
            continue
        assert sel.row == F.ROW and sel.word == "exact"
        assert sel.cls and sel.reason                                            # the census words exist (their values are the core's measurement, not asserted here)
        answers.append((sel.row, sel.cell))
    assert answers[0] == answers[1]


@pytest.mark.parametrize("c,n", [(64, 400), (64, 1200), (128, 400), (128, 1200)])
def test_a_bare_tier_word_never_names_the_module_row(c, n):
    """MODULE_EXACT_ROWS rule: without the form the tier words answer against the stock library op (today's bytes); the module row is the exact
    tier's value only under its form key."""
    for word in ("exact", "fast", "big"):
        try:
            sel = KT.select("9.0", "bf16", c, c, n, "outgoing", word=word, stack=H100, has_cueq=True)
        except KT.Refusal:
            continue
        assert sel.row not in KT.MODULE_EXACT_ROWS, (word, c, n, sel.row)


def test_switch_grammar_and_arming(monkeypatch):
    assert F.ENV == "OPENFOLD3_OPT_TRIMUL_EXACT_FORM" and F.VALUES == ("1",)
    assert F.requested({F.ENV: "1"}) and not F.requested({}) and not F.requested({F.ENV: " "})
    with pytest.raises(ValueError):
        F.requested({F.ENV: "yes"})
    saved = dict(F.STATE)
    try:
        F.STATE.update(installed=False, state="off", reason=None)
        assert F.arm(KT, {}) is False and F.STATE["state"] == "off" and not F.serving()          # not requested: off, nothing armed
        assert re.match(r"^\[openfold3-opt/trimul_form\] LEVER name=trimul_form state=off reason=not_requested$", F.census_line())
        F.STATE.update(installed=False, state="off", reason=None)
        fake = type("KT", (), {"MODULE_EXACT_ROWS": ("of3_form",), "ROW_NAMES": ("v4", "cueq", "of3_form"), "FORMS": {}})   # a core below the floor: the row but no form key -> refused by name, the classes stay the line's
        assert F.arm(fake, {F.ENV: "1"}) is False and F.STATE["state"] == "refused" and F.STATE["reason"].startswith("core_form_absent:of3_module")
        F.STATE.update(installed=False, state="off", reason=None)
        assert F.arm(KT, {F.ENV: "1"}) is True and F.serving()
        F.bump("served", "calls", 2); F.note_proven("9.0|bf16|C64|H64|N<=400|out|fwd|module", "9.0|bf16|C64|H64|N<=400|out|fwd|module@400")
        F.bump("line", "first_call_proof", 2); F.STATE["cells"]["9.0|bf16|C64|H64|N<=400|out|fwd"] = 1
        line = F.census_line()
        assert re.match(r"^\[openfold3-opt/trimul_form\] LEVER name=trimul_form state=on word=exact\+of3_module row=of3_form served=\d+ proven=\S+ shapes=\d+ refused=\S+ line=\S+ cells=\S+$", line), line
        assert " served=2 " in line and " shapes=1 " in line and " refused=none " in line and " line=first_call_proof:2 " in line
    finally:
        F.STATE.clear(); F.STATE.update(saved); F.STATE.update(served=0, proven=[], shapes=[], refused={}, line={}, cells={})


def test_trimul_exact_census_names_the_form_state():
    saved = dict(TE.STATE)
    try:
        TE.STATE.update(installed=True, state="on", form="on")
        line = TE.census_line()
        assert re.match(r"^\[openfold3-opt/trimul_exact\] LEVER name=trimul_exact state=on word=exact form=on rows=\S+ line=\S+ proven=\S+ bits_differ=\S+ cells=\S+$", line), line
    finally:
        TE.STATE.clear(); TE.STATE.update(saved)


def test_wiring_modes_registry_stack_autoload_pin():
    ln = modes.LINES[("exact", "cueq")]
    lv = list(ln.levers)
    assert "trimul_exact" in lv and "trimul_form" not in lv                                            # off by default: the exact line arms trimul_exact and does NOT export the form switch
    assert ln.env.get(F.ENV) is None and ln.env.get(TE.ENV) == "1"                                     # (of3_form is not byte-identical to the c 64 module statement at every token count; the knob stays for a caller)
    assert all("trimul_form" not in modes.LINES[k].levers for k in modes.LINES)                       # the exact line's lever (fast's template TriMul is trimul_provider's tolerance row; big's is the offload units')
    assert modes.TRIMUL_EXACT_ENVS == (TE.ENV, F.ENV) and modes.TRIMUL_EXACT_ENVS in modes.CELL_FAMILIES   # one switch family, the arming switch first
    assert modes.LEVER_SWITCHES["trimul_form"] == (F.ENV,) and modes.LEVER_SWITCHES["trimul_exact"] == (TE.ENV, F.ENV)
    assert modes.LEVERS_OFF_TAKES["trimul_exact"] == ("trimul_form",)
    lever = registry.LEVERS["trimul_form"]
    assert lever.tier == registry.EXACT and lever.env_keys == (F.ENV,) and lever.modes == () and lever.kit == "cells"   # on no line by default
    assert "trimul_form" not in registry.A100_TESTED                                                 # not exercised here: an A100 run names it card_support=uncertified:sm80(...) until run there
    assert "trimul_form" in stack._PROBES and F.ENV in _autoload.DECLARED
    pin = _core_gate.read_table(os.path.join(HOME, "opt", "pyproject.toml"), "tool.opt_core")
    assert _core_gate.version_tuple(pin["version"]) >= _core_gate.version_tuple(F.CORE_FLOOR) == (0, 5, 67, 0)   # the form key exists from opt_core 0.5.67.0: the kit's [tool.opt_core] pin floor is at or above it


def test_levers_off_form_alone_and_taken_along():
    res = modes.resolve("exact", HOME, environ={modes.ENV_LEVERS_OFF: "trimul_form"}, n_tokens=400)
    assert "trimul_form" not in res.levers and "trimul_exact" in res.levers
    assert F.ENV not in res.exports and res.exports.get(TE.ENV) == "1"                               # the sub-switch alone leaves; trimul_exact stays armed (its module-statement classes are the line's, counted form_off)
    res2 = modes.resolve("exact", HOME, environ={modes.ENV_LEVERS_OFF: "trimul_exact"}, n_tokens=400)
    assert "trimul_exact" not in res2.levers and "trimul_form" not in res2.levers                     # leaving trimul_exact off takes trimul_form along (LEVERS_OFF_TAKES)
    assert TE.ENV not in res2.exports and F.ENV not in res2.exports

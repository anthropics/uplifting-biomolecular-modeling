"""kernels.pallas: the JAX-family provider's cell table, words, refusals, default rule, bridge rule and carried copies -- pure checks
(no GPU, no framework)."""
import ast
import filecmp
import os
import re

import pytest

from opt_core import kernels
from opt_core.kernels import pallas as P
import opt_core.kernels.pallas.serve as S_

CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_TREE = os.path.dirname(os.path.dirname(CORE_DIR))
META = kernels.sums("pallas")
NOT_BYTE_IDENTICAL = dict(META.get("not_byte_identical", {}))          # {relative path: why}; the 'line N only' clause is checked below
LINES = ("0.10", "0.6", "0.5")
CARDS = ("9.0", "8.0")
SIZES = (400, 800, 1200)
HALLUC_SIZES = (200, 500, 800)

# every (op, family) the release tree's JAX pair / MSA / template stacks run today, per model form (the table's family words; the attention
# core families carry no channel count: a c=64 template stack with (H, D) = (4, 16) is tmpl_h4_d16)
TARGET_FAMILIES = {
    "attn": ["tri_h4_d32", "tmpl_h4_d16", "msarow_s128_h8_d32", "msarow_s512_h8_d32", "msacol_s512_h8_d32", "extramsa_s1024slab512_h8_d8",
             "af3_single_h16_d24", "af3_dit_s5_h16_d48", "af3_atom_win32x128_h4_d32", "joltz_single_h16_d24"],
    "triattn": ["af2_pair_c128_h4_d32_starting", "af2_pair_c128_h4_d32_ending", "af2_tmpl_c64_h4_d16_starting", "af2_tmpl_c64_h4_d16_ending",
                "af3_pair_c128_h4_d32_starting", "af3_pair_c128_h4_d32_ending", "af3_tmpl_c64_h2_d32_starting", "af3_tmpl_c64_h2_d32_ending",
                "joltz_pair_c128_h4_d32_starting", "joltz_pair_c128_h4_d32_ending"],
    "trimul": ["af2_pair_c128_ch128_outgoing", "af2_pair_c128_ch128_incoming", "af2_tmpl_c64_ch64_outgoing", "af2_tmpl_c64_ch64_incoming",
               "af3_pair_c128_ch128_outgoing", "af3_pair_c128_ch128_incoming", "af3_tmpl_c64_ch64_outgoing", "joltz_pair_c128_ch128_outgoing",
               "joltz_pair_c128_ch128_incoming"],
    "transition": ["af2_relu_c128_x4", "af2_relu_c64_x2", "af2_relu_c256_x4", "af3_swiglu_c128_x4", "af3_swiglu_c64_x2", "af3_swiglu_c384_x4",
                   "af3_swiglu_c768_x2_s5", "joltz_swiglu_c128_x4", "joltz_swiglu_c384_x4"],
    "ln": ["pair_c128", "tmpl_c64", "msa_c256_s512", "msa_c64_s1024", "single_c384", "atom_c128", "dit_c768_s5"],
    "opm": ["af2_opm_cm256_c32_f128_S128", "af2_opm_cm256_c32_f128_S512", "af3_opm_cm64_c32_f128_S1024"],
    "glut": ["af3_trimul_glu_c128_out256"],
}
FORWARD_ONLY = ("fpf_block", "fpf_trimul", "fpf_core", "fpf_transition", "glut", "opm_two_launch", "triattn_xla", "mlp_transition", "native_xla")


def _cells():
    return P.CELLS


def test_the_face_imports_the_standard_library_only_at_top_level():
    for rel, allowed in (("__init__.py", {"json", "os", "re", "typing"}), ("serve.py", {"math", "os", "typing"})):
        tree = ast.parse(open(os.path.join(P.HERE, rel), encoding="utf-8").read())
        tops = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                tops.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                tops.add((node.module or "").split(".")[0])
        assert tops <= allowed, (rel, sorted(tops - allowed))


def test_table_schema_and_cell_keys():
    T = P.TABLE
    assert T["schema"].startswith("pallas_cells/v1")
    for k in ("default_rule", "exact_rule", "tie_rule", "words", "rows", "lines", "families", "bridge", "refusal_evidence", "cells", "tuned", "stacks"):
        assert k in T, k
    assert set(P.OPS) == set(T["families"]), (P.OPS, list(T["families"]))
    key_re = re.compile(r"^(0\.10|0\.6|0\.5)\|(9\.0|8\.0)\|(bf16|f32)\|(attn|triattn|trimul|transition|ln|opm|glut)\|([A-Za-z0-9_]+)\|N<=(\d+)\|(fwd|fwdbwd)$")
    assert len(_cells()) >= 1400
    for k, c in _cells().items():
        m = key_re.match(k)
        assert m, k
        assert m.group(5) in T["families"][m.group(4)], k
        assert c["stack"] in T["stacks"], k
        assert isinstance(c["measured"], bool)
        for field in ("ms", "x_stock", "peak_mib", "cls"):
            assert isinstance(c[field], dict), (k, field)
        for field in ("refused", "unmeasured"):
            assert isinstance(c.get(field, {}), dict), (k, field)
        if c["measured"]:
            assert c["fast_order"][-1] == "xla" and c["fast"] == c["fast_order"][0], k
            for a in list(c["ms"]) + c["fast_order"] + [c["exact"], c["big"]]:
                row = P.parse_arm(a)[0]
                assert row in P.ROW_NAMES, (k, a)
                ip, setting = P.parse_arm(a)[1], P.parse_arm(a)[2]
                assert ip in (None,) + P.F32_WORDS, (k, a)
                assert setting is None or setting in P.SETTING_WORDS, (k, a)
            order = list(c["fast_order"])
            if c.get("tie_rule"):                                              # the default-flip rule kept the incumbent word at the head on a tie:
                tr = c["tie_rule"]                                             # it leads the order although the tie's leader measured (insignificantly) faster
                assert order[0] == tr["kept"] == c["fast"] and tr["leader"] in c["ms"] and tr["kept"] in c["ms"], k
                assert c["ms"][tr["kept"]] / c["ms"][tr["leader"]] < 1.05 or tr.get("iqr_overlap"), k
                order = sorted(order[:-1], key=lambda a: c["ms"][a]) + ["xla"] if c.get("stock_failed") else sorted(order[:-1], key=lambda a: -c["x_stock"][a]) + ["xla"]
            if c.get("stock_failed"):                                          # the stock arm did not run at this size: timed arms ranked by ms
                assert "xla" not in c["ms"] and c["fast"] != "xla", k
                assert order[:-1] == sorted(c["ms"], key=lambda a: c["ms"][a]), k
            else:
                assert "xla" in c["ms"] and c["stock_ms"] == c["ms"]["xla"], k
                for a in order[:-1]:
                    assert c["x_stock"][a] >= 1.0, (k, a)                       # only arms that beat the stock statement are tier candidates
                xs = [c["x_stock"][a] for a in order[:-1]]
                assert xs == sorted(xs, reverse=True), k                        # fastest first
        else:
            assert c["fast"] == c["exact"] == c["big"] == "xla" and c["fast_order"] == ["xla"], k


def test_exact_floor_and_parity_flags():
    """exact = an arm measured bitwise the stock op AND >= x1.00, else the stock statement by name; a named winner inside +-3 % carries a parity flag."""
    n_exact = 0
    for k, c in _cells().items():
        if not c["measured"]:
            continue
        e = c["exact"]
        if e != "xla":
            n_exact += 1
            assert c["cls"][e] == "exact", (k, e)
            assert c.get("stock_failed") or c["x_stock"][e] >= 1.0, (k, e)
        for a, cl in c["cls"].items():
            if cl == "exact" and not c.get("stock_failed") and (c["x_stock"].get(a) or 0) >= 1.0:
                assert c["exact"] != "xla", (k, a)                             # a bitwise arm at or above parity is never passed over for the stock op
        for role, txt in (c.get("parity") or {}).items():
            a = c[role]
            assert a != "xla" and 0.97 <= c["x_stock"][a] <= 1.03 and txt.startswith(a), (k, role)
        for role in ("fast", "exact", "big"):
            a = c[role]
            if a != "xla" and not c.get("stock_failed") and 0.97 <= c["x_stock"][a] <= 1.03:
                assert role in (c.get("parity") or {}), (k, role)
    assert n_exact >= 10                                                        # the GLU prologue and the stock-body OPM rows are bitwise where measured


def test_most_pallas_rows_are_tolerance_class_and_say_so():
    tol_rows, exact_rows = set(), set()
    for c in _cells().values():
        for a, cl in c["cls"].items():
            (exact_rows if cl == "exact" else tol_rows if cl == "tol" else set()).add(P.parse_arm(a)[0])
    assert {"pallas_attn", "cd_triatt", "fpf_block", "fpf_trimul", "cd_trimul", "cd_ln"} <= tol_rows
    assert exact_rows <= {"glut", "opm_stockbody", "tokamax_glu_T", "xla_subbatch4", "xla_sdpa", "cudnn", "tokamax", "tokamax_glu", "rowshared", "opm_reassoc", "cd_opm"} | exact_rows
    for r in ("glut", "opm_stockbody"):
        assert P.ROWS[r]["cls"] == "exact", r
    for r in ("pallas_attn", "cd_triatt", "fpf_block", "fpf_trimul", "cd_trimul", "rowshared", "cd_ln", "cd_transition", "mlp_transition", "fpf_transition"):
        assert P.ROWS[r]["cls"] == "fast", r


def test_rows_block_states_every_row_and_its_fallback():
    for r in P.ROW_NAMES:
        if r.startswith("proj+"):
            assert r.split("+", 1)[1] in P.ROWS, r
            continue
        d = P.ROWS[r]
        for k in ("module", "ops", "cls", "backward", "jax_lines", "cc", "dtypes", "fallback", "refusals"):
            assert k in d, (r, k)
        assert d["fallback"] is None or d["fallback"] in P.ROWS, (r, d["fallback"])
        assert P.has_backward(r) == (r not in FORWARD_ONLY) or d["cls"] == "stock", r
    assert P.ROWS["xla"]["fallback"] is None
    for r in FORWARD_ONLY:
        assert not P.has_backward(r), r


@pytest.mark.parametrize("line", LINES)
@pytest.mark.parametrize("cc", CARDS)
def test_every_target_family_has_a_cell_or_a_named_stock_answer_on_every_stack(line, cc):
    """the table covers every JAX pair / MSA / template sub-layer the release tree runs, per jax line and card: a measured cell (rows timed against
    the stock statement), a cell closed n/a by rule (every arm 'unmeasured' with the rule's words), or -- for a family this line's model forms do
    not run -- the tier words name the stock statement with a note (never a guess, never a refusal)."""
    for op, fams in TARGET_FAMILIES.items():
        for fam in fams:
            sizes = HALLUC_SIZES if fam.startswith("joltz") else SIZES
            for n in sizes:
                for direction in ("fwd", "fwdbwd"):
                    key, cell, note = P.cell_for(line, cc, "bf16", op, fam, n, direction)
                    sel = P.select(line, cc, "bf16", op, fam, n, direction, word="fast")
                    assert sel.candidates[-1] == "xla"
                    if cell is None:
                        assert (sel.arm == "xla" and "no measured cell" in sel.note) or (sel.arm in ("triattn_xla", "proj+triattn_xla", "triattn_xla@vjp", "proj+triattn_xla@vjp") and "bridge" in sel.note), (key, note)
                    elif not cell["measured"]:
                        assert sel.arm == "xla" and (cell.get("unmeasured") or cell.get("refused")), key
                    else:
                        assert sel.arm in cell["fast_order"] or sel.arm in ("triattn_xla", "proj+triattn_xla", "triattn_xla@vjp", "proj+triattn_xla@vjp"), (key, sel.arm)


def test_measured_coverage_per_line_and_card():
    cov = P.coverage()
    assert cov["cells"] == len(_cells()) and cov["measured"] >= 1200
    for line in LINES:
        for cc in CARDS:
            for op in ("attn", "triattn", "trimul", "transition", "ln"):
                n = cov["by_line_cc_op"].get("%s|%s|%s" % (line, cc, op), 0)
                assert n > 0, (line, cc, op)
    # the winners are spread over the carried rows (the table is not a one-row table)
    assert {"rowshared", "cd_triatt", "fpf_block", "cd_trimul", "cd_ln"} <= set(cov["fast_winners"]), cov["fast_winners"]


# ----------------------------------------------------------------------------------------------------------------- the default rule
@pytest.mark.parametrize("word,op,fam", [
    ("fpf_block", "triattn", "af3_pair_c128_h4_d32_starting"), ("fpf_block", "triattn", "af2_tmpl_c64_h4_d16_ending"), ("fpf_trimul", "trimul", "af3_pair_c128_ch128_outgoing"),
    ("fpf_trimul@hi", "trimul", "af2_pair_c128_ch128_incoming"), ("pallas_attn", "attn", "tri_h4_d32"), ("pallas_attn", "attn", "msarow_s512_h8_d32"),
    ("glut", "glut", "af3_trimul_glu_c128_out256"), ("fpf_transition", "transition", "af3_swiglu_c128_x4"), ("cd_triatt", "attn", "tri_h4_d32"),
    ("cd_trimul", "trimul", "af2_pair_c128_ch128_outgoing"), ("cd_ln", "ln", "pair_c128"), ("rowshared", "attn", "msarow_s512_h8_d32"), ("rowshared@r4", "attn", "msarow_s512_h8_d32"),
    ("xla", "attn", "tri_h4_d32"), ("xla_sdpa", "attn", "tri_h4_d32"), ("tokamax@triton", "attn", "tri_h4_d32"), ("opm_stockbody", "opm", "af2_opm_cm256_c32_f128_S512"),
    ("proj+pallas_attn", "triattn", "af2_pair_c128_h4_d32_starting"), ("proj+cd_triatt", "triattn", "af2_pair_c128_h4_d32_ending"),
])
def test_a_row_word_serves_exactly_that_row_whatever_the_cell_measured(word, op, fam):
    """default rule: what a kit binding that module today runs does not change underneath it -- the row word is the switch, the measured
    winner is reached only through a tier word."""
    for line in LINES:
        for cc in CARDS:
            if word.startswith("tokamax") and line != "0.10":
                continue
            sel = P.select(line, cc, "bf16", op, fam, 800, "fwd", word=word)
            assert sel.arm == word and sel.row == P.parse_arm(word)[0] and sel.candidates == [word], (line, cc, sel)
            base = P.parse_arm(word)[2]
            assert sel.config == (P.setting_config(base) if base else {}), (word, sel.config)      # no tuned launch setting rides on a bare row word


def test_row_words_outside_every_measured_cell_still_serve_as_named():
    sel = P.select("0.4.30", "7.5", "bf16", "attn", "tri_h4_d32", 5000, "fwd", word="pallas_attn")
    assert sel.arm == "pallas_attn" and sel.cell is None and "no measured" in sel.note
    sel = P.select("0.10.2", "9.0", "bf16", "attn", "tri_h4_d32", 5000, "fwd", word="fast")
    assert sel.cell_key.endswith("N<=1536|fwd") or sel.cell_key.endswith("N<=1280|fwd") or sel.cell_key.endswith("N<=1200|fwd")          # above the largest measured size: the largest
    sel = P.select("0.7.1", "9.0", "bf16", "attn", "tri_h4_d32", 800, "fwd", word="fast")
    assert sel.arm == "xla" and "no measured cell" in sel.note                                    # an unmeasured jax line: the stock statement, by name


def test_tier_words_name_measured_arms_and_prefer_narrows_in_the_cells_order():
    hits = 0
    for k, c in _cells().items():
        if not c["measured"] or c["fast"] == "xla":
            continue
        line, cc, dt, op, fam, nb, direction = k.split("|")
        n = int(nb[3:])
        if c.get("n_floor") and n < int(c["n_floor"]["below"]):                      # a program floor covers the whole bucket: the stock statement by rule (its own test)
            assert P.select(line, cc, dt, op, fam, n, direction, word="fast").arm == c["n_floor"]["arm"], k
            continue
        sel = P.select(line, cc, dt, op, fam, n, direction, word="fast")
        assert sel.cell_key == k
        first_admitted = [a for a in c["fast_order"] if P.static_refusal(P.parse_arm(a)[0], op=op, fam=fam, cc=cc, dtype=dt, direction=direction, jax_line=line, cell=c, arm=a) is None][0]
        assert sel.arm in (first_admitted,) + ("triattn_xla", "proj+triattn_xla", "triattn_xla@vjp", "proj+triattn_xla@vjp"), (k, sel.arm, first_admitted)
        ex = P.select(line, cc, dt, op, fam, n, direction, word="exact")
        assert ex.arm == c["exact"] or (ex.arm == "xla" and P.static_refusal(P.parse_arm(c["exact"])[0], op=op, fam=fam, cc=cc, dtype=dt, direction=direction, jax_line=line, cell=c, arm=c["exact"]) is not None), k
        rows_beating = [P.parse_arm(a)[0] for a in c["fast_order"][:-1]]
        last = rows_beating[-1]
        pr = P.select(line, cc, dt, op, fam, n, direction, word="fast", prefer=(last,))
        assert P.parse_arm(pr.arm)[0] in (last, "xla"), (k, pr.arm)
        nob = P.select(line, cc, dt, op, fam, n, direction, word="fast", prefer=("no_such_row",))
        assert nob.arm == "xla" and "none of prefer" in nob.note
        hits += 1
    assert hits >= 500


def test_f32_words_pick_arms_at_that_operand_precision():
    sel = P.select("0.5.3", "9.0", "f32", "trimul", "af2_pair_c128_ch128_outgoing", 800, "fwd", word="tf32")
    assert sel.arm == "xla" or P.parse_arm(sel.arm)[1] == "tf32" or P.parse_arm(sel.arm)[0] in P.STOCK_ROWS, sel.arm
    with pytest.raises(P.Refusal) as e:
        P.select("0.5.3", "9.0", "bf16", "trimul", "af2_pair_c128_ch128_outgoing", 800, "fwd", word="tf32")
    assert e.value.kind == P.UNKNOWN_WORD
    with pytest.raises(P.Refusal) as e:
        P.select("0.5.3", "9.0", "bf16", "trimul", "af2_pair_c128_ch128_outgoing", 800, "fwd", word="no_such_row")
    assert e.value.kind == P.UNKNOWN_WORD and "rows" in str(e.value)
    with pytest.raises(P.Refusal) as e:
        P.select("0.5.3", "9.0", "bf16", "trimul", "af2_pair_c128_ch128_outgoing", 800, "fwd", word="pallas_attn")   # a row of another op
    assert e.value.kind == P.UNKNOWN_WORD


# ----------------------------------------------------------------------------------------------------------------- refusals by name
@pytest.mark.parametrize("row,op,fam", [("fpf_block", "triattn", "af3_pair_c128_h4_d32_starting"), ("fpf_trimul", "trimul", "af3_pair_c128_ch128_outgoing"),
                                        ("fpf_core", "attn", "tri_h4_d32"), ("fpf_transition", "transition", "af3_swiglu_c128_x4"), ("glut", "glut", "af3_trimul_glu_c128_out256"),
                                        ("opm_two_launch", "opm", "af2_opm_cm256_c32_f128_S512"), ("triattn_xla", "attn", "tri_h4_d32"), ("mlp_transition", "transition", "af2_relu_c128_x4"),
                                        ("proj+fpf_core", "triattn", "af2_pair_c128_h4_d32_starting"), ("proj+triattn_xla", "triattn", "af2_pair_c128_h4_d32_starting")])
def test_forward_only_rows_refuse_a_differentiated_call_by_name(row, op, fam):
    with pytest.raises(P.Refusal) as e:
        P.select("0.10.2", "9.0", "bf16", op, fam, 400, "fwdbwd", word=row)
    assert e.value.kind == P.NO_BACKWARD and e.value.row in (row, row.split("+")[-1]) and e.value.fallback, (row, e.value)
    assert P.has_backward(e.value.fallback) or e.value.fallback == "xla", (row, e.value.fallback)
    # and a tier word never names them for a differentiated call
    for k, c in _cells().items():
        if k.endswith("|fwdbwd") and c["measured"]:
            line, cc, dt, op_, fam_, nb, direction = k.split("|")
            sel = P.select(line, cc, dt, op_, fam_, int(nb[3:]), direction, word="fast")
            assert P.has_backward(sel.row, P.parse_arm(sel.arm)[2]), (k, sel.arm)


def test_mlp_transition_under_a_gradient_only_at_its_reference_backward_setting():
    sel = P.select("0.10.2", "9.0", "bf16", "transition", "af2_relu_c128_x4", 400, "fwdbwd", word="mlp_transition@bwd_reference")
    assert sel.arm == "mlp_transition@bwd_reference" and sel.config == {"bwd": "reference"}


@pytest.mark.parametrize("word,op,fam", [("cudnn", "attn", "tri_h4_d32"), ("tokamax@cudnn", "attn", "tri_h4_d32"), ("proj+cudnn", "triattn", "af2_pair_c128_h4_d32_starting")])
def test_cudnn_is_refused_for_masked_differentiated_calls_and_serves_unmasked_ones(word, op, fam):
    """measured through the face: with a key mask, cuDNN's VJP returns gradients 40-57x the reference scale from fully-masked query rows (cosine
    0.16-0.18) -> refused by name under fwdbwd when a key mask is given (the default: cells were measured masked); unmasked differentiated calls
    serve (cosine >= 0.9999); forward always serves; tier words never pick a cuDNN arm for a differentiated cell."""
    with pytest.raises(P.Refusal) as e:
        P.select("0.10.2", "9.0", "bf16", op, fam, 400, "fwdbwd", word=word)
    assert e.value.kind == P.CUDNN_SHARED_BIAS_GRAD and e.value.fallback == "xla"
    assert P.select("0.10.2", "9.0", "bf16", op, fam, 400, "fwdbwd", word=word, key_masked=False).arm == word
    assert P.select("0.10.2", "9.0", "bf16", op, fam, 400, "fwd", word=word).arm == word
    for k, c in _cells().items():
        if k.endswith("|fwdbwd") and c["measured"]:
            line, cc, dt, op_, fam_, nb, direction = k.split("|")
            for km in (True, False):
                sel = P.select(line, cc, dt, op_, fam_, int(nb[3:]), direction, word="fast", key_masked=km)
                assert "cudnn" not in sel.arm, (k, sel.arm)


def test_head_dim_8_is_padded_to_16_for_every_row_but_the_stock_statement_and_unpadded_timings_are_never_candidates():
    fam = "extramsa_s1024slab512_h8_d8"
    hz = 0
    for k, c in _cells().items():
        if "|%s|" % fam in k and c["measured"]:
            for a in c.get("hazard_unpadded", {}):
                assert a not in c["fast_order"] or a in c["ms"], k                        # a padded timing may exist under the same arm; the unpadded one is never it
            hz += bool(c.get("hazard_unpadded"))
    assert hz > 0
    for word in ("pallas_attn", "cd_triatt", "rowshared", "rowshared@r4", "tokamax@triton", "cudnn", "fast"):
        for line in LINES:
            for cc in CARDS:
                if word.startswith("tokamax") and line != "0.10":
                    continue
                sel = P.select(line, cc, "bf16", "attn", fam, 400, "fwd", word=word)
                if sel.row != "xla":
                    assert sel.pad_head_dim_to == 16, (word, line, cc, sel)             # every row but the stock statement is served zero-padded to 16
                else:
                    assert sel.pad_head_dim_to is None, sel
    assert P.PAD_HEAD_DIM_TO == 16
    assert P.select("0.10.2", "9.0", "bf16", "attn", "tri_h4_d32", 400, "fwd", word="pallas_attn").pad_head_dim_to is None


@pytest.mark.parametrize("fam", ["af3_single_h16_d24", "af3_dit_s5_h16_d48", "joltz_single_h16_d24"])
def test_head_dims_that_are_not_powers_of_two_are_refused_by_name(fam):
    for word in ("pallas_attn", "cd_triatt", "rowshared"):
        with pytest.raises(P.Refusal) as e:
            P.select("0.10.2", "9.0", "bf16", "attn", fam, 400, "fwd", word=word)
        assert e.value.kind == P.HEAD_DIM_NOT_POW2 and e.value.fallback in ("xla", "xla_sdpa", "cudnn", "tokamax"), (word, e.value)
    sel = P.select("0.10.2", "9.0", "bf16", "attn", fam, 400, "fwd", word="fast")
    assert sel.row in P.STOCK_ROWS, sel


def test_per_window_bias_cells_refuse_the_shared_bias_rows_by_name():
    fam = "af3_atom_win32x128_h4_d32"
    for word in ("pallas_attn", "cd_triatt", "rowshared", "cudnn"):
        with pytest.raises(P.Refusal) as e:
            P.select("0.10.2", "9.0", "bf16", "attn", fam, 400, "fwd", word=word)
        assert e.value.kind == P.BIAS_PER_WINDOW, (word, e.value)


def test_a100_f32_triangle_multiplication_design_kit_row_serves_with_its_measured_tiles():
    """cc 8.0, f32, c_hidden 128: cd_trimul's module-default forward tiles run x0.21 / 0.31 of the stock statement (measured); the row word serves them
    (default rule); the tuned word @t32w8 (x2.9 / 4.2 measured, same process; fpf_trimul:tf32 x3.3 / 5.4) is a cell arm and what tier words attach
    when they land on the row; f32 backward there on the jax 0.5 line fails to launch -> refused by name; bf16 and the c 64 template stack serve."""
    for line in LINES:
        assert P.select(line, "8.0", "f32", "trimul", "af2_tmpl_c64_ch64_outgoing", 800, "fwd", word="cd_trimul").arm == "cd_trimul"
        for fam in ("af2_pair_c128_ch128_outgoing", "af3_pair_c128_ch128_incoming", "joltz_pair_c128_ch128_outgoing"):
            sel = P.select(line, "8.0", "f32", "trimul", fam, 800, "fwd", word="cd_trimul")
            assert sel.arm == "cd_trimul" and not sel.config, (line, fam, sel)                     # the row word: the module's own tiles
            sel = P.select(line, "8.0", "f32", "trimul", fam, 800, "fwd", word="cd_trimul@t32w8")
            assert sel.config == {"t1": 32, "w1": 8, "t3": 32, "w3": 8}
            assert P.select(line, "8.0", "bf16", "trimul", fam, 800, "fwdbwd", word="cd_trimul").arm == "cd_trimul"
            if line == "0.5":
                with pytest.raises(P.Refusal) as e:
                    P.select(line, "8.0", "f32", "trimul", fam, 800, "fwdbwd", word="cd_trimul")
                assert e.value.kind == P.LAUNCH_FAILS_SM80 and e.value.fallback == "fpf_trimul_xlabwd:tf32"
            else:
                assert P.select(line, "8.0", "f32", "trimul", fam, 800, "fwdbwd", word="cd_trimul").arm == "cd_trimul"
                assert P.select(line, "9.0", "f32", "trimul", fam, 800, "fwdbwd", word="cd_trimul").arm == "cd_trimul"
    # the measured cells: the tuned arm enters where this provider timed it, behind fpf_trimul:tf32 (same-process order), ahead of the statement
    c = _cells()["0.5|8.0|f32|trimul|af2_pair_c128_ch128_outgoing|N<=800|fwd"]
    assert c["fast_order"][:2] == ["fpf_trimul:tf32", "cd_trimul@t32w8"] and c["x_stock"]["cd_trimul"] < 1.0 < c["x_stock"]["cd_trimul@t32w8"] < c["x_stock"]["fpf_trimul:tf32"]
    assert P.select("0.5.3", "8.0", "f32", "trimul", "af2_pair_c128_ch128_outgoing", 800, "fwd", word="tf32", prefer=("cd_trimul",)).arm == "cd_trimul@t32w8"
    assert P.select("0.5.3", "8.0", "f32", "trimul", "af2_pair_c128_ch128_outgoing", 800, "fwd", word="tf32").arm == "fpf_trimul:tf32"
    for k, c in _cells().items():
        line, cc, dt, op, fam, nb, direction = k.split("|")
        if op == "trimul" and cc == "8.0" and dt == "f32" and c["measured"] and "_ch128_" in fam:
            assert P.parse_arm(P.select(line, cc, dt, op, fam, int(nb[3:]), direction, word="fast").arm)[0] != "cd_trimul" or "@t32w8" in P.select(line, cc, dt, op, fam, int(nb[3:]), direction, word="fast").arm, k
            x = c["x_stock"].get("cd_trimul")
            assert x is None or x < 1.0, (k, x)                                # the module-default tiles: never at parity on that card in f32 at c 128


def test_the_design_kit_transition_serves_on_sm80_with_the_face_tile_and_refuses_wide_channels_nowhere_else():
    """cc 8.x, C >= 128: the module's 64-row tile fails to launch (measured); the face installs (32 rows, 4 warps) there (bitwise the same output;
    x1.02 / 1.07 forward, x0.88 / 0.86 forward+backward of the statement measured on the jax 0.5 line) -> the row serves by every word; tier
    words keep the measured order (mlp_transition / xla ahead)."""
    for line in LINES:
        for direction in ("fwd", "fwdbwd"):
            assert P.select(line, "8.0", "bf16", "transition", "af2_relu_c128_x4", 400, direction, word="cd_transition").arm == "cd_transition"
        assert P.select(line, "8.0", "bf16", "transition", "af2_relu_c64_x2", 400, "fwd", word="cd_transition").arm == "cd_transition"
        assert P.select(line, "9.0", "bf16", "transition", "af2_relu_c128_x4", 400, "fwdbwd", word="cd_transition").arm == "cd_transition"
    c = _cells()["0.5|8.0|bf16|transition|af2_relu_c128_x4|N<=400|fwd"]
    assert "cd_transition" not in (c.get("refused") or {}) and c["x_stock"]["cd_transition"] >= 1.0 and c["fast"] == "mlp_transition"
    c = _cells()["0.5|8.0|bf16|transition|af2_relu_c128_x4|N<=800|fwdbwd"]
    assert c["x_stock"]["cd_transition"] < 1.0 and c["fast"] == "xla"
    assert S_.SM80_TRANSITION_TILE == (32, 4)



def test_layer_norm_rows_win_only_where_measured_at_or_above_x1_05_else_the_stock_statement_is_named():
    for k, c in _cells().items():
        line, cc, dt, op, fam, nb, direction = k.split("|")
        if op != "ln" or not c["measured"]:
            continue
        sel = P.select(line, cc, dt, op, fam, int(nb[3:]), direction, word="fast")
        if sel.arm != "xla":
            assert c["x_stock"][sel.arm] >= 1.0, k
        x = c["x_stock"].get("cd_ln")
        if x is not None and x < 1.0:
            assert sel.arm == "xla", k


def test_measured_launch_failures_are_refusals_by_name_in_their_cells():
    seen = set()
    for k, c in _cells().items():
        line, cc, dt, op, fam, nb, direction = k.split("|")
        for arm, why in (c.get("refused") or {}).items():
            row = P.parse_arm(arm)[0]
            if row in P.STOCK_ROWS or row in P.NOT_SERVED:
                continue
            with pytest.raises(P.Refusal) as e:
                P.select(line, cc, dt, op, fam, int(nb[3:]), direction, word=arm)
            seen.add(e.value.kind.split(":")[0])
    assert {"no_backward", "launch_fails", "head_dim_not_pow2"} <= seen or {"no_backward", "launch_fails"} <= seen, seen


def test_levers_off_words_switch_rows_off_by_name(monkeypatch):
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "pallas:cd_triatt,unrelated_word")
    assert P.levers_off() == ["cd_triatt"]
    with pytest.raises(P.Refusal) as e:
        P.select("0.6.0", "9.0", "bf16", "attn", "tri_h4_d32", 500, "fwdbwd", word="cd_triatt")
    assert e.value.kind == P.LEVERS_OFF and e.value.fallback == "pallas_attn"
    sel = P.select("0.6.0", "9.0", "bf16", "attn", "tri_h4_d32", 500, "fwdbwd", word="fast")
    assert sel.row != "cd_triatt"
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "pallas")
    for word in ("pallas_attn", "fpf_block"):
        with pytest.raises(P.Refusal):
            P.select("0.10.2", "9.0", "bf16", ("attn" if word == "pallas_attn" else "triattn"), ("tri_h4_d32" if word == "pallas_attn" else "af3_pair_c128_h4_d32_starting"), 400, "fwd", word=word)
    sel = P.select("0.10.2", "9.0", "bf16", "attn", "tri_h4_d32", 400, "fwd", word="fast")
    assert sel.row in P.STOCK_ROWS, sel
    assert P.select("0.10.2", "9.0", "bf16", "attn", "tri_h4_d32", 400, "fwd", word="xla").arm == "xla"


def test_engine_module_arms_are_timed_but_not_served_by_the_face():
    with pytest.raises(P.Refusal) as e:
        P.select("0.5.3", "9.0", "bf16", "triattn", "af2_pair_c128_h4_d32_starting", 800, "fwd", word="xla_subbatch4")
    assert e.value.kind == P.NOT_SERVED_BY_FACE
    for k, c in _cells().items():
        if c["measured"]:
            line, cc, dt, op, fam, nb, direction = k.split("|")
            for w in ("fast", "big"):
                assert P.select(line, cc, dt, op, fam, int(nb[3:]), direction, word=w).row not in P.NOT_SERVED, (k, w)


# ----------------------------------------------------------------------------------------------------------------- the bridge rule
def test_the_bridge_row_is_preferred_for_triangle_attention_forward_where_its_kernels_measured_ahead():
    B = P.BRIDGE["cells"]
    assert B and all(re.match(r"^(0\.10|0\.6|0\.5)\|(9\.0|8\.0)\|(bf16|f32)\|D\d+\|H\d+$", k) for k in B), list(B)[:3]
    ahead = [k for k, b in B.items() if b["ahead_of_pallas"]]
    assert ahead, "the bridge measured ahead of the fused Pallas core somewhere"
    for k, b in B.items():
        line, cc, dt, D, H = k.split("|")
        fam = "%s_h%s_d%s" % ("tri" if D == "D32" else "tmpl", H[1:], D[1:])
        if fam not in P.families("attn"):
            continue
        sel = P.select(line, cc, dt, "attn", fam, 800, "fwd", word="fast")
        if b["ahead_of_pallas"]:
            assert sel.arm == "triattn_xla" and len(sel.candidates) >= 2 and sel.candidates[-1] == "xla", (k, sel)
            for r in b["ahead_of_pallas"]:
                assert b[r]["x_vs_pallas_core"] is None or b[r]["x_vs_pallas_core"] > 1.03, (k, r)
            no_bridge = P.select(line, cc, dt, "attn", fam, 800, "fwd", word="fast", prefer=tuple(r for r in P.ROW_NAMES if "triattn_xla" not in r))
            assert no_bridge.arm != "triattn_xla", (k, no_bridge)                                  # a kit without jax.ffi reaches the Pallas fallback by prefer=
        else:
            assert sel.arm != "triattn_xla", (k, sel)
        with pytest.raises(P.Refusal) as e:                                                          # forward only
            P.select(line, cc, dt, "attn", fam, 800, "fwdbwd", word="triattn_xla")
        assert e.value.kind == P.NO_BACKWARD
    # module level: 'proj+triattn_xla' heads the triangle-attention module's fast order on the same stacks
    some = False
    for k, b in B.items():
        line, cc, dt, D, H = k.split("|")
        if not b["ahead_of_pallas"] or D != "D32" or H != "H4":
            continue
        for fam in ("af2_pair_c128_h4_d32_starting", "af3_pair_c128_h4_d32_ending", "joltz_pair_c128_h4_d32_starting"):
            key, cell, _ = P.cell_for(line, cc, dt, "triattn", fam, 800, "fwd")
            if cell and cell["measured"]:
                assert P.select(line, cc, dt, "triattn", fam, 800, "fwd", word="fast").arm == "proj+triattn_xla", (k, fam)
                some = True
    assert some


def test_the_bridge_differentiable_row_is_reached_by_word_only_and_preferred_where_its_cells_say():
    B = P.BRIDGE["cells"]
    pref = [k for k, b in B.items() if (b.get("fwdbwd") or {}).get("preferred")]
    assert pref and all(k.split("|")[0] in ("0.6", "0.10") for k in pref), pref        # measured on the jax.ffi lines with the Pallas backward importable
    for k, b in B.items():
        fb = b.get("fwdbwd")
        if not fb:
            continue
        line, cc, dt, D, H = k.split("|")
        fam = "tri_h%s_d%s" % (H[1:], D[1:])
        sel = P.select(line, cc, dt, "attn", fam, 500, "fwdbwd", word="fast")
        if fb["preferred"]:
            assert fb["x_vs_xla"] > 1.0 and (fb["x_vs_best_pallas"] is None or fb["x_vs_best_pallas"] >= 0.97), (k, fb)
            assert sel.arm == "triattn_xla@vjp" and sel.config == {"vjp": "auto"} and len(sel.candidates) >= 2, (k, sel)
            assert P.has_backward(P.parse_arm(sel.candidates[1])[0], P.parse_arm(sel.candidates[1])[2]), (k, sel.candidates)   # the named fallback differentiates
        else:
            assert sel.arm != "triattn_xla@vjp", (k, sel)
        with pytest.raises(P.Refusal) as e:                                        # the bare row stays forward-only
            P.select(line, cc, dt, "attn", fam, 500, "fwdbwd", word="triattn_xla")
        assert e.value.kind == P.NO_BACKWARD
        assert P.select(line, cc, dt, "attn", fam, 500, "fwdbwd", word="triattn_xla@vjp").arm == "triattn_xla@vjp"
    for line in ("0.5",):                                                          # no bridge fwd+bwd cell on that line: the Pallas rows serve differentiated calls
        sel = P.select(line, "9.0", "bf16", "attn", "tri_h4_d32", 500, "fwdbwd", word="fast")
        assert sel.row in ("cd_triatt", "pallas_attn", "rowshared"), sel


# ----------------------------------------------------------------------------------------------------------------- tuned launch settings (this provider's own checks)
def test_tuned_settings_are_applied_by_tier_words_only_where_this_provider_measured_them_ahead():
    """table 'tuned': {setting word: {config, ops, measured: {'<line>|<cc>': x over the row's default launch}, ...}}; a tier word attaches the
    config on (line, cc) pairs measured at >= x1.03; the row word keeps the row's own launch (default rule)."""
    for w, ent in P.TUNED.items():
        assert w in P.SETTING_WORDS, w
        assert isinstance(ent.get("config"), dict) and ent["config"] == P.setting_config(w), w
        assert isinstance(ent.get("measured"), dict) and ent.get("ops"), w
        row = P.parse_arm(w)[0]
        for lc, x in ent["measured"].items():
            line, cc = lc.split("|")
            assert line in LINES and cc in CARDS and isinstance(x, (int, float)), (w, lc)
            dt_ = (ent.get("dtypes") or ["bf16"])[0]
            fam = {"trimul": ("af2_pair_c128_ch128_outgoing"), "attn": "tri_h4_d32", "triattn": "af2_pair_c128_h4_d32_starting"}[ent["ops"][0]]
            hit = P.tuned_for(row, line, cc, ent["ops"][0], dt_, fam)
            if x >= 1.03:
                assert hit is not None and P.TUNED[hit]["measured"][lc] >= x - 1e-9, (w, lc, hit)
            # a probe cell: the tier word carries the config, the row word does not
            op = ent["ops"][0]
            sel = P.select(line, cc, dt_, op, fam, 800, "fwd", word=("fast" if dt_ == "bf16" else "tf32"), prefer=(row,))
            if sel.row == row and x >= 1.03:
                assert all(sel.config.get(k) == v for k, v in P.TUNED[hit]["config"].items()), (w, lc, sel)
                assert "tuned:" in sel.note or sel.arm == hit                            # attached by the tier word, or the cell's own arm IS the setting
            rw = P.select(line, cc, dt_, op, fam, 800, "fwd", word=row)
            assert not any(k in rw.config for k in ent["config"]), (w, lc, rw.config)

def test_setting_words_carry_their_launch_config():
    assert P.setting_config("pallas_attn@s3") == {"num_stages": 3}
    assert P.setting_config("pallas_attn@bq64bk32") == {"bq": 64, "bk": 32}
    assert P.setting_config("pallas_attn@bwd_tf32") == {"bwd_f32_precision": "tf32"}
    assert P.setting_config("fpf_trimul@w4") == {"w1": 4, "w2": 4}
    assert P.setting_config("rowshared@r2") == {"rows": 2, "order": "grouped"} and P.setting_config("rowshared@r4") == {"rows": 4, "order": "grouped"}
    sel = P.select("0.5.3", "8.0", "bf16", "attn", "msarow_s512_h8_d32", 800, "fwd", word="pallas_attn@bq64bk32")
    assert sel.config == {"bq": 64, "bk": 32} and sel.row == "pallas_attn"
    sel = P.select("0.6.0", "9.0", "f32", "attn", "tri_h4_d32", 500, "fwdbwd", word="pallas_attn:tf32@bwd_tf32")
    assert sel.input_precision == "tf32" and sel.config == {"bwd_f32_precision": "tf32"}
    # the f32 backward precision word is load-bearing: the module default (ieee backward) never wins a differentiated f32 cell; the tf32 backward arm does where measured
    wins = [k for k, c in _cells().items() if c["measured"] and k.split("|")[2] == "f32" and k.endswith("|fwdbwd") and "@bwd_tf32" in c["fast"]]
    assert wins, "pallas_attn:tf32@bwd_tf32 measured fastest in some differentiated f32 cell"


# ----------------------------------------------------------------------------------------------------------------- META census and carried copies
def test_meta_lists_every_file_and_the_restated_lines():
    files = set(META["files"])
    on_disk = set(os.path.relpath(os.path.join(r, f), P.HERE) for r, ds, fs in os.walk(P.HERE) for f in fs if "__pycache__" not in r and not f.endswith(".pyc"))
    assert files == on_disk
    assert {"PALLAS_CELLS.json", "__init__.py", "serve.py", "cd_trimul/trimul_pallas.py", "cd_layers/layers_ln.py",
            "cd_layers/layers_opm.py", "cd_layers/layers_transition.py", "rowshared_flash_pallas.py", "mlp_transition/mlp_transition_pallas.py", "opm_pallas/opm_pallas.py"} <= files
    for sub in ("cd_trimul", "cd_layers", "mlp_transition", "opm_pallas"):
        assert sub + "/NOTICE" in files, sub
    assert "rowshared_flash_pallas.NOTICE" in files
    assert not os.path.isdir(os.path.join(P.HERE, "cd_triatt"))                    # the triangle-attention pair is the core's kernels/pallas_triatt (imported, not copied twice)
    assert os.path.isfile(os.path.join(os.path.dirname(P.HERE), "pallas_triatt", "triatt_attn.py"))
    assert set(NOT_BYTE_IDENTICAL) == set()
    for rel, why in NOT_BYTE_IDENTICAL.items():
        m = re.match(r"line (\d+) only", why)
        assert m, why
        ln = int(m.group(1))
        line = open(os.path.join(P.HERE, rel), encoding="utf-8").read().split("\n")[ln - 1]
        assert line.lstrip().startswith("from . import "), (rel, line)
    for n in META["runtime_imports"]:
        assert n in kernels.names(), n


def _donors():
    """(carried relative path, donor absolute path) for every kit copy present beside common/ in this checkout: any kit's kernels package
    holding a file of the same name (the design kit's modules; an inference kit's older third-party copy lives under another path)."""
    import glob
    out = []
    for rel in ("cd_trimul/trimul_pallas.py", "cd_layers/layers_ln.py", "cd_layers/layers_opm.py"):
        for donor in sorted(glob.glob(os.path.join(RELEASE_TREE, "*", "opt", "*_opt", "kernels", os.path.basename(rel)))):
            if "from . import provider" in open(donor, encoding="utf-8").read():        # an ADAPTER bound to this provider's row (no kernel source of its own): not a copy
                continue
            out.append((rel, donor))
    return out


@pytest.mark.parametrize("rel,donor", _donors() or [pytest.param(None, None, marks=pytest.mark.skip("no donor kit beside common/"))])
def test_carried_files_are_byte_identical_to_the_kit_copies_but_the_restated_line(rel, donor):
    core = os.path.join(P.HERE, rel)
    if rel in NOT_BYTE_IDENTICAL:
        ln = int(re.match(r"line (\d+) only", NOT_BYTE_IDENTICAL[rel]).group(1))
        a = open(donor, encoding="utf-8").read().split("\n"); b = open(core, encoding="utf-8").read().split("\n")
        assert len(a) == len(b) and [i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y] == [ln], rel
    else:
        assert filecmp.cmp(donor, core, shallow=False), rel


def test_the_rowshared_kernel_reaches_the_carried_pallas_attn_kernel_by_a_package_relative_import():
    src = open(os.path.join(P.HERE, "rowshared_flash_pallas.py"), encoding="utf-8").read()
    assert "from ..pallas_attn import af2_flash_pallas" in src or "from ..pallas_attn.af2_flash_pallas import" in src or "..pallas_attn" in src
    assert os.path.isfile(os.path.join(os.path.dirname(P.HERE), "pallas_attn", "af2_flash_pallas.py"))


def test_a_program_floor_naming_a_row_serves_it_under_fast_and_big_and_leaves_exact_on_the_cells_exact_arm():
    """program_floors.attn_bf16_af2_subbatch128_512: cc 9.0, jax 0.5 line, bf16 triangle attention (tri_h4_d32, tmpl_h4_d16) below 512 tokens the
    Pallas flash row serves under fast / big (an adopting program's end-to-end floor at its 128-row sub-batch chunks: the bridge from 512 keys);
    exact names the cell's exact arm (the stock statement), never the floor's tolerance row; cc 8.0 carries no floor (the bridge heads there)."""
    fl = P.TABLE["program_floors"]["attn_bf16_af2_subbatch128_512"]
    assert (fl["below_tokens"], fl["arm"], fl["cc"], fl["lines"]) == (512, "pallas_attn", ["9.0"], ["0.5"])
    for fam in fl["families"]:
        for n in (199, 400, 511):
            assert P.select("0.5.3", "9.0", "bf16", "attn", fam, n, "fwd", word="fast").arm == "pallas_attn", (fam, n)
            assert P.select("0.5.3", "9.0", "bf16", "attn", fam, n, "fwd", word="big").arm == "pallas_attn", (fam, n)
            assert P.select("0.5.3", "9.0", "bf16", "attn", fam, n, "fwd", word="exact").arm == "xla", (fam, n)
            assert P.parse_arm(P.select("0.5.3", "8.0", "bf16", "attn", fam, n, "fwd", word="fast").arm)[0] == "triattn_xla", (fam, n)
        for n in (512, 800, 1200):
            assert P.parse_arm(P.select("0.5.3", "9.0", "bf16", "attn", fam, n, "fwd", word="fast").arm)[0] == "triattn_xla", (fam, n)
        assert P.select("0.5.3", "9.0", "bf16", "attn", fam, 199, "fwd", word="triattn_xla").arm == "triattn_xla"      # the row word is unaffected (default rule)


def test_program_floors_serve_the_stock_statement_below_the_floor_under_tier_words_only():
    """1.6.1 of the bridge / this table: an adopting program's measured size floor (program_floors) is a cell field: below it tier words name the stock
    statement; at or above it the measured order stands; row words are unaffected."""
    T = P.TABLE
    fl = T["program_floors"]["trimul_f32_design_320"]
    assert fl["below_tokens"] == 320 and fl["arm"] == "xla" and set(fl["lines"]) == {"0.10"}
    for fam in fl["families"]:
        for cc in ("9.0", "8.0"):
            for direction in ("fwd", "fwdbwd"):
                key, cell, _ = P.cell_for("0.10", cc, "f32", "trimul", fam, 200, direction)
                if cell is None or not cell["measured"]:
                    continue
                assert cell.get("n_floor", {}).get("below") == 320, key
                sel = P.select("0.10", cc, "f32", "trimul", fam, 200, direction, word="fast")
                assert sel.arm == "xla" and "program floor" in sel.note, (key, sel.arm)
                sel = P.select("0.10", cc, "f32", "trimul", fam, 319, direction, word="fast")
                assert sel.arm == "xla", key
                sel = P.select("0.10", cc, "f32", "trimul", fam, 320, direction, word="fast")            # the N<=500 cell's own order from the floor up
                k5, c5, _ = P.cell_for("0.10", cc, "f32", "trimul", fam, 320, direction)
                assert sel.arm == (c5["fast"] if c5 and c5["measured"] else "xla"), (k5, sel.arm)
                assert P.select("0.10", cc, "f32", "trimul", fam, 200, direction, word="cd_trimul").arm == "cd_trimul"   # row word: served as named


def test_the_jax_06_line_cc_80_cells_inherit_the_05_line_and_the_template_incoming_family_is_covered():
    cells = P.TABLE["cells"]
    n = 0
    for k, c in cells.items():
        line, cc, dt, op, fam, nb, d = k.split("|")
        if line == "0.6" and cc == "8.0" and c.get("inherited"):
            sib = cells["|".join(["0.5", cc, dt, op, fam, nb, d])]
            assert c["measured"] and c["fast_order"] == sib["fast_order"] and c["ms"] == sib["ms"] and c["measured_on"] == "A100-80GB:jax0.5.3", k
            n += 1
    assert n >= 150
    assert "af3_tmpl_c64_ch64_incoming" in P.TABLE["families"]["trimul"]
    for line in ("0.10", "0.6", "0.5"):
        for cc in ("9.0", "8.0"):
            for N in (400, 800, 1200):
                sel = P.select(line, cc, "bf16", "trimul", "af3_tmpl_c64_ch64_incoming", N, "fwd", word="fast")
                out = P.select(line, cc, "bf16", "trimul", "af3_tmpl_c64_ch64_outgoing", N, "fwd", word="fast")
                assert sel.arm == out.arm and sel.cell.get("inherited", "").startswith(out.cell_key), (line, cc, N, sel.arm, out.arm)


def test_af3_form_triangle_multiplication_on_cc_80_names_the_fused_block_row_by_program_evidence():
    """An adopting structure program measured cd_trimul slower than its fused Pallas pair block INSIDE its program on cc 8.0 (+0.165 / +0.22 s at 832 / 1216
    tokens) although the op-level race here has cd_trimul ahead: for the AF3 forms on that card the tier words pass over cd_trimul (tier_excluded +
    program_evidence), naming fpf_trimul (or its @hi setting) where it beats the statement, else the statement; cd_trimul keeps serving by row word;
    cc 9.0 is unaffected."""
    cells = P.TABLE["cells"]; n = 0
    assert "program_evidence_rule" in P.TABLE
    for k, c in cells.items():
        line, cc, dt, op, fam, nb, direction = k.split("|")
        if op != "trimul" or not fam.startswith("af3_") or not c["measured"]:
            continue
        if cc == "8.0":
            assert c.get("tier_excluded") == ["cd_trimul"] and "cd_trimul" in c.get("program_evidence", {}), k
            assert all(P.parse_arm(a)[0] != "cd_trimul" for a in c["fast_order"]), k
            sel = P.select(line, cc, dt, op, fam, int(nb[3:]), direction, word="fast")
            assert P.parse_arm(sel.arm)[0] in ("fpf_trimul", "xla"), (k, sel.arm)
            assert P.select(line, cc, dt, op, fam, int(nb[3:]), direction, word="cd_trimul").arm == "cd_trimul", k      # the row word: served as named
            n += 1
        elif dt == "bf16" and direction == "fwd":
            assert P.parse_arm(c["fast"])[0] in ("cd_trimul", "native_xla"), k                                               # cc 9.0 stands as measured
    assert n == 48                                                                   # 36 cells + their 12 N<=1280 extensions on cc 8.0


def test_big_is_fasts_order_minus_measured_memory_costs_and_the_bridge_heads_it_for_triangle_attention():
    """big changes a row only for a MEASURED memory cost (peak over the stock statement by more than max(8 MiB, 2 %)): on every measured cell the big word
    serves fast's arm unless that arm carries such a cost; for the triangle-attention families (tri_* / tmpl_* cores, the pair / template module families)
    on both cards big == fast, the bridge arm included where it heads fast."""
    T = P.TABLE; cells = T["cells"]
    assert "big_rule" in T and "max(8 MiB, 2 %)" in T["big_rule"]
    n_tri = 0
    for k, c in cells.items():
        if not c["measured"]:
            continue
        line, cc, dt, op, fam, nb, direction = k.split("|")
        n = int(nb[3:])
        if c.get("n_floor") and n < int(c["n_floor"]["below"]):
            continue
        costly = P.memory_cost_arms(c)
        want = ([a for a in c["fast_order"] if a not in costly] or ["xla"])[0]
        assert c["big"] == want, (k, c["big"], want)
        f = P.select(line, cc, dt, op, fam, n, direction, word="fast"); b = P.select(line, cc, dt, op, fam, n, direction, word="big")
        if f.arm not in costly:
            assert b.arm == f.arm, (k, f.arm, b.arm)
        if op in ("attn", "triattn") and re.match(r"^(tri_|tmpl_|af3_pair_c128_h4|af2_pair_c128_h4|af3_tmpl_c64_h|af2_tmpl_c64_h|joltz_pair_c128_h4|joltz_tmpl)", fam):
            if f.arm in costly:                                                     # fast's arm itself carries a measured cost: big names the next arm of the order
                assert b.arm != f.arm and b.arm not in costly, (k, f.arm, b.arm)
            else:
                assert b.arm == f.arm, (k, f.arm, b.arm)
            n_tri += 1
    assert n_tri >= 150


def test_af3_family_cells_extend_to_the_1280_bucket():
    cells = P.TABLE["cells"]
    ext = [k for k, c in cells.items() if k.split("|")[5] == "N<=1280" and c.get("extended_from")]
    assert len(ext) >= 60
    for k in ext:
        src = cells[cells[k]["extended_from"]]
        assert cells[k]["fast_order"] == src["fast_order"] and cells[k]["pending"], k
    for line in ("0.10", "0.5"):
        for cc in ("9.0", "8.0"):
            for op, fam in (("trimul", "af3_pair_c128_ch128_outgoing"), ("trimul", "af3_pair_c128_ch128_incoming"), ("attn", "tri_h4_d32")):
                key, cell, note = P.cell_for(line, cc, "bf16", op, fam, 1280, "fwd")
                assert key.split("|")[5] == "N<=1280" and cell is not None, (line, cc, op, fam, key)
                assert P.select(line, cc, "bf16", op, fam, 1216, "fwd", word="fast").arm == P.select(line, cc, "bf16", op, fam, 1200, "fwd", word="fast").arm


def test_an_unmeasured_cc_inherits_the_nearest_measured_columns_portable_rows_under_tier_words():
    """cc 10.3 / 12.0 (no measured column) -> the 9.0 column's source-compiled Pallas / Triton rows for fast / big with the census token; 8.6 -> the 8.0
    column's; never the prebuilt per-arch bridge rows (native_xla, triattn_xla), never a library op while a Triton row remains, never the statement while an
    arch-portable row beats it in the inherited cell; exact = the statement."""
    T = P.TABLE; cells = T["cells"]
    assert "unmeasured_cc_rule" in T and P.inherit_cc("10.3") == "9.0" and P.inherit_cc("12.0") == "9.0" and P.inherit_cc("8.6") == "8.0"
    assert P.inherit_cc("9.0") is None and P.inherit_cc("8.0") is None and P.inherit_cc("7.5") is None
    checked = 0
    for k, c in cells.items():
        if not c["measured"]:
            continue
        line, cc, dt, op, fam, nb, direction = k.split("|")
        n = int(nb[3:])
        if c.get("n_floor") and n < int(c["n_floor"]["below"]):
            continue
        for new_cc in (("10.3", "12.0") if cc == "9.0" else ("8.6",)):
            for w in ("fast", "big"):
                sel = P.select(line, new_cc, dt, op, fam, n, direction, word=w)
                assert sel.cc == new_cc and sel.cell_key == k, (k, new_cc, sel.cell_key)
                assert P.INHERITED_CC_TOKEN in sel.note and "(from %s)" % cc in sel.note, (k, new_cc, sel.note)
                assert P.portable_arm(sel.arm), (k, new_cc, w, sel.arm)                       # never native_xla / triattn_xla on another arch
                base = c["fast_order"] if w == "fast" else ([a for a in c["fast_order"] if a not in P.memory_cost_arms(c)] or ["xla"])
                portable = [a for a in base if P.portable_arm(a) and P.static_refusal(P.parse_arm(a)[0], op=op, fam=fam, cc=new_cc, dtype=dt, direction=direction, jax_line=line, cell=c, arm=a) is None]
                is_triton = lambda a: a != "xla" and not P._library_arm(a) and (P.parse_arm(a)[0] not in P.STOCK_ROWS or a.startswith("tokamax@triton"))  # noqa: E731
                triton = [a for a in portable if is_triton(a)]
                if triton:
                    assert not P._library_arm(sel.arm), (k, new_cc, w, sel.arm)                # never a library op while a Triton row remains
                    expected = [a for a in portable if not P._library_arm(a)][0]
                    assert sel.arm == expected, (k, new_cc, w, sel.arm, expected)             # the source column's resolution, minus non-portable / library arms
                    if is_triton(base[0]) and P.portable_arm(base[0]):
                        assert sel.arm == base[0], (k, new_cc, w, sel.arm, base[0])           # a real kernel at the source column -> the same kernel here
                else:
                    assert sel.arm in portable or sel.arm == "xla", (k, new_cc, w, sel.arm)
            assert P.select(line, new_cc, dt, op, fam, n, direction, word="exact").arm == "xla", (k, new_cc)
            checked += 1
    assert checked >= 1000


def test_measured_columns_resolve_exactly_as_before_the_unmeasured_cc_rule():
    """The 9.0 / 8.0 resolutions are untouched by the inheritance glue: every measured cell's tier words serve what the cell (and the bridge / big /
    floor rules) say, with no inheritance token."""
    cells = P.TABLE["cells"]
    for k, c in cells.items():
        if not c["measured"]:
            continue
        line, cc, dt, op, fam, nb, direction = k.split("|")
        n = int(nb[3:])
        for w in ("fast", "exact", "big"):
            sel = P.select(line, cc, dt, op, fam, n, direction, word=w)
            assert P.INHERITED_CC_TOKEN not in str(sel.note), (k, w)
            assert sel.cell_key == k or (c.get("n_floor") and n < int(c["n_floor"]["below"])), (k, w, sel.cell_key)


def test_cc_10_3_with_ONE_injected_cell_still_inherits_the_portable_row_for_every_OTHER_key():
    """Inheritance is decided PER KEY, never per cc: a 10.3 part that owns exactly one measured cell (one family / dtype / direction / line) uses it
    for that key and STILL inherits the 9.0 column's portable rows -- with the census token -- for every other key (no key falls to the statement by
    name because the cc owns a cell elsewhere).  Same for an 8.9 part with one injected cell vs the 8.0 column."""
    import copy
    T = P.TABLE; cells = T["cells"]
    src9 = "0.10|9.0|bf16|trimul|af2_pair_c128_ch128_outgoing|N<=800|fwd"; src8 = "0.10|8.0|bf16|ln|pair_c128|N<=800|fwd"
    assert src9 in cells and src8 in cells
    inj9 = src9.replace("|9.0|", "|10.3|"); inj8 = src8.replace("|8.0|", "|8.9|")
    saved_stacks = copy.deepcopy(T["stacks"])
    try:
        for inj, src, stack in ((inj9, src9, "B300:injected"), (inj8, src8, "L40S:injected")):
            c = copy.deepcopy(cells[src]); c["stack"] = stack
            c["fast_order"] = [a for a in c["fast_order"] if P.portable_arm(a)] or ["xla"]; c["fast"] = c["fast_order"][0]; c["big"] = c["fast"]
            cells[inj] = c
            T["stacks"][stack] = {"cc": inj.split("|")[1], "card": stack.split(":")[0]}
        P._CELL_COLS = None
        assert "10.3" in P.measured_ccs() and "8.9" in P.measured_ccs()
        assert P.inherit_columns("10.3") == ["9.0"] and P.inherit_columns("8.9") == ["8.0"] and P.inherit_columns("12.0")[:2] == ["10.3", "9.0"]
        # the injected key: its own cell, no token
        sel = P.select("0.10", "10.3", "bf16", "trimul", "af2_pair_c128_ch128_outgoing", 800, "fwd", word="fast")
        assert sel.cell_key == inj9 and P.INHERITED_CC_TOKEN not in sel.note, sel
        sel = P.select("0.10", "8.9", "bf16", "ln", "pair_c128", 800, "fwd", word="fast")
        assert sel.cell_key == inj8 and P.INHERITED_CC_TOKEN not in sel.note, sel
        # every OTHER measured key of the source columns: still inherited (portable rows, token, partial_cc), never the statement by name for lack of a cell
        n = 0
        for k, c in list(cells.items()):
            line, cc, dt, op, fam, nb, direction = k.split("|")
            if not c["measured"] or cc not in ("9.0", "8.0") or k in (src9, src8):
                continue
            ntok = int(nb[3:])
            if c.get("n_floor") and ntok < int(c["n_floor"]["below"]):
                continue
            new_cc = "10.3" if cc == "9.0" else "8.9"
            if (line, dt, op, fam, direction) in (tuple(src9.split("|")[i] for i in (0, 2, 3, 4, 6)), tuple(src8.split("|")[i] for i in (0, 2, 3, 4, 6))):
                continue                                                          # other sizes of the injected key resolve inside the injected cell's family grid
            for w in ("fast", "big"):
                sel = P.select(line, new_cc, dt, op, fam, ntok, direction, word=w)
                assert sel.cell_key == k and P.INHERITED_CC_TOKEN in sel.note and "(from %s)" % cc in sel.note and "partial_cc" in sel.note, (k, new_cc, w, sel.cell_key, sel.note[:120])
                assert P.portable_arm(sel.arm), (k, new_cc, w, sel.arm)
                base = c["fast_order"] if w == "fast" else ([a for a in c["fast_order"] if a not in P.memory_cost_arms(c)] or ["xla"])
                if base[0] != "xla" and P.portable_arm(base[0]) and not P._library_arm(base[0]) and P.static_refusal(P.parse_arm(base[0])[0], op=op, fam=fam, cc=new_cc, dtype=dt, direction=direction, jax_line=line, cell=c, arm=base[0]) is None:
                    assert sel.arm == base[0], (k, new_cc, w, sel.arm, base[0])   # a real kernel at the source column -> the same kernel, cell or no cell elsewhere on this cc
            n += 1
        assert n >= 500
    finally:
        cells.pop(inj9, None); cells.pop(inj8, None); T["stacks"].clear(); T["stacks"].update(saved_stacks); P._CELL_COLS = None
    assert "10.3" not in P.measured_ccs()


def test_the_reference_backward_of_mlp_transition_in_bf16_is_refused_by_name_on_the_jax_06_line_and_cd_transition_serves_bf16_only():
    """Measured wrong-gradient class (refusal_evidence): mlp_transition@bwd_reference, bf16, differentiated, jax 0.6 -> refused by name on both cards
    (the row word too); f32, the forward, and the jax 0.5 / 0.10 lines are served.  cd_transition: bf16 activations only.  No tier cell names the refused
    arm as a bf16 fwd+bwd candidate, so no 9.0 / 8.0 resolution moves."""
    for cc in ("9.0", "8.0"):
        r = P.static_refusal("mlp_transition", op="transition", fam="af2_relu_c64_x2", cc=cc, dtype="bf16", direction="fwdbwd", jax_line="0.6", cell=None, arm="mlp_transition@bwd_reference")
        assert r is not None and r.kind == P.WRONG_GRADIENT and r.fallback == "cd_transition", (cc, r)
        assert P.static_refusal("mlp_transition", op="transition", fam="af2_relu_c64_x2", cc=cc, dtype="f32", direction="fwdbwd", jax_line="0.6", cell=None, arm="mlp_transition:tf32@bwd_reference") is None
        assert P.static_refusal("mlp_transition", op="transition", fam="af2_relu_c64_x2", cc=cc, dtype="bf16", direction="fwd", jax_line="0.6", cell=None, arm="mlp_transition@bwd_reference") is None
        for line in ("0.5", "0.10"):
            assert P.static_refusal("mlp_transition", op="transition", fam="af2_relu_c64_x2", cc=cc, dtype="bf16", direction="fwdbwd", jax_line=line, cell=None, arm="mlp_transition@bwd_reference") is None
        r = P.static_refusal("cd_transition", op="transition", fam="af2_relu_c64_x2", cc=cc, dtype="f32", direction="fwd", jax_line="0.6", cell=None, arm="cd_transition")
        assert r is not None and r.kind == P.DTYPE_NOT_SERVED
        assert P.static_refusal("cd_transition", op="transition", fam="af2_relu_c64_x2", cc=cc, dtype="bf16", direction="fwdbwd", jax_line="0.6", cell=None, arm="cd_transition") is None
    cells = P.TABLE["cells"]
    assert not [k for k, c in cells.items() if c["measured"] and k.split("|")[2] == "bf16" and k.split("|")[6] == "fwdbwd"
                and any(a.startswith("mlp_transition") and "bwd_reference" in a for a in c["fast_order"][:-1])]
    assert "mlp_transition" in P.TABLE["refusal_evidence"]


def test_an_inherited_row_whose_launch_raises_is_stepped_aside_by_name_once_and_never_retried():
    """unmeasured-cc launch guard: on an INHERITED selection a candidate whose build / launch raises (RuntimeError here; anything but an out-of-memory)
    is marked dead for the process, the walk serves the next candidate / the statement, the census carries the error token, and a second call does not
    retry the dead row; on a MEASURED column the same raw error propagates (unchanged); an out-of-memory is re-raised."""
    from opt_core.kernels.pallas import serve as S
    sel = P.select("0.10", "10.3", "bf16", "trimul", "af2_pair_c128_ch128_outgoing", 800, "fwd", word="fast")
    assert P.INHERITED_CC_TOKEN in sel.note and len(sel.candidates) >= 2 and sel.candidates[-1] == "xla"
    bad = sel.candidates[0]
    calls = []
    def one(arm, row, ip, setting):
        calls.append(arm)
        if arm == bad:
            raise RuntimeError("ptxas: sm_103 not supported by this tile")
        return "served:" + arm
    S.DEAD_ARMS.clear(); S.PROBED_ARMS.clear()
    out = S._walk(sel, "triangle_multiplication", one, strict=False)
    assert out == "served:" + sel.candidates[1] and calls == [bad, sel.candidates[1]]
    assert S.DEAD_ARMS[("triangle_multiplication", bad, "10.3")] == "error:RuntimeError"
    assert S.LAST_REFUSAL and "RuntimeError" in S.LAST_REFUSAL["kind"] or S.COUNTS.get("refused:%s:error:RuntimeError" % P.parse_arm(bad)[0])
    calls.clear()
    out2 = S._walk(sel, "triangle_multiplication", one, strict=False)                       # second call: the dead row is not retried
    assert out2 == out and calls == [sel.candidates[1]]
    # the probe hook: a probe that raises marks the arm dead before the real call
    S.DEAD_ARMS.clear(); S.PROBED_ARMS.clear(); calls.clear()
    def probe(arm):
        raise RuntimeError("probe launch failed")
    good = lambda arm, row, ip, setting: (calls.append(arm), "served:" + arm)[1]  # noqa: E731
    try:
        import jax  # noqa: F401
        have_jax = True
    except ImportError:
        have_jax = False
    if have_jax:
        out3 = S._walk(sel, "triangle_multiplication", good, strict=False, probe=probe)
        assert out3 == "served:xla" and ("triangle_multiplication", bad, "10.3") in S.DEAD_ARMS
    # an out-of-memory is re-raised, not converted
    S.DEAD_ARMS.clear(); S.PROBED_ARMS.clear()
    def oom(arm, row, ip, setting):
        raise MemoryError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate")
    with pytest.raises(MemoryError):
        S._walk(sel, "triangle_multiplication", oom, strict=False)
    # a MEASURED column: the raw error propagates unchanged
    sel9 = P.select("0.10", "9.0", "bf16", "trimul", "af2_pair_c128_ch128_outgoing", 800, "fwd", word="fast")
    assert P.INHERITED_CC_TOKEN not in str(sel9.note)
    with pytest.raises(RuntimeError):
        S._walk(sel9, "triangle_multiplication", one if sel9.candidates[0] == bad else (lambda arm, row, ip, setting: (_ for _ in ()).throw(RuntimeError("x"))), strict=False)
    S.DEAD_ARMS.clear(); S.PROBED_ARMS.clear()

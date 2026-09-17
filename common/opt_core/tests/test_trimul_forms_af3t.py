"""kernels.trimul: form-keyed exact cells (forms / form_cells: select(word='exact', form=...)) and row af3t_form -- pure checks (no GPU)."""
import ast
import json
import os

import pytest

from opt_core.kernels import trimul as T

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opt_core", "kernels", "trimul")
AF3T_STACK = "H100:2.13.0+cu130/3.7.1/nocueq"
OF3_STACK = "H100:2.10.0+cu128/3.6.0/cueq0.10.0"


def test_forms_table_and_constants_agree():
    t = T.table()
    assert set(T.FORMS) == set(t["forms"]) == {"of3_module", "af3t_module"}
    for form, row in T.FORMS.items():
        assert t["forms"][form]["row"] == row and row in T.MODULE_EXACT_ROWS and row in T.EXACT_ROWS and row in T.ROW_NAMES
    assert "form_rule" in t and "form_cells" in t
    for key, cell in t["form_cells"].items():                                # '<cell key>+<form>': never a key of "cells"; names its form and row; per-stack class bitwise
        base, _, form = key.partition("+")
        assert form in T.FORMS and cell["form"] == form and cell["exact"] == T.FORMS[form]
        assert key not in t["cells"] and base.count("|") == 6 and base.endswith("|fwd")
        assert cell["ref_stack"] in cell["stacks"]
        for st, rec in cell["stacks"].items():
            assert rec["class"].startswith("bitwise")
            if "x_module" in rec:                                            # capture-replay records: timed per pair, the row ahead of the module
                assert rec["x_module"] > 1.0
            else:                                                            # vouch-only records (byte pass + dense sweep at sizes above the captures, or another card): the vouch itself
                assert (rec.get("vouched") or {}).get("N") or (rec.get("vouched") or {}).get("N_class"), (key, st)
    of3 = [k for k in t["form_cells"] if k.endswith("+of3_module")]; af3 = [k for k in t["form_cells"] if k.endswith("+af3t_module")]
    assert len(of3) == 36 and len(af3) >= 18 and len(af3) % 6 == 0          # of3: bf16/tf32/f32z_bf16 x c64/c128 x 3 sizes x out/in; af3t: (bf16 c64, bf16 c128, f32z_bf16 c128) x sizes x out/in per card


def test_no_form_resolution_is_unchanged_and_form_cells_stay_out_of_it():
    for key in T.table()["cells"]:
        assert "+" not in key
    fam = T.cell_family("9.0", "bf16", 64, 64, "out", "fwd")
    assert all("+" not in k for k in fam.values())
    assert T.select((9, 0), "bf16", 128, 128, 400, "outgoing", word="exact").row in ("tmk3_exact", "cueq", "tx_sm90a_exact", "native_exact")   # any plain exact-class row / the stock op, never a module-exact row
    assert T.select((9, 0), "bf16", 64, 64, 400, "outgoing", word="exact").row not in T.MODULE_EXACT_ROWS
    for w in ("fast", "big"):                                              # form keys only the exact tier
        assert T.select((9, 0), "bf16", 128, 128, 400, "outgoing", word=w, form="af3t_module").row == T.select((9, 0), "bf16", 128, 128, 400, "outgoing", word=w).row


@pytest.mark.parametrize("form,row,stack,shapes", [
    ("of3_module", "of3_form", OF3_STACK, [("bf16", 64), ("bf16", 128), ("tf32", 64), ("tf32", 128), ("f32z_bf16", 64), ("f32z_bf16", 128)]),
    ("af3t_module", "af3t_form", AF3T_STACK, [("bf16", 64), ("bf16", 128), ("f32z_bf16", 128)]),
])
def test_exact_under_a_form_resolves_to_the_module_row_where_proven(form, row, stack, shapes):
    """VOUCH TABLE with CLASSES (CORE >= 0.5.78.0): served at the listed capture sizes (400 / 800 / 1200) AND, on the densely swept (stack, precision,
    width) classes, at every token count the class admits WITH THE LAYOUT THE CLASS NAMES: of3_module bf16 -> N % 16 == 0: the cell's fast layout,
    other N in [16, 1200]: `replica`; af3t_module (all three classes) -> N % 8 == 0: hm_fused, other N in [2, 1200]: `upstream`.  Unswept
    precisions (of3_module tf32 / f32z_bf16) keep the list only; every other stack is refused by name."""
    swept = {("of3_module", "bf16"), ("af3t_module", "bf16"), ("af3t_module", "f32z_bf16")}
    alln = {"of3_module": "replica", "af3t_module": "upstream"}
    for prec, c in shapes:
        for d in ("outgoing", "incoming"):
            for n in (400, 800, 1200):
                s = T.select((9, 0), prec, c, c, n, d, word="exact", form=form, stack=stack)
                assert s.row == row and s.word == "exact" and s.cell.endswith("+" + form) and s.cls.startswith("bitwise") and s.stack == stack and s.size_measured is True
                assert "vouched on %s" % stack in s.reason and s.config in ("hm_fused", "hm_view")          # aligned sizes: the fast layout, explicit per cell
                s2 = T.select((9, 0), prec, c, c, n, d, word="exact+" + form, stack=stack)   # the one-string spelling (bindings that pass words through)
                assert s2 == s
            for n in [x for x in (17, 61, 199, 399, 401, 600, 995, 1004) if x % (16 if form == "of3_module" else 8)]:   # token counts outside the aligned class
                if (form, prec) in swept:
                    s = T.select((9, 0), prec, c, c, n, d, word="exact", form=form, stack=stack)
                    assert s.row == row and s.config == alln[form] and "N class mod 1" in s.reason, (form, prec, c, n, d, s.config, s.reason)
                else:
                    with pytest.raises(T.Refusal) as e:
                        T.select((9, 0), prec, c, c, n, d, word="exact", form=form, stack=stack)
                    assert e.value.kind.startswith("form_vouch_not_recorded(shape=") and ("N=%d" % n) in e.value.kind and e.value.row == row
            for n in ((1201, 1536, 4096) if form == "of3_module" else (2049, 2304, 4096)):   # above every class and list: refused everywhere (af3t: vouched to 2048 from the C3b pass)
                with pytest.raises(T.Refusal) as e:
                    T.select((9, 0), prec, c, c, n, d, word="exact", form=form, stack=stack)
                assert e.value.kind.startswith("form_vouch_not_recorded(shape=")
            if form == "af3t_module":                                                      # C3b: every N to 2048 on this stack (1537 unaligned -> upstream; 1536 aligned -> hm_fused)
                assert T.select((9, 0), prec, c, c, 1537, d, word="exact", form=form, stack=stack).config == "upstream"
                assert T.select((9, 0), prec, c, c, 1536, d, word="exact", form=form, stack=stack).config == "hm_fused"
            if (form, prec) in swept:                                                      # aligned classes below the first bucket
                s = T.select((9, 0), prec, c, c, 64, d, word="exact", form=form, stack=stack)
                assert s.config in ("hm_fused", "hm_view") and "N class mod" in s.reason
    lo = 17 if form == "of3_module" else 2                                                 # the every-N class floor (the sweeps' smallest unaligned size)
    assert T.select((9, 0), "bf16", 64, 64, lo, "outgoing", word="exact", form=form, stack=stack).config == alln[form]
    with pytest.raises(T.Refusal):
        T.select((9, 0), "bf16", 64, 64, 15 if form == "of3_module" else 1, "outgoing", word="exact", form=form, stack=stack)
    for st in (None, "A100:2.9.0+cu128/3.5.0/cueq0.10.0", (OF3_STACK if stack == AF3T_STACK else AF3T_STACK)):   # NOT vouched: any other stack, or no stack word
        with pytest.raises(T.Refusal) as e:
            T.select((9, 0), "bf16", 128, 128, 400, "outgoing", word="exact", form=form, stack=st)
        assert "stack not vouched" in e.value.kind
    with pytest.raises(T.Refusal) as e:                                                      # the row word by name obeys the same table
        T.select((9, 0), "bf16", 64, 64, 1201 if form == "of3_module" else 2049, "incoming", word=row, stack=stack)
    assert e.value.kind.startswith("form_vouch_not_recorded")
    assert T.select((9, 0), "bf16", 64, 64, 400, "incoming", word=row, stack=stack).row == row
    assert T.select((9, 0), "bf16", 64, 64, 61, "incoming", word=row, stack=stack).config == alln[form]     # by name: the class's layout too


def test_vouch_table_records_exactly_the_measured_capture_sizes():
    t = T.table()
    assert "form_vouch_rule" in t and "form_vouch_not_recorded" in t["form_vouch_rule"] and "replica" in t["form_vouch_rule"]
    n_cls = 0
    for k, fc in t["form_cells"].items():
        bucket = int(k.split("|N<=")[1].split("|")[0])
        assert fc.get("config") in ("hm_fused", "hm_view"), (k, fc.get("config"))            # the fast layout, explicit on every form cell
        for st, rec in fc["stacks"].items():
            v = rec["vouched"]
            assert bucket in v["N"] or bucket == 2048, (k, st, v["N"])                       # the capture size of the cell (+ byte-pass sizes inside the bucket; the N<=2048 cells are byte/sweep-vouched)
            assert set(v["N"]) <= {400, 448, 800, 832, 1200, 1216, 1536, 2048}, (k, st, v["N"])
            assert T.vouched(bucket, rec) is not None
            for c in v.get("N_class", []):                                                   # classes only from dense sweeps, each with its evidence
                n_cls += 1
                assert {"mod", "rem", "min", "max", "measured"} <= set(c) and "bitwise" in c["measured"] and "x 2 directions" in c["measured"]
                assert (c["mod"] in (8, 16) and "config" not in c) or (c["mod"] == 1 and c["config"] in ("replica", "upstream"))
    assert n_cls >= 2 * 30                                                                   # 12 of3_module bf16 records + 18 af3t records, two classes each (+ the vouch-only cells' classes)
    rec = {"vouched": {"N": [400], "N_class": [{"mod": 16, "rem": 0, "min": 64, "max": 1216, "except": [512]}, {"mod": 1, "rem": 0, "min": 16, "max": 1200, "config": "replica"}]}}
    assert T.vouched(400, rec) == {"N": 400, "config": None} and T.vouched(64, rec)["mod"] == 16 and T.vouched(1216, rec)["mod"] == 16
    assert T.vouched(512, rec)["config"] == "replica" and T.vouched(61, rec)["config"] == "replica" and T.vouched(1232, rec) is None and T.vouched(8, rec) is None
    assert T.vouched(48, {"vouched": {"N_class": {"mod": 16, "rem": 0, "min": 64, "max": 128}}}) is None          # a single dict class still reads


def test_prove_mode_selection_and_its_refusal_without_a_module():
    """`exact+<form>:prove`: outside the vouch table a Selection in PROVE mode (proven per call shape against module= at serve time); inside the table
    the ordinary vouched Selection; serving without module= is refused by name."""
    s = T.select((9, 0), "bf16", 64, 64, 1300, "outgoing", word="exact+of3_module:prove", stack=OF3_STACK)
    assert s.row == "of3_form" and s.cell.endswith("+of3_module:prove") and s.reason.startswith("prove per call shape") and s.size_measured is False
    s = T.select((9, 0), "bf16", 64, 64, 1300, "outgoing", word="exact", form="of3_module:prove", stack=OF3_STACK)
    assert s.cell.endswith("+of3_module:prove")
    s = T.select((9, 0), "bf16", 64, 64, 400, "outgoing", word="exact+of3_module:prove", stack=OF3_STACK)
    assert s.cell.endswith("+of3_module") and s.size_measured is True                       # vouched: no proof needed
    torch = pytest.importorskip("torch")
    w = {k: torch.zeros((64, 64)) if k.startswith("w_") else torch.zeros(64) for k in T.WEIGHT_KEYS}
    sel = T.select((9, 0), "bf16", 64, 64, 8, "outgoing", word="exact+of3_module:prove", stack=OF3_STACK)
    with pytest.raises(T.Refusal) as e:
        T.triangle_multiplication(torch.zeros((1, 8, 8, 64), dtype=torch.bfloat16), None, direction="outgoing", weights=w, selection=sel, cache={})
    assert "prove_needs_module" in e.value.kind


def test_exact_under_a_form_refuses_by_name_where_unproven():
    for form in T.FORMS:
        for q in [("bf16", 256, 256, 400), ("bf16", 128, 128, 1201), ("fp32", 128, 128, 400), ("bf16", 64, 128, 400)]:
            with pytest.raises(T.Refusal) as e:
                T.select((9, 0), q[0], q[1], q[2], q[3], "outgoing", word="exact", form=form)
            assert (e.value.kind.startswith("form_vouch_not_recorded(") or "c_hidden=" in e.value.kind) and e.value.fallback in ("cueq", "torch_math")   # admission refusals (width) precede the vouch table
    with pytest.raises(T.Refusal):
        T.select((9, 0), "tf32", 64, 64, 400, "outgoing", word="exact", form="af3t_module")     # measured for of3_module only
    with pytest.raises(ValueError):
        T.select((9, 0), "bf16", 64, 64, 400, "outgoing", word="exact", form="nonsense")
    with pytest.raises(ValueError):
        T.select((9, 0), "bf16", 64, 64, 400, "outgoing", word="exact+nonsense")
    with pytest.raises(T.Refusal):
        T.select((7, 5), "bf16", 64, 64, 400, "outgoing", word="exact", form="af3t_module")      # no 7.5 form cell


def test_af3t_form_registration_admission_and_rows_entry():
    assert "af3t_form" in T.ROW_NAMES and "af3t_form" in T.EXACT_ROWS and "af3t_form" in T.MODULE_EXACT_ROWS
    t = T.table(); r = t["rows"]["af3t_form"]
    assert r["class"] == "exact" and r["backward"] is False and r["fallback"] == "torch_math" and "MODULE" in r["exact_vs"]
    assert "af3t_form" in t["tiers"]["exact"] and AF3T_STACK in t["stacks"]
    assert r["configs"]["default"] == "hm_fused" and set(r["configs"]["words"]) == {"upstream", "hm_view", "hm_copy", "hm_fused"}
    assert r["numerics"]["vs_module"].startswith("bitwise: 36/36")
    T.admits("af3t_form", (9, 0), "bf16", 128, 128, 400)
    T.admits("af3t_form", (8, 0), "f32z_bf16", 64, 64, 3000)
    for args, kind in [(((9, 0), "bf16", 64, 128, 400), "c_hidden=128!=c_z=64"), (((7, 5), "bf16", 128, 128, 400), "cc:"), (((9, 0), "fp16", 128, 128, 400), "dtype")]:
        with pytest.raises(T.Refusal) as e:
            T.admits("af3t_form", *args)
        assert kind in str(e.value)
    with pytest.raises(T.Refusal):
        T.admits("af3t_form", (9, 0), "bf16", 128, 128, 400, backward=True)
    s = T.select((9, 0), "bf16", 128, 128, 800, "incoming", word="af3t_form", stack=AF3T_STACK)
    assert s.row == "af3t_form" and s.word == "af3t_form"


def test_af3t_in_form_records():
    t = T.table(); seen = 0
    for C, prec, form in ((64, "bf16", "af3t_template"), (128, "bf16", "af3t_pairformer"), (128, "f32z_bf16", "af3t_confidence")):
        for nb in (400, 800, 1200):
            for d in ("out", "in"):
                cell = t["cells"]["9.0|%s|C%d|H%d|N<=%d|%s|fwd" % (prec, C, C, nb, d)]
                rec = cell["in_form"]["stacks"][AF3T_STACK][form]
                assert all(rec["capture"]["replay_check"].values())              # the replay form proven: module here == the engine's recorded outputs
                rows = rec["states"]["as_captured"]["rows"]
                for cfg in ("af3t_form", "af3t_form:config=upstream", "af3t_form:config=hm_view", "af3t_form:config=hm_copy", "exact+af3t_module", "module_under_exact_line_levers"):
                    assert rows[cfg]["class"].startswith("bitwise"), (C, prec, nb, d, cfg)
                assert rows["af3t_form"]["x_e2e_vs_module_under_levers"] >= 1.05 and rows["af3t_form"]["x_e2e_vs_module"] > 1.5
                assert not rows["v4"].get("class", "x").startswith("bitwise") if "v4" in rows and "class" in rows["v4"] else True
                assert cell["class"]["af3t_form"][AF3T_STACK].startswith("bitwise")
                assert t["form_cells"]["9.0|%s|C%d|H%d|N<=%d|%s|fwd+af3t_module" % (prec, C, C, nb, d)]["stacks"][AF3T_STACK]["x_statement"] >= 1.05
                seen += 1
    assert seen == 18


def test_af3t_form_module_imports_only_inside_functions_and_meta_census():
    src = open(os.path.join(HERE, "af3t_form.py"), encoding="utf-8").read()
    top = [n for n in ast.parse(src).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert top == [], [ast.dump(n) for n in top]
    meta = json.load(open(os.path.join(os.path.dirname(HERE), "META", "trimul.json"), encoding="utf-8"))
    assert "af3t_form.py" in meta["core_added"] and "of3_form.py" in meta["core_added"]


def test_af3t_form_refusals_with_tensors():
    torch = pytest.importorskip("torch")
    from opt_core.kernels.trimul import af3t_form as AF
    w = {k: torch.zeros((64, 64)) if k.startswith("w_") else torch.zeros(64) for k in T.WEIGHT_KEYS}
    z = torch.zeros((8, 8, 64))
    assert AF.supported(z, None, w).startswith("device:")
    assert AF.CONFIGS == ("upstream", "hm_view", "hm_copy", "hm_fused") and AF.DEFAULT_CONFIG == "upstream"   # from 0.5.78.0: the every-N-exact layout


def test_of3_module_form_cells_carry_the_measured_layout_config_and_select_serves_it():
    """From 0.5.78.0 row of3_form's DEFAULT layout is `replica` (bitwise at every swept N); the fast layouts are explicit per form cell (hm_fused; the
    c 64 N<=400 bf16 cells hm_view, host-bound there) and served at the aligned sizes; a caller's config word still wins."""
    from opt_core.kernels.trimul import of3_form as OF
    t = T.table()
    assert OF.DEFAULT_CONFIG == "replica" and "replica" in OF.CONFIGS and t["rows"]["of3_form"]["default"].startswith("replica") and "replica" in t["rows"]["of3_form"]["configs"]
    assert "speed_of3_h100_r8" in t["rows"]["of3_form"]["configs"]["replica"] and t["rows"]["of3_form"]["configs"]["replica"]["measured"]
    assert "inference_mode" in t["rows"]["of3_form"] and "0 errors" in t["rows"]["of3_form"]["inference_mode"]
    hmview = sorted(k for k, v in t["form_cells"].items() if v.get("config") == "hm_view")
    assert hmview == ["9.0|bf16|C64|H64|N<=400|in|fwd+of3_module", "9.0|bf16|C64|H64|N<=400|out|fwd+of3_module"], hmview
    s = T.select((9, 0), "bf16", 64, 64, 400, "outgoing", word="exact", form="of3_module", stack=OF3_STACK)
    assert s.row == "of3_form" and s.config == "hm_view"
    s = T.select((9, 0), "bf16", 64, 64, 800, "incoming", word="exact", form="of3_module", stack=OF3_STACK)
    assert s.row == "of3_form" and s.config == "hm_fused"                    # explicit now
    s = T.select((9, 0), "bf16", 64, 64, 400, "outgoing", word="exact", form="of3_module", config="upstream", stack=OF3_STACK)
    assert s.config == "upstream"                                           # a caller's config word wins
    for k, v in t["form_cells"].items():                                    # every of3_module cell refreshed by the later run; x_module still > 2
        if k.endswith("+of3_module"):
            st = v["stacks"][v["ref_stack"]]
            assert st.get("run") == "of3_h100_r6" and st["x_module"] > 2.0, k


def test_form_selection_never_steps_aside_to_the_library_rows():
    with pytest.raises(T.Refusal) as e:
        T.select((9, 0), "bf16", 64, 64, 400, "outgoing", word="exact", form="af3t_module", exclude=("af3t_form",), stack=AF3T_STACK)
    assert "no other row states this module" in str(e.value)

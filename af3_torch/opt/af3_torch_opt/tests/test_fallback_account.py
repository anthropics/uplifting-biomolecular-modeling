"""The fallback census verdict is derived per item from the kernels' DECLARED gates (registry.EXPECTED_FALLBACKS): a size gate
(`fallback:N<101`, the threshold the kernel prints) is expected on an item iff the item's padded token count is below it; shape gates
are item-independent; anything else — an undeclared word, a size word on an item the gate does not cover, a kernel_error, calls no
item accounts for — stays UNEXPECTED and the pred is REFUSED:fallback exactly as before."""
import json

from af3_torch_opt import cli, registry


SMALL = {"name": "001_dsdna24", "seed": 1, "bucket": 64, "fallbacks": {"trimul": {"fallback:N<101": 736, "fallback:c=64": 44}, "transition": {"fallback:c=384,stock-row": 548}}}
BIG = {"name": "002_prot400", "seed": 1, "bucket": 448, "fallbacks": {"trimul": {"fallback:c=64": 44}, "transition": {"fallback:c=384,stock-row": 548}}}


def _events(*recs):
    """The pass census the model process would report: the items' counters summed."""
    ev = {}
    for r in recs:
        for lever, counts in r["fallbacks"].items():
            for k, v in counts.items():
                ev.setdefault(lever, {}); ev[lever][k] = ev[lever].get(k, 0) + v
    return ev


def test_size_gate_is_read_from_the_kernels_word():
    assert registry.size_gate_threshold("trimul", "fallback:N<101") == 101 and registry.size_gate_threshold("triattn", "fallback:N<16") == 16
    assert registry.size_gate_threshold("trimul", "fallback:c=64") is None and registry.size_gate_threshold("transition", "fallback:N<101") is None   # not a size gate / not declared for that lever
    assert registry.fallback_expected("trimul", "fallback:N<101", 64) and not registry.fallback_expected("trimul", "fallback:N<101", 448)
    assert not registry.fallback_expected("trimul", "fallback:N<101", None)                      # unattributable to an item: not expected
    assert registry.fallback_expected("trimul", "fallback:c=64", None) and registry.fallback_expected("transition", "fallback:c=384,stock-row", 4608)   # shape gates: any item
    assert not registry.fallback_expected("transition", "fallback:c=128,rows<64", 64) and not registry.fallback_expected("trimul", "fallback:shape", 64)


def test_small_item_every_trimul_call_size_gated_passes():
    """(1) a 24-token item (bucket 64): its trimul calls all take N<101 by the declared gate -> expected == observed, nothing unexpected."""
    acct = registry.fallback_account([SMALL], _events(SMALL))
    assert acct["unexpected"] == {}
    assert acct["expected"] == {"trimul": {"fallback:N<101": 736, "fallback:c=64": 44}, "transition": {"fallback:c=384,stock-row": 548}}
    assert acct["levers"]["trimul"] == {"observed": 780, "accounted": 780, "size_gated": {"fallback:N<101": ["001_dsdna24"]}}
    assert acct["levers"]["transition"] == {"observed": 548, "accounted": 548, "size_gated": {}}          # (4) the transition gate: an exact declared key, expected on any item


def test_big_item_passes_clean_and_refuses_an_undeclared_or_uncovered_word():
    """(2) a 448-token item: no size-gated events -> passes; an undeclared census word, or N<101 on an item the gate does not cover, is unexpected."""
    acct = registry.fallback_account([BIG], _events(BIG))
    assert acct["unexpected"] == {} and acct["levers"]["trimul"] == {"observed": 44, "accounted": 44, "size_gated": {}}
    bad = {**BIG, "fallbacks": {"trimul": {"fallback:c=64": 44, "fallback:shape": 2}}}
    assert registry.fallback_account([bad], _events(bad))["unexpected"] == {"trimul": {"fallback:shape": 2}}
    uncovered = {**BIG, "fallbacks": {"trimul": {"fallback:N<101": 5}}}
    acct = registry.fallback_account([uncovered], _events(uncovered))
    assert acct["unexpected"] == {"trimul": {"fallback:N<101": 5}} and acct["levers"]["trimul"]["size_gated"] == {}
    kerr = {**BIG, "fallbacks": {"triattn": {"kernel_error": 1}}}
    assert registry.fallback_account([kerr], _events(kerr))["unexpected"] == {"triattn": {"kernel_error": 1}}


def test_mixed_pass_names_only_the_small_item():
    """(3) a pass over the 24-token and the 448-token item: expected N<101 = the small item's calls exactly, named; the big item contributes none."""
    acct = registry.fallback_account([SMALL, BIG], _events(SMALL, BIG))
    assert acct["unexpected"] == {}
    assert acct["expected"]["trimul"] == {"fallback:N<101": 736, "fallback:c=64": 88}
    assert acct["levers"]["trimul"] == {"observed": 824, "accounted": 824, "size_gated": {"fallback:N<101": ["001_dsdna24"]}}
    two_seeds = registry.fallback_account([SMALL, {**SMALL, "seed": 2}, BIG], _events(SMALL, SMALL, BIG))
    assert two_seeds["expected"]["trimul"]["fallback:N<101"] == 1472 and two_seeds["levers"]["trimul"]["size_gated"] == {"fallback:N<101": ["001_dsdna24"]}


def test_calls_beyond_the_items_are_judged_without_a_size():
    """Census counts no item accounts for (build-time or unattributed calls): shape gates stay expected, a size gate does not."""
    ev = _events(SMALL); ev["trimul"]["fallback:N<101"] += 10; ev["trimul"]["fallback:c=64"] += 4
    acct = registry.fallback_account([SMALL], ev)
    assert acct["unexpected"] == {"trimul": {"fallback:N<101": 10}}
    assert acct["expected"]["trimul"] == {"fallback:N<101": 736, "fallback:c=64": 48} and acct["levers"]["trimul"]["observed"] == 794 and acct["levers"]["trimul"]["accounted"] == 784
    assert registry.fallback_account([], {"trimul": {"fallback:N<101": 3}})["unexpected"] == {"trimul": {"fallback:N<101": 3}}   # no item records at all (an older forward.json): as before, refused


def test_pass_level_split_keeps_its_meaning():
    exp, unexp = registry.fallback_split({"trimul": {"fallback:c=64": 40, "fallback:N<101": 7}})
    assert exp == {"trimul": {"fallback:c=64": 40}} and unexp == {"trimul": {"fallback:N<101": 7}}
    exp, unexp = registry.fallback_split({"trimul": {"fallback:N<101": 7}}, bucket=64)
    assert exp == {"trimul": {"fallback:N<101": 7}} and unexp == {}


def _input(tmp, name):
    d = tmp / "in"; d.mkdir(exist_ok=True); p = d / f"{name}.json"
    p.write_text(json.dumps({"name": name, "sequences": [], "modelSeeds": [1]})); return str(p)


def test_pred_small_item_size_gated_fallback_is_active_not_partial(box, monkeypatch, capsys):
    """Through pred: the model process reports trimul N<101 fallbacks on a bucket-64 item -> rc 0, no PARTIAL, the FALLBACK line names the
    verdict, the counts and the item; the same events on a bucket-256 item -> REFUSED:fallback, rc 3, as before."""
    monkeypatch.setenv("STUB_FALLBACK", "trimul:N<101"); monkeypatch.setenv("STUB_BUCKETS", "s=64")
    out = box["tmp"] / "out"
    rc = cli.main(["pred", "--mode", "fast", "--json_path", _input(box["tmp"], "s"), "--json_path", _input(box["tmp"], "b"), "--output_dir", str(out)])
    err = capsys.readouterr().err
    assert rc == 0, err[-3000:]
    assert "[af3-torch-opt] FALLBACK lever=trimul fallback_Nlt101=3 expected=1 dead=0 observed=3 accounted=3 size_gated=Nlt101:s" in err and "partial=none" in err
    M = cli.last_run()
    assert M["partial"] is None and M["fallbacks_expected"] == {"trimul": {"fallback:N<101": 3}}
    monkeypatch.setenv("STUB_BUCKETS", "s=256")
    out2 = box["tmp"] / "out2"
    rc = cli.main(["pred", "--mode", "fast", "--json_path", _input(box["tmp"], "s"), "--output_dir", str(out2)])
    err = capsys.readouterr().err
    assert rc == 3 and "[af3-torch-opt] FALLBACK lever=trimul fallback_Nlt101=3 expected=0 dead=0 observed=3 accounted=0 size_gated=none" in err and "partial=REFUSED:fallback" in err
    assert "[af3-torch-opt] PARTIAL refused=fallback rc=3 opt_out=--allow-partial stock=--mode off" in err   # a refused partial names its one opt-out flag (and the stock route)


def test_opm_arch_step_aside_is_accounted_on_a_card_without_cells():
    """Lever opm's kernels carry cells for capability majors 8 and 9 (af3t_msa.OPM_CELLS); on any other card the lever steps aside BY NAME to the
    stock statements and counts `fallback:arch=sm_<cc>` per call. That word is declared coverage (registry.EXPECTED_FALLBACKS['opm'], pattern
    `fallback:arch=sm_{sm}`): accounted on any item, so the pred is not partial there — while the same word naming a measured major (sm_90, sm_80,
    sm_86) stays unexpected (the gate cannot fire on those cards), as does any other opm word."""
    import os, re
    from af3_torch_opt import stack
    assert registry.arch_gate_word("opm", "fallback:arch=sm_103") == "sm_103" and registry.arch_gate_word("opm", "fallback:arch=sm_100") == "sm_100"
    assert registry.arch_gate_word("opm", "fallback:arch=sm_120") == "sm_120"
    for measured in ("fallback:arch=sm_90", "fallback:arch=sm_80", "fallback:arch=sm_86"):
        assert registry.arch_gate_word("opm", measured) is None and not registry.fallback_expected("opm", measured, 1216), measured
    assert registry.arch_gate_word("trimul", "fallback:arch=sm_103") is None and not registry.fallback_expected("trimul", "fallback:arch=sm_103", 1216)   # declared for opm only
    assert not registry.fallback_expected("opm", "fallback:shape", 1216) and not registry.fallback_expected("opm", "fallback:arch=sm_", 1216)
    assert registry.size_gate_threshold("opm", "fallback:arch=sm_103") is None                     # not a size gate: never lands in size_gated
    item = {**BIG, "bucket": 1216, "fallbacks": {"opm": {"fallback:arch=sm_103": 44}, "trimul": {"fallback:c=64": 44}}}
    acct = registry.fallback_account([item, {**item, "name": "003_prot1200"}, {**item, "name": "004_prot1200"}], _events(item, item, item))
    assert acct["unexpected"] == {}
    assert acct["expected"]["opm"] == {"fallback:arch=sm_103": 132} and acct["levers"]["opm"] == {"observed": 132, "accounted": 132, "size_gated": {}}
    assert registry.fallback_account([], {"opm": {"fallback:arch=sm_103": 132}})["levers"]["opm"] == {"observed": 132, "accounted": 132, "size_gated": {}}   # unattributed calls: the card decides, still expected
    bad = {**item, "fallbacks": {"opm": {"fallback:arch=sm_90": 1}}}
    assert registry.fallback_account([bad], _events(bad))["unexpected"] == {"opm": {"fallback:arch=sm_90": 1}}
    exp, unexp = registry.fallback_split({"opm": {"fallback:arch=sm_100": 132, "kernel_error:RuntimeError": 1}})
    assert exp == {"opm": {"fallback:arch=sm_100": 132}} and unexp == {"opm": {"kernel_error:RuntimeError": 1}}
    src = open(os.path.join(stack.forward_dir(), "af3t", "kernels", "af3t_msa.py"), encoding="utf-8").read()   # the word's spelling and the measured majors are the kernel module's
    assert 'if cc not in OPM_CELLS: return "arch=sm_%d%d" % torch.cuda.get_device_capability(msa.device)' in src
    majors = tuple(int(m) for m in re.findall(r"^\s+(\d+): dict\(BA=", src[src.index("OPM_CELLS = {"):], re.M))
    assert majors and tuple(sorted(majors)) == tuple(sorted(registry.ARCH_CELL_MAJORS["opm"])), (majors, registry.ARCH_CELL_MAJORS)


def test_a_stock_row_answered_by_number_is_a_named_aside():
    """When the shared core's provider answers a STOCK row by number for a cell (the measured winner there is a library / torch statement of the op),
    the adapters serve the module's own statement BY NAME and count it: trimul / tmpl_trimul / trimul_exact `fallback:stock_row:<row>`, triangle
    attention `fallback:refused:<row>:stock_row`, transition `fallback:c=<c>,stock-row`. Correct outputs, declared coverage: accounted on any item,
    exit 0. Refusals of KERNEL rows, a bare `refused`, kernel errors and every other word stay unexpected (exit 3)."""
    for lever, key in (("trimul", "fallback:stock_row:torch_math"), ("tmpl_trimul", "fallback:stock_row:cueq"), ("trimul_exact", "fallback:stock_row:native:f32in"),
                       ("triattn", "fallback:refused:sdpa:stock_row"), ("triattn", "fallback:refused:sdpa_upcast:stock_row"),
                       ("transition", "fallback:c=128,stock-row"), ("transition", "fallback:c=64,stock-row"), ("transition", "fallback:c=384,stock-row")):
        assert registry.fallback_expected(lever, key, 1216) and registry.fallback_expected(lever, key, None), (lever, key)
    assert registry.stock_row_word("trimul", "fallback:stock_row:torch_math") == "torch_math" and registry.stock_row_word("triattn", "fallback:refused:sdpa:stock_row") == "sdpa"
    assert registry.stock_row_word("trimul_exact", "fallback:stock_row:native:f32in") == "native:f32in" and registry.stock_row_word("trimul", "fallback:stock_row:triattn_native@v10") == "triattn_native@v10"
    for lever, key in (("triattn", "fallback:refused:triattn_native@v10:NoPrebuilt"), ("triattn", "fallback:refused:fast:no_cell:bf16:D32"), ("triattn", "fallback:refused:sdpa:stock_row "),
                       ("trimul", "fallback:refused"), ("trimul", "kernel_error"), ("trimul", "kernel_error:RuntimeError"), ("trimul", "fallback:stock_row:"), ("trimul", "fallback:stock_row:torch math"),
                       ("transition", "fallback:c=128,no_launch_pin"), ("transition", "fallback:c=128,no_cell"), ("glu_proj", "fallback:stock_row:torch_math"), ("opm", "fallback:refused:sdpa:stock_row"),
                       ("transition", "fallback:stock_row:torch_swiglu"), ("triattn", "fallback:no-provider"), ("trimul", "fallback:differs:c=128")):
        assert not registry.fallback_expected(lever, key, 1216), (lever, key)
        assert registry.stock_row_word(lever, key) is None, (lever, key)
    assert registry.size_gate_threshold("trimul", "fallback:stock_row:torch_math") is None and registry.arch_gate_word("trimul", "fallback:stock_row:torch_math") is None
    item = {**BIG, "bucket": 256, "fallbacks": {"trimul": {"fallback:stock_row:torch_math": 96, "fallback:c=64": 4}, "triattn": {"fallback:refused:sdpa:stock_row": 96},
                                               "transition": {"fallback:c=128,stock-row": 48, "fallback:c=64,stock-row": 6, "fallback:c=384,stock-row": 44}}}
    acct = registry.fallback_account([item, {**item, "name": "003_prot200"}], _events(item, item))
    assert acct["unexpected"] == {}
    for lever in ("trimul", "triattn", "transition"):
        assert acct["levers"][lever]["observed"] == acct["levers"][lever]["accounted"] > 0 and acct["levers"][lever]["size_gated"] == {}, (lever, acct["levers"][lever])
    assert acct["expected"]["triattn"] == {"fallback:refused:sdpa:stock_row": 192} and acct["expected"]["transition"]["fallback:c=128,stock-row"] == 96
    mixed = {**item, "fallbacks": {"triattn": {"fallback:refused:sdpa:stock_row": 96, "fallback:refused:triattn_native@v10:NoPrebuilt": 2}, "trimul": {"kernel_error": 1}}}
    assert registry.fallback_account([mixed], _events(mixed))["unexpected"] == {"triattn": {"fallback:refused:triattn_native@v10:NoPrebuilt": 2}, "trimul": {"kernel_error": 1}}
    exp, unexp = registry.fallback_split({"trimul": {"fallback:stock_row:torch_math": 10, "fallback:refused": 3}})
    assert exp == {"trimul": {"fallback:stock_row:torch_math": 10}} and unexp == {"trimul": {"fallback:refused": 3}}


def test_stock_row_words_are_the_adapters_spellings():
    """Lock the three census spellings at their source (opt/forward/af3t/kernels/af3_kernels.py) so a reworded counter fails here and not as a
    partial pred: the TriMul adapter counts `stock_row:<row>` under the census name of the width (trimul / tmpl_trimul / trimul_exact), the triangle
    attention adapter records `<row>:stock_row` at bind time and counts `refused:<that>` per call, the transition adapter counts `c=<c>,stock-row`."""
    import os
    from af3_torch_opt import stack
    src = open(os.path.join(stack.forward_dir(), "af3t", "kernels", "af3_kernels.py"), encoding="utf-8").read()
    assert '_fallback(lever_c, "stock_row:%s" % sel.row)' in src and 'lever_c = "tmpl_trimul" if (C == 64 and lever == "trimul") else lever' in src
    assert '_TRIATTN_PROV["refused"] = "%s:stock_row" % sel.row' in src and src.count('_fallback(lever, "refused:%s" % _TRIATTN_PROV["refused"])') >= 2
    assert '_fallback(lever, "c=%d,stock-row" % C)' in src
    for lever in ("trimul", "tmpl_trimul", "trimul_exact"):
        assert "fallback:stock_row:{row}" in registry.EXPECTED_FALLBACKS[lever], lever
    assert "fallback:refused:{row}:stock_row" in registry.EXPECTED_FALLBACKS["triattn"]
    assert {"fallback:c=384,stock-row", "fallback:c=128,stock-row", "fallback:c=64,stock-row"} <= set(registry.EXPECTED_FALLBACKS["transition"])

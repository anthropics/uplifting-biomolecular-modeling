"""One LEVER line per registry lever per process in the shared per-lever grammar (opt_core.report.lever_line): key order
name, state, [reason], impl, origin, [evidence ...], mode, lever=<registry name> last; reasons are single tokens; no blank inside any value."""
from protenix_opt import registry, report as R


def _rep(**over):
    rep = {"active": True, "mode": "fast", "levers_applied": ["k2b_flash_triattention", "trimul_core", "lazy_init"], "levers_fallback": ["fastln_prebuilt"],
           "levers_not_in_arm": ["mk_pf"],
           "fallback_reasons": {"fastln_prebuilt": "no fastln_prebuilt*/manifest.json for torch 2.13.0", "mk_pf": "not in the 9.0|3.3 row"}}
    rep.update(over); return rep


def _keys(line):
    return [kv.split("=", 1)[0] for kv in line.split(" LEVER ", 1)[1].split(" ")]


def _by_lever(lines):
    return {l.rsplit(" lever=", 1)[1]: l for l in lines}


def test_tables_cover_the_registry():
    assert set(R.STRATEGY_IDS) == set(registry.LEVERS) == set(R.IMPL)
    assert all(v[1] in ("core", "kit") for v in R.IMPL.values()) and all(" " not in v[0] for v in R.IMPL.values())
    from opt_core import report as CR
    for s in set(R.STRATEGY_IDS.values()):
        CR.strategy_form(s)                                                                        # the LEVER line's own rule: a family id F<k>.<name> or LOCAL.protenix_v2.<name>


def test_one_line_per_registry_lever_in_key_order():
    lines = R.lever_lines(_rep())
    assert [l.rsplit(" lever=", 1)[1] for l in lines] == list(registry.LEVERS)                     # every lever once, registry order
    for l in lines:
        ks = _keys(l)
        assert ks[:2] == ["name", "state"] and ks[-2:] == ["mode", "lever"] and "impl" in ks and ks[ks.index("impl") + 1:ks.index("impl") + 3] == ["origin", "strategy"]
        assert ks[2] in ("reason", "impl") and (ks[2] == "impl" or ks[3] == "impl")
        assert all(" " not in kv.split("=", 1)[1] for kv in l.split(" LEVER ", 1)[1].split(" "))
    assert R.lever_lines({"active": False, "mode": "fast"}) == [] and R.lever_lines(None) == []



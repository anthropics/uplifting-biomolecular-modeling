"""The accounted small-input state (report.gate_state): a run whose every item sits at or below a lever's size gate — the trimul levers at
N_token <= a trimul token gate in accounts that carry one (kits before 0.2.38: 100), the transition levers under TRANSITION_GATE_ROWS (4096 pair rows, N_token <= 63) — served
no call BY DESIGN: the lever prints `state=skipped reason=below_gate … gate=<word> n=<largest item>` and the run exits 0, never PARTIAL.
An item above the gate with no served call stays partial (exit 3)."""
import re
import pytest

from protenix_v1_opt import modes, report as R
from protenix_v1_opt.tests._ditfast_account import DF_OK as _DF_OK

EXACT = modes.resolve("exact"); FAST = modes.resolve("fast")
GATES = {"trimul_tokens": 100, "transition_rows": 4096}


def _exact_account(n_items_tokens, trimul_served=0, xtr_served=0):
    """The kit account of an exact run over items of the given sizes: every TriMul call `stock:N` unless served, every pair transition
    `stock:C=128` unless served; gblock serves (N > 16), sg/hoist serve."""
    return {"cfg": {"trimul": "exact", "gblock": True, "xtr": True, "sg": True, "hoist": True}, "gates": dict(GATES),
            "counts": {"dit": {"apb:fp16": 48}, "pf": {"t2:pf@9.0": 96}, "opm": {"t2:opm@9.0": 8}, "pwa": {"t2:pwa@9.0": 6}, "pf": {"t2:pf@9.0": 96}, "opm": {"t2:opm@9.0": 8}, "pwa": {"t2:pwa@9.0": 6}, "atom": {"apb:tf32rn": 12}, "trimul": ({"exact": trimul_served} if trimul_served else {}) | {"stock:N": 40},
                       "triattn": {"gblock": 96}, "transition": ({"xtr:C=128": xtr_served} if xtr_served else {}) | {"stock:C=128": 40, "stock:C=384": 22}},
            "keep_pool": {"installed": True, "skipped_total": 3, "passed_total": 1, "errors": 0}, "summary_hostidx": {"installed": True, "samples": 5, "delegated": 0, "errors": 0}, "lazy_init": {"installed": True, "constructs": 1, "lazy_construct_s": 3.9, "patched": 12}, "ditfast": _DF_OK, "templ": {"template_dedupe": {"installed": True, "calls": 1, "evaluated": 2, "reused": 2, "stock": {}}, "tmpl_triatt": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul_exact": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_xtr": {"on": True, "routed_total": 4, "stock": {}, "fallback": {}}, "tmpl_pairfused": {"on": True, "calls": 8}}, "dit_attn_exact": {"installed": True, "calls": 72, "routes": {"kernel": 72}}, "apb": {"dit": {"engaged": True, "fp16": True, "opd": "fp16", "cell_key": "9.0", "installed_on": 24, "calls": 48}, "atom": {"engaged": True, "opd": "tf32rn", "cell_key": "9.0", "installed_on": 6, "calls": 12}}, "trunk2": {"pf": {"engaged": True, "cell_key": "9.0", "installed_on": 52, "calls": 96}, "opm": {"engaged": True, "cell_key": "9.0", "installed_on": 4, "calls": 8}, "pwa": {"engaged": True, "cell_key": "9.0", "installed_on": 3, "calls": 6}}, "trunk2": {"pf": {"engaged": True, "cell_key": "9.0", "installed_on": 52, "calls": 96}, "opm": {"engaged": True, "cell_key": "9.0", "installed_on": 4, "calls": 8}, "pwa": {"engaged": True, "cell_key": "9.0", "installed_on": 3, "calls": 6}}, "sampler": {"graphs": True, "prep": {"on": True, "parts": "rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec", "poison": "ok", "stats": {}, "aside": None}, "hoist_installed": True, "sampler": {"replays": 100}, "hoist": {"hits": 10, "records": 1}}}


def test_gate_state_words():
    assert R.gate_state(0, {"word": "n<=100", "tokens": 100}, [{"N_token": 61}])["state"] == "gate-off"
    assert R.gate_state(0, {"word": "n<=100", "tokens": 100}, [{"N_token": 61}, {"N_token": 300}])["state"] == "partial"     # an item above the gate and nothing served
    assert R.gate_state(0, {"word": "n<=100", "tokens": 100}, [{"N_token": None}])["state"] == "partial"                     # unsized: never excused
    assert R.gate_state(7, {"word": "n<=100", "tokens": 100}, [{"N_token": 300}])["state"] == "served"
    assert R.gate_state(0, None, [{"N_token": 61}]) is None and R.gate_state(0, {"word": "x"}, []) is None                    # an account without the gates: no state (the older rule applies)
    assert R._kit_gates({"gates": GATES}) == {"trimul": {"word": "n<=100", "tokens": 100}, "transition": {"word": "rows<4096", "tokens": 63}}


def test_a_61_token_exact_run_is_accounted_not_partial(capsys):
    rep = {"mode": "exact", "levers": _exact_account([61]), "items": [{"N_token": 61}]}
    v = R.verdict(rep, "exact", EXACT.levers, allow_partial=False)
    assert v["partial"] == [] and v["exit_code"] == R.EXIT_OK, v
    assert v["evidence"]["exact"]["gate"]["state"] == "gate-off" and v["evidence"]["xtr"]["gate"]["state"] == "gate-off"
    lines = R.lever_lines(v["evidence"], rep)
    assert re.fullmatch('\\[protenix\\-v1\\-opt\\]\\ LEVER\\ name=F2\\.fpf_trimul_exact\\ state=skipped\\ reason=below_gate\\ impl=opt_core/kernels/\\w+\\ origin=core\\ strategy=F2\\.fpf_trimul_exact\\ served=0\\ gated=40\\ gated_by=stock:N:40\\ gate=n<=100\\ n=61\\ lever=exact', lines[0]), lines[0]   # impl: class-tolerant (the shared core's TriMul provider path)
    assert lines[2].endswith(" served=0 row=none host=gblock aside=no_core_call tier=exact lever=triexact") and " state=skipped reason=aside " in lines[2]   # the provider word: gblock made no core call at 61 tokens (by design)
    assert lines[3] == "[protenix-v1-opt] LEVER name=LOCAL.fused_transition state=skipped reason=below_gate impl=opt_core/kernels/transition origin=core strategy=LOCAL.fused_transition served=0 gated=62 gated_by=stock:C=128:40,stock:C=384:22 gate=rows<4096 n=61 lever=xtr"
    R.log_exit_lines(v)
    err = capsys.readouterr().err
    assert "PARTIAL" not in err and "NOT ACTIVE" not in err


def test_an_80_token_run_serves_transitions_and_accounts_the_trimul_only():
    rep = {"mode": "exact", "levers": _exact_account([80], xtr_served=40), "items": [{"N_token": 80}]}
    v = R.verdict(rep, "exact", EXACT.levers, allow_partial=False)
    assert v["partial"] == [] and v["evidence"]["exact"]["gate"]["state"] == "gate-off" and v["evidence"]["xtr"]["gate"]["state"] == "served"


def test_an_item_above_the_gate_with_nothing_served_stays_partial():
    rep = {"mode": "exact", "levers": _exact_account([61, 300]), "items": [{"N_token": 61}, {"N_token": 300}]}
    v = R.verdict(rep, "exact", EXACT.levers, allow_partial=False)
    assert v["partial"] == ["exact", "xtr"] and v["exit_code"] == R.EXIT_NOT_ACTIVE
    lines = R.lever_lines(v["evidence"], rep)
    assert " state=skipped reason=served0 " in lines[0] and " gate=" not in lines[0]


def test_an_account_without_gates_keeps_the_old_rule():
    acc = _exact_account([61]); acc.pop("gates")
    v = R.verdict({"mode": "exact", "levers": acc, "items": [{"N_token": 61}]}, "exact", EXACT.levers, allow_partial=False)
    assert v["partial"] == ["exact", "xtr"]


def test_the_kit_file_names_the_gates_the_report_reads():
    import ast, os
    from protenix_v1_opt import kit as K
    src = open(K.levers_file()).read(); t = ast.parse(src)
    vals = {n.targets[0].id: n.value.value for n in t.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("TRIMUL_GATE_TOKENS", "TRANSITION_GATE_ROWS")}
    assert vals == {"TRANSITION_GATE_ROWS": 4096}                                   # kit 0.2.38: the TriMul levers have no size gate (the provider's cells decide per key; a refused key keeps the stock statement by name)
    assert '"gates": {"transition_rows": TRANSITION_GATE_ROWS, "triattn_ceiling_tokens": TRIATTN_CEILING_TOKENS, "triattn_gate_tokens": TRIATTN_GATE_TOKENS}' in src      # describe() carries them to the report

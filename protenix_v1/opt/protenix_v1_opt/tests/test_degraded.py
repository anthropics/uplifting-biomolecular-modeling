"""The DEGRADED event: a lever whose kernel served every call on a slower path of identical numerics is named in the record and on
stderr — today gflash's flash_triattn direct launcher (opt_core/kernels/flash_triattn.py fast_launch_stats) — and is neither a fallback
nor partial (the exit rule is unchanged by it)."""
from protenix_v1_opt import report as R


def _levers(disabled, why=None):
    return {"cfg": {"trimul": "fast", "gflash": True}, "counts": {"dit": {"apb:fp16": 48}, "atom": {"apb:tf32rn": 12}, "trimul": {"fast": 96}, "triattn": {"gflash": 96}},
            "flash_triattn_launch": {"disabled": disabled, "why_disabled": why, "fast_launches": 0 if disabled else 1200, "jit_launches": 1296 if disabled else 96, "cached_keys": 3}}


def test_fast_launch_disabled_is_a_named_degraded_event_not_partial():
    ev = R.kit_evidence(_levers(True, "launch mismatch (attn)"), "fast", ("gflash",))
    assert ev["gflash"]["served"] == 96 and ev["gflash"]["fallback"] == {}
    assert ev["gflash"]["degraded"] == {"fast_launch": {"state": "disabled", "why": "launch mismatch (attn)", "fast_launches": 0, "jit_launches": 1296}}
    assert R.partial_of(ev) == ([], None)
    lines = R.degraded_lines(ev)
    assert lines == [f"{R.PREFIX} DEGRADED gflash fast_launch=disabled why='launch mismatch (attn)' fast_launches=0 jit_launches=1296"]


def test_fast_launch_on_prints_nothing():
    ev = R.kit_evidence(_levers(False), "fast", ("gflash",))
    assert "degraded" not in ev["gflash"] and R.degraded_lines(ev) == []


def test_no_launch_record_prints_nothing():
    lv = _levers(False); lv.pop("flash_triattn_launch")
    ev = R.kit_evidence(lv, "fast", ("gflash",))
    assert "degraded" not in ev["gflash"]


def test_log_exit_lines_order(capsys):
    v = {"partial": [], "partial_reason": None, "allow_partial": False, "exit_code": 0,
         "evidence": R.kit_evidence(_levers(True, "launch: RuntimeError('x')"), "fast", ("gflash",))}
    R.log_exit_lines(v)
    err = capsys.readouterr().err.strip().splitlines()
    assert err[0].startswith(f"{R.PREFIX} DEGRADED gflash fast_launch=disabled why=") and all(" LEVER " in l for l in err[1:-2])
    assert err[-2].startswith(f"{R.PREFIX} TEMPLATE event=exit templates=") and err[-1].startswith(f"{R.PREFIX} ITEMS total=")   # the template record, then the item census close the exit lines
    assert [l for l in err if l.endswith("lever=gflash")] == [f"{R.PREFIX} LEVER name=F1.flash_triatt state=on impl=opt_core/attn/pair_fused.py origin=core strategy=F1.flash_triatt served=96 degraded=fast_launch lever=gflash"]

"""The EXIT line carries ONE per-call fallback census word, `fallbacks=none|<label>:<reason>=<n>,…`, composed from every loaded kit module's own
record (report.FALLBACK_SOURCES) — a degraded route that served an input is a named event; the rows fail closed on a non-none word."""
import sys, types

from openfold3_ob0_opt import report


def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _clear():
    for name, _l, _r in report.FALLBACK_SOURCES:
        sys.modules.pop(name, None)


def test_none_when_no_kit_module_is_loaded():
    _clear()
    assert report.fallback_census() == {} and report.fallback_word() == "none"
    line = report.exit_tally_line({"active": True, "mode": "fast", "line": None, "levers_applied": [], "levers_unavailable": [], "levers_pending": []}).split("\n")[0]
    assert line.count(" fallbacks=") == 1 and " fallbacks=none" in line


def test_word_names_every_source_with_a_count():
    _clear()
    try:
        _mod("of3_graphs", fallbacks=lambda: {"refused": 2, "capture_error": 0}, _STATE={"enabled": True, "stats": {"captures": 3, "replays": 40, "eager_steps": 2, "fallbacks": 2, "fallback:refused": 2}})
        _mod("of3t_levers", fallbacks=lambda: {"paircache:cap": 1})
        _mod("of3_offload", census=lambda: {"fallbacks": {"pageable_pin": 4}, "pageable": {}, "conf_census": {}, "conf": {"path": "chunked"}})
        _mod("openfold3_ob0_opt.cells.rollout", fallbacks=lambda: {})
        assert report.fallback_census() == {"cuda_graphs:refused": 2, "trunk_kernels:paircache:cap": 1, "offload:pageable_pin": 4}
        assert report.fallback_word() == "cuda_graphs:refused=2,offload:pageable_pin=4,trunk_kernels:paircache:cap=1"
        c = report.kit_counters()
        assert c["fallbacks"] == report.fallback_word() and c["captures"] == 3 and c["replays"] == 40 and "fallback:refused" not in c   # the graphs' numbers stay, their fallbacks live in the word
        line = report.exit_tally_line({"active": True, "mode": "fast", "line": None}).split("\n")[0]
        assert line.count(" fallbacks=") == 1
    finally:
        _clear()


def test_an_unreadable_source_is_named_not_hidden():
    _clear()
    try:
        def boom():
            raise RuntimeError("x")
        _mod("of3t_levers", fallbacks=boom)
        assert report.fallback_word() == "trunk_kernels:unreadable=1"
    finally:
        _clear()

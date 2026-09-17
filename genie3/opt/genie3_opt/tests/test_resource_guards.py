"""Resource-exhaustion guards: two layers of the same discipline, no silent fallback to a smaller or slower configuration.

CAPACITY (pre-run): before each batch the driver measures one eager forward at batch 1 on the batch's first design (the probe: this
input, this line's model as configured) and projects the batch's need as resident + CAPACITY_MULT[regime] × B × probe; a batch that
would not fit the device is refused by name BEFORE it runs (the driver's exit 5; the package's exit 3, NOT ACTIVE at this batch on this card,
with the largest batch that fits named) — never reduced. The
measured cases (binder problem `07_h1`, 612 tokens padded to 670: exact at batch 16 ran inside a 37.4 GiB captured-graph pool plus the eager
pre-part; fast at batch 16 died with 57.9 GiB in pools) are the `graph` regime's reference cases.

OOM (in-run): no served handler catches an OOM to run a smaller or slower configuration (a fallback needs at least the memory of the
fused path it replaces, so an out-of-memory is the caller's to see). The served package's broad handlers are bookkeeping only (the
census as data — probes, pin / evidence / tally reads, the stock proof); L7's wrapper (`trimul._forward`), the carried drivers
(`driver/g3fast.py`, `g3cap/g3cap.py`) and their patch module have no exception handler at all; a mocked OOM driven through L7's wrapper and
a `design` pass on the modes' driver line propagates / fails loud instead of falling back. The classifier is the
shared core's `opt_core.oom.is_oom`; the kit spells none of its own.

CPU only throughout: the capacity verdict is pure arithmetic (the pass runs the stub driver); the OOM checks mock the exception, never
raise a real one."""

import ast
import os
import re
import types

import pytest
from opt_core.oom import is_oom

from genie3_opt import design, g3batch, stack, trimul
from genie3_opt.tests import _stubs


# ===== CAPACITY: the batched line's pre-run guard =====
def test_regimes_and_multipliers():
    assert g3batch.capacity_regime(True) == "graph" and g3batch.capacity_regime(False) == "eager" and set(g3batch.CAPACITY_MULT) == {"eager", "graph"}
    m = g3batch.CAPACITY_MULT
    assert 1.3 <= m["eager"] < m["graph"] <= 2.2                                             # an eager forward peaks at 1.31 × B × probe → 1.35; the kept pool (1.16–1.35) + the pre-part beside it → graph 2.0
    assert 0.8 <= g3batch.CAPACITY_HEADROOM <= 0.92 and g3batch.EXIT_CAPACITY == 5


def test_the_measured_oom_case_is_refused_with_margin_and_the_measured_fits_fit():
    probe = 2.02                                                                             # GB: the measured batch-1 probe at 670 padded tokens
    need16, fits16, _ = g3batch.capacity_verdict(16, probe, resident_gb=0.9, total_gb=79.2, regime="graph")
    assert fits16 and 64 <= need16 <= 67, need16                                             # exact at batch 16 ran (pool 37.4 GiB + the pre-part): fits, inside the card
    fast16 = g3batch.capacity_verdict(16, 2.68, resident_gb=0.9, total_gb=79.2, regime="graph")            # fast (unchunked transition, probe 2.68 GB) at batch 16 died with 57.9 GiB in pools + the pre-part: refused, batch 8 named
    assert not fast16[1] and fast16[0] > 79.2 and fast16[2] == 8, fast16                     # refused outright (need above the whole card, not inside the headroom band): the measured OOM
    assert g3batch.capacity_verdict(8, 2.68, resident_gb=0.9, total_gb=79.2, regime="graph")[1]            # fast at batch 8 ran
    line = g3batch.capacity_line(16, 670, "graph", 2.68, fast16[0], 0.9, 79.2, fast16[1], fast16[2])
    assert line.startswith("[genie3-opt] CAPACITY batch=16 n_token=670 regime=graph probe_gb=2.68 need_gb=") and "verdict=refused: set generation.dataset.batch_size <= 8" in line
    assert g3batch.capacity_verdict(1, 80.0, resident_gb=1.0, total_gb=79.2, regime="eager")[2] is None    # nothing smaller than batch 1 to suggest
    kept = g3batch.capacity_verdict(16, probe, resident_gb=35.0, total_gb=79.2, regime="graph")           # kept graph pools of other shapes already resident count against the batch
    assert not kept[1] and kept[2] == 8, kept


def test_kept_graphs_yield_before_a_batch_that_fits_by_itself_is_refused():
    """The measured two-batch case (A100-SXM4-80GB, unconditional L=800, batch 8, 16 designs, exact): batch 1 fits (need 46.4 of
    0.9 × 79.3); before batch 2 the kept graph of batch 1 holds 29.5 GB, so the same projection reads 75.5 — refused with the pool resident,
    ok once it yields. The guard frees kept graphs oldest-first and judges again; it refuses only what does not fit on its own; with no
    cache (eager line) or an empty one the verdict is capacity_verdict's, untouched."""
    class FakeCache:                                                                         # GraphCache's shape: an ordered {sig: [graphs]} store and a count
        def __init__(self, pools):
            import collections
            self.store = collections.OrderedDict((f"sig{i}", [gb]) for i, gb in enumerate(pools)); self.n = len(pools)
    def meter(cache, base=0.8, total=79.3):                                                  # resident = context + the pools the cache still keeps
        return lambda: (base + sum(sum(v) for v in cache.store.values()), total)
    c = FakeCache([29.5])
    need, fits, b_ok, resident, total, evicted = g3batch.capacity_verdict_yielding(8, 2.82, "graph", c, meter(c))
    assert fits and evicted == 1 and c.n == 0 and not c.store and abs(resident - 0.8) < 1e-9 and 45.5 < need < 46.5, (need, fits, evicted)
    c = FakeCache([29.5])                                                                    # a batch that does not fit even alone: everything yields, then the refusal names the smaller batch
    need, fits, b_ok, resident, total, evicted = g3batch.capacity_verdict_yielding(8, 6.0, "graph", c, meter(c))
    assert not fits and evicted == 1 and b_ok == 4 and c.n == 0, (need, fits, b_ok, evicted)
    c = FakeCache([10.0, 10.0, 10.0])                                                        # oldest first, and only as many as the batch needs
    need, fits, b_ok, resident, total, evicted = g3batch.capacity_verdict_yielding(8, 2.82, "graph", c, meter(c))
    assert fits and evicted == 1 and c.n == 2 and list(c.store) == ["sig1", "sig2"], (need, evicted, list(c.store))
    c = FakeCache([10.0])                                                                    # fits with the pool resident: nothing yields (the kept graph stays a reuse opportunity)
    assert g3batch.capacity_verdict_yielding(8, 1.12, "graph", c, meter(c))[1:] == (True, 4, 10.8, 79.3, 0) and c.n == 1
    for empty in (None, FakeCache([])):                                                      # no cache / nothing kept: capacity_verdict's own answer
        need, fits, b_ok, resident, total, evicted = g3batch.capacity_verdict_yielding(16, 2.68, "graph", empty, lambda: (0.9, 79.2))
        assert (need, fits, b_ok) == g3batch.capacity_verdict(16, 2.68, 0.9, 79.2, "graph") and evicted == 0 and not fits
    small = FakeCache([17.0]); m40 = meter(small, total=39.4)                                # a 40 GB card, batch 2 of a request whose batch 1 fitted (batch 8, probe 1.6 GB: need 26.4 of 35.5): refused while batch 1's
    assert not g3batch.capacity_verdict(8, 1.6, m40()[0], 39.4, "graph")[1]                 # 17 GB pool is counted resident (43.4), fits once it yields
    fit40 = g3batch.capacity_verdict_yielding(8, 1.6, "graph", small, m40)
    assert fit40[1] and fit40[5] == 1 and small.n == 0 and 26.0 < fit40[0] < 26.8, fit40
    assert g3batch.evict_oldest_graph(None) is False and g3batch.evict_oldest_graph(FakeCache([])) is False


def test_small_inputs_are_never_refused():
    for b, probe in ((8, 0.083), (16, 0.083), (32, 0.083), (16, 0.25), (8, 0.45), (8, 1.2)):          # batch-1 probes of the unconditional rungs (L100 measured 0.083 GB; the longer rungs scaled by tokens²)
        assert g3batch.capacity_verdict(b, probe, resident_gb=1.0, total_gb=79.2, regime="graph")[1], (b, probe)


def test_probe_slice_takes_the_first_design_and_drops_hoist_keys():
    torch = pytest.importorskip("torch")
    bd = {"gt_atom_positions": torch.zeros(4, 7, 3), "token_mask": torch.ones(4, 7), "_g3fast_cond_group_max": 3, "_g3cap_hoist": True, "pair": torch.zeros(4, 7, 7, 2)}
    b1 = g3batch.probe_slice(bd, 4)
    assert b1["gt_atom_positions"].shape == (1, 7, 3) and b1["pair"].shape == (1, 7, 7, 2) and b1["_g3fast_cond_group_max"] == 3 and "_g3cap_hoist" not in b1
    assert g3batch.probe_slice({"x": torch.zeros(1, 5)}, 1)["x"].shape == (1, 5)


def test_a_refused_batch_is_not_active_by_name(tmp_path, monkeypatch, capsys):
    """The pass cannot activate at this batch on this card: exit 3 (NOT ACTIVE), the batch size that fits named — never a silent smaller batch."""
    _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setenv("STUB_FAIL", "capacity")
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2, batch=16), str(tmp_path / "cap"), "exact")
    assert rc == 3 and man["status"] == "refused", man
    assert man["driver_pass"]["rc"] == 5 and man["outputs"]["n_pdb"] == 0
    assert man["capacity_note"] == "batch 2 at 7 tokens needs ≈75.9 GB: refused before running — set generation.dataset.batch_size <= 8 in the request (or run --mode off)"   # the stub's batch: 2 designs, its token stand-in 7
    err = capsys.readouterr().err
    assert "[genie3-opt] CAPACITY mode=exact batch=2 n_token=7 need_gb=75.9 verdict=refused suggest_batch=8" in err
    assert "[genie3-opt] NOT ACTIVE: capacity — batch 2 at 7 tokens" in err
    assert "capacity=refused" in open(man["driver_pass"]["log"]).read()
    stack.reset_for_tests()


def test_probe_seconds_are_inside_the_batch_wall(tmp_path, monkeypatch):
    """The probe is per-batch work of the line: every per_batch record carries probe_s as a component of wall_s, and timings.probe_s totals it."""
    import json, os
    _stubs.box(str(tmp_path), monkeypatch)
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2, batch=2), str(tmp_path / "p"), "exact")
    assert rc == 0, man
    tj = json.load(open(os.path.join(str(tmp_path / "p"), "timings.json")))
    assert all("probe_s" in p and p["probe_s"] <= p["wall_s"] for p in tj["per_batch"]) and tj["per_batch"]
    stack.reset_for_tests()


# ===== OOM: propagates on every served path, never caught to fall back =====
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                       # opt/genie3_opt
TREE = os.path.dirname(os.path.dirname(PKG))                                             # genie3/
BROAD_NAMES = {"Exception", "BaseException", "RuntimeError"}

# the served package's broad handlers, every one bookkeeping (file, enclosing function): a new broad handler fails here until it is classed
SERVED_BOOKKEEPING = {("stack.py", "gpu_probe"), ("stack.py", "activate"), ("design.py", "read_timings"), ("design.py", "read_evidence"),
                      ("report.py", "tally_from_timings"), ("report.py", "_print_exit_tally"),
                      ("stock_cli.py", "core_proof"), ("stock_cli.py", "main")}



class OutOfMemoryError(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError on a box without torch (is_oom knows the class by name and the message by text)."""


def _broad_handlers(path):
    """(enclosing function, line) of every `except` whose type is absent, Exception, BaseException or a tuple holding one of BROAD_NAMES."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    found = []

    def names(t):
        if t is None:
            return {"<bare>"}
        if isinstance(t, ast.Tuple):
            return set().union(*(names(e) for e in t.elts))
        return {t.id} if isinstance(t, ast.Name) else {ast.unparse(t)}

    def walk(node, func):
        for child in ast.iter_child_nodes(node):
            f = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else func
            if isinstance(child, ast.ExceptHandler) and (names(child.type) & (BROAD_NAMES | {"<bare>"})):
                found.append((func, child.lineno))
            walk(child, f)

    walk(tree, "<module>")
    return found


def test_is_oom_is_the_core_classifier_and_the_kit_spells_none():
    assert is_oom(OutOfMemoryError("CUDA out of memory (mock)")) and is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")) and is_oom(MemoryError())
    assert not is_oom(RuntimeError("CUDA error: an illegal memory access was encountered")) and not is_oom(ValueError("x"))
    for root, _d, files in os.walk(os.path.join(TREE, "opt")):                              # package + carried tree: no classifier of the kit's own
        for f in files:
            if f.endswith((".py", ".sh")):
                assert not re.search(r"^\s*def is_oom\b", open(os.path.join(root, f), encoding="utf-8", errors="replace").read(), re.M), os.path.join(root, f)


def test_served_broad_handlers_are_bookkeeping_only():
    found = set()
    for f in sorted(os.listdir(PKG)):
        if f.endswith(".py"):
            found |= {(f, func) for func, _line in _broad_handlers(os.path.join(PKG, f))}
    assert found == SERVED_BOOKKEEPING, sorted(found ^ SERVED_BOOKKEEPING)
    driver_dir = os.path.join(TREE, "opt", "forward", "fast_inference", "driver")
    cap_driver = os.path.join(TREE, "opt", "forward", "g3cap", "g3cap.py")
    batched = os.path.join(PKG, "g3batch.py")                                                  # the modes' driver: no exception handler at all (its LEVER line prints from `finally`)
    for served in (trimul.__file__, os.path.join(driver_dir, "g3fast.py"), os.path.join(driver_dir, "g3fast_patches.py"), cap_driver, batched):
        handlers = [n for n in ast.walk(ast.parse(open(served, encoding="utf-8").read())) if isinstance(n, ast.ExceptHandler)]
        assert handlers == [], (served, [h.lineno for h in handlers])                     # the lever wrapper and the carried drivers: no handler at all
    assert {func for func, _l in _broad_handlers(os.path.join(TREE, "stock", "check_pins.py"))} == {"_git"}   # the pins probe: git unreadable -> None
    pth = open(os.path.join(TREE, "opt", "genie3_opt_autoload.pth"), encoding="utf-8").read()   # the finder guard (the core's template): fail-closed, exit 3
    assert pth.count("except BaseException as _e") == 1 and "_o._exit(3)" in pth and "except SystemExit as _x" in pth


def test_mocked_oom_propagates_through_the_lever_wrapper(monkeypatch):
    """L7's adapter is one call into the shared core's ladder (opt_core.trimul.Lever.serve — whose own tests hold the OOM / error / named-
    fallback contract): an OOM raised while serving propagates through `_forward` unchanged (no stock reroute in the kit), so does any other
    error the ladder lets out; the module, the direction word, `residual=False` and the module's own forward reach the ladder as the Call."""
    seen = {}

    class FakeLever:
        def __init__(self, exc=None):
            self.exc = exc

        def serve(self, call):
            seen["call"] = call
            if self.exc is not None:
                raise self.exc
            return call.orig()

    monkeypatch.setitem(trimul._ORIG, "forward", lambda self, z, mask=None: ("stock-forward", z, mask))
    out_mod, in_mod = types.SimpleNamespace(_outgoing=True), types.SimpleNamespace(_outgoing=False)
    monkeypatch.setitem(trimul.STATE, "lever", FakeLever(OutOfMemoryError("CUDA out of memory (mock)")))
    with pytest.raises(OutOfMemoryError) as e:
        trimul._forward(out_mod, "z", "mask")
    assert is_oom(e.value) and seen["call"].direction == "outgoing" and seen["call"].residual is False and seen["call"].module is out_mod
    monkeypatch.setitem(trimul.STATE, "lever", FakeLever(RuntimeError("CUDA error: an illegal memory access was encountered (mock)")))
    with pytest.raises(RuntimeError) as e:
        trimul._forward(in_mod, "z", None)
    assert not is_oom(e.value) and seen["call"].direction == "incoming" and seen["call"].mask is None
    monkeypatch.setitem(trimul.STATE, "lever", FakeLever(None))                                  # the ladder's named-fallback route ends in the module's own forward (Call.orig)
    assert trimul._forward(out_mod, "z", "mask") == ("stock-forward", "z", "mask")


def test_mocked_oom_fails_the_design_pass_loud(tmp_path, monkeypatch, capsys):
    """The served entry above the wrapper: a `design` pass whose driver dies of OOM is `failed` with the driver's rc and the OOM in its log —
    never `ok`, never `partial`, nothing rerouted — on the exact and the fast line alike."""
    src = "class OutOfMemoryError(RuntimeError):\n    pass\nraise OutOfMemoryError('CUDA out of memory (mock)')\n"
    b = _stubs.box(str(tmp_path), monkeypatch)
    kit_driver = os.path.join(b["tree"], "opt", "forward", "fast_inference", "driver", "g3fast.py")   # the resident driver dies of OOM too
    open(kit_driver, "w", encoding="utf-8").write(src)
    batched = os.path.join(b["tree"], "opt", "genie3_opt", "g3batch.py")                              # the modes' driver (exact and fast): the batched capture line dies of OOM — `design` records failed
    open(batched, "w", encoding="utf-8").write(src)
    for mode in ("exact", "fast"):
        design.stack.reset_for_tests(); design._report.reset_for_tests()
        out = str(tmp_path / f"out_oom_{mode}")
        rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2, seed=0), out, mode)
        assert rc != 0 and man["status"] == "failed" and man["driver_pass"]["rc"] != 0 and not man.get("levers_fallback"), (mode, man)
        assert "OutOfMemoryError: CUDA out of memory (mock)" in open(man["driver_pass"]["log"], encoding="utf-8").read(), mode


def test_check_pins_reports_stack_drift_and_refuses_only_the_checkout(monkeypatch, capsys):
    """stock/check_pins.py (run.sh install's one gate): a torch / lightning / numpy off the pinned stack is REPORTED (a NOTE line, exit 0 — the kit runs on the
    installed stack and names it); a checkout that is not the pin byte for byte is the one refusal (exit 3): that changes what "stock" means."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("g3_check_pins", os.path.join(TREE, "stock", "check_pins.py"))
    cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)
    monkeypatch.setattr(cp, "find_root", lambda root=None: ("/x/genie3", "env"))
    detail = {"checkout": {"pinned": True, "root": "/x/genie3", "files_checked": 188, "archive": "genie3-d77ae5ac.tar.gz", "git_head": None, "git_modified": None}, "stack": {}}
    monkeypatch.setattr(cp, "check", lambda pins, root=None, stack=True: (["stack: torch 2.8.0 != pinned 2.7.1 (PINS.json pinned_stack)"], detail))
    cp.main(["--quiet"])                                                                          # returns: reported, not refused
    assert "check_pins: NOTE stack: torch 2.8.0 != pinned 2.7.1 (PINS.json pinned_stack) — reported, not refused" in capsys.readouterr().err
    monkeypatch.setattr(cp, "check", lambda pins, root=None, stack=True: (["genie3: git HEAD 01234567 at /x/genie3 is not the pin d77ae5ac", "stack: numpy 2.3.0 != pinned 2.2.6 (PINS.json pinned_stack)"], detail))
    with pytest.raises(SystemExit) as ex:
        cp.main(["--quiet"])
    err = capsys.readouterr().err
    assert ex.value.code == 3 and "check_pins: genie3: git HEAD 01234567" in err and "check_pins: NOTE stack: numpy" in err


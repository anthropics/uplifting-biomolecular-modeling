"""The offload port's logging-only diagnostics (`opt/forward/offload/of3o/of3_offload.py`): the row-chunked tensor statistics (`_chunk_stats`),
the >= 2^31-element output observer (`_BigCheckMode`, OF3O_BIGCHECK=1: it logs, it never changes a value), the pairformer-block cadence and abort
probe (OF3O_LOG_EVERY_BLOCKS, OF3O_ABORT_AFTER_BLOCKS), the layer words of `apply_core` (OF3O_LAYER=0: the stock trunk / rollout under phase
timers and no lever; 2: refused by name), and the resolver's reading of the diagnostic switches (no line's exports; a preset one under a mode is refused by
name). CPU; the hook-level cases run in a fresh interpreter (apply_core patches upstream's classes process-wide)."""
import json
import math
import os
import subprocess
import sys

import pytest

from openfold3_ob0_opt import modes
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()
OF3O = os.path.join(HOME, "opt", "forward", "offload", "of3o")
DIAG_SWITCHES = {"OF3O_BIGCHECK": "1", "OF3O_ZSTATS": "1", "OF3O_LOG_EVERY_BLOCKS": "8", "OF3O_ABORT_AFTER_BLOCKS": "40"}
REFUSAL = "OF3O_LAYER=2 (the pair representation streamed through host rows) is not part of this tree; the big mode runs OF3O_LAYER=1"


def _events(text):
    """The port's `[of3o] {json}` event records in a stderr capture, in order."""
    out = []
    for line in text.splitlines():
        if line.startswith("[of3o] {"):
            out.append(json.loads(line[len("[of3o] "):]))
    return out


@pytest.fixture(scope="module")
def O():
    pytest.importorskip("torch")
    sys.path.insert(0, OF3O)
    try:
        import of3_offload
        yield of3_offload
    finally:
        sys.path.remove(OF3O)


def test_the_diagnostic_switches_are_no_line_s_exports_and_a_preset_one_is_refused_under_a_mode():
    """No line exports or unsets them (off unless set); under a kit mode a preset one is refused by name like every add-on variable the line
    does not set (modes.resolve: NOT ACTIVE, `unset it`) — they are reachable on the hook directory's own PYTHONPATH route (the offload README)."""
    for (mode, line), ln in modes.LINES.items():
        for name in DIAG_SWITCHES:
            assert name not in ln.env and name not in ln.unset, (mode, line, name)
    assert "OF3O_CHUNK" not in modes.LINES[("big", "resident")].unset                  # no unit of this tree reads a chunk-plan switch: the line names none
    res = modes.resolve("big", HOME, environ=dict(DIAG_SWITCHES), n_tokens=2565)
    assert res.line == "resident"
    for name, val in DIAG_SWITCHES.items():
        assert f"{name}={val!r} preset: not a switch of the big/resident line (the mode sets its own switches; unset it)" in res.conflicts, (name, res.conflicts)
    assert modes.resolve("big", HOME, environ={}, n_tokens=2565).conflicts == []


def test_chunk_stats_equal_the_full_tensor_statistics(O):
    import torch
    g = torch.Generator().manual_seed(7)
    t = torch.randn(1, 300, 9, 5, generator=g, dtype=torch.float32) * 3.0          # 300 leading rows -> chunks of 128, 128, 44 along dim 1 (the first dim > 1)
    st = O._chunk_stats(t)
    ref = t.double()
    assert st["numel"] == t.numel() and st["shape"] == [1, 300, 9, 5] and st["n_nan"] == 0 and st["n_inf"] == 0 and st["first_bad_row"] is None
    assert math.isclose(st["mean"], float(ref.mean()), abs_tol=2e-6)                     # fp64 sums per chunk, rounded to 6 places
    assert math.isclose(st["std"], float(ref.var(unbiased=False).sqrt()), rel_tol=1e-6)
    assert st["absmax"] == round(float(ref.abs().max()), 4)
    assert st["row_chunk_absmax"] == [round(float(t[:, i:i + 128].abs().max()), 4) for i in (0, 128, 256)]
    bad = t.clone(); bad[0, 130, 2, 1] = float("nan"); bad[0, 299, 0, 0] = float("inf"); bad[0, 5, 0, 0] = float("-inf")
    sb = O._chunk_stats(bad)
    assert (sb["n_nan"], sb["n_inf"], sb["first_bad_row"]) == (1, 2, 0)                 # first_bad_row: the start row of the first chunk holding a nan/inf
    fin = torch.nan_to_num(bad, nan=0.0, posinf=0.0, neginf=0.0).double()
    assert math.isclose(sb["mean"], float(fin.sum() / bad.numel()), abs_tol=2e-6)        # the finite statement: bad elements enter as zeros, the count stays the tensor's


def test_bigcheck_logs_big_outputs_once_per_signature_counts_bad_ones_and_changes_no_value(O, capfd):
    import torch
    g = torch.Generator().manual_seed(11)
    a = torch.randn(4, 40, generator=g); b = torch.randn(4, 40, generator=g); small = torch.randn(3, 5, generator=g); c = a - 10.0
    want = [a + b, a * 2.0, a + b, a + b, a + b, torch.sqrt(c), small + 1.0]             # outside the observer
    m = O._BigCheckMode(); m.LIMIT = 100                                                  # this instance: outputs of >= 100 elements are "big" (the class: 2^31)
    capfd.readouterr()
    with m:
        got = [a + b, a * 2.0, a + b, a + b, a + b, torch.sqrt(c), small + 1.0]
    err = capfd.readouterr().err
    assert len(want) == len(got)
    for x, y in zip(want, got):
        assert torch.equal(torch.nan_to_num(x, nan=-1.0), torch.nan_to_num(y, nan=-1.0))  # logging only: every value as without the observer (nan positions included)
    ev = _events(err)
    firsts = [e for e in ev if e["event"] == "bigop_first"]; checks = [e for e in ev if e["event"] == "bigop_check"]
    summ = m.summary()
    assert summ["n_big_outputs"] == 6 and summ["n_bad"] == 1, summ                       # the 4x40 = 160-element outputs: add x4, mul, sqrt (all nan); small + 1 (15 elements) is not big
    assert summ["n_signatures"] == len(firsts) == 3, (summ, firsts)                     # (add, [4,40], f32), (mul, [4,40], f32), (sqrt, [4,40], f32): one bigop_first each
    adds = [e for e in checks if e["op"] == "add"]
    assert [e["call"] for e in adds] == [1, 2, 3]                                         # checked on the first 3 calls per signature; the 4th add (call 4) is neither checked nor logged
    sq = [e for e in checks if e["op"] == "sqrt"]
    assert len(sq) == 1 and sq[0]["n_nan"] == 160 and sq[0]["first_bad_row"] == 0        # a nan output is counted (n_bad) and logged with its statistics
    assert all("row_chunk_absmax" not in e for e in checks)


def test_bigcheck_summary_lists_signatures_by_call_count(O):
    import torch
    m = O._BigCheckMode(); m.LIMIT = 10
    x = torch.ones(2, 8)
    with m:
        for _ in range(5):
            x * 3.0
        x + 1.0
    s = m.summary()
    assert s["n_big_outputs"] == 6 and s["n_signatures"] == 2 and s["n_bad"] == 0
    assert [(r["op"], r["calls"]) for r in s["signatures"]] == [("mul", 5), ("add", 1)] and s["signatures"][0]["shape"] == [2, 8] and s["signatures"][0]["dtype"] == "torch.float32"


def test_zstats_selects_cuda_pair_tensors_only(O, capfd):
    """`_zstats_log` picks the pair (N x N x C) and single tensors among a block's CUDA arguments/outputs; CPU tensors are never selected, the
    block record is still logged (n, phase)."""
    import torch
    z = torch.zeros(1, 64, 64, 8); s = torch.zeros(1, 64, 16)
    assert O._find_pair_single([z, s]) == (None, None)                                     # CPU tensors: not candidates
    capfd.readouterr()
    O.STATE["phase"] = "trunk"
    try:
        O._zstats_log(5, (s, z), {}, (s, z))
    finally:
        O.STATE["phase"] = None
    ev = [e for e in _events(capfd.readouterr().err) if e["event"] == "zstats"]
    assert len(ev) == 1 and ev[0]["n"] == 5 and ev[0]["phase"] == "trunk" and "z" not in ev[0] and "s" not in ev[0]


def test_the_block_cadence_and_the_abort_probe_read_their_switches(O, capfd, monkeypatch):
    """OF3O_LOG_EVERY_BLOCKS=k: a `pairformer_block` line on the first 3 blocks and every k-th; OF3O_ABORT_AFTER_BLOCKS=n: the trunk stops by name after
    n blocks with an `abort_probe` record (the wrapper is installed on a stand-in forward here; apply_core installs it on upstream's)."""
    import torch
    PF = pytest.importorskip("openfold3.core.model.latent.pairformer")
    orig = PF.PairFormerBlock.forward
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)                    # the wrapper synchronizes the device per block; no device here
    monkeypatch.setenv("OF3O_LOG_EVERY_BLOCKS", "5"); monkeypatch.setenv("OF3O_ABORT_AFTER_BLOCKS", "7"); monkeypatch.delenv("OF3O_ZSTATS", raising=False)
    z = torch.ones(1, 4, 4, 2)
    try:
        PF.PairFormerBlock.forward = lambda self, *a, **k: a[0]                              # the stand-in: returns its first argument
        O.wrap_pairformer_block()
        O.STATE["block_times"] = []; O.STATE["phase"] = "trunk"
        capfd.readouterr()
        with torch.no_grad():
            for i in range(6):
                assert PF.PairFormerBlock.forward(None, z) is z                             # the wrapped forward hands the block's own output through
            with pytest.raises(RuntimeError, match="OF3O_ABORT_AFTER_BLOCKS probe: stopping after 7 pairformer blocks"):
                PF.PairFormerBlock.forward(None, z)
        ev = _events(capfd.readouterr().err)
        assert [e["n"] for e in ev if e["event"] == "pairformer_block"] == [1, 2, 3, 5]      # the first 3 and every 5th
        ab = [e for e in ev if e["event"] == "abort_probe"]
        assert len(ab) == 1 and ab[0]["blocks"] == 7 and len(ab[0]["block_times"]) == 7 and len(O.STATE["block_times"]) == 7
    finally:
        PF.PairFormerBlock.forward = orig
        O.STATE["block_times"] = []; O.STATE["phase"] = None


PROBE = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
import of3_offload as O
from openfold3.core.model.latent import base_blocks as BB
from openfold3.projects.of3_all_atom import model as M
O.apply_core()
print("PROBE_JSON " + json.dumps({"layer": O.LAYER, "run_trunk": M.OpenFold3.run_trunk.__name__, "rollout": M.OpenFold3._rollout.__name__,
                                  "pairblock_forward_is_stock": BB.PairBlock.forward is O._ORIG["PairBlock.forward"],
                                  "model_forward_is_stock": M.OpenFold3.forward is O._ORIG["OpenFold3.forward"]}))
"""


def _probe(layer):
    """apply_core under OF3O_LAYER=<layer> in a fresh interpreter that sees the packages this process sees (minus the kit's and the units' variables)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OF3", "OPENFOLD3_OB0_OPT", "BFTP_"))}
    env["OF3O_LAYER"] = str(layer)
    env["PYTHONPATH"] = os.pathsep.join([os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))] + [d for d in sys.path if d])
    r = subprocess.run([sys.executable, "-c", PROBE, OF3O], capture_output=True, text=True, env=env, cwd=os.sep, timeout=600)
    rows = [l for l in r.stdout.splitlines() if l.startswith("PROBE_JSON ")]
    return r, (json.loads(rows[0][len("PROBE_JSON "):]) if rows else None), _events(r.stderr)


@pytest.fixture(scope="module")
def upstream():
    pytest.importorskip("torch")
    pytest.importorskip("openfold3.projects.of3_all_atom.model", reason="the openfold3 wheel is not installed: apply_core patches upstream's classes")


def test_layer_2_is_refused_by_name(upstream):
    r, rec, ev = _probe(2)
    assert r.returncode != 0 and rec is None, (r.returncode, r.stdout[-1000:])
    assert "RuntimeError: " + REFUSAL in r.stderr, r.stderr[-3000:]
    assert not any(e["event"] == "applied_core" for e in ev)                                 # refused before the install is recorded


def test_layer_0_times_the_stock_trunk_and_rollout_and_installs_no_lever(upstream):
    r, rec, ev = _probe(0)
    assert r.returncode == 0 and rec is not None, (r.returncode, r.stderr[-3000:])
    assert rec == {"layer": 0, "run_trunk": "run_trunk_timed", "rollout": "rollout_timed", "pairblock_forward_is_stock": True, "model_forward_is_stock": True}, rec
    applied = [e for e in ev if e["event"] == "applied_core"]
    assert len(applied) == 1 and applied[0]["layer"] == 0 and applied[0]["env"].get("OF3O_LAYER") == "0"


def test_layer_1_installs_the_levers_and_no_stock_phase_timer(upstream):
    r, rec, ev = _probe(1)
    assert r.returncode == 0 and rec is not None, (r.returncode, r.stderr[-3000:])
    assert rec == {"layer": 1, "run_trunk": "run_trunk", "rollout": "_rollout", "pairblock_forward_is_stock": False, "model_forward_is_stock": False}, rec

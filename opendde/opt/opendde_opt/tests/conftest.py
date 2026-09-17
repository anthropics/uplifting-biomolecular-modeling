"""The tests import the kit's own modules (the ACCEL shim, fpf_engines, fpf_trimul) from opt/forward/: no bytecode is written into the
carried tree. The tests that ACTIVATE line S1 run the kit's FPF TriMul bytes, which import torch and triton at import: on a box
without them those tests are skipped BY NAME (never red, never silently green)."""
import importlib.util
import os
import sys

import pytest

try:                                                                       # a session without a CUDA device runs the row-sharded line's tri-attention under the core's opt-out
    import torch as _torch                                                 # ROWPAIR_TRIATT_CORE=torch (the carried flash kernel needs a CUDA device; without the opt-out the core
    if not _torch.cuda.is_available():                                     # refuses the lever BY NAME — test_tp_kernels asserts that refusal and manages the variable itself)
        os.environ.setdefault("ROWPAIR_TRIATT_CORE", "torch")
except Exception:                                                          # noqa: BLE001
    pass

from opendde_opt import _core_gate, _producers as _prod


def _gate_refusal():
    """The template gate's NOT ACTIVE line for the opt_core importable here vs the kit's pin, or None on a match."""
    import io
    import os
    try:
        _core_gate.gate(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "__init__.py"), "opendde-opt", stream=io.StringIO())
    except _core_gate.CoreGateRefused as e:
        return e.line.strip()
    return None

sys.dont_write_bytecode = True


@pytest.fixture(autouse=True)
def _chunk_lift_plan_reset():
    """The chunk lever's ceiling plan is process state (cli.pred plans every call): a test that drove `pred` must not compose the next test's lines."""
    from opendde_opt import chunklift
    chunklift._reset()
    yield
    chunklift._reset()


@pytest.fixture(autouse=True)
def _house_lever_state_reset():
    """The LayerNorm-stream lever's bind state and the ablation switch's plan are process state too: a test that armed the loader patch (then had
    its meta-path hook restored away by a fixture) or resolved a line under MODEL_OPT_LEVERS_OFF must not colour the next test's LEVER rows."""
    from opendde_opt import ablate, lnstream, lncore, tmpldedup, keeppool, stepgraph, schedhost, structoksync
    lnstream._reset(); lncore._reset(); ablate._reset(); tmpldedup._reset(); keeppool._reset(); stepgraph._reset(); schedhost._reset(); structoksync._reset()
    from opendde_opt import writer_overlap, prefetch, zprephoist
    writer_overlap._reset(); prefetch._reset(); zprephoist._reset()
    yield
    lnstream._reset(); lncore._reset(); ablate._reset(); tmpldedup._reset(); keeppool._reset(); stepgraph._reset(); schedhost._reset(); structoksync._reset()
    writer_overlap._reset(); prefetch._reset(); zprephoist._reset()


@pytest.fixture(autouse=True)
def _card_memory_unprobed(monkeypatch):
    """big's two size gates default by the RUNNING card's memory (big.gate_default over big.card_memory_mib, probed once per process):
    this box's own GPU — or its absence — decides nothing in a test. Every test starts on the probed, no-GPU memo (the 80 GB card's defaults,
    the values configs/h100.env restates); a test of the card branches sets the memo or the probe (stack.gpu_info) itself."""
    from opendde_opt import big
    monkeypatch.setitem(big._CARD, "probed", True)
    monkeypatch.setitem(big._CARD, "memory_mib", None)
    yield

NEEDS_FPF_BYTES = {                                                        # test name -> why it needs the kit's FPF bytes importable
    "test_exact_arms_the_kits_own_shim_and_wraps_get_default_runner": "activates S1 in-process (fpf_engines.enable_from_env on the kit's bytes)",
    "test_exact_strips_preset_switches_before_the_shim_reads_them": "activates S1 in-process",
    "test_overrides_are_exported_after_the_line_and_recorded": "activates S1 in-process",
    "test_env_route_fires_after_the_trigger_and_arms_the_kit_shim": "activates S1 through the autoload finder in a subprocess",
}
FPF_BYTES_IMPORTABLE = all(importlib.util.find_spec(m) is not None for m in ("torch", "triton"))


def pytest_collection_modifyitems(config, items):
    older = _gate_refusal() or _prod.refusal()                             # a mismatched / older / absent shared core: the package refuses every entry route by name
    if older:                                                              # (_producers); its unit tests are then moot except the gate's own (test_core_gate.py) —
        skip = pytest.mark.skip(reason=f"older core installed: {older} (only tests/test_core_gate.py runs until the re-pin to >= {_prod.MIN_CORE})")
        for item in items:                                                 # skipped BY NAME with the refusal sentence, never red for a reason the gate states
            if "test_core_gate.py" not in str(item.fspath):
                item.add_marker(skip)
    if FPF_BYTES_IMPORTABLE:
        return
    for item in items:
        why = NEEDS_FPF_BYTES.get(item.name)
        if why:
            item.add_marker(pytest.mark.skip(reason=f"{why}; the kit's FPF TriMul bytes import torch and triton, not installed here (opt[test] pins triton; torch = the pinned stack)"))


def pytest_configure(config):
    config.addinivalue_line("markers", "pending_rebase: the test sees the registry's real testing of the pinned opendde (no test testing)")
    config.addinivalue_line("markers", "stack_gate: the test sees the real box-start stack assertion (stack.stack_mismatch) instead of a passing one")


@pytest.fixture(autouse=True)
def weights_digest_memo(tmp_path_factory, monkeypatch):
    """The weights digest memo of this test session lives in a temporary directory of its own (manifest.DIGEST_DIR_ENV), never the box's cache."""
    d = str(tmp_path_factory.mktemp("weights_digest"))
    monkeypatch.setenv("MODEL_OPT_WEIGHTS_DIGEST_DIR", d)
    monkeypatch.delenv("MODEL_OPT_WEIGHTS_SHA256", raising=False)                 # the launcher->rank digest relay starts empty in every test
    monkeypatch.delenv("LAYERNORM_TYPE", raising=False)                          # the stock-side environment a CLI call exports (settings.STOCK_ENV) starts unset in every test and is restored after it
    monkeypatch.setenv("MODEL_OPT_TEST_KERNELS_UNCONFIRMED", "1")                  # lncensus.TEST_UNCONFIRMED_ENV: the stub upstream of this suite carries no opendde.model.triangular.layers, so the
    return d                                                                     # KERNELS census is `unconfirmed` here — reported by name instead of refused (exit 5); must-be-absent on every timing run route


@pytest.fixture(autouse=True)
def tested_pin(request, monkeypatch):
    """Every carried lever tested on the tree's pin (stock/PINS.json) for this test's process — the activation-mechanics tests run as on a
    tested tree. Tests marked ``pending_rebase`` see registry.TESTED_ON as the tree carries it (the gate's own tests). Fresh interpreters
    a test starts are never touched: they test explicitly with ``_stubs.ADMIT_ALL`` or see the real table."""
    from opendde_opt import registry, stack
    from opendde_opt.tests import _stubs
    if not request.node.get_closest_marker("stack_gate"):                       # a CPU test box is not the pinned stack: the assertion passes here unless a test asks for it
        monkeypatch.setattr(stack, "stack_mismatch", lambda tree=None: None)
    if request.node.get_closest_marker("pending_rebase"):
        return None
    monkeypatch.setitem(registry.TESTED_ON, _stubs.PINNED, frozenset(registry.LEVERS))
    for k in list(registry.LINES_ON_HOLD):
        monkeypatch.delitem(registry.LINES_ON_HOLD, k)
    return _stubs.PINNED


@pytest.fixture
def weights_root(tmp_path, monkeypatch):
    """OPENDDE_ROOT_DIR -> a stub frozen-weights root of this test's own (_stubs.weights_root): `pred` passes its boot gate on any box."""
    from opendde_opt.tests import _stubs
    root = _stubs.weights_root(str(tmp_path / "weights_root"))
    monkeypatch.setenv("OPENDDE_ROOT_DIR", root)
    return root


def run_sharded_or_skip(*args, **kwargs):
    """opt_core.mem.rowpair.launch.run_sharded, skipped BY NAME (never red) when gloo's CPU TCP transport cannot
    initialize at all in this sandbox: torch.distributed.init_process_group -> ProcessGroupGloo fails at the OS/
    syscall level with 'Operation not permitted', before any rank ever reaches its entry function. Confirmed
    deterministic across repeated runs and unaffected by GLOO_SOCKET_IFNAME=lo -- a fixed property of this box's
    network policy, not a kit correctness question. Any OTHER RankFailed (a rank that DID reach its entry function
    and hit a real assertion, a different exception, or a timeout) is a genuine result and is re-raised untouched."""
    from opt_core.mem.rowpair import launch
    try:
        return launch.run_sharded(*args, **kwargs)
    except launch.RankFailed as e:
        detail = e.detail or ""
        if "gloo/transport/tcp/device.cc" in detail and "Operation not permitted" in detail:
            last_line = detail.strip().splitlines()[-1] if detail.strip() else str(e)
            pytest.skip(f"gloo's CPU TCP transport is refused at the OS level in this sandbox (rank {e.rank}: {last_line}) "
                        "-- a network-sandbox restriction on this box, not a kit correctness issue")
        raise

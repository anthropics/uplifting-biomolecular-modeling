"""Shared fixtures: every test runs in its own temporary directory that holds stock-shaped input files under the names the tests pass
(``x``, ``in.json``, ``a.json``, ``b.json``, ``FAIL``, ``IMPORT_KIT``: one entry each), so a stub ``pred`` writes the stock output layout for
them and the output census counts it.

Also the ``needs_gloo`` marker: some sandboxes refuse the raw socket option gloo's TCP transport needs (``Operation not permitted`` well
after import time, not caught by an import-only capability check) -- the tests that actually spawn a multi-rank ``gloo`` process group
skip by name on that one signature so a host that can create one still runs them.

And ``needs_durable_install``: the stock route's "clean subprocess" (``protenix_opt.stock_pred``, run ``python -s -m ...`` with PYTHONPATH
stripped of every kit-carrying entry) needs ``protenix_opt`` importable with NO PYTHONPATH at all -- a real ``pip install -e opt``, not
merely reachable via this process's own PYTHONPATH. A sandbox that only ever adds ``opt/`` to PYTHONPATH (never a durable site install)
fails that specific subprocess import; skip by name rather than fake a passing "clean" run. Deliberately NOT fixed by installing a
package-finding ``.pth`` into this interpreter's own site: doing so would leak into any ``venv.EnvBuilder(system_site_packages=True)``
test elsewhere in the suite (see test_rf6_pth_gate.py, which specifically tests venvs that must NOT see the kit pre-installed)."""
import os
import subprocess
import sys
import textwrap

import pytest

from protenix_opt.tests import _stock_stub

STUB_INPUTS = ("x", "in.json", "a.json", "b.json", "FAIL", "IMPORT_KIT")

_GLOO_PROBE = textwrap.dedent("""
    import os
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "0"
    import torch.distributed as dist
    try:
        dist.init_process_group(backend="gloo", rank=0, world_size=1, store=dist.HashStore())
        dist.destroy_process_group()
        print("PROBE_OK")
    except RuntimeError as e:
        print("PROBE_FAIL" if "Operation not permitted" in str(e) else f"PROBE_OTHER:{e}")
""")


def _can_use_gloo() -> bool:
    """Whether this host can actually stand up a ``gloo`` process group (a one-rank probe in a subprocess: real device setup, no
    torch.distributed global state left behind in this process either way)."""
    try:
        r = subprocess.run([sys.executable, "-c", _GLOO_PROBE], capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001 -- any probe failure means "can't use it here"
        return False
    return r.returncode == 0 and r.stdout.strip() == "PROBE_OK"


GLOO_OK = _can_use_gloo()
needs_gloo = pytest.mark.skipif(not GLOO_OK, reason="gloo TCP device setup refuses 'Operation not permitted' in this sandbox (no raw socket option here)")


def _has_durable_protenix_opt_install() -> bool:
    """Whether ``protenix_opt`` is importable in a fresh ``sys.executable`` subprocess with PYTHONPATH removed entirely -- what the
    stock 'clean subprocess' route actually gets (a real ``pip install -e opt``), never what this test process's own PYTHONPATH offers."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        r = subprocess.run([sys.executable, "-s", "-c", "import protenix_opt"], capture_output=True, env=env, timeout=30)
    except Exception:  # noqa: BLE001 -- any probe failure means "can't use it here"
        return False
    return r.returncode == 0


HAS_DURABLE_INSTALL = _has_durable_protenix_opt_install()
needs_durable_install = pytest.mark.skipif(
    not HAS_DURABLE_INSTALL,
    reason="protenix_opt is reachable only via this process's own PYTHONPATH, not a durable site install (pip install -e opt); "
           "the stock 'clean subprocess' route strips PYTHONPATH and needs the real thing",
)


@pytest.fixture(autouse=True)
def _stub_inputs_cwd(tmp_path, monkeypatch):
    for name in STUB_INPUTS:
        _stock_stub.input_json(str(tmp_path / name), names=(name.split(".")[0],))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(_stock_stub.SHORT_ENV, raising=False)

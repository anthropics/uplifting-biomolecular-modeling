"""Environment-dependent skip markers: ``requires_core`` (``boltz2_opt`` + ``opt_core`` actually pip-installed —
findable by a subprocess with no ``PYTHONPATH`` override, not merely reachable via this run's own dev
``PYTHONPATH``) and ``requires_gpu`` (a real CUDA device). A marked test is a named skip on a box that lacks
the precondition, never a silent failure and never deleted — it runs for real wherever the box has both.

Deliberately absent: any marker gated on common/opt_core's own carried-bytes-vs-its-own-sums-manifest
consistency. That is not a per-box environment fact (unlike a missing GPU or a dev-only PYTHONPATH) -- it is
a content fact about a specific, version-pinned dependency, true or false identically on every box that
checks out the same commit. A skip gated on "the content matches" would by construction skip exactly when
the content does NOT match -- i.e. exactly when there is a real defect for the test to report. Tests that
hit this go through as ordinary failures instead."""
import os
import subprocess
import sys

import pytest

_CORE_INSTALLED = None
_GPU_AVAILABLE = None
requires_core = pytest.mark.requires_core   # the markers as importable names (`from .conftest import requires_gpu`): the same named skips
requires_gpu = pytest.mark.requires_gpu


def _core_installed() -> bool:
    """``boltz2_opt``/``opt_core`` findable by a subprocess with no ``PYTHONPATH`` override: genuinely
    pip-installed, not merely reachable via this run's own dev ``PYTHONPATH``. Probed once per session."""
    global _CORE_INSTALLED
    if _CORE_INSTALLED is None:
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        # no -S: site processing must stay on, or site-packages (where any pip/-e install lives) is never on
        # sys.path and this probe would report "not installed" even on a properly provisioned box -- the
        # mirrored tests (test_autoload._run, test_stock_route's subprocess) don't pass -S either.
        r = subprocess.run([sys.executable, "-c", "import boltz2_opt, opt_core"], env=env, capture_output=True, timeout=30)
        _CORE_INSTALLED = r.returncode == 0
    return _CORE_INSTALLED


def _gpu_available() -> bool:
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        try:
            import torch
            _GPU_AVAILABLE = bool(torch.cuda.is_available())
        except Exception:
            _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_core(reason): needs boltz2_opt/opt_core actually pip-installed (not just this run's dev PYTHONPATH)")
    config.addinivalue_line("markers", "requires_gpu(reason): needs a real CUDA device")


def _skip_reason(mark, default):
    return mark.kwargs.get("reason") or (mark.args[0] if mark.args else None) or default


def pytest_collection_modifyitems(config, items):
    for item in items:
        core_mark = item.get_closest_marker("requires_core")
        if core_mark is not None and not _core_installed():
            item.add_marker(pytest.mark.skip(reason=_skip_reason(core_mark, "requires_core: boltz2_opt/opt_core not pip-installed on this box")))
        gpu_mark = item.get_closest_marker("requires_gpu")
        if gpu_mark is not None and not _gpu_available():
            item.add_marker(pytest.mark.skip(reason=_skip_reason(gpu_mark, "requires_gpu: no CUDA device on this box")))

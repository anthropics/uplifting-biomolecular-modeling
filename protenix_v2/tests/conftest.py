"""Shared fixture for this dir: ``needs_durable_opt_core`` -- ``configs/*.env`` computes its JIT cache key by running
``python -m protenix_opt._stackkey`` before any RF-6-specific code runs; that probe's own core-pin gate needs ``opt_core``
importable on whatever ``python`` sources the config (a real ``pip install`` of ``common/opt_core``, not merely reachable via this
process's own PYTHONPATH). A sandbox that never durably installs it fails at that precondition -- skip by name rather than let an
unrelated JIT-key refusal masquerade as the test's own assertion."""
import os
import subprocess
import sys

import pytest


def _has_durable_opt_core_install() -> bool:
    """Whether ``opt_core`` is importable in a fresh ``sys.executable -s`` subprocess with PYTHONPATH removed entirely."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        r = subprocess.run([sys.executable, "-s", "-c", "import opt_core"], capture_output=True, env=env, timeout=30)
    except Exception:  # noqa: BLE001 -- any probe failure means "can't use it here"
        return False
    return r.returncode == 0


HAS_DURABLE_OPT_CORE = _has_durable_opt_core_install()
needs_durable_opt_core = pytest.mark.skipif(
    not HAS_DURABLE_OPT_CORE,
    reason="opt_core is reachable only via this process's own PYTHONPATH, not a durable site install; configs/*.env's own "
           "stack-key probe needs the real thing before any RF-6-specific code runs",
)

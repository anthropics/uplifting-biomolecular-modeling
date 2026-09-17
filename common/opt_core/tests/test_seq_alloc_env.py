"""opt_core.seq.alloc_env: the host malloc tunables (plan, ensure, stamp, line fields); the CUDA allocator policy is opt_core.mem.torch_alloc's."""
import ast
import sys

import pytest

from opt_core.seq import alloc_env as ae


def test_host_malloc_plan_forms():
    T = ae.HOST_MALLOC_TUNABLES
    assert T == {"MALLOC_MMAP_MAX_": "0", "MALLOC_TRIM_THRESHOLD_": "-1", "MALLOC_TOP_PAD_": "536870912"}
    off = ae.plan_host_malloc("KIT_MALLOC", {"KIT_MALLOC": "0"})
    assert off["active"] is False and "KIT_MALLOC=0" in off["source"]
    fresh = ae.plan_host_malloc("KIT_MALLOC", {})
    assert fresh["active"] is True and fresh["source"].startswith("launcher:") and fresh["present"] == {k: None for k in T}
    pre = ae.plan_host_malloc("KIT_MALLOC", dict(T))
    assert pre["active"] is True and pre["source"].startswith("explicit env")
    st = ae.stamp_host_malloc("KIT_MALLOC", {"KIT_MALLOC": "0"})
    assert st == {"tunables": {k: None for k in T}, "active": False, "source": off["source"]}
    assert ae.line_fields(st) == {"host_malloc": "off (" + off["source"] + ")"}
    assert ae.host_malloc_line(st).startswith("host_malloc=off (KIT_MALLOC=0")
    assert ae.line_fields({"active": True, "source": "s"}) == {"host_malloc": "on (s)"}


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc mallopt")
def test_host_malloc_ensure_applies_on_glibc():
    env = {}
    st = ae.ensure_host_malloc("KIT_MALLOC", env)
    assert st["active"] is True and st["mallopt"] == {k: 1 for k in ae.HOST_MALLOC_TUNABLES} and env == ae.HOST_MALLOC_TUNABLES
    assert ae.host_malloc_line(st) == "host_malloc=on (launcher: mallopt in-process + the env exported for exec'd children)"
    assert ae.ensure_host_malloc("KIT_MALLOC", {"KIT_MALLOC": "0"})["mallopt"] is None


def test_cuda_allocator_policy_is_not_duplicated_here():
    assert not any(n in dir(ae) for n in ("cuda_alloc_conf", "apply_cuda_alloc_conf", "AllocConfLate", "ALLOC_CONF_VAR"))
    from opt_core.mem import torch_alloc                     # the one owner of PYTORCH_CUDA_ALLOC_CONF
    assert torch_alloc.ENV == "PYTORCH_CUDA_ALLOC_CONF"


def test_source_parses_at_python_3_8_and_top_imports_are_stdlib():
    src = open(ae.__file__, encoding="utf-8").read()
    ast.parse(src, feature_version=(3, 8))
    tops = [n for n in ast.parse(src).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {(n.module if isinstance(n, ast.ImportFrom) else n.names[0].name) for n in tops}
    assert names <= {"__future__", "os", "typing"}, names

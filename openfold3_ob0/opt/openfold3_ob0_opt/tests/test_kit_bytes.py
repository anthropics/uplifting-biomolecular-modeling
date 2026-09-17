"""The carried kit bytes: the tree's patched served-path modules carry the out-of-memory guard as whole lines (byte identity of a carried
file is the git commit, not a runtime hash re-check); the flash triangle-attention module is the shared core's carried file byte for byte.
"""
import os


from openfold3_ob0_opt.tests import _stubs
from openfold3_ob0_opt.manifest import sha256_file

HOME = _stubs.tree_home()


def sha(p):
    return sha256_file(p, strict=True)


# the tree's PATCHES of carried files: every served-path fallback around a kernel or a CUDA graph re-raises an out-of-memory before it
# reroutes (`from opt_core.oom import is_oom` + `if is_oom(e): raise` as the handler's first statement) — whole lines added, nothing else
# (``test_oom_guard_in_every_patched_module``). <path relative to opt/forward>: the CUDA-graph capture's eager fallback; the trunk-kernels
# add-on reroutes nothing (the pair stacks' triangle attention is the pair cells' — the core's provider by the tier word).
PATCHED = frozenset({                                                 # the served-path modules that carry the out-of-memory guard (opt_core.oom.is_oom): an OOM is re-raised, never rerouted
    "fast_inference/of3_levers/of3_graphs.py",
})


def test_oom_guard_in_every_patched_module():
    """Every served-path module of PATCHED that is in the tree imports the core's out-of-memory predicate and re-raises on it before any
    fallback (`if is_oom(e):` ahead of the reroute) — an out-of-memory error propagates to the caller, it is never served by another route."""
    import re
    root = os.path.join(HOME, "opt", "forward")
    present = [rel for rel in sorted(PATCHED) if os.path.isfile(os.path.join(root, rel))]
    assert present == sorted(PATCHED), [rel for rel in sorted(PATCHED) if rel not in present]   # every served-path module is in the tree
    for rel in present:
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        assert "from opt_core.oom import is_oom" in src and re.search(r"^\s*if is_oom\(\w+\):", src, re.M), rel


def test_no_triangle_attention_module_under_the_addons_name():
    hook = os.path.join(HOME, "opt", "forward", "trunk_kernels", "of3t_hook")
    assert not os.path.exists(os.path.join(hook, "of3t_flash_triattn.py")) and not os.path.exists(os.path.join(hook, "of3t_triatt_route.py"))



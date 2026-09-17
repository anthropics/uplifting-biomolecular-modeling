"""The carried kit bytes: the four carried add-ons ship one stock configuration, the tree's patched served-path modules carry the
out-of-memory guard as whole lines (the bytes of a carried file are the git commit's, not a runtime hash re-check).
"""
import os

from openfold3_opt import modes
from openfold3_opt.tests import _stubs
from openfold3_opt.manifest import sha256_file

HOME = _stubs.tree_home()


def sha(p):
    return sha256_file(p, strict=True)


# the tree's PATCHES of carried files: every served-path fallback around a kernel or a CUDA graph
# re-raises an out-of-memory before it reroutes (`from opt_core.oom import is_oom` + `if is_oom(e): raise` as the handler's first statement) —
# whole lines added, nothing else: dropping the lines that name `is_oom` reproduces the kit's file byte for byte
# (``test_patched_files_carry_the_oom_lines``). <path relative to opt/forward>: the served-path module with a reroute (the CUDA-graph
# capture's eager fallback); the trunk-kernels add-on reroutes nothing (its triangle attention is the pair cells' — the core's provider).
PATCHED = frozenset({
    "fast_inference/of3_levers/of3_graphs.py",
})


def test_the_kits_ship_one_stock_configuration():
    """Every add-on's config/stock_predict.yml is the one kernels-off configuration (modes.KERNELS_OFF_YAML, the fast-inference kit's: the base
    of the fast and big lines): the same settings in every copy; the trunk-kernels add-on's copy differs from the fast-inference kit's in its comment lines only."""
    import yaml
    ref = os.path.join(HOME, modes.KERNELS_OFF_YAML)
    text = {kit: open(os.path.join(HOME, modes.KITS[kit], "config", "stock_predict.yml"), encoding="utf-8").read() for kit in modes.KITS if kit not in modes.PORTS}   # the port ships the pinned-chunk yamls only
    assert all(yaml.safe_load(t) == yaml.safe_load(open(ref, encoding="utf-8")) for t in text.values())

    def body(t):
        return [l for l in t.splitlines() if not l.lstrip().startswith("#")]
    assert all(body(t) == body(open(ref, encoding="utf-8").read()) for t in text.values())
    byte_identical = {kit for kit, t in text.items() if sha(os.path.join(HOME, modes.KITS[kit], "config", "stock_predict.yml")) == sha(ref)}
    assert byte_identical == {"fast_inference"}, byte_identical


def test_patched_files_carry_the_oom_lines():
    """Every served-path module of PATCHED is in the tree and carries the out-of-memory guard as whole lines only (the import and the
    re-raise as each served fallback's first statement) — no hash comparison (the bytes are the git commit's)."""
    root = os.path.join(HOME, "opt", "forward")
    for rel in sorted(PATCHED):
        p = os.path.join(root, rel)
        assert os.path.isfile(p), rel
        lines = open(p, encoding="utf-8").read().splitlines(keepends=True)
        oom = [l for l in lines if "is_oom" in l]
        assert oom and all(("import is_oom" in l) or l.strip().startswith("if is_oom(e): raise") for l in oom), rel
    served = sorted(PATCHED)
    assert len(served) == 1 and sum(open(os.path.join(root, r), encoding="utf-8").read().count("if is_oom(e):") for r in served) == 1
    hook = os.path.join(HOME, modes.KITS["trunk_kernels"], "of3t_hook")
    assert not os.path.exists(os.path.join(hook, "of3t_flash_triattn.py"))                   # no triangle-attention module under the add-on's name: the provider's rows are the core's
    assert "is_oom" not in open(os.path.join(hook, "of3t_levers.py"), encoding="utf-8").read()  # nothing rerouted there



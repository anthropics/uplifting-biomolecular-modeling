"""Per-card rows: configs/a100.env has configs/h100.env's shape less big's two size gates, which the package sizes by the running card's memory
(big.gate_default; no configuration probes the card), and names every per-card difference. The kit carries no architecture-keyed tile row of
its own (the TriMul and triangle-attention rows are the core providers' per-card cells). CPU only."""
import os
import re

from opendde_opt import big, stack

TREE = stack.tree_root()
OPT = os.path.join(TREE, "opt")
ARMT = os.path.join(OPT, "forward", "fast_inference", "levers", "ARMT")


def _exports(path):
    out = {}
    for line in open(path).read().splitlines():
        m = re.match(r"export ([A-Z_]+)=\$\{\1:-([^}]*)\}", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def test_a100_env_has_h100_envs_shape():
    h = _exports(os.path.join(TREE, "configs", "h100.env")); a = _exports(os.path.join(TREE, "configs", "a100.env"))
    gates = {var for var, _default in big.SIZE_GATES.values()}
    assert h and gates <= set(h) and set(a) == set(h) - gates                                   # the same variables, filled when unset — less big's two size gates: a100.env leaves them to the package (big.gate_default: by the running card's memory; a value it exported would win over the card)
    assert {k for k in a if a[k] != h[k]} == {"MODEL_OPT_TARGET_GPU"} and (h["MODEL_OPT_TARGET_GPU"], a["MODEL_OPT_TARGET_GPU"]) == ("H100", "A100")
    assert {var: h[var] for var in gates} == {var: default for var, default in big.SIZE_GATES.values()}   # h100.env restates the package's defaults of a card of 64 GiB and over: an H100 plans identically with or without --config
    body = lambda p: [l for l in open(p).read().splitlines() if l.strip() and not l.lstrip().startswith("#") and not l.startswith("export ")]
    assert body(os.path.join(TREE, "configs", "a100.env")) == body(os.path.join(TREE, "configs", "h100.env"))   # the refusal / JIT-root logic is the same text


def test_no_config_probes_the_card_the_package_sizes_bigs_gates():
    """big's two size gates default by the running card INSIDE the package (big.gate_default: device 0's total memory read the way the package
    reads it, stack.gpu_info — under 64 GiB the 40 GB A100's 1160 / 1856, else 1400 / 2565; test_big_adapter covers both branches): no
    configuration probes the card or computes a gate, a100.env exports neither gate, h100.env restates the >= 64 GiB defaults (the test above)."""
    for f in ("a100.env", "h100.env"):
        code = [l for l in open(os.path.join(TREE, "configs", f)).read().splitlines() if not l.lstrip().startswith("#")]
        assert not any("gpu_info" in l or "nvidia-smi" in l or "_CARD_MIB" in l for l in code), f      # no probe statement in a configuration
    a100 = open(os.path.join(TREE, "configs", "a100.env")).read()
    assert not any(l.startswith(f"export {var}=") or f"{var}:=" in l for l in a100.splitlines() for var, _d in big.SIZE_GATES.values())
    assert "big.py gate_default" in a100 and "1160" in a100 and "1856" in a100                  # the per-card list names where the decision lives and its values
    assert callable(stack.gpu_info) and "memory_mib" in stack.gpu_info()                          # the probe's reading exists (None off a GPU: the 80 GB values)
    assert big.SMALL_CARD_MIB == 64 * 1024 and all(0 < int(big.SMALL_CARD_GATES[lv]) < int(d) for lv, (_v, d) in big.SIZE_GATES.items())   # the 64 GiB class boundary; the 40 GB values below the 80 GB defaults


def test_a100_env_names_every_per_card_difference():
    """configs/a100.env is the one place that lists the per-card differences from H100 (anything not listed = identical)."""
    txt = open(os.path.join(TREE, "configs", "a100.env")).read()
    for name in ("cueq_cache_shipped", "opt_core.kernels.triattn", "opt_core.kernels.trimul"):
        assert name in txt, name
    assert "anything not named here runs the identical code path and lever set as on H100" in txt

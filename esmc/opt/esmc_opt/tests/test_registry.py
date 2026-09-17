"""The kit tree and its registries agree with the carried bytes: the lever registry names a class/tier/switches/dir that exists and is
imported by the mode table; the kits' own environment-switch names (registry.KIT_SWITCHES, ESMC_* / HOST_CLASS_*) are all
accounted for, and PINS.json's forbidden-environment list is their exact union; the kits' root modules import on CPU without torch; the
deterministic recipe lives in the kits' own bytes."""
import json
import os
import re
import subprocess
import sys

import pytest

from esmc_opt import modes, registry, stack

from ._paths import KITS, TREE

ENV_RE = re.compile(r"""(?:os\.environ(?:\.get)?\s*[\(\[]|os\.getenv\s*\(|environ\.get\s*\()\s*["']([A-Z][A-Z0-9_]+)["']""")
OWN_PREFIXES = ("MO_SDK_", "ESMC_", "HOST_CLASS_")


def test_kits_package_imports_without_torch():
    code = ("import sys, importlib; sys.path.insert(0, sys.argv[1]);"
            "[importlib.import_module(m) for m in ('esmc_opt.kits', 'esmc_opt._oom')];"
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-I", "-c", code, os.path.dirname(os.path.dirname(KITS))], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip() == "False"


# ------------------------------------------------------------------------------------------------------------- the lever registry
def test_levers_are_complete():
    for name, lv in registry.LEVERS.items():
        for k in ("dir", "class", "tier", "switches", "what", "apply", "regime", "variants", "route_condition"):
            assert lv.get(k), f"lever {name}: {k} missing"
        assert lv["tier"] in ("exact", "tier2"), (name, lv["tier"])
        assert lv["class"] in ("datapath", "serving", "forward"), (name, lv["class"])
        assert os.path.isdir(os.path.join(KITS, lv["dir"])), f"lever {name}: kit dir {lv['dir']} missing"


def test_lever_order_covers_registry():
    assert set(registry.LEVER_ORDER) == set(registry.LEVERS)


# ------------------------------------------------------------------------------------------------------------------ the mode table
def test_modes_table():
    assert tuple(modes.MODES) == ("off", "exact")             # no fast mode for this model (no tier-2 composition); one stock-class mode
    assert tuple(modes.STOCK_MODES) == ("off",)
    assert modes.DEFAULT_MODE == "exact"
    assert modes.VARIANTS == ("300m", "600m", "6b")
    assert modes.REGIMES == ("b1", "batched") and modes.DEFAULT_REGIME == "b1"


@pytest.mark.parametrize("regime", modes.REGIMES)
@pytest.mark.parametrize("variant", modes.VARIANTS)
def test_exact_is_pipe_plus_fused_at_every_size_in_every_regime(variant, regime):
    r = modes.resolve("exact", variant, regime)
    assert r["levers"] == ("pipe", "fused") and r["levers_out"] == {}, r          # one lever set, every size, every call shape


def test_no_graphs_lever():
    assert "graphs" not in registry.LEVERS and tuple(registry.LEVER_ORDER) == ("pipe", "fused") and modes.KIT_MODES["exact"] == ("pipe", "fused")


def test_three_modes_only():
    with pytest.raises(modes.ResolveError):
        modes.resolve("turbo", "6b", "b1")
    assert set(modes.KIT_MODES) == {"off", "exact"} == set(modes.MODES)
    assert modes.check_variant("esmc_6b") == "6b" and modes.check_variant("ESMC_300m") == "300m" and modes.check_variant("600m") == "600m"   # the SDK's model names name the same sizes
    for m in modes.STOCK_MODES:                                # a stock-class mode resolves to no lever on every (variant, regime)
        for v in modes.VARIANTS:
            assert modes.resolve(m, v, "b1")["levers"] == () and modes.resolve(m, v, "batched")["levers"] == ()
    assert modes.regime_of_batch(1) == "b1" and modes.regime_of_batch(8) == "batched"


def test_unknown_mode_variant_regime_refused():
    for bad in (("turbo", "6b", "b1"), ("exact", "7b", "b1"), ("exact", "6b", "b7")):
        with pytest.raises(modes.ResolveError):
            modes.resolve(*bad)


def test_describe_names_every_lever_in_and_out():
    for v in modes.VARIANTS:
        for rg in modes.REGIMES:
            d = modes.describe("exact", v, rg)
            assert v in d and rg in d
            r = modes.resolve("exact", v, rg)
            for lv in r["levers"]:
                assert lv in d
            for lv in r["levers_out"]:
                assert lv in d


# ------------------------------------------------------------------------------------------------------------- the switch names
def _kit_env_names():
    names = {}
    for dp, dn, fn in os.walk(KITS):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dp, f)
            for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
                for m in ENV_RE.finditer(line):
                    names.setdefault(m.group(1), []).append(f"{os.path.relpath(p, KITS)}:{i}")
    return names


def test_registry_switches_cover_the_kits_own_names():
    names = _kit_env_names()
    own = {n for n in names if n.startswith(OWN_PREFIXES)}
    allowed = set(registry.KIT_SWITCHES) | set(stack.PACKAGE_ENV) | set(registry.KIT_BOOKKEEPING_ENV)
    missing = sorted(n for n in own if n not in allowed)
    assert not missing, {n: names[n] for n in missing}
    for n in registry.KIT_SWITCHES:          # every registered switch is read by the carried bytes (the registry is not a wish list)
        assert n in names or n == registry.KIT_ENV, n


def test_pins_must_be_absent_is_the_union():
    p = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
    se = p["stock_environment"]
    union = set(registry.KIT_SWITCHES) | set(stack.PACKAGE_ENV) | set(se.get("numerics_must_be_absent") or [])   # the kits' switches + the package's one env name + the numerics var
    assert "HF_HOME" in (se.get("allowed_exceptions") or []), "the weights location (upstream's own HF_HOME) passes into the stock arm"
    assert set(se["must_be_absent"]) == union, sorted(set(se["must_be_absent"]) ^ union)
    assert "must_be_absent" not in p, "one list: stock_environment.must_be_absent"
    assert "env" not in p.get("deterministic_recipe", {}), "PINS.json points at it"


# ------------------------------------------------------------------------------------------------- the fused kit's per-shape composition
def test_fused_composition_per_shape():
    """fused SERVED_CONFIGS / ITEM_SERVES: the three ESM C shapes are served; the composition is fixed per hidden size — 6B: every seam;
    960 / 1152 / 2560: every seam (the residual+LN kernel mirrors the TE config of each width; the q/k kernel the ragged ATen form)."""
    import importlib
    F = importlib.import_module("esmc_opt.kits.fused")
    assert set(F.SERVED_CONFIGS) == set(modes.VARIANTS)
    on6, out6 = F.composition_for(2560)
    assert [m.rsplit(".", 1)[-1] for m, _ in on6] == ["residual_ln", "qk_rotary", "thin"] and out6 == []
    assert dict(on6)["esmc_opt.kits.residual_ln"] == ("residual", "residual_ln_qkv")
    for h in (960, 1152):
        on, out = F.composition_for(h)
        assert [m.rsplit(".", 1)[-1] for m, _ in on] == ["residual_ln", "qk_rotary", "thin"], on
        assert dict(on)["esmc_opt.kits.residual_ln"] == ("residual", "residual_ln_qkv")
        assert out == [], out
    on, out = F.composition_for(512)                                                              # below TE's <1024,4,1,16> range: the LN seam named out by shape
    assert [n for n, _ in out] == ["residual_ln.residual_ln_qkv"], out

    class _Cfg:
        def __init__(self, **kw): self.__dict__.update(kw)

    class _M:
        def __init__(self, **kw): self.config = _Cfg(**kw)
    g = F.model_gate(_M(hidden_size=1152, num_hidden_layers=36, num_attention_heads=18))
    assert not g["refused"] and g["size"] == "600m" and [m.rsplit(".", 1)[-1] for m, _ in g["composition"]] == ["residual_ln", "qk_rotary", "thin"]
    g = F.model_gate(_M(hidden_size=1280, num_hidden_layers=33, num_attention_heads=20))          # not an ESM C shape: refused by name, nothing composed
    assert g["refused"] and g["composition"] == [] and "serves the ESM C shapes" in g["reasons"][0]

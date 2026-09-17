"""The shared core (opt_core) as this package adopts it: the pin in pyproject.toml is a version floor the imported core must meet, the
build backend is the core's template byte-for-byte, the path shim resolves the pinned copy, and a stock process holds no core module
beyond the proof machinery (opt_core.stock_proof.CORE_ALLOWED_IN_STOCK). The kit's own lines and manifest keys are locked by the other
test files; this one locks the adoption's seams."""
import os
import re
import subprocess
import sys

from protenix_opt import _core, stack                                   # first: _core makes opt_core importable when it is not installed
import opt_core
from opt_core import gates, stock_proof

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # protenix_v2/opt


def test_pin_is_the_imported_core():
    """The kit's ONE pin gate (_core.gate = the shared template's _core_gate on the kit's pyproject) passes in this environment: the
    installed core's version is at least the pinned floor, and the core's own live comparison (opt_core.gates.core_pin_check) agrees
    on the stable facts (version) that both sides expose regardless of what else core_pin_check.details carries."""
    facts = _core.gate()
    pinned = facts["pinned"]
    assert pinned["path"] and pinned["version"]
    assert facts["installed"]["version"] and facts["installed"]["package_dir"]
    assert gates.version_tuple(facts["installed"]["version"]) >= gates.version_tuple(pinned["version"])
    g = gates.core_pin_check(os.path.join(OPT, "pyproject.toml"))
    assert g.ok, g.reason
    assert g.details["pinned"]["version"] == pinned["version"] and g.details["imported"]["version"] == facts["installed"]["version"]


def test_pinned_path_is_the_release_tree_copy():
    p = _core.pinned_path()
    assert os.path.normpath(p) == os.path.normpath(os.path.join(OPT, "..", "..", "common", "opt_core")), p
    assert os.path.isfile(os.path.join(p, "opt_core", "__init__.py")), "the pinned copy is beside the tree (common/opt_core)"


def test_gpu_probe_dict_keeps_the_kit_shape(monkeypatch):
    monkeypatch.setattr(gates, "nvidia_smi_probe", lambda keys=None: {k: {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0", "sm": "sm90", "memory_mib": 81559, "probe": "nvidia-smi"}[k] for k in keys})
    assert list(stack.gpu_probe_smi()) == list(stack.GPU_KEYS) == ["name", "sm", "cc", "probe"]


def test_stock_process_holds_only_the_proof_machinery(tmp_path):
    """`python -s -m protenix_opt.stock_pred` (the stock child) imports opt_core and opt_core.stock_proof — nothing else of the core."""
    code = ("import sys, protenix_opt.stock_pred; "
            "print(sorted(m for m in sys.modules if m == 'opt_core' or m.startswith('opt_core.')))")
    env = dict({k: v for k, v in os.environ.items() if not k.startswith("PROTENIX_OPT")}, PYTHONPATH=OPT, PYTHONDONTWRITEBYTECODE="1")   # the stock route's environment: the kit's switches stripped
    r = subprocess.run([sys.executable, "-s", "-c", code], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-800:]
    assert eval(r.stdout.strip()) == sorted(stock_proof.CORE_ALLOWED_IN_STOCK), r.stdout



def test_core_served_kernels_route_to_the_core_copies_and_the_tree_carries_none():
    """KERNEL_ROUTES: the block prologue / epilogue / fused transition / glue v2 / MK-PF / K2B packages (kits.CORE_SERVED_KERNELS) are served from
    the core's carried copies — resolved by name to opt_core/kernels/<name>, every file the core's own sums() names as carried is present there
    (nothing re-hashed) and no copy under the kit's third_party."""
    from opt_core import kernels
    from protenix_opt import kits
    assert set(kits.CORE_SERVED_KERNELS) <= set(stack.KERNEL_ROUTES)
    fpf_home = stack.kit_home()
    meta = list(sys.meta_path); env0 = dict(os.environ)
    try:
        out = stack._route_kernels(fpf_home)
        for name in kits.CORE_SERVED_KERNELS:
            k = out[name]
            assert k["ok"], k["reason"]
            core_copy = os.path.join(_core.pinned_path(), "opt_core", "kernels", name)
            assert os.path.realpath(k["resolved"]) == os.path.realpath(k["core_copy"]) == os.path.realpath(core_copy), k
            assert k["routed"] and not k["already_imported"]
            assert not os.path.exists(os.path.join(fpf_home, "third_party", name)), f"third_party/{name}: the tree carries no copy (kits.CORE_SERVED_KERNELS)"
            files = kernels.sums(name)["files"]
            assert files, (name, "a carried kernel with no files")
            for rel in files:
                assert os.path.isfile(os.path.join(core_copy, rel)), (name, rel)
        for name in ("fpf_glue_v2", "fpf_mkpf"):                                   # their run-time imports are the routed core copies too
            ri = out[name]["runtime_imports"]
            assert set(ri) == {"fpf_triatt_pro", "fpf_triatt_epi", "fpf_transition"} and all(ri.values()), ri
        k2b_cells = os.path.join(kits.CORE_KERNELS_DIR, "fpf_triatt_k2b", "K2B_CELLS.json")   # K2B's launch-cell table: the core copy's, exported at activation
        assert out["fpf_triatt_k2b"]["exports"] == {"PF_TRIATTN_TABLE": k2b_cells} and os.environ["PF_TRIATTN_TABLE"] == k2b_cells and os.path.isfile(k2b_cells)
        os.environ["PF_TRIATTN_TABLE"] = "/caller/own_cells.json"                    # a caller's pre-set table wins (stack.CALLER_EXPORTS) and is the recorded export
        sys.meta_path[:] = meta
        assert stack._route_kernels(fpf_home)["fpf_triatt_k2b"]["exports"] == {"PF_TRIATTN_TABLE": "/caller/own_cells.json"} and os.environ["PF_TRIATTN_TABLE"] == "/caller/own_cells.json"
    finally:
        sys.meta_path[:] = meta; os.environ.clear(); os.environ.update(env0)
    for name in kits.CORE_SERVED_KERNELS:
        assert name not in sys.modules, "the check resolves without importing"


def test_kernel_census_reads_a_late_import():
    """``kernels.<name>.imported_from`` is re-read from sys.modules at reconcile time: a kernel the levers import lazily (fpf_trimul_v4 at
    the first served call) reads None at activation end and its file afterwards."""
    import types
    name = "protenix_opt_test_late_kernel"
    rep = {"mode": "fast", "active": True, "kernels": {name: {"ok": True, "imported_from": None}}}
    assert stack.kernel_census(dict(rep))["kernels"][name]["imported_from"] is None
    mod = types.ModuleType(name); mod.__file__ = "/routed/copy/__init__.py"
    sys.modules[name] = mod
    try:
        assert stack.kernel_census(rep)["kernels"][name]["imported_from"] == "/routed/copy/__init__.py"
        rep["kernels"][name]["imported_from"] = None
        assert stack.reconcile({"kernels": {name: {"imported_from": None}}, "levers_applied": []}, {})["kernels"][name]["imported_from"] == "/routed/copy/__init__.py"
    finally:
        del sys.modules[name]


def test_unit_labels_are_the_units_own():
    """kits.KITS labels are the units' own version words: FlashPairformer's is the `kit=` token of env.sh's KIT_SPEC line, PTX_TP's carries
    the ptx_tp package version."""
    import re as _re
    from protenix_opt import kits, tp
    env_sh = open(os.path.join(kits.kit_dir("flashpairformer"), "env.sh"), encoding="utf-8").read()
    assert _re.search(r"KIT_SPEC kit=(\S+)", env_sh).group(1) == kits.KITS["flashpairformer"].label
    init = open(os.path.join(tp.unit_dir(), "ptx_tp", "__init__.py"), encoding="utf-8").read()
    assert kits.KITS["PTX_TP"].label.endswith("_v" + _re.search(r'__version__ = "([^"]+)"', init).group(1))

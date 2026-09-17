"""The K2B (fpf_triatt_k2b/K2B_CELLS.json) and GLUE (fpf_glue_v2/GLUE_CELLS.json) tables select, for every compute capability, device name and triton the
tables name, exactly the cells opt_core 0.5.18.11 selected: the fixtures hold that release's load-bearing projection of each table (the keys the loaders
read), and the loaders' own resolution statements are run over both.  CPU-only: no device, no triton.

Loader facts this test pins (read them there):
- fpf_triatt_k2b/triatt_k2b.py `_pf_resolve_cells`: `by_cc[<cc>]` -> `entries[<key>]`; else `by_name` substring of the device name; a table without
  `by_cc` resolves top-level device-name keys not starting with '_' (v1).  `_pf_load_device_table` then reads `entry[bf16|fp32][<D>]` = [(max_seq, cfg)]
  and prints `status[:60]`.  Nothing under `__doc__` is read.
- fpf_glue_v2/__init__.py `resolve_cells`: `by_cc_triton['<cc>|<M.m>']`, else `by_cc['<cc>']` (named unverified when the triton is not in
  `triton_tested`), -> `entries[<entry>]` families that carry a `cfg` (their `cfg` / `shape` are what `install` launches).  `schema`, `image`,
  `device`, `note`, `what`, `certificate`, `fallback_v1_hazard` are not read."""
import ast
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), "opt_core")
K2B_JSON = os.path.join(CORE, "kernels", "fpf_triatt_k2b", "K2B_CELLS.json")
K2B_SRC = os.path.join(CORE, "kernels", "fpf_triatt_k2b", "triatt_k2b.py")
GLUE_JSON = os.path.join(CORE, "kernels", "fpf_glue_v2", "GLUE_CELLS.json")
K2B_RELEASED = os.path.join(HERE, "fixtures", "k2b_cells_selection_0_5_18_11.json")
GLUE_RELEASED = os.path.join(HERE, "fixtures", "glue_cells_selection_lifted_rows.json")   # + the 8.0|3.7 entry sm80_t37 lifted from a kit A100 pass
DEVICE_NAMES = ("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA H100 80GB HBM3", "NVIDIA H200", "NVIDIA B200", "NVIDIA B300", "NVIDIA GB200",
                "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA L40S", "NVIDIA GeForce RTX 4090", "Tesla T4")
CCS = ((7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3), (12, 0), (12, 1))


def _k2b_resolver():
    """`_pf_resolve_cells` compiled from the loader's own source (the module imports triton at import time; the function is pure)."""
    tree = ast.parse(open(K2B_SRC).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pf_resolve_cells")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), K2B_SRC, "exec"), ns)
    return ns["_pf_resolve_cells"]


def _served(entry):
    """What `_pf_load_device_table` builds from an entry: {dtype: {D: [(max_seq, cfg minus '_' keys)]}} (its lines after the resolution)."""
    if entry is None:
        return None
    out = {}
    for dt_name in ("bf16", "fp32"):
        for d_str, entries in (entry.get(dt_name) or {}).items():
            out.setdefault(dt_name, {})[int(d_str)] = [(int(ms), {k: v for k, v in dict(cfg).items() if not str(k).startswith("_")}) for ms, cfg in entries]
    return out


def test_k2b_cells_select_the_released_cells_for_every_capability_and_device_name():
    resolve = _k2b_resolver()
    shipped = json.load(open(K2B_JSON)); released = json.load(open(K2B_RELEASED))
    released_v2 = {"by_cc": released["by_cc"], "by_name": released["by_name"], "entries": released["entries"]}
    assert shipped["by_cc"] == released["by_cc"] and shipped["by_name"] == released["by_name"]
    assert {e: {k: v for k, v in ent.items() if k != "status"} for e, ent in shipped["entries"].items()} == released["entries"]
    n = 0
    for cc in CCS:
        for name in DEVICE_NAMES:
            k_new, e_new, how_new = resolve(shipped, cc, name); k_old, e_old, how_old = resolve(released_v2, cc, name)
            assert (k_new, how_new) == (k_old, how_old) and _served(e_new) == _served(e_old), (cc, name, k_new, k_old)
            n += 1
    # a v1 reader (a table handed over without by_cc): top-level device-name keys
    v1_new = {k: v for k, v in shipped.items() if k not in ("by_cc", "by_name", "entries", "schema")}
    v1_old = dict(released["v1"], __doc__={"_": "not read: starts with '_'"})
    for name in DEVICE_NAMES:
        k_new, e_new, how_new = resolve(v1_new, (9, 0), name); k_old, e_old, how_old = resolve(v1_old, (9, 0), name)
        assert (k_new, how_new) == (k_old, how_old) and _served(e_new) == _served(e_old), (name, k_new, k_old)
    assert n == len(CCS) * len(DEVICE_NAMES)


def test_glue_cells_select_the_released_cells_for_every_capability_and_triton(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from opt_core import kernels as CK
    CK.route("fpf_glue_v2")                                   # the package under its routed top-level name (torch only; triton is imported lazily)
    import fpf_glue_v2 as G
    released = json.load(open(GLUE_RELEASED))
    old_path = str(tmp_path / "released.json"); json.dump(released, open(old_path, "w"))
    shipped = json.load(open(GLUE_JSON))
    assert shipped.get("by_cc_triton", {}) == released["by_cc_triton"] and shipped.get("by_cc", {}) == released["by_cc"] and shipped.get("triton_tested", []) == released["triton_tested"]
    tritons = sorted(set(released["triton_tested"]) | {k.split("|")[1] for k in released["by_cc_triton"]} | {"3.3", "3.6", "3.7", "3.9"})
    monkeypatch.delenv("FPF_GLUE_V2_ALLOW_TRITON", raising=False)
    n = 0
    for cc in ("%d.%d" % c for c in CCS):
        for tmm in tritons:
            new = G.resolve_cells(path=GLUE_JSON, cc=cc, triton=tmm); why_new = G._STATE["why"]
            old = G.resolve_cells(path=old_path, cc=cc, triton=tmm); why_old = G._STATE["why"]
            assert {f: (v["cfg"], tuple(v.get("shape") or ())) for f, v in new.items()} == {f: (v["cfg"], tuple(v.get("shape") or ())) for f, v in old.items()}, (cc, tmm)
            assert why_new.split(" (unknown capability")[0] == why_old.split(" (unknown capability")[0], (cc, tmm, why_new, why_old)
            n += 1
    assert n == len(CCS) * len(tritons)

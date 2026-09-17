"""Carried kernels: the copy is the sums file's bytes; routing is by NAME (a route exposes exactly one name, never the directory); a kept
copy is never shadowed; a neighbouring version is refused by name; the required exports are gated before import. No torch: nothing here
imports a kernel — the routed imports are exercised through stub packages under the same names."""
import os
import shutil
import sys
import textwrap
import types

import pytest

from opt_core import kernels

NAMES = kernels.names()


@pytest.fixture(autouse=True)
def clean_routes(monkeypatch):
    """Every test starts with no route and no kernel name in sys.modules; whatever it routes is forgotten afterwards."""
    monkeypatch.setattr(sys, "meta_path", [f for f in sys.meta_path if not isinstance(f, kernels.RouteFinder)])
    for n in NAMES + ["fpf_trimul_v4.kernels"]:
        monkeypatch.delitem(sys.modules, n, raising=False)
    yield
    for n in NAMES + ["fpf_trimul_v4.kernels"]:                 # a kit copy a test imported from its tmp_path must not outlive the test: later suites
        sys.modules.pop(n, None)                                # hold `import <name>` to the core copy (kernels.verify_carry)


def test_the_carried_kernels():
    assert NAMES == ["apb", "apb_attn", "atom_window", "dtk_kernels", "flash_triattn", "fpf_glue_v2", "fpf_mkpf", "fpf_pallas", "fpf_pallas_f32", "fpf_transition",
                     "fpf_transition_v2", "fpf_triatt_epi", "fpf_triatt_k2b", "fpf_triatt_pro", "fpf_trimul", "fpf_trimul_rows", "fpf_trimul_v4", "gather_attn", "ln", "ln_proj",
                     "lnl_fused", "pallas", "pallas_attn", "pallas_glut", "pallas_triatt", "rfd_layernorm", "templ_embed", "transition", "triattn", "triattn_exact", "triattn_xla", "trimul",
                     "trimul_esm_shapes", "trimul_xla"]


@pytest.mark.parametrize("name", NAMES)
def test_core_copy_matches_its_own_metadata(name, tmp_path):
    assert kernels.verify_carry(name) == []                    # nothing routed/imported: falls back to the core's own copy, held to itself -- vacuous
    # on its own (core vs itself can never differ) -- the behavioural check is a REAL neighbour, deliberately mutated, held by explicit path:
    doc = kernels.sums(name)
    core = kernels.carried_path(name)
    if doc["kind"] == "package":
        dest = os.path.join(str(tmp_path), name)                            # resolve() finds a PACKAGE by <path-entry>/<name>/
        shutil.copytree(core, dest, ignore=shutil.ignore_patterns("__pycache__"))
        target = os.path.join(dest, sorted(f for f in doc["files"] if f.endswith(".py"))[0])
    else:
        target = os.path.join(str(tmp_path), name + ".py")                  # resolve() finds a MODULE by <path-entry>/<name>.py directly
        shutil.copy2(core, target)
    with open(target, "a") as fh:
        fh.write("\n# mutated for this test\n")
    problems = kernels.verify_carry(name, [str(tmp_path)])
    assert problems and any("differs" in p for p in problems), (name, problems)   # the mutation IS caught -- verify_carry is not a no-op
    doc = kernels.sums(name)
    root = kernels.carried_path(name)
    if doc["kind"] == "package":
        on_disk = sorted(os.path.relpath(os.path.join(r, f), root) for r, ds, fs in os.walk(root) for f in fs if "__pycache__" not in r)
        assert on_disk == sorted(doc["files"]), "a file inside a carried package that the sums file does not list"
        for f in doc["not_carried"]:
            assert not os.path.exists(os.path.join(root, f))
    else:                                                                        # a module: <name>.py plus its sidecars (<name>.LICENSE, <name>.NOTICE ...) beside META/
        assert os.path.isfile(root) and name + ".py" in doc["files"]
        assert all(f.startswith(name + ".") and os.path.isfile(os.path.join(os.path.dirname(root), f)) for f in doc["files"]), doc["files"]
    assert set(doc) >= {"kind", "version", "license", "files", "not_carried", "exports", "runtime_imports", "mechanism"}
    for n in doc["runtime_imports"]:
        assert n in NAMES


def test_sums_files_name_no_addon_record_words():
    for name in NAMES:
        doc = kernels.sums(name)
        assert "kit_of" + "_record" not in doc and "twins" not in doc


def test_route_exposes_exactly_one_name(tmp_path):
    empty = [str(tmp_path)]
    assert all(kernels.resolve(n, empty) is None for n in NAMES)
    kernels.route("fpf_trimul_v4")
    assert kernels.routed() == ["fpf_trimul_v4"]
    assert kernels.resolve("fpf_trimul_v4", empty) == kernels.carried_path("fpf_trimul_v4")
    assert kernels.resolve("fpf_trimul", empty) is None and kernels.resolve("flash_triattn", empty) is None
    assert sum(isinstance(f, kernels.RouteFinder) for f in sys.meta_path) == 1 and isinstance(sys.meta_path[0], kernels.RouteFinder)
    kernels.route("flash_triattn")
    assert kernels.routed() == ["flash_triattn", "fpf_trimul_v4"] and sum(isinstance(f, kernels.RouteFinder) for f in sys.meta_path) == 1
    assert kernels.unroute("flash_triattn") is True and kernels.unroute("flash_triattn") is False and kernels.routed() == ["fpf_trimul_v4"]


def test_route_check_ok_with_or_without_the_kit_table(tmp_path, monkeypatch):
    kernels.route("fpf_trimul_v4")
    monkeypatch.delenv("FPF_TRIMUL_V4_CELLS", raising=False)
    g = kernels.route_check("fpf_trimul_v4", path=[str(tmp_path)])                          # routed, no kit table exported: the package's own table serves every part
    assert g.ok, g.reason
    assert g.details["exports"] == []
    monkeypatch.setenv("FPF_TRIMUL_V4_CELLS", str(tmp_path / "missing.json"))               # an exported table that does not exist is still a named problem
    g = kernels.route_check("fpf_trimul_v4", path=[str(tmp_path)])
    assert not g.ok and "no such file" in g.reason
    monkeypatch.setenv("FPF_TRIMUL_V4_CELLS", __file__)                                      # any existing file satisfies the gate (the loader parses it)
    g = kernels.route_check("fpf_trimul_v4", path=[str(tmp_path)])
    assert g.ok, g.reason
    assert g.details["differing"] == [] and g.details["missing"] == [] and g.details["routed"] and g.details["exports"] == []
    assert g.details["resolved"] == kernels.carried_path("fpf_trimul_v4") and not g.details["already_imported"]
    assert g.details["runtime_imports"] == {}                              # fpf_trimul_v4 imports no other carried kernel at run time
    assert "fpf_trimul_v4" not in sys.modules and "fpf_trimul" not in sys.modules, "route_check imported a kernel"


def test_route_check_reports_a_routed_runtime_import(tmp_path):
    kernels.route("fpf_glue_v2")
    kernels.route("fpf_transition")
    g = kernels.route_check("fpf_glue_v2", path=[str(tmp_path)])
    assert g.ok and g.details["runtime_imports"] == {"fpf_triatt_pro": None, "fpf_triatt_epi": None,        # not routed, not on the path
                                                     "fpf_transition": kernels.carried_path("fpf_transition")}


@pytest.mark.parametrize("name", ["flash_triattn", "fpf_trimul"])
def test_route_check_without_exports_is_ok(name, tmp_path):
    kernels.route(name)
    g = kernels.route_check(name, path=[str(tmp_path)])
    assert g.ok and g.details["exports"] == [] and g.details["resolved"] == kernels.carried_path(name)


def test_unrouted_name_is_not_importable(tmp_path):
    g = kernels.route_check("fpf_trimul", path=[str(tmp_path)])
    assert not g.ok and "not importable" in g.reason and "kernels.route('fpf_trimul')" in g.reason


def _kit_copy(tmp_path, name, mutate):
    """A kit's own copy of a package kernel beside the core's: the same bytes except `mutate` (relpath -> bytes)."""
    kit = tmp_path / "kit"
    kit.mkdir(exist_ok=True)
    shutil.copytree(kernels.carried_path(name), kit / name, ignore=shutil.ignore_patterns("__pycache__"))
    for rel, data in mutate.items():
        (kit / name / rel).write_bytes(data)
    return kit


def test_a_kept_copy_is_never_shadowed(tmp_path, monkeypatch):
    """The shadowing case: the kit keeps its own fpf_transition (different transition.py) and routes only fpf_glue_v2, which imports it at run time."""
    kit = _kit_copy(tmp_path, "fpf_transition", {"transition.py": b"# the kit's own version\n"})
    monkeypatch.syspath_prepend(str(kit))
    kernels.route("fpf_glue_v2")
    assert kernels.resolve("fpf_transition") == str(kit / "fpf_transition")              # the kit's copy, not the core's
    g = kernels.route_check("fpf_glue_v2")
    assert g.ok and g.details["runtime_imports"]["fpf_transition"] == str(kit / "fpf_transition")
    import fpf_transition                                                                   # a real import of the kit's copy (no torch: __init__ is empty)
    assert os.path.dirname(os.path.abspath(fpf_transition.__file__)) == str(kit / "fpf_transition")
    g = kernels.route_check("fpf_transition")                                               # asking the core about the kit's copy names the difference
    assert not g.ok and g.details["differing"] == ["transition.py"] and g.details["already_imported"] and not g.details["routed"]


def test_route_wins_over_a_neighbouring_copy_on_sys_path(tmp_path, monkeypatch):
    kit = _kit_copy(tmp_path, "fpf_trimul", {"kernels.py": b"# neighbour\n"})
    monkeypatch.syspath_prepend(str(kit))
    kernels.route("fpf_trimul")
    assert kernels.resolve("fpf_trimul") == kernels.carried_path("fpf_trimul")
    import fpf_trimul
    assert os.path.dirname(os.path.abspath(fpf_trimul.__file__)) == kernels.carried_path("fpf_trimul")
    assert fpf_trimul.__path__ == [kernels.carried_path("fpf_trimul")]
    g = kernels.route_check("fpf_trimul")
    assert g.ok and g.details["already_imported"] and g.details["routed"]


def test_route_refuses_a_neighbouring_version_by_name(tmp_path, monkeypatch):
    other = tmp_path / "fpf_trimul_v4"
    shutil.copytree(kernels.carried_path("fpf_trimul_v4"), other, ignore=shutil.ignore_patterns("__pycache__"))
    with open(other / "trimul.py", "ab") as fh:
        fh.write(b"\n# one more byte\n")
    os.remove(other / "kexp.py")
    monkeypatch.setenv("FPF_TRIMUL_V4_CELLS", __file__)
    g = kernels.route_check("fpf_trimul_v4", path=[str(tmp_path)])                          # not routed: the neighbour resolves
    assert not g.ok and g.details["resolved"] == str(other)
    assert g.details["differing"] == ["trimul.py"] and g.details["missing"] == ["kexp.py"]
    assert "differing [trimul.py]" in g.reason and "missing [kexp.py]" in g.reason and kernels.sums("fpf_trimul_v4")["version"] in g.reason


def test_route_refuses_a_differing_single_module(tmp_path):
    (tmp_path / "flash_triattn.py").write_bytes(open(os.path.join(kernels.KERNELS_DIR, "flash_triattn.py"), "rb").read() + b"\n")
    g = kernels.route_check("flash_triattn", path=[str(tmp_path)])
    assert not g.ok and g.details["differing"] == ["flash_triattn.py"]


def test_already_imported_copy_answers_with_its_own_location(tmp_path, monkeypatch):
    identical = tmp_path / "fpf_trimul"
    shutil.copytree(kernels.carried_path("fpf_trimul"), identical, ignore=shutil.ignore_patterns("__pycache__"))
    stub = types.ModuleType("fpf_trimul")
    stub.__file__ = str(identical / "__init__.py")
    monkeypatch.setitem(sys.modules, "fpf_trimul", stub)
    g = kernels.route_check("fpf_trimul")
    assert g.ok and g.details["already_imported"] and g.details["resolved"] == str(identical)
    (identical / "kernels.py").write_bytes(b"# not the carried bytes\n")
    g = kernels.route_check("fpf_trimul")
    assert not g.ok and g.details["differing"] == ["kernels.py"]


def test_exports_contract():
    assert kernels.exports("fpf_trimul_v4") == {}                                             # the kit table is optional: a kit with none exports nothing
    env = kernels.exports("fpf_trimul_v4", cells="rel/cells.json")
    assert env == {"FPF_TRIMUL_V4_CELLS": os.path.abspath("rel/cells.json")}
    assert kernels.exports("flash_triattn") == {} and kernels.exports("fpf_trimul") == {}
    assert kernels.exports_check("fpf_trimul_v4", {}) == []
    assert kernels.exports_check("fpf_trimul_v4", {"FPF_TRIMUL_V4_CELLS": __file__}) == []
    assert kernels.sums("fpf_trimul_v4")["not_carried"] == {}                               # table.json is carried: the package's one table; the export is the kit's OWN table
    with pytest.raises(KeyError):
        kernels.sums("no_such_kernel")


def test_carried_files_are_exempt_but_the_routing_module_is_house_written():
    carried = kernels.carried_files()
    assert "fpf_trimul_v4/trimul.py" in carried and "flash_triattn.py" in carried and "__init__.py" not in carried
    text = open(os.path.join(kernels.KERNELS_DIR, "__init__.py"), encoding="utf-8").read()
    assert "import torch" not in text and "import triton" not in text
    assert not any(tok in text for tok in ("sys.path.insert", "sys.path.append", "sys.path[", "sys.path ="))    # routing never touches sys.path


def test_lnl_fused_reads_its_carried_tile_table_by_default():
    """The module's default tile-table path is the carried data file beside it (the PF_LNL_TILES export is an optional override, not load-bearing)."""
    src = open(os.path.join(kernels.KERNELS_DIR, "lnl_fused.py"), encoding="utf-8").read()
    assert '"lnl_fused.tiles_by_arch.json"' in src
    assert os.path.isfile(os.path.join(kernels.KERNELS_DIR, "lnl_fused.tiles_by_arch.json"))
    spec = kernels.sums("lnl_fused")["exports"]["PF_LNL_TILES"]
    assert spec["path"] is True and spec["required"] is False


def test_lnl_fused_serves_pinned_tiles_and_never_benchmarks_outside_the_sweep():
    """QoL rule (no runtime tuning): lnl_fused's two autotuned kernels take their tile from the carried per-arch table WITHOUT a search --
    the autotuner's cache is a _PinnedTiles that answers every key (the row's config for a key it names, the row's principal config
    otherwise); no row for the capability = the module's default tile alone; only PF_LNL_AUTOTUNE_ALL=1 (the sweep that writes the table)
    searches.  Exercised here on the pure-Python pieces (no triton): _PinnedTiles + _row_pins against the shipped table's 9.0 / 8.0 rows."""
    import ast, json
    KERNELS_DIR = kernels.KERNELS_DIR
    src = open(os.path.join(KERNELS_DIR, "lnl_fused.py"), encoding="utf-8").read()
    assert "autotuner.cache = _PINS[kernel_name]" in src and "return full[:1]" in src and 'PF_LNL_AUTOTUNE_ALL' in src
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in ("_PinnedTiles", "_row_pins")]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "lnl_pure", "exec"), ns)

    class Cfg:                                                                   # stands in for triton.Config (kwargs, num_warps, num_stages by value)
        def __init__(self, kwargs, num_warps, num_stages): self.kwargs, self.num_warps, self.num_stages = kwargs, num_warps, num_stages

    table = json.load(open(os.path.join(KERNELS_DIR, "lnl_fused.tiles_by_arch.json")))
    for cc in ("9.0", "8.0"):
        for kern in ("_fused_transition_kernel", "_ln_linear_kernel"):
            row = table[cc][kern]
            distinct = {}
            for cfg in row.values():
                sig = (tuple(sorted((k, v) for k, v in cfg.items() if k not in ("num_warps", "num_stages"))), cfg["num_warps"], cfg["num_stages"])
                distinct.setdefault(sig, Cfg(dict(sig[0]), sig[1], sig[2]))
            configs = list(distinct.values())
            pins, principal = ns["_row_pins"](row, configs)
            assert len(pins) == len(row) and principal in configs
            cache = ns["_PinnedTiles"](pins, principal)
            for key_str, cfg in row.items():                                    # a tabled key: ITS config, no benchmark (the autotuner sees `key in cache` True)
                key = ast.literal_eval(key_str)
                assert key in cache and cache[key].kwargs == {k: v for k, v in cfg.items() if k not in ("num_warps", "num_stages")}
                assert (cache[key].num_warps, cache[key].num_stages) == (cfg["num_warps"], cfg["num_stages"])
            probe = ("no", "such", "key")
            assert probe in cache and cache[probe] is principal and cache.untabled == [str(probe)]



def test_ln_qkvg_tile_by_cc_is_the_8_0_cell():
    """kernels.atom_window: the one-pass TF32 ln_qkvg launch tile when the caller names none = 32 rows x 8 warps on cc 8.0 (the 128-row tile
    exceeds its shared memory), 128 x 8 elsewhere; tf32x3 keeps 16 rows on every card."""
    import pytest
    aw = pytest.importorskip("opt_core.kernels.atom_window")
    assert aw.LN_QKVG_TILE_BY_CC == {"8.0": (32, 8)}
    assert aw.ln_qkvg_tile("8.0") == (32, 8) and aw.ln_qkvg_tile("9.0") == (128, 8) and aw.ln_qkvg_tile("12.0", num_warps=4) == (128, 4)
    assert aw.ln_qkvg_tile("8.0", "tf32x3") == (16, 8) and aw.ln_qkvg_tile("9.0", "tf32x3", 4) == (16, 4)

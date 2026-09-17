"""The mode table locks: exact's flag set string-equal to the base kit README's production line; the fast row; no lever subtraction;
the default mode = fast on every route; upstream's typed keys composed onto the driver line (the driver's three constants replaced,
the rest verbatim); no lever switch anywhere but modes.py / registry.py; the deterministic recipe's one override; the design route's
resolutions byte-locked; one mode axis, no driver families."""
import argparse
import dataclasses
import inspect
import json
import os
import re

import pytest

from rfdiffusion1_opt import design, det, modes, registry, stack
from rfdiffusion1_opt.registry import KIT_BASE, LEVERS

README_LINE_RE = re.compile(r"--fastpath \S+ --prep \d --einsum-route \d --cudagraph \d")
OVERRIDE_FORMS = (("defaults", None), ("no-traj-not-cautious", ["inference.write_trajectory=False", "inference.cautious=False"]))   # upstream's defaults; the two knobs typed off
DESIGN_ROUTE_FIXTURE = os.path.join(os.path.dirname(__file__), "design_route_resolutions.json")


def _read(kit, rel):
    return open(os.path.join(stack.kit_dir(kit), *rel.split("/")), encoding="utf-8").read()


def test_exact_flags_equal_the_documented_line_with_the_kits_whole_forward_graph():
    """exact = the base driver's lever flags with the whole-forward graph W1 (`--fullgraph 1`, not the per-block `--cudagraph 1`), exactly as
    CHANGES.md 'Mode composition' states them for `exact`; the line has no fixed head before them."""
    assert modes.K_LINE == "--fastpath chain_breaks,full_graph,rbf,msa_index --prep 1 --einsum-route 1 --fullgraph 1"
    changes = open(os.path.join(stack.tree_root(), "CHANGES.md"), encoding="utf-8").read()
    row = [l for l in changes.splitlines() if l.startswith("| `exact` |")]
    assert len(row) == 1 and f"`{modes.K_LINE}`" in row[0] and "`RFD_PDBIO=1`" in row[0], row
    assert README_LINE_RE.search(modes.K_LINE.replace("--fullgraph 1", "--cudagraph 1"))   # the grammar the regex names
    assert not hasattr(modes, "DRIVER_COMMON")                                  # no `--mode time --module-timers 0` head: the driver has one mode and no timers


def test_exact_row():
    r = modes.resolve("exact")
    assert r.levers == ("U1", "C1", "P", "E_einsum", "W1", "IO1") and not hasattr(r, "without")
    assert r.flags == tuple(modes.K_FLAGS) and r.env == {"RFD_PDBIO": "1"} and r.tier == 1 and r.attach == "driver"
    assert r.driver_chain == ((KIT_BASE, "drivers/rfd_bench.py"),)
    assert r.line == "RFD_PDBIO=1 <fast_inference>/drivers/rfd_bench.py " + modes.K_LINE
    assert [i for i, lv in LEVERS.items() if lv.switch[0] == "env"] == ["IO1", "T2"] and registry.env(modes.K_LEVERS) == {"RFD_PDBIO": "1"}   # two environment switches: this package's PDB-writer lever IO1 (on every base line) and the SE(3) add-on's own variable (T2)
    assert LEVERS["IO1"].kit == registry.PKG and LEVERS["IO1"].tier == 1 and os.path.isfile(os.path.join(os.path.dirname(modes.__file__), LEVERS["IO1"].kit_file))
    assert not hasattr(modes, "SHIM_DEFAULTS") and not hasattr(registry, "KIT_SE3K")


def test_off_row_is_stock():
    r = modes.resolve("off")
    assert r.levers == () and r.flags == () and r.env == {} and r.attach == "stock-cli" and r.tier is None


def test_one_mode_axis():
    """The table is keyed by mode alone: one driver line (two kit directories: the base kit and the SE(3) add-on), no second axis on the
    resolution, no row that runs under another mode's name, no per-case override channel; the refusal line names the mode and the reasons."""
    assert set(registry.KIT_DIRS) == {registry.KIT_BASE, registry.KIT_SE3} and {k for k, _ in modes.MODES["exact"].driver_chain} == {registry.KIT_BASE}
    assert [f.name for f in dataclasses.fields(modes.Mode)] == ["name", "tier", "levers", "attach", "driver_chain", "flags", "env", "note"]
    assert [f.name for f in dataclasses.fields(modes.Resolution)] == ["mode", "tier", "levers", "attach", "driver_chain", "flags", "env", "settings", "note", "served"]
    assert list(inspect.signature(modes.resolve).parameters) == ["mode", "overrides", "served", "det"] and list(inspect.signature(modes.default_mode).parameters) == ["served"]
    assert design.UNSERVED_FMT == "mode={mode} cannot serve {reasons} — refused by name, nothing ran; `--mode off` runs this request on upstream's command line (scripts/run_inference.py)" and not hasattr(design, "DECLINED_FMT")


def test_fast_row():
    """fast = exact's line + the tolerance-tier levers T2 (the SE(3) add-on's row RFD_SE3FAST=t2, no driver flag), K2
    (--triton-ln 1) and TF32 (--tf32 1), each behind its own switch; every mode word has a row."""
    assert set(modes.MODES) == {"off", "exact", "fast"}
    r, e = modes.resolve("fast"), modes.resolve("exact")
    assert modes.TOLERANCE_LEVERS == ("T2", "K2", "TF32") and all(LEVERS[l].tier == 2 for l in modes.TOLERANCE_LEVERS)
    assert r.levers == e.levers + modes.TOLERANCE_LEVERS == ("U1",) + modes.FAST_LEVERS and r.tier == 2 and r.attach == "driver"
    assert r.flags == e.flags + ("--triton-ln", "1", "--tf32", "1") == tuple(modes.FAST_FLAGS) and r.driver_chain == e.driver_chain
    assert r.env == {"RFD_PDBIO": "1", "RFD_SE3FAST": "t2"} and e.env == {"RFD_PDBIO": "1"}
    assert e.line.startswith("RFD_PDBIO=1 <fast_inference>/") and r.line == "RFD_PDBIO=1 RFD_SE3FAST=t2 " + e.line[len("RFD_PDBIO=1 "):] + " --triton-ln 1 --tf32 1"
    assert modes.NUMERICS["exact"]["policy"] == "fp32_strict" and modes.NUMERICS["fast"]["policy"] == "tf32"
    assert modes.numerics("fast")["matmul_tf32"] is True and modes.numerics("fast")["cudnn_tf32"] is True and modes.numerics("exact")["matmul_tf32"] is False
    assert LEVERS["T2"].kit == registry.KIT_SE3 and LEVERS["T2"].tier == 2 and LEVERS["T2"].evidence.startswith(r"^\[rfd_se3fast\] v0\.3\.0 applied: mode=t2")
    assert LEVERS["K2"].switch == ("flag", "--triton-ln", "1") and LEVERS["TF32"].switch == ("flag", "--tf32", "1") and LEVERS["TF32"].tier == 2
    assert modes.MODE_NAMES == ("off", "exact", "fast") == tuple(modes.MODES)                     # every mode word has a row, in the table's order
    assert not hasattr(modes, "KIT_COMPOSITIONS") and not hasattr(modes.Mode, "open")               # a subset of fast is `design --without`, never a mode name


def test_no_lever_subtraction_and_no_compositions():
    """A mode runs its row whole: no `--without` (resolve takes no such argument, Resolution has no such field) and no composition vocabulary."""
    for gone in ("COMPOSITION_POINTER", "check_composition", "SERVE_COMPOSITIONS", "DEFAULT_COMPOSITION", "KIT_COMPOSITIONS", "parse_without"):
        assert not hasattr(modes, gone), gone
    assert "without" not in inspect.signature(modes.resolve).parameters and "without" not in [f.name for f in dataclasses.fields(modes.Resolution)]
    assert modes.resolve("fast").levers == modes.resolve("exact").levers + modes.TOLERANCE_LEVERS


def test_default_mode_is_fast_on_every_route():
    """DEFAULT_MODE_RULE: no --mode = fast on the design route and on the packed line; a request the kit line cannot serve is refused by name
    (design.refusals) under the default mode as under a typed one — never a silent fall-back to another mode or to the stock command line."""
    assert modes.DEFAULT_MODE == "fast" and not hasattr(modes, "DEFAULT_ORDER")
    assert modes.default_mode() == modes.default_mode(served=True) == modes.default_mode(served=False) == "fast"
    assert modes.resolve(None).mode == "fast" and modes.resolve(None, served=True).mode == "fast"
    assert modes.resolve(None) == modes.resolve("fast") and modes.resolve(None, served=True) == modes.resolve("fast", served=True)
    assert modes.DEFAULT_MODE_RULE == "no --mode = fast, on every route (design, the served line --pack K); --mode or RFDIFFUSION1_OPT names another"
    assert design.refusals(None, []) == [] and design.refusals("exact", ["diffuser.partial_T=10"]) == [] and design.refusals("off", ["inference.symmetry=C3"]) == []   # partial diffusion is served; off never refuses a request
    assert design.refusals(None, ["inference.symmetry=C3"]) == design.refusals("fast", ["inference.symmetry=C3"]) != []          # no mode word = the default kit line: it refuses what fast refuses
    assert not hasattr(design, "family_usage")                                                                  # no usage error for a stock input under any mode name


def test_unknown_mode():
    with pytest.raises(modes.ModeError, match="unknown mode"):
        modes.resolve("turbo")
    with pytest.raises(modes.ModeError, match="not a hydra override"):
        modes.resolve("exact", overrides=["paper"])                                    # the driver line composes Hydra overrides only
    odd = ["paper", "--config-name=symmetry", "+inference.x=1", "~inference.cautious", "inference.write_trajectory=maybe"]
    off = modes.resolve("off", overrides=odd)                                          # the stock command line takes every token as typed (upstream's parser answers for them): nothing refused, nothing composed
    assert off.settings == modes.Settings(tuple(odd), (), ()) and (off.settings.stock_overrides, off.settings.driver_flags, off.settings.compose_overrides) == (tuple(odd), (), ())
    for m in ("exact", "fast"):                                                        # a kit line: any KEY=VALUE passes — verbatim for the stock arm, composed onto the driver's configuration
        r = modes.resolve(m, overrides=["inference.final_step=5"])
        assert r.settings.stock_overrides == ("inference.final_step=5",) and r.settings.compose_overrides[-1] == "inference.final_step=5", m
    with pytest.raises(modes.ModeError, match="expects True|False"):
        modes.resolve("exact", overrides=["inference.write_trajectory=maybe"])         # a driver-constant key must be a hydra boolean on the driver line (the driver switch --no-traj follows from it)
    with pytest.raises(modes.ModeError, match=re.escape("inference.deterministic expects True|False (hydra boolean), got 'sometimes'")):
        modes.resolve("exact", overrides=["inference.deterministic=sometimes"])


def test_registry_switches_are_the_drivers_flags():
    src = _read(KIT_BASE, "drivers/rfd_bench.py")
    argflags = set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', src))
    for lid in ("C1", "P", "E_einsum", "W1", "K2"):
        assert LEVERS[lid].switch[0] == "flag" and LEVERS[lid].switch[1] in argflags, lid
    assert "--cudagraph" not in argflags and "--triton-attn" not in argflags and "--jit-warmup" not in src and "G1" not in LEVERS and "C3" not in LEVERS   # the base driver carries no lever outside its rows
    assert set(LEVERS) == {"U1", "C1", "P", "E_einsum", "W1", "IO1", "T2", "K2", "TF32"} and all(lv.kit in registry.KIT_DIRS or lv.kit == registry.PKG for lv in LEVERS.values())   # every registered lever is a lever of the one line


def test_every_mode_lever_has_a_switch_and_a_class():
    for m, md in modes.MODES.items():
        for lid in md.levers:
            lv = LEVERS[lid]
            assert lv.switch and lv.class_4 in ("forward", "datapath", "serving", "orchestration"), lid
            if lv.class_4 == "forward":
                assert lv.tier in (1, 2) and lv.evidence, lid


def test_exact_is_tier_1_only():
    for lid in modes.MODES["exact"].levers:
        assert LEVERS[lid].tier in (1, None), lid


def test_config_carries_no_lever_switch():
    cfg = open(os.path.join(stack.tree_root(), "configs", "h100.env"), encoding="utf-8").read()
    switches = {lv.switch[1] for lv in LEVERS.values() if lv.switch[0] in ("flag", "env")}
    for s in switches | {"--fullgraph", "--triton-ln", "RFD_TRITON_LN", "RFD_PREP", "RFD_FASTPATH", "RFDIFFUSION1_OPT", "--det"}:
        rx = re.escape(s) + (r"\b" if s.startswith("--") else r"=")          # a flag as a token; a variable as an assignment `NAME=`
        assert not re.search(r"(^|[^A-Za-z0-9_-])" + rx, cfg, re.M), f"{s} in configs/h100.env"
    assert "MODEL_OPT=" in cfg and "SE3K" not in cfg


def test_upstream_knobs_pass_through():
    """No presets: upstream's defaults when nothing is typed; the typed KEY=VALUE verbatim on the stock arm; on the driver line the launch's
    value (typed, else upstream's default; `inference.deterministic=True` under det) on the driver's three constant keys, then every other typed
    key verbatim except the target keys the case row carries (TARGET_KEYS)."""
    assert modes.settings_of() == modes.settings_of([], "stock-cli") == modes.Settings((), (), ())      # the stock arm: nothing typed, nothing appended — upstream's defaults
    d = modes.settings_of(attach="driver")
    assert d.stock_overrides == () and d.driver_flags == ("--no-traj", "0")
    assert d.compose_overrides == ("inference.write_trajectory=True", "inference.cautious=True", "inference.deterministic=False")
    assert modes.DRIVER_FIXED == ("inference.write_trajectory", "inference.cautious", "inference.deterministic")
    assert modes.UPSTREAM_DEFAULTS == {"inference.write_trajectory": "True", "inference.cautious": "True", "inference.deterministic": "False"}
    yaml = open(os.path.join(stack.tree_root(), "stock", "src", "config", "inference", "base.yaml"), encoding="utf-8").read().splitlines()
    lines = stack.pins()["cli_defaults"]["lines"]                                                            # stock/PINS.json cites upstream's default lines: base.yaml:13, :17, :21
    assert [lines[k] for k in modes.DRIVER_FIXED] == [13, 17, 21]
    assert yaml[12].strip() == "write_trajectory: True" and yaml[16].strip() == "cautious: True" and yaml[20].strip() == "deterministic: False"
    assert all(str(stack.pins()["cli_defaults"]["values"][k]) == modes.UPSTREAM_DEFAULTS[k] for k in modes.DRIVER_FIXED)   # the composed defaults ARE upstream's
    k = modes.settings_of(["inference.write_trajectory=False", "inference.cautious=False"], "driver")
    assert k.stock_overrides == ("inference.write_trajectory=False", "inference.cautious=False") and k.driver_flags == ("--no-traj", "1")
    assert modes.settings_of(["inference.write_trajectory=False"]).stock_overrides == ("inference.write_trajectory=False",) and modes.settings_of(["inference.write_trajectory=False"]).driver_flags == ()   # the stock arm: as typed, no driver switch
    assert modes.settings_of(["inference.cautious=false"], "driver").compose_overrides == ("inference.write_trajectory=True", "inference.cautious=False", "inference.deterministic=False")
    assert modes.settings_of(attach="driver", det=True).compose_overrides[2] == "inference.deterministic=True" == det.SEED_OVERRIDE       # --det 1: the seed composed on the driver line
    assert modes.settings_of(["inference.deterministic=0"], "driver", det=True).compose_overrides[2] == "inference.deterministic=False"   # the typed key wins over the recipe's default
    assert modes.settings_of(["inference.deterministic=true"]).stock_overrides == ("inference.deterministic=true",)              # ... and reaches the stock arm as typed (det.stock_overrides appends nothing over it)
    assert modes.settings_of(det=True) == modes.Settings((), (), ())                                                              # the stock arm's seed is det.stock_overrides' to append (design.stock_overrides), not settings_of's
    # every other typed key: verbatim onto the driver's configuration, in the typed order; the target keys are the case row's and are not composed twice
    typed = ["inference.input_pdb=/in/x.pdb", "contigmap.contigs=[A1-10/0 5-5]", "ppi.hotspot_res=[A1]", "inference.output_prefix=/o/p", "inference.num_designs=3",
             "inference.design_startnum=2", "inference.model_directory_path=/w", "inference.final_step=5", "inference.cautious=False", "denoiser.noise_scale_ca=0.5", "inference.ckpt_override_path=/w/x.pt"]
    t = modes.settings_of(typed, "driver")
    assert t.compose_overrides == ("inference.write_trajectory=True", "inference.cautious=False", "inference.deterministic=False", "inference.final_step=5", "denoiser.noise_scale_ca=0.5", "inference.ckpt_override_path=/w/x.pt")
    assert t.stock_overrides == tuple(typed) and set(modes.TARGET_KEYS) == {o.split("=", 1)[0] for o in typed[:7]}
    assert modes.settings_of(typed).stock_overrides == tuple(typed) and modes.settings_of(typed).compose_overrides == ()   # the stock arm: verbatim, nothing composed
    with pytest.raises(modes.ModeError, match="not a hydra override"):
        modes.settings_of(["--config-name=symmetry"], "driver")                           # a flag is not an override token on the driver line (design.refuse_unserved names such a request before settings_of sees it)
    t2 = modes.settings_of(["+inference.x=1", "~ppi.hotspot_res", "++inference.cautious=False", "++inference.num_designs=3", "diffuser.partial_T=10"], "driver")
    assert t2.compose_overrides == ("inference.write_trajectory=True", "inference.cautious=False", "inference.deterministic=False", "+inference.x=1", "~ppi.hotspot_res", "diffuser.partial_T=10")   # +/++/~ forms composed verbatim; a fixed / target key's ++ form is its row's value
    assert modes.settings_of(["--config-name", "symmetry", "+inference.x=1"]).stock_overrides == ("--config-name", "symmetry", "+inference.x=1")   # ... and rides verbatim on the stock arm
    src = _read(KIT_BASE, "drivers/rfd_bench.py")
    assert 'p.add_argument("--no-traj", type=int, default=1' in src                      # the driver's own default is the shape the kit composes


def test_table_rows():
    rows = modes.table()
    keys = {r["mode"] for r in rows}
    assert keys == set(modes.MODE_NAMES) and all(set(r) <= {"mode", "defined", "tier", "levers", "flags", "env", "attach", "driver_chain", "note", "fact", "served"} for r in rows)   # modes only: no lever-set (composition) row, no second axis
    for r in rows:
        assert "composition" not in r and "kit_composition" not in r, r
        if r.get("served") and r["mode"] in modes.NOT_SERVED:
            assert r["defined"] is False and r["fact"], r
        else:
            assert r["defined"] is True and "fact" not in r, r
    assert [r["mode"] for r in rows if r.get("served") and r["defined"]] == ["exact", "fast"]      # the packed line: the exact and fast rows
    assert set(modes.NOT_SERVED) == {"off"} and [r["mode"] for r in rows if r.get("served") and not r["defined"]] == ["off"]


def test_dgl_backend_has_one_copy():
    """stack.DGL_BACKEND is the one place the tree writes DGLBACKEND: the driver environment and the stock arm's environment both take it from
    there (the driver's own setdefault covers a hand-started driver); no config carries it."""
    for cfg_name in ("h100.env", "a100.env"):
        cfg = open(os.path.join(stack.tree_root(), "configs", cfg_name), encoding="utf-8").read()
        assert "DGLBACKEND=" not in cfg, cfg_name
    assert stack.DGL_BACKEND == ("DGLBACKEND", "pytorch") and not hasattr(det, "DGL_BACKEND") and not hasattr(det, "driver_env")
    from rfdiffusion1_opt.tests._stubs import H100
    env, _ = stack.driver_environment(modes.resolve("exact"), H100, environ={"PATH": "/usr/bin"})
    senv, _absent, _dropped = design.stock_environment({"PATH": "/usr/bin", "RFD_ROOT": "/opt/rfd", "WEIGHTS": "/w"})
    assert env["DGLBACKEND"] == senv["DGLBACKEND"] == "pytorch"
    pkg = os.path.dirname(modes.__file__)
    writers = sorted(f for f in os.listdir(pkg) if f.endswith(".py") and re.search(r"[\"']DGLBACKEND[\"']", open(os.path.join(pkg, f), encoding="utf-8").read()))
    assert writers == ["stack.py"], writers                                              # the literal is written once in the package


def test_det_recipe_is_upstreams_seed_override_alone():
    """det.py: the deterministic recipe = upstream's own per-design seed (`inference.deterministic=True`: run_inference.py seeds every RNG
    with design_startnum + i_des — the string the kit driver hard-codes, rfd_bench.py), composed on both arms under `--det 1` and nowhere under
    the default `--det 0`; a typed value of the key wins on both arms. No cross-host BLAS pinning, no schedule-cache split, no replay-check
    cadence live here: the recipe is one override and the environment builders take no recipe switch."""
    drv = _read(KIT_BASE, "drivers/rfd_bench.py")
    assert '"inference.deterministic=True"' in drv and det.SEED_OVERRIDE == "inference.deterministic=True"
    assert det.LEVELS == (0, 1) and det.DEFAULT_LEVEL == 0 and set(det.LEVEL_WHAT) == {0, 1}
    assert "unseeded default on every arm" in det.LEVEL_WHAT[0] and det.SEED_OVERRIDE in det.LEVEL_WHAT[1] and det.SEED_KEY == "inference.deterministic"
    assert det.stock_overrides() == [] and det.stock_overrides(False) == [] and det.stock_overrides(True) == ["inference.deterministic=True"]
    assert det.stock_overrides(True, typed=["inference.deterministic=False"]) == [] and det.stock_overrides(True, typed=("diffuser.T=25",)) == [det.SEED_OVERRIDE]   # typed wins: nothing appended over it
    assert modes.settings_of(attach="driver", det=True).compose_overrides[-1] == det.SEED_OVERRIDE and modes.settings_of(attach="driver").compose_overrides[-1] == "inference.deterministic=False"   # the driver arm: the same key through DRIVER_FIXED
    public = {n for n in dir(det) if not n.startswith("_") and n not in ("annotations", "List", "Optional", "Sequence", "Tuple", "Dict")}
    assert public == {"SEED_OVERRIDE", "SEED_KEY", "HOST_CLASS_RULE", "LEVELS", "DEFAULT_LEVEL", "LEVEL_WHAT", "stock_overrides"}, public   # the recipe's whole surface: no environment line, no schedule split, no replay-check cadence
    assert "one CPU host class" in det.HOST_CLASS_RULE
    yaml = open(os.path.join(stack.tree_root(), "stock", "src", "config", "inference", "base.yaml"), encoding="utf-8").read().splitlines()
    assert yaml[20].strip() == "deterministic: False"                                          # base.yaml:21, upstream's default: no seeding


def test_numerics_words():
    """The numerics words, the RF family's form: exact declares fp32_strict (both TF32 flags off by the driver, matmul precision / autocast at torch's
    defaults), off is untouched; one pass's driver record read back names every differing key (never refused)."""
    from rfdiffusion1_opt.modes import NUMERICS, MODE_NAMES, numerics
    from rfdiffusion1_opt.report import numerics_words, _kv
    assert set(NUMERICS) <= set(MODE_NAMES) and NUMERICS["off"] == {"policy": "untouched"}
    assert _kv(numerics_words(numerics("exact"))) == "numerics=fp32_strict numerics_source=declared matmul=highest matmul_tf32=off cudnn_tf32=off autocast=none"
    assert _kv(numerics_words(numerics("off"))) == "numerics=untouched numerics_source=declared"
    rec = {"param_dtype": "torch.float32", "allow_tf32_matmul": False, "cudnn_tf32": False, "autocast_enabled": False}   # the driver's per-case precision block
    n = numerics("exact", rec)
    assert n["source"] == "torch" and n["mismatch"] == [] and "numerics_mismatch" not in numerics_words(n)
    n = numerics("exact", dict(rec, cudnn_tf32=True, allow_tf32_matmul=True, param_dtype="torch.float16"))
    assert n["mismatch"] == ["matmul_tf32", "cudnn_tf32", "dtype"] and numerics_words(n)["numerics_mismatch"] == "matmul_tf32,cudnn_tf32,dtype"
    for f in NUMERICS["exact"]["set_by"].split(" (")[0].split("; "):                 # the driver's TF32 assignments, cited by line
        rel, lines = f.rsplit(":", 1)
        a, b = (int(x) for x in lines.split("-"))
        from rfdiffusion1_opt import stack
        import os
        src = open(os.path.join(stack.tree_root(), rel), encoding="utf-8").read().splitlines()[a - 1:b]
        assert [l.split("=")[0].strip() for l in src] == ["torch.backends.cuda.matmul.allow_tf32", "torch.backends.cudnn.allow_tf32"], (rel, src)


def test_help_lists_exactly_the_shipped_modes(monkeypatch, capsys):
    """`--help` offers the modes the kit ships (off | exact | fast) and nothing else; an unknown name, asked for by flag, environment or
    enable(), is refused by name on the one path (modes.resolve → the NOT ACTIVE line, exit 3) before any work; fast resolves to its row."""
    from rfdiffusion1_opt import cli
    from rfdiffusion1_opt.tests._stubs import good_box
    subs = next(a for a in cli.build_parser()._actions if isinstance(a, argparse._SubParsersAction)).choices
    with_mode = {verb: [a for a in sp._actions if "--mode" in a.option_strings] for verb, sp in subs.items()}
    assert {v for v, acts in with_mode.items() if acts} == set(cli.VERBS)
    for verb, acts in with_mode.items():
        for a in acts:
            assert a.help.startswith("off | exact | fast; ") and "default: fast;" in a.help and a.choices is None, (verb, a.help)   # no argparse choices: a name is refused by the package, by name; the default is fast
    assert "[--mode off|exact|fast]" in cli.__doc__.splitlines()[0]
    assert {o for sp in subs.values() for a in sp._actions for o in a.option_strings} == {"-h", "--help", "--mode", "--out_dir", "--pack", "--det", "--dry-run", "--json"}   # the kit's whole flag surface (--out_dir is warm's scratch directory); everything else on the line is upstream's
    assert {o for a in subs["design"]._actions for o in a.option_strings} == {"-h", "--help", "--mode", "--pack", "--det", "--dry-run"}   # design: the input is upstream's overrides, nothing else
    assert modes.resolve("fast").levers[-3:] == modes.TOLERANCE_LEVERS
    good_box(monkeypatch)
    rep = stack.activate("fast", [], dry_run=True)                                                  # --mode fast
    assert rep["reason"] is None and rep["mode"] == "fast" and not rep.get("would_refuse") and "requested" not in rep and "fallback" not in rep
    stack.reset_for_tests(); monkeypatch.setenv("RFDIFFUSION1_OPT", "fast")                         # the environment form of the mode
    rep = stack.activate(None, [], dry_run=True)
    assert rep["reason"] is None and rep["mode"] == "fast"
    stack.reset_for_tests(); monkeypatch.delenv("RFDIFFUSION1_OPT")
    import rfdiffusion1_opt
    rep = rfdiffusion1_opt.enable("fast")                                                         # enable(): armed as the fast row
    assert rep["active"] and rep["mode"] == "fast" and list(inspect.signature(rfdiffusion1_opt.enable).parameters) == ["mode", "overrides", "strict", "trigger"]
    stack.reset_for_tests(); capsys.readouterr()
    assert cli.main(["check", "--mode", "turbo"]) == 3                                              # an unknown name: refused by name, exit 3
    err = capsys.readouterr().err
    assert err.startswith("[rfdiffusion1-opt] NOT ACTIVE: unknown mode 'turbo' (expected off|exact|fast)") and "(mode=turbo)" in err and err.count("NOT ACTIVE") == 1, err
    stack.reset_for_tests(); capsys.readouterr()
    assert cli.main(["check", "--mode", "fast"]) == 0                                               # the row resolves (the DRY-RUN line)
    assert "[rfdiffusion1-opt] DRY-RUN mode=fast attach=driver tier=2 " in capsys.readouterr().err
    stack.reset_for_tests()


# ----------------------------------------------------------------------------------------------------------------- the design route's resolutions, locked
def _render_design_routes() -> dict:
    """Every (mode, typed-overrides form) row `design` can run, the default-mode rule and defaults,
    the packed line's refusal facts (NOT_SERVED) and the numerics table — rendered for a byte-for-byte lock against
    design_route_resolutions.json (modes.resolve(served=False) via dataclasses.asdict; the served line is asserted apart in
    test_the_served_line_moves_no_design_route_byte)."""
    out = {}
    for m in modes.MODE_NAMES:
        for sname, ov in OVERRIDE_FORMS:
            r = modes.resolve(m, ov)
            assert not r.served, (m, sname)
            d = dataclasses.asdict(r)
            d["settings"] = {"driver_flags": list(r.settings.driver_flags), "stock_overrides": list(r.settings.stock_overrides),
                             "compose_overrides": list(r.settings.compose_overrides)}
            out[f"{m}/{sname}"] = d
    out["DEFAULT_MODE_RULE"] = modes.DEFAULT_MODE_RULE
    out["default"] = modes.default_mode()
    out["default_served"] = modes.default_mode(True)
    out["NOT_SERVED"] = dict(modes.NOT_SERVED)
    out["NUMERICS"] = modes.NUMERICS
    return json.loads(json.dumps(out, sort_keys=True, default=list))


def test_design_route_resolutions_are_the_locked_ones():
    want = json.load(open(DESIGN_ROUTE_FIXTURE, encoding="utf-8"))
    got = _render_design_routes()
    assert sorted(got) == sorted(want)
    for key in sorted(want):
        assert got[key] == want[key], key


def test_the_served_line_moves_no_design_route_byte():
    """The served resolution differs from the design route's exact row in exactly: the note's served clause and `served` — driver, flags,
    levers, env, tier, attach, settings equal."""
    d, s = modes.resolve("exact"), modes.resolve("exact", served=True)
    assert (s.levers, s.env, s.tier, s.attach, s.settings, s.driver_chain) == (d.levers, d.env, d.tier, d.attach, d.settings, d.driver_chain)
    assert s.flags == d.flags and modes.serve_flags("exact") == () and s.line == d.line
    assert s.note.startswith(d.note + "; served: ") and (s.served, d.served) == (True, False)

"""The stock pins reference the one stock configuration; the registry's sources point at files in the tree; the hook files define
the finder classes hooks.py looks for."""
import os


from openfold3_ob0_opt import _autoload, cli, env, modes, registry, stack
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()
FINDER_CLASS = {"fast_inference": "_LeverFinder", "trunk_kernels": "_Finder", "offload": "_PostImportFinder", "tp": "_TPFinder",
                "confhead": "_PostImportFinder", "tp_rowpair": "_TPFinder", "cells": "_PostImportFinder"}   # the finder class of every hook directory the tree can carry (carried add-ons and the package's own hooks)
FINDER_OF_KIT = {k: FINDER_CLASS[k] for k in list(modes.KITS) + list(modes.PACKAGE_HOOKS)}                    # the hook directories THIS tree declares (a KeyError here = a hook directory without a named finder class)


def test_hook_files_define_the_finder_classes():
    present = [kit for kit in FINDER_OF_KIT if os.path.isfile(os.path.join(modes.hook_dir(HOME, kit), modes.HOOK_FILE))]
    assert present == list(FINDER_OF_KIT), [k for k in FINDER_OF_KIT if k not in present]                       # every declared hook directory carries its hook file
    for kit in present:
        src = open(os.path.join(modes.hook_dir(HOME, kit), modes.HOOK_FILE), encoding="utf-8").read()
        cls = FINDER_OF_KIT[kit]
        assert f"class {cls}" in src, kit


def test_stock_pins_name_the_one_stock_configuration():
    import yaml
    p = env.pins(HOME)
    assert p["stock_config"]["file"] == modes.STOCK_YAML.replace(os.sep, "/") and p["stock_config"]["det_file"] == modes.STOCK_DET_YAML.replace(os.sep, "/")
    assert p["stock_config"]["kernels_off_file"] == modes.KERNELS_OFF_YAML.replace(os.sep, "/") and p["stock_config"]["shipped_file"] == modes.SHIPPED_YAML.replace(os.sep, "/")
    stock = yaml.safe_load(open(os.path.join(HOME, modes.STOCK_YAML)))
    ev = stock["model_update"]["custom"]["settings"]["memory"]["eval"]
    assert p["stock_config"]["sets"]["model_update.custom.settings.memory.eval"] == ev                       # the pins restate the stock file's eval block exactly (kernel triple + chunk plan)
    assert p["stock_config"]["sets"]["pl_trainer_args.precision"] == stock["pl_trainer_args"]["precision"] == "bf16-mixed"
    assert set(cli.yaml_kernels_on(os.path.join(HOME, modes.STOCK_YAML))) == {"use_triton_triangle_kernels", "use_cueq_triangle_kernels"}   # stock: the predict preset + the cuEquivariance triangle kernels (DS4Sci off)
    assert modes.STOCK_DET_YAML == modes.STOCK_YAML                                                          # one stock: the det form is the same file under --det 1
    assert cli.yaml_kernels_on(os.path.join(HOME, modes.KERNELS_OFF_YAML)) == ()
    assert yaml.safe_load(open(os.path.join(HOME, modes.KERNELS_OFF_YAML)))["pl_trainer_args"] == {"precision": "32-true"}   # the big lines' base names its trainer precision (a bf16 line adds its own file)
    assert p["openfold3_version"] == _stubs.PIN and len(p["upstream"]["commit"]) >= 40                       # the pin and upstream's commit of it, read from the one file (stock/PINS.json)
    se = p["stock_environment"]
    assert any(_autoload.ENV.startswith(x) for x in se["must_be_absent_prefixes"])                          # the package's own mode variable is refused on the stock route
    third_party = []
    for lv in registry.LEVERS.values():                                                                      # every lever's KIT switch (a kit prefix or the package's own) is covered by a must-be-absent prefix …
        for k in lv.env_keys:
            if not (k.startswith(modes.SWITCH_PREFIXES) or k.startswith(_autoload.ENV)):
                third_party.append((lv.name, k)); continue                                                   # a third-party variable a lever sets (torch's allocator config): stock's surface, never refused
            assert any(k.startswith(x) for x in se["must_be_absent_prefixes"]), (lv.name, k)
    assert all(k == "PYTORCH_CUDA_ALLOC_CONF" for _, k in third_party), third_party                         # the one such variable in the table; a new one is named here deliberately
    assert not [k for k in se.get("upstream_reads", {}) if k in env.must_be_absent(HOME)]                    # … and no name upstream itself reads is a prefix of the refusal list verbatim
    assert {"OPENFOLD3_OB0_CKPT", "OPENFOLD_CACHE"} <= set(se["reads"])
    assert "image" not in p["pinned_stack"] and isinstance(p["pinned_stack"]["tested_on"], str)          # the one documentation line of the tested hardware/stack; no gate reads it
    for section, key in (("stock_config", "sha256"), ("source", "tree_sha256"), ("pinned_stack", "freeze_sha256")):
        assert key not in p[section], (section, key)                                                         # no digest of a file whose identity is the tree itself
    assert p["wheel"]["pypi"] == "openfold3==" + p["openfold3_version"] and len(p["wheel"]["sha256"]) == 64 and p["wheel"]["bytes"] > 0   # the PyPI wheel `run.sh install` fetches when the tree arrives without it: one digest + size, judged before it is placed (stock/install_upstream.py)
    assert stack.pinned_weights(HOME) == (p["weights"]["file"], p["weights"]["sha256"])                      # the one digest this tree keeps: the downloaded checkpoint


def test_registry_sources_exist():
    """Every lever's registry `source` names files that exist in the tree (the add-ons' files under their table's directory, a package lever's under opt/openfold3_ob0_opt/)."""
    checked = 0
    for name, lv in registry.LEVERS.items():
        assert lv.kit in modes.KITS or lv.kit in modes.PACKAGE_HOOKS or lv.kit == stack.PACKAGE, name
        for ref in lv.source.split(";"):
            path = ref.strip().split(":")[0].split(" ")[0]
            if path.split("/")[0] in modes.KITS:
                path = os.path.join(modes.KITS[path.split("/")[0]], *path.split("/")[1:])
            elif lv.kit in modes.PACKAGE_HOOKS or lv.kit == stack.PACKAGE:                                       # a package lever (confhead): its sources relative to the tree (opt/openfold3_ob0_opt/...)
                assert path.startswith("opt/openfold3_ob0_opt/"), (name, ref)
            else:
                path = os.path.join(modes.KITS[lv.kit], path)
            assert os.path.isfile(os.path.join(HOME, path)), (name, ref)
            checked += 1
    assert checked > 0


def test_check_weights_refuses_other_bytes(tmp_path):
    """stock/check_pins.py check_weights(pins, path): the one digest this tree keeps of a downloaded, external thing (stock/PINS.json
    "weights") — refuses by name on a checkpoint that does not hash to it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_pins", os.path.join(HOME, "stock", "check_pins.py"))
    cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cp)
    pins = env.pins(HOME)
    bad_ckpt = tmp_path / "not_the_weights.pt"
    bad_ckpt.write_bytes(b"not the pinned checkpoint bytes")
    bad, detail = cp.check_weights(pins, str(bad_ckpt))
    assert bad and "NOT the pinned weights" in bad[0] and detail["is_pinned"] is False
    missing = tmp_path / "absent.pt"
    bad, detail = cp.check_weights(pins, str(missing))
    assert bad and "no such file" in bad[0] and detail["exists"] is False


def test_registry_markers_are_printed_by_the_kits():
    """Each lever's marker (the stderr prefix the kit prints when installed) is a string in the kit's code."""
    checked = 0
    for name, lv in registry.LEVERS.items():
        if not lv.marker:
            continue
        roots = ([os.path.join(HOME, modes.KITS[lv.kit])] if lv.kit in modes.KITS else [os.path.join(HOME, "opt", "openfold3_ob0_opt", n + ".py") for n in stack.ACTIVATION_MODULES] if lv.kit == stack.PACKAGE   # an activation-installed lever: the package modules stack.enable installs
                 else [modes.hook_dir(HOME, lv.kit), os.path.join(HOME, "opt", "openfold3_ob0_opt", lv.kit + ".py"), os.path.join(HOME, "opt", "openfold3_ob0_opt", lv.kit)])   # a package lever: its hook dir, module or package
        roots = [r for r in roots if os.path.exists(r)]
        assert roots, (name, lv.kit)
        text = ""
        for top in roots:
            walk = [(os.path.dirname(top), [], [os.path.basename(top)])] if os.path.isfile(top) else os.walk(top)
            for root, _, files in walk:
                for f in files:
                    if f.endswith(".py"):
                        text += open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        head = lv.marker.split(" ")[0]
        assert head in text, (name, lv.marker)
        checked += 1
    assert checked > 0

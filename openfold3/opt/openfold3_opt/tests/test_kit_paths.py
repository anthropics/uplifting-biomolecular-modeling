"""The stock pins reference the one stock configuration; the registry's sources point at files in the tree; the hook files define
the finder classes hooks.py looks for."""
import os

from openfold3_opt import cli, env, hooks, modes, registry, stack
from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()


def test_hook_files_define_the_finder_classes():
    want = {"fast_inference": "_LeverFinder", "trunk_kernels": "_Finder", "offload": "_PostImportFinder",
            "confhead": "_PostImportFinder", "tp_rowpair": "_TPFinder", "cells": "_PostImportFinder"}
    assert set(want) == set(modes.KITS) | set(modes.PACKAGE_HOOKS)                          # every hook directory's finder class is named here (carried add-ons and the package's own hooks)
    for kit, cls in want.items():
        src = open(os.path.join(modes.hook_dir(HOME, kit), modes.HOOK_FILE), encoding="utf-8").read()
        assert f"class {cls}" in src and cls in hooks.FINDER_CLASSES, kit


def test_stock_pins_name_the_one_stock_configuration():
    p = env.pins(HOME)
    assert p["stock_config"]["file"] == modes.STOCK_YAML.replace(os.sep, "/") and p["stock_config"]["det_file"] == modes.STOCK_DET_YAML.replace(os.sep, "/")
    assert p["stock_config"]["kernels_off_file"] == modes.KERNELS_OFF_YAML.replace(os.sep, "/")
    assert cli.yaml_kernel_flags(os.path.join(HOME, modes.STOCK_YAML)) == {"use_deepspeed_evo_attention": True, "use_cueq_triangle_kernels": True}      # stock: the predict preset as shipped + the cuEquivariance triangle kernels
    import yaml
    for yml in (modes.STOCK_YAML, modes.STOCK_DET_YAML):                                                                                              # one stock: bf16-mixed precision on both the speed form and the det form
        assert yaml.safe_load(open(os.path.join(HOME, yml)))["pl_trainer_args"] == {"precision": "bf16-mixed"}, yml
    assert yaml.safe_load(open(os.path.join(HOME, modes.KERNELS_OFF_YAML))).get("pl_trainer_args") is None                                            # the kit lines' base carries no trainer block (fast adds its own, fast_bf16_predict.yml)
    assert cli.yaml_kernel_flags(os.path.join(HOME, modes.STOCK_DET_YAML)) == {"use_deepspeed_evo_attention": False, "use_cueq_triangle_kernels": True} # stock under the det recipe: the DS4Sci attention off
    assert cli.yaml_kernel_flags(os.path.join(HOME, modes.KERNELS_OFF_YAML)) == {"use_deepspeed_evo_attention": False, "use_cueq_triangle_kernels": False}
    assert p["openfold3_version"] == "0.4.1" and p["upstream"]["commit"].startswith("d12f5955")
    for pref in ("OF3_", "OF3T_", "OF3FPF_", "OPENFOLD3_OPT", "CUBLAS_WORKSPACE_CONFIG"):
        assert pref in p["stock_environment"]["must_be_absent_prefixes"]
    assert set(p["stock_environment"]["reads"]) == {"OPENFOLD3_CKPT", "OPENFOLD_CACHE"}
    assert "image" not in p["pinned_stack"] and isinstance(p["pinned_stack"]["tested_on"], str)   # the one documentation line of the tested hardware/stack; no gate reads it
    for section, key in (("wheel", "sha256"), ("stock_config", "sha256"), ("source", "tree_sha256"), ("pinned_stack", "freeze_sha256")):
        assert key not in p[section], (section, key)                                         # no digest of an in-repo file outside "weights"
    assert stack.pinned_weights(HOME) == (p["weights"]["file"], p["weights"]["sha256"])      # the one digest this tree keeps: the downloaded checkpoint


def test_registry_sources_exist():
    for name, lv in registry.LEVERS.items():
        for ref in lv.source.split(";"):
            path = ref.strip().split(":")[0].split(" ")[0]
            if path.split("/")[0] in modes.KITS:
                path = os.path.join(modes.KITS[path.split("/")[0]], *path.split("/")[1:])
            elif lv.kit in modes.PACKAGE_HOOKS:                                                  # a package lever (confhead): its sources relative to the tree (opt/openfold3_opt/...)
                assert path.startswith("opt/openfold3_opt/"), (name, ref)
            else:
                path = os.path.join(modes.KITS[lv.kit], path)
            assert os.path.isfile(os.path.join(HOME, path)), (name, ref)
        assert lv.kit in modes.KITS or lv.kit in modes.PACKAGE_HOOKS, name


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
    for name, lv in registry.LEVERS.items():
        if not lv.marker:
            continue
        text = ""
        roots = [os.path.join(HOME, modes.KITS[lv.kit])] if lv.kit in modes.KITS else [modes.hook_dir(HOME, lv.kit), os.path.join(HOME, "opt", "openfold3_opt", lv.kit + ".py"), os.path.join(HOME, "opt", "openfold3_opt", lv.kit)]   # a package lever: its hook dir + its module or package: its hook + module
        for top in roots:
            walk = [(os.path.dirname(top), [], [os.path.basename(top)])] if os.path.isfile(top) else os.walk(top)
            for root, _, files in walk:
                for f in files:
                    if f.endswith(".py"):
                        text += open(os.path.join(root, f), encoding="utf-8", errors="replace").read()
        head = lv.marker.split(" ")[0]
        assert head in text, (name, lv.marker)

"""stock/PINS.json agrees with the lever tree's tables; check_pins.py runs standalone and names drift; the registry reads both."""
import json
import os
import subprocess

from gpnstar_opt import registry
from gpnstar_opt.tests._paths import CHECK_PINS, PINS_JSON, PY, env_clean, tree_or_skip


def test_pins_schema_and_agreement_with_the_lever_tables():
    tree_or_skip()
    pins = json.load(open(PINS_JSON))
    up = pins["upstream"]["gpn"]
    assert up["commit"] == registry.gpn_commit() and len(up["commit"]) == 40 and up["install"].endswith("@" + up["commit"])
    for k in ("torch", "transformers", "gpn"):
        assert k in pins["pins"]
    assert registry.cross_check(pins) == []
    models = registry.models()
    assert registry.PRIMARY in models and models[registry.PRIMARY]["repo"] == "songlab/gpn-star-hg38-v100-200m"
    ck = registry.checkpoint(registry.PRIMARY, pins)
    assert ck["rev"] == models[registry.PRIMARY]["revision"] and ck["files"], "the primary model's file digests are pinned"
    assert all(len(v["sha256"]) == 64 for v in ck["files"].values()) and "model.safetensors" in ck["files"]


def test_registry_resolves_keys_repos_and_snapshot_dirs():
    assert registry.resolve(None) == registry.PRIMARY == "v100-200m"
    assert registry.resolve("songlab/gpn-star-hg38-v100-200m") == "v100-200m" == registry.resolve("gpn-star-hg38-v100-200m")
    assert registry.resolve("/w/hub/models--songlab--gpn-star-ce11-n135-25m/snapshots/078ad60c23bbaf40fbeae89fb509a21320a71bbb") == "ce11-25m"
    assert registry.resolve("someone/else-model") is None
    assert registry.label("v100-200m") == "gpn-star-hg38-v100-200m@0c949f13"
    snap = registry.snapshot_dir("v100-200m", "/w")
    assert snap == "/w/hub/models--songlab--gpn-star-hg38-v100-200m/snapshots/0c949f132d35619a3eb188b402848c998a3313ae"


def test_cross_check_names_a_disagreeing_pins_file():
    pins = {"upstream": {"gpn": {"commit": "0" * 40}}, "weights": {"songlab/gpn-star-hg38-v100-200m": {"snapshot_commit": "1" * 40}}}
    bad = registry.cross_check(pins)
    assert len(bad) == 2 and "stock commit" in bad[0] and "revision" in bad[1]


def test_check_pins_runs_standalone(tmp_path):
    """stock/check_pins.py is standard-library only and runnable alone: on this CPU test environment (no gpn installed) the package
    check refuses (exit 3); an empty weights directory refuses (exit 3); a bad flag is a usage error (exit 2)."""
    tree_or_skip()
    p = subprocess.run([PY, "-I", CHECK_PINS], capture_output=True, text=True, env=env_clean(), timeout=120)
    assert p.returncode in (0, 3), (p.stdout, p.stderr)
    assert (p.stdout + p.stderr).strip(), "the check says what it found"
    hf = tmp_path / "hf"
    hf.mkdir()
    p = subprocess.run([PY, "-I", CHECK_PINS, "--skip-package", "--weights", str(hf)], capture_output=True, text=True, env=env_clean(), timeout=120)
    assert p.returncode == 3, (p.stdout, p.stderr)
    p = subprocess.run([PY, "-I", CHECK_PINS, "--bogus"], capture_output=True, text=True, env=env_clean(), timeout=60)
    assert p.returncode == 2


def test_pins_list_h100_and_a100_as_classes_and_the_card_gate_reads_them():
    tree_or_skip()
    pins = json.load(open(PINS_JSON))
    from gpnstar_opt import stack
    names, caps = stack.pins_card_table(pins)
    assert caps[("9.0", 81559)] == "h100" and caps[("8.0", 81920)] == "a100" and caps[("8.0", 40960)] == "a100"
    assert names["NVIDIA A100 80GB PCIe"] == "a100" and names["NVIDIA A100-PCIE-40GB"] == "a100"
    g = stack.card_gate({"name": "NVIDIA A100-SXM4-40GB", "mib": 40536, "cc": "8.0", "sm": "sm80"}, {}, pins)
    assert g == {"class": "a100", "words": []}
    g = stack.card_gate({"name": "NVIDIA L4", "mib": 22731, "cc": "8.9", "sm": "sm89"}, {}, pins)
    assert g["class"] is None and g["words"] == ["card=untested(NVIDIA_L4,22731MiB,sm89)"]


def test_pins_variants_are_served_models():
    tree_or_skip()
    pins = json.load(open(PINS_JSON))
    served = registry.models()
    for key in (pins.get("variants") or {}):
        assert key in served, key
    for repo, w in pins["weights"].items():
        if isinstance(w, dict) and w.get("snapshot_commit"):
            key = registry.resolve(repo)
            assert key is not None and served[key]["revision"] == w["snapshot_commit"], repo

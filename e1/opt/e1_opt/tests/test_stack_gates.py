"""The gates: the card gate by name AND memory (the kit's own card names), the stack key grammar,
the kernel snapshot search order (the kernels cache first, then the HF hub), the nvidia-smi probe, the must-be-absent set."""
import os

from e1_opt import stack, stock_score
from e1_opt.tests import _stubs


def test_card_names_are_the_kits_own_pin_table_names():
    pins = _stubs.pins_or_skip()
    assert set(stack.CARDS) <= set(pins.TRITON_AUTOTUNE_PIN), "the card gate's names must be spelled as the kit's pin table spells them"
    assert {cls for cls, _ in stack.CARDS.values()} == {"h100", "h200", "a100"}
    assert set(stack.CARD_CAPABILITY.values()) == {cls for cls, _ in stack.CARDS.values()}   # every class is reachable by capability too
    for name, (cls, _) in stack.CARDS.items():
        assert pins.gpu_class_key(name) == cls
    for name in ("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-40GB", "NVIDIA A100-PCIE-40GB", "A100-80GB", "A100-40GB"):
        assert pins.gpu_class_key(name) == "a100", name          # every A100 part is the a100 class (the kit's class-keyed A3 table: A3 off on all of them)
    for name, cls in (("NVIDIA H100 80GB HBM3", "h100"), ("NVIDIA H100 NVL", "h100"), ("NVIDIA H200", "h200"), ("NVIDIA H100 PCIe", "nvidia h100 pcie")):
        assert pins.gpu_class_key(name) == cls, name             # the H100 / H200 keys unchanged (an H100 PCIe stays an unknown class)


def test_card_class_and_notes(monkeypatch):
    """The card gate resolves the CLASS (a tested card by name, or the same silicon by capability + memory) and NAMES everything else
    about the card in `notes` — a card of no tested class, a memory reading off its class's nominal, a config targeting another class —
    never refusing: `ok` is False only without a GPU."""
    monkeypatch.delenv(stack.ENV_TARGET_GPU, raising=False)
    name, (cls, mib) = next(iter(stack.CARDS.items()))
    OK = lambda c: {"ok": True, "class": c, "notes": [], "reason": None}
    assert stack.card_gate({"name": name, "mib": mib}) == OK(cls)
    r = stack.card_gate({"name": name, "mib": mib - int(0.03 * mib)})                     # a tested name reading a memory 3 % off its nominal: the class, named
    assert r["ok"] and r["class"] == cls and len(r["notes"]) == 1 and f"reports {mib - int(0.03 * mib)} MiB, its class {cls} has {mib} MiB" in r["notes"][0]
    r = stack.card_gate({"name": "NVIDIA H100 NVL", "mib": 97871, "cc": "9.0"})            # no tested class has this (cc, memory): runs as no class, named
    assert r["ok"] and r["class"] is None and r["notes"][0].startswith("card 'NVIDIA H100 NVL' (cc 9.0, 97871 MiB) is outside the tested classes (a100|h100|h200)")
    r = stack.card_gate({"name": "NVIDIA H100 PCIe", "mib": mib})
    assert r["ok"] and r["class"] is None and "outside the tested classes" in r["notes"][0]
    # by capability: the A100 80GB under its PCIe name (cc 8.0, 81920 MiB) and both A100 40GB parts (cc 8.0, 40960 MiB) are class a100, nothing to name;
    # a cc 8.0 card of another memory size (an A30: 24576 MiB) is of no tested class — it runs, named
    assert stack.card_gate({"name": "NVIDIA A100 80GB PCIe", "mib": 81920, "cc": "8.0"}) == OK("a100")
    assert stack.card_gate({"name": "NVIDIA A100-SXM4-80GB", "mib": 81920, "cc": "8.0"}) == OK("a100")
    assert stack.card_gate({"name": "NVIDIA A100-SXM4-40GB", "mib": 40960, "cc": "8.0"}) == OK("a100")
    assert stack.card_gate({"name": "NVIDIA A100-PCIE-40GB", "mib": 40536, "cc": "8.0"}) == OK("a100")   # an older driver's reading of the 40 GB part
    assert stack.card_gate({"name": "NVIDIA A100-SXM4-40GB", "mib": 40441, "cc": "8.0"}) == OK("a100")   # torch's total_memory measured on the 40 GB part (1.3 % under nominal)
    assert stack.card_gate({"name": "NVIDIA A100-SXM4-80GB", "mib": 81050, "cc": "8.0"}) == OK("a100")   # a reading 1.1 % under nominal (in tolerance)
    r = stack.card_gate({"name": "NVIDIA A30", "mib": 24576, "cc": "8.0"})
    assert r["ok"] and r["class"] is None and "outside the tested classes" in r["notes"][0] and "cc 8.0, 24576 MiB" in r["notes"][0]
    r = stack.card_gate({"name": "NVIDIA A100-SXM4-80GB", "mib": 61440, "cc": "8.0"})     # a tested name, a memory no A100 has: the class by name, the reading named
    assert r["ok"] and r["class"] == "a100" and "reports 61440 MiB, its class a100 has 81920 MiB" in r["notes"][0]
    g = stack.card_gate(None)
    assert not g["ok"] and g["reason"].startswith("no GPU visible") and g["notes"] == []   # the one refusal: nothing can run
    monkeypatch.setenv(stack.ENV_TARGET_GPU, "a100")                                     # configs/a100.env's target class holds on either memory size
    assert stack.card_gate({"name": "NVIDIA A100-SXM4-40GB", "mib": 40960, "cc": "8.0"}) == OK("a100")
    assert stack.card_gate({"name": "NVIDIA A100-SXM4-80GB", "mib": 81920, "cc": "8.0"}) == OK("a100")
    monkeypatch.setenv(stack.ENV_TARGET_GPU, "h200")                                     # a config targeting another class than the card's: named, never refused
    r = stack.card_gate({"name": name, "mib": mib})
    assert r["ok"] and r["class"] == cls and r["notes"] == [f"the config targets h200 (MODEL_OPT_TARGET_GPU), this card is class {cls}: the config's constants are used as given"]
    monkeypatch.setenv(stack.ENV_TARGET_GPU, cls.upper())
    assert stack.card_gate({"name": name, "mib": mib}) == OK(cls)


def test_stack_key_grammar(monkeypatch):
    monkeypatch.delitem(__import__("sys").modules, "torch", raising=False)
    assert stack.stack_key({"sm": "sm90"}, {"torch": "2.8.0+cu128"}) == "torch2.8.0-cu128-sm90"
    assert stack.stack_key(None, {"torch": "2.8.0"}) == "torch2.8.0-cuNA-smNA"
    assert stack.stack_key({"sm": "sm100"}, {"torch": None}) == "torchNA-cuNA-sm100"


def test_fake_nvidia_smi_probe(monkeypatch, tmp_path):
    d = _stubs.write_fake_nvidia_smi(str(tmp_path), "NVIDIA H200", 143771, "9.0")
    monkeypatch.setenv("PATH", d + os.pathsep + os.environ.get("PATH", ""))
    g = stack.probe_gpu_nvidia_smi()
    assert g == {"name": "NVIDIA H200", "mib": 143771, "cc": "9.0", "sm": "sm90", "source": "nvidia-smi"}
    assert stack.card_gate(g)["class"] == "h200"


def test_kernel_snapshot_search_order(monkeypatch, tmp_path):
    pins = _stubs.pins_or_skip()
    caches = _stubs.write_stub_caches(str(tmp_path), pins, "300m")
    monkeypatch.setenv("KERNELS_CACHE", caches["kernels_cache"])
    monkeypatch.setenv("HF_HOME", caches["hf_home"])
    k = stack.kernel_snapshot(pins)
    assert k["present"] and k["root"] == caches["kernels_cache"] and k["layer_norm_py_sha256_ok"] is False   # a placeholder file: present, drifted
    assert stack.kernel_cache_roots()[0] == caches["kernels_cache"] and stack.kernel_cache_roots()[-1] == os.path.join(caches["hf_home"], "hub")
    monkeypatch.delenv("KERNELS_CACHE")
    k = stack.kernel_snapshot(pins)
    assert not k["present"] and k["searched"] == [os.path.join(caches["hf_home"], "hub", "models--" + pins.KERNEL["repo"].replace("/", "--"), "snapshots", pins.KERNEL["rev"])]
    monkeypatch.setitem(pins.KERNEL, "layer_norm_py_sha256", caches["kernel_sha256"])
    monkeypatch.setenv("HF_KERNELS_CACHE", caches["kernels_cache"])
    assert stack.kernel_snapshot(pins)["layer_norm_py_sha256_ok"] is True


def test_must_be_absent_sets_agree_between_stack_and_the_stock_runner():
    assert stack.MUST_BE_ABSENT_PREFIXES == stock_score.MUST_BE_ABSENT_PREFIXES
    assert stack.KIT_MODULE_PREFIXES == stock_score.KIT_MODULE_PREFIXES and stack.KIT_DIR_MARKERS == stock_score.KIT_DIR_MARKERS
    from e1_opt import report
    import re
    pat = re.compile(report.ENV_CHECK_PATTERN)
    for p in stack.MUST_BE_ABSENT_PREFIXES:
        assert pat.match(p + "_X"), p
    assert not pat.match("HF_HOME=x") and not pat.match("TRITON_CACHE_DIR=x")


def test_paths_follow_model_opt(monkeypatch, tmp_path):
    monkeypatch.setenv(stack.ENV_HOME, str(tmp_path))
    assert stack.tree_home() == str(tmp_path) and stack.forward_root() == str(tmp_path / "opt" / "forward")
    assert stack.kit_entry().endswith(os.path.join("kits", "eager", "__init__.py")) and not hasattr(stack, "kit_script")
    monkeypatch.delenv(stack.ENV_HOME)
    assert stack.tree_home() == os.path.dirname(stack.opt_home())


def test_card_gate_memory_within_two_percent():
    """The card's memory reading is its class's when within 2 % of the class's nominal nvidia-smi total (an older driver's ECC carve-out and
    torch's total_memory read up to ~1.5 % lower); a reading further off is NAMED (a note), never refused; no reading at all is named too."""
    from e1_opt import stack
    assert stack.CARD_MIB_TOLERANCE == 0.02
    name, (cls, want) = next(iter(stack.CARDS.items()))
    for mib in (want, want - int(0.006 * want), want - int(0.015 * want)):
        assert stack.card_gate({"name": name, "mib": mib}) == {"ok": True, "class": cls, "notes": [], "reason": None}
    g = stack.card_gate({"name": name, "mib": want - int(0.03 * want)})
    assert g["ok"] and g["class"] == cls and len(g["notes"]) == 1 and "reports" in g["notes"][0]
    g = stack.card_gate({"name": name, "mib": None})
    assert g["ok"] and g["class"] == cls and "reports None MiB" in g["notes"][0]


def test_the_kit_reads_no_environment_variable_of_its_own():
    """The carried kit (opt/forward/engines/e1/kits/) reads NO environment name of its own — no mode switch, no card value, no optional
    add-on: the only `E1_KIT*` token in its sources is the bare prefix the package's clean-environment proof scans for, and the package
    names no card values (`_names` has no KIT_CARD_VALUES). A name the kit grows fails here."""
    import os, re
    from e1_opt import _names, stack
    kits = os.path.join(stack.tree_home(), "opt", "forward", "engines", "e1", "kits")
    found, getenv = set(), set()
    for d, _, fs in os.walk(kits):
        for f in fs:
            if f.endswith(".py") and "/tests" not in d:
                src = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
                found |= set(re.findall(r"\bE1_KIT[A-Z0-9_]*", src))
                getenv |= set(re.findall(r"os\.environ(?:\.get)?[\[(]\s*[\"']([A-Z0-9_]+)", src)) | set(re.findall(r"os\.getenv\(\s*[\"']([A-Z0-9_]+)", src))
    assert found <= {"E1_KIT"}, sorted(found)
    assert getenv <= {"HF_HOME", "HF_HUB_CACHE", "KERNELS_CACHE", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
                      "USE_FLASH_ATTN", "FAST_BLOCK_MASK", "MODEL_OPT_TARGET_GPU"}, sorted(getenv)   # the caches', the recipe's, upstream's own switches, and the config's target-class name — no kit switch
    assert not hasattr(_names, "KIT_CARD_VALUES") and not hasattr(stack, "hand_card_values")


def test_a_prefixed_variable_the_kit_never_reads_is_not_a_refusal(monkeypatch, tmp_path, capsys):
    """The package composes the kit's line itself and the kit reads no switch: a caller's own `E1_KIT*` / `E1_OPT_*` bookkeeping variable
    is not a refusal under a kit mode (the stock subprocess strips and proves absent every such prefix regardless: test_off_env_proof)."""
    from e1_opt import cli, report
    box = _stubs.setup_box(monkeypatch, tmp_path)
    _stubs.patch_pins_for_stub(monkeypatch, box["pins"], box["variant"], box["caches"])
    monkeypatch.setenv("E1_KIT_HOME_VAR", "x"); monkeypatch.setenv("KIT_PYTHONPATH", "opt"); monkeypatch.setenv("E1_OPT_KIT", "/somewhere")
    rc = cli.main(["check", "--variant", box["variant"], "--mode", "exact"])
    line = capsys.readouterr().out.splitlines()[0]
    assert rc == 0 and report.RE_DRY_RUN.match(line).group("would_refuse") == "none", line

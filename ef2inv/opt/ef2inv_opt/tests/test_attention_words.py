"""The attention words and everything keyed on flash-attn, on a CPU box without flash-attn or the stack: the one reader over fake upstream
modules (flags true / false, bound callables, a flash attention class on a live model), the words' grammar incl. det_attn, the fail-loud
guard's refusal sentence (EF2INV_REQUIRE_FAST_ENV, every word incl. esmc_mlp), the det recipe's attention element and ESM-C's RoPE pin on fake modules, the
launcher's argv, stock/check_pins.py's flash-attn pin with stubbed metadata, the PINS image block, configs/h100.env's export."""
import functools
import json
import os
import subprocess
import types

import pytest

from . import _paths as P
from .. import attention as AT, det as D, patches as PT, launch as L, modes as MD, report as R, cli


def _fn(module, name):
    f = types.FunctionType((lambda *a, **k: (a, k)).__code__, {}, name)
    f.__module__ = module; f.__qualname__ = name
    return f


class _TE:                                            # transformer_engine.pytorch as the fork imports it (`te`): the three classes it builds blocks from
    class LayerNormMLP: pass
    class LayerNormLinear: pass
    class Linear: pass


class _PyTorchLayerNormMLP: pass
class _PyTorchLayerNormLinear: pass


def fake_modules(flash: bool, xformers: bool = True, rotary=None, te: bool = True):
    """{name: module} as the fork imports them: with flash-attn / transformer_engine (flags True, callables and `te` bound) or without."""
    rotary = flash if rotary is None else rotary
    C = types.ModuleType(AT.FOLD_MODULE)
    C.FLASH_ATTN_AVAILABLE = flash
    C.flash_attn_func = _fn("flash_attn.flash_attn_interface", "flash_attn_func") if flash else None
    C.flash_attn_varlen_func = _fn("flash_attn.flash_attn_interface", "flash_attn_varlen_func") if flash else None
    E = types.ModuleType(AT.ESMC_MODULE)
    E._xformers_available = xformers; E._flash_attn_available = flash; E._flash_attn_rotary_available = rotary
    E.flash_attn_func = _fn("flash_attn.flash_attn_interface", "flash_attn_func") if flash else None
    E.flash_attn_varlen_qkvpacked_func = _fn("flash_attn.flash_attn_interface", "flash_attn_varlen_qkvpacked_func") if flash else None
    E.apply_triton_rotary = _fn("flash_attn.ops.triton.rotary", "apply_rotary") if rotary else None
    class MultiHeadAttention:                       # noqa: D401 — stand-ins for the fork's two attention classes
        pass
    class _FlashMultiHeadAttention:
        pass
    E.MultiHeadAttention, E._FlashMultiHeadAttention = MultiHeadAttention, _FlashMultiHeadAttention
    E._te_available = te; E.te = _TE if te else None
    E._PyTorchLayerNormMLP, E._PyTorchLayerNormLinear = _PyTorchLayerNormMLP, _PyTorchLayerNormLinear
    return {AT.FOLD_MODULE: C, AT.ESMC_MODULE: E}


def te_model(n_blocks=2):
    return _Model([m for _ in range(n_blocks) for m in (_TE.LayerNormLinear(), _TE.Linear(), _TE.LayerNormMLP())])


def torch_model(n_blocks=2):
    return _Model([m for _ in range(n_blocks) for m in (_PyTorchLayerNormLinear(), _PyTorchLayerNormMLP())])


class _Model:
    def __init__(self, subs, use_flash=False):
        self._subs = list(subs); self._use_flash_attn = use_flash
    def modules(self):
        return iter([self] + self._subs)


# ---- the reader and the words -------------------------------------------------------------------------------------------------
KW = dict(flash_version="2.8.3", te_ver="2.15.0")


def test_words_on_the_pinned_image():
    mods = fake_modules(True)
    rec = AT.read(esmc_models=[te_model()], modules=mods, **KW)
    assert rec["words"] == {"esmc_mlp": "te", "esmc_attn": "xformers", "esmc_rope": "flash_triton", "fold_atom_attn": "flash_attn"}   # xformers stays ESM-C's kernel
    assert AT.words(rec) == "esmc_mlp=te esmc_attn=xformers esmc_rope=flash_triton fold_atom_attn=flash_attn"
    assert rec["fold"]["flash_attn_func"] == "flash_attn.flash_attn_interface.flash_attn_func" and rec["flash_attn_version"] == "2.8.3" and rec["transformer_engine_version"] == "2.15.0"
    assert rec["esmc"]["te_modules"] == 6 and rec["esmc"]["torch_fallback_modules"] == 0
    assert AT.require_problems(rec, "im-NEW", "im-BASE") == [] and rec["require_exempt"] == []
    assert rec["words"] == AT.FAST_ENV                                                                                 # the pinned stack's values, word for word


def test_esmc_mlp_reads_the_built_modules_not_the_flag():
    mods = fake_modules(True)                                                                                          # te importable ...
    rec = AT.read(esmc_models=[torch_model()], modules=mods, **KW)                                                     # ... but the blocks hold the torch fallbacks
    assert rec["words"]["esmc_mlp"] == "torch" and rec["esmc"]["torch_fallback_modules"] == 4
    assert "esmc_mlp=torch (expected te" in AT.require_problems(rec, "im-NEW")[0]
    assert AT.read(esmc_models=[te_model(), torch_model()], modules=mods, **KW)["words"]["esmc_mlp"] == "mixed"
    assert AT.read(modules=mods, **KW)["words"]["esmc_mlp"] == "te"                                                    # no model passed (check): the flag + the bound te classes
    assert AT.read(modules=fake_modules(True, te=False), **KW)["words"]["esmc_mlp"] == "torch"


def test_words_on_the_base_image():
    rec = AT.read(esmc_models=[torch_model()], modules=fake_modules(False, te=False), flash_version=None, te_ver=None)
    assert AT.words(rec) == "esmc_mlp=torch esmc_attn=xformers esmc_rope=torch fold_atom_attn=sdpa"
    rec = AT.read(modules=fake_modules(False, xformers=False, te=False), flash_version=None)
    assert rec["words"]["esmc_attn"] == "sdpa"
    rec = AT.read(modules=fake_modules(True, xformers=False), **KW)
    assert rec["words"]["esmc_attn"] == "flash_attn"                                                                  # no xformers: the fork's second choice
    assert "esmc_attn=flash_attn (expected xformers" in AT.require_problems(rec)[0]                                    # fast, but not the pinned stack's kernel: refused


def test_words_unread_without_the_stack():
    rec = AT.read(modules={}, flash_version=None, te_ver=None)
    assert AT.words(rec) == "esmc_mlp=unread esmc_attn=unread esmc_rope=unread fold_atom_attn=unread"
    assert AT.require_problems(rec, "im-NEW")                                                                          # unread is refused under REQUIRE, never passed


def test_flash_class_bound_on_a_live_model_reads_flash_attn():
    mods = fake_modules(True)
    E = mods[AT.ESMC_MODULE]
    plain = _Model([E.MultiHeadAttention(), E.MultiHeadAttention()])
    assert AT.read(esmc_models=[plain], modules=mods, **KW)["words"]["esmc_attn"] == "xformers"
    flash = _Model([E._FlashMultiHeadAttention()])
    rec = AT.read(esmc_models=[plain, flash], modules=mods, **KW)
    assert rec["words"]["esmc_attn"] == "flash_attn" and rec["esmc"]["flash_mha_modules"] == 1 and rec["esmc"]["models_read"] == 2
    assert AT.read(esmc_models=[_Model([], use_flash=True)], modules=mods, **KW)["words"]["esmc_attn"] == "flash_attn"


def test_det_attn_word_and_line_grammar():
    mods = fake_modules(True)
    det_rec = {"det_attn": "flash_det"}
    rec = AT.read(det=det_rec, modules=mods, **KW)
    assert AT.words(rec) == "esmc_mlp=te esmc_attn=xformers esmc_rope=flash_triton fold_atom_attn=flash_attn det_attn=flash_det"
    assert R.attn_line(AT.words(rec), rec, True) == ("ATTN esmc_mlp=te esmc_attn=xformers esmc_rope=flash_triton fold_atom_attn=flash_attn det_attn=flash_det "
                                                   "flash_attn=2.8.3 transformer_engine=2.15.0 xformers=none require_fast_env=1")
    base = AT.read(modules=fake_modules(False, te=False), flash_version=None, te_ver=None)
    assert R.attn_line(AT.words(base), base, False).endswith("fold_atom_attn=sdpa flash_attn=none transformer_engine=none xformers=none require_fast_env=0")
    E = mods[AT.ESMC_MODULE]; PT.pin_esmc_rope(module=E)
    pinned = AT.read(esmc_models=[te_model()], modules=mods, **KW)
    assert AT.require_problems(pinned, "im-NEW", rope_pin="torch") == [] and pinned["require_exempt"][0].startswith("esmc_rope=torch (upstream fix EF2INV-0003")
    assert R.attn_line(AT.words(pinned), pinned, True).endswith("require_fast_env=1 exempt=esmc_rope=torch")           # the exemption is named on the line
    unfixed = AT.read(esmc_models=[te_model()], modules=mods, **KW)                                                      # the same words WITHOUT upstream fix EF2INV-0003 requested: esmc_rope=torch is not exempt
    probs = AT.require_problems(unfixed, "im-NEW", rope_pin=None)
    assert len(probs) == 1 and "esmc_rope" in probs[0] and not unfixed["require_exempt"]


# ---- the fail-loud guard --------------------------------------------------------------------------------------------------------
def test_require_guard_words_by_name():
    rec = AT.read(esmc_models=[torch_model()], modules=fake_modules(False, te=False), flash_version=None, te_ver=None)   # = the base environment without the two wheels
    probs = AT.require_problems(rec, None, None, rope_pin="torch")                                                       # the callers' positional form (cli, stock_design)
    assert len(probs) == 1 and probs == AT.require_problems(rec, rope_pin="torch")
    s = probs[0]
    assert s.startswith("fast environment (EF2INV_REQUIRE_FAST_ENV=1) NOT present: ") and "image" not in s and "refused" not in s and s.endswith("the levers engage on what is bound")
    for needle in ("flash_attn is not installed", "transformer_engine is not installed", "does not carry the pinned stack",
                   "esmc_mlp=torch (expected te; transformers.models.esmc.modeling_esmc._te_available=False)",
                   "fold_atom_attn=sdpa (expected flash_attn; transformers.models.esmfold2.modeling_esmfold2_common.FLASH_ATTN_AVAILABLE=False)"):
        assert needle in s, (needle, s)
    assert "esmc_rope" not in s and rec["require_exempt"]                                                               # torch RoPE under the torch pin: exempt, named apart
    unpinned = AT.require_problems(rec, rope_pin="flash_triton")
    assert "esmc_rope=torch (expected flash_triton" in unpinned[0]                                                     # asked flash_triton, got torch: refused
    rec = AT.read(esmc_models=[te_model()], modules=fake_modules(True), flash_version=None, te_ver="2.15.0")           # module bound it but no distribution metadata: still not the stack
    assert "flash_attn is not installed" in AT.require_problems(rec)[0]


def test_require_var_is_the_cli_s_and_never_an_arm_s(monkeypatch):
    assert AT.REQUIRE_VAR == "EF2INV_REQUIRE_FAST_ENV" and AT.REQUIRE_VAR.startswith("EF2INV_")                        # the package prefix: stripped from every arm (launch.arm_env) and proven absent
    monkeypatch.delenv(AT.REQUIRE_VAR, raising=False); assert cli.require_fast_env() is False
    monkeypatch.setenv(AT.REQUIRE_VAR, "1"); assert cli.require_fast_env() is True
    monkeypatch.setenv(AT.REQUIRE_VAR, "0"); assert cli.require_fast_env() is False
    env = L.arm_env(MD.MODES["off"], P.PINS["stock_environment"]["must_be_absent_prefixes"], base={AT.REQUIRE_VAR: "1", "PATH": "/bin"})
    assert AT.REQUIRE_VAR not in env


def test_h100_env_exports_the_requirement():
    r = subprocess.run(["bash", "-c", f"set -a; . {os.path.join(P.ROOT, 'configs', 'h100.env')} >/dev/null 2>&1; echo $EF2INV_REQUIRE_FAST_ENV"], capture_output=True, text=True,
                       env={"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp")})
    assert r.stdout.strip() == "1", r
    r = subprocess.run(["bash", "-c", f"set -a; EF2INV_REQUIRE_FAST_ENV=0; . {os.path.join(P.ROOT, 'configs', 'h100.env')} >/dev/null 2>&1; echo $EF2INV_REQUIRE_FAST_ENV"], capture_output=True, text=True,
                       env={"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp")})
    assert r.stdout.strip() == "0", r                                                                                    # an explicit 0 before sourcing is kept


# ---- the det recipe's attention element -----------------------------------------------------------------------------------------
def test_det_attn_flash_det_rebinds_deterministic_partials():
    mods = fake_modules(True)
    orig = mods[AT.FOLD_MODULE].flash_attn_varlen_func
    rec = D.apply_attn(modules=mods)
    C, E = mods[AT.FOLD_MODULE], mods[AT.ESMC_MODULE]
    for fn in (C.flash_attn_func, C.flash_attn_varlen_func, E.flash_attn_func, E.flash_attn_varlen_qkvpacked_func):
        assert isinstance(fn, functools.partial) and fn.keywords == {"deterministic": True}
    assert C.flash_attn_varlen_func.func is orig and C.FLASH_ATTN_AVAILABLE is True and E._flash_attn_available is True
    assert E._xformers_available is False and rec["flags"][AT.ESMC_MODULE + "._xformers_available"] is False           # ESM-C attention leaves xformers under the recipe
    assert len(rec["rebound"]) == 5 and rec["det_attn"] == "flash_det" and AT.ESMC_MODULE + "._xformers_available=False" in rec["rebound"]
    _, kw = C.flash_attn_varlen_func("q", "k", "v", softmax_scale=0.5, window_size=(64, 64))                          # the fork's call form gains deterministic=True
    assert kw == {"softmax_scale": 0.5, "window_size": (64, 64), "deterministic": True}
    assert D.apply_attn(modules=mods)["rebound"] == []                                                    # idempotent: never a partial of a partial
    assert AT.read(modules=mods, **KW)["fold"]["flash_attn_func"] == "flash_attn.flash_attn_interface.flash_attn_func(deterministic=True)"
    det = AT.read(modules=mods, det=rec, **KW)
    assert AT.words(det) == "esmc_mlp=te esmc_attn=flash_attn esmc_rope=flash_triton fold_atom_attn=flash_attn det_attn=flash_det"      # esmc_attn: the module's flash_attn_func(deterministic=True) branch
    assert AT.require_problems(det, "im-NEW") == [] and det["require_exempt"] == ["esmc_attn=flash_attn (det_attn=flash_det: xformers bypassed under the det recipe for flash_attn_func(deterministic=True))"]
    assert R.attn_line(AT.words(det), det, True).endswith("det_attn=flash_det flash_attn=2.8.3 transformer_engine=2.15.0 xformers=none require_fast_env=1 exempt=esmc_attn=flash_attn")
    speed = AT.read(modules=fake_modules(True), **KW)                                                                   # a speed pass (no det record): esmc_attn=flash_attn without xformers is refused, never exempt
    speed["words"]["esmc_attn"] = "flash_attn"
    assert "esmc_attn=flash_attn (expected xformers" in AT.require_problems(speed, "im-NEW")[0]


def test_det_attn_without_flash_attn_is_a_no_op_and_choices():
    mods = fake_modules(False, xformers=False)
    assert D.apply_attn(modules=mods)["rebound"] == []
    base = fake_modules(False)                                                                                          # the base image: xformers only — the recipe still moves ESM-C attention off it (F.scaled_dot_product_attention)
    assert D.apply_attn(modules=base)["rebound"] == [AT.ESMC_MODULE + "._xformers_available=False"] and AT.words(AT.read(modules=base)) .split()[1] == "esmc_attn=sdpa"
    assert D.DET_ATTN == "flash_det" and D.apply_attn(modules=fake_modules(True))["det_attn"] == D.DET_ATTN     # one attention element; its word rides the ATTN line
    assert D.apply(0) == {} and "det_attn" in D.LEVEL_WHAT[1]


# ---- ESM-C's RoPE pin -----------------------------------------------------------------------------------------------------------
def test_esmc_rope_pin():
    E = fake_modules(True)[AT.ESMC_MODULE]
    rec = PT.pin_esmc_rope(module=E)
    assert rec["applied"] is True and rec["as_imported"] is True and rec["effective"] == "torch" and E._flash_attn_rotary_available is False
    assert rec["name"] == "esmc_rope_pin" and rec["impl"] == "torch" and "modeling_esmc._flash_attn_rotary_available" in rec["target"]
    assert AT.read(modules={AT.ESMC_MODULE: E, AT.FOLD_MODULE: fake_modules(True)[AT.FOLD_MODULE]}, **KW)["words"]["esmc_rope"] == "torch"
    rec = PT.pin_esmc_rope(module=E)                                                                          # already torch: nothing to rebind, still effective
    assert rec["applied"] is False and rec["effective"] == "torch"
    E2 = fake_modules(False)[AT.ESMC_MODULE]                                                                    # a box where flash-attn's kernel was never bound: nothing to rebind, the record says so
    rec = PT.pin_esmc_rope(module=E2)
    assert rec["applied"] is False and rec["as_imported"] is False and rec["effective"] == "torch" and E2._flash_attn_rotary_available is False
    assert PT.ESMC_ROPE_FIX_IMPL == "torch" and "impl" not in PT.pin_esmc_rope.__code__.co_varnames[:PT.pin_esmc_rope.__code__.co_argcount]   # one implementation: no selector


# ---- the launcher's argv -----------------------------------------------------------------------------------------------------------
def test_argv_carries_the_fast_env_flag_only_when_set_and_no_element_selector():
    """The det recipe's attention element is det.DET_ATTN and ESM-C's RoPE has no selector of its own (upstream fix
    EF2INV-0003 rides `--upstream-fix`, test_upstream_fix): no element selector rides the arm argv; the fast-env requirement bit does, only when set."""
    import inspect
    kw = dict(cookbook_stock="/x/binder_design.py", target_name="c", target_sequence="ACDEFGHIK", binder_len=80, seed=3, out="/o", python="py")
    base = L.argv_for(MD.MODES["exact"], P.ROOT, **kw)
    assert "--det-attn" not in base and "--esmc-rope" not in base and "--require-fast-env" not in base
    assert not {"det_attn", "esmc_rope"} & set(inspect.signature(L.argv_for).parameters)
    argv = L.argv_for(MD.MODES["exact"], P.ROOT, det=1, require_fast_env=1, **kw)
    assert argv[argv.index("--require-fast-env") + 1] == "1" and "--det-attn" not in argv and "--esmc-rope" not in argv


# ---- stock/check_pins.py: the flash-attn pin refuses the base image by name ------------------------------------------------------------
def _check_pins_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_pins_under_test_fa", os.path.join(P.ROOT, "stock", "check_pins.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_check_pins_names_the_base_image_and_runs():
    """The pinned stack's two wheels are warn-and-run: another version, or none, is one `stack <key>=<got> (pinned <want>) NOT PINNED — …` line
    per wheel (facts['stack_lines'] in check()), never a fail."""
    cp = _check_pins_module(); sor = P.PINS["pinned_stack"]
    installed = {"flash-attn": sor["flash_attn"], "transformer_engine": sor["transformer_engine"]}
    facts = {}
    assert cp.dist_pins_check(P.PINS, facts, version_of=installed.get) == [] and facts == {"flash_attn_version": sor["flash_attn"], "transformer_engine_version": sor["transformer_engine"]}
    notes = cp.dist_pins_check(P.PINS, facts, version_of=lambda name: None)                                            # the base environment: neither distribution
    assert notes == [f"stack {k}=None (pinned {sor[k]}) {cp.STACK_NOT_PINNED_NOTE} "
                     f"(the sha256-pinned {k} wheel of stock/PINS.json pinned_stack.image_recipe; STOCK.md §Pinned stack)" for k in ("flash_attn", "transformer_engine")]
    assert all("NOT PINNED" in n for n in notes) and not [n for n in notes if "im-" in n or "image " in n or "REFUSED" in n or "NOT MET" in n]
    assert cp.dist_pins_check(P.PINS, {}, version_of={"flash-attn": "2.8.1", "transformer_engine": sor["transformer_engine"]}.get) == [notes[0].replace("=None", "=2.8.1")]
    assert len(cp.dist_pins_check(P.PINS, {}, version_of={"flash-attn": sor["flash_attn"], "transformer_engine": "2.15.0"}.get)) == 1        # the local version tag is part of the pin


def test_check_pins_names_an_unset_weights_root(monkeypatch):
    """No stored weights path: with neither --hf-home nor HF_HOME the weights gate refuses by the variable's name and digests nothing."""
    cp = _check_pins_module()
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(cp, "dist_pins_check", lambda pins, facts, version_of=None: [])
    monkeypatch.setattr(cp, "esm_check", lambda *a, **k: [], raising=False)
    fails, facts = cp.check({"weights": P.PINS["weights"], "transformers_fork": {"commit": "", "installed_files_sha256": {}}}, weights=False)
    named = [f for f in fails if f.startswith("HF_HOME is not set — see README.md §Variables")]      # the variable is named, never a path
    assert len(named) == 1 and "/" not in named[0].split("biohub/*")[0] and facts["hf_home"] is None and facts["weights"] == {} and facts["weights_lines"] == []
    assert cp.dist_pins_check({"weights": {}}, {}, version_of=lambda name: None) == []                                # no stack pin, nothing to meet


def test_pins_image_block():
    sor = P.PINS["pinned_stack"]
    assert sor["flash_attn"] == "2.8.3" and sor["transformer_engine"].startswith("2.15.0") and sor["torch"] == "2.11.0+cu128" and sor["cuda"] == "12.8" and sor["python"] == "3.12.14"   # two distributions added; the stack unchanged
    assert sor["image_recipe"]["image_env"]["XFORMERS_IGNORE_FLASH_VERSION_CHECK"] == "1" and sor["image_recipe"]["image_env"]["NVTE_FRAMEWORK"] == "pytorch"
    te = sor["image_recipe"]["transformer_engine_wheels"]
    assert te["files"] and all(len(te["sha256"][f]) == 64 for f in te["files"]) and "2.15.0" in te["source"]
    rec = sor["image_recipe"]
    assert "image" not in sor and "base" not in rec and "interpreter" not in sor                                       # the stack is pinned by versions and digests only; the tested-on note is documentation nothing reads
    assert sor["tested_on"].startswith("the pinned stack") and "key" not in sor and "im-" not in json.dumps(sor)   # the stack is named by its versions alone — no label of any other kind
    assert not [w for w in P.PINS["weights"].values() if "hf_home" in w]                                                # the weights root is the operator's HF_HOME, never a stored path
    w = rec["flash_attn_wheel"]
    assert w["file"].startswith("flash_attn-2.8.3") and w["file"].endswith(".whl") and len(w["sha256"]) == 64 and int(w["sha256"], 16) >= 0
    assert "pip install --no-deps" in rec["layer"] and "flash-attn==2.8.3" in w["source"]


def test_reader_keeps_stdout_clean(capsys, tmp_path, monkeypatch):
    """`check --json` is one JSON document: an upstream module that prints at import (the fork's auto_docstring notices) is imported by the
    reader with its stdout sent to stderr."""
    pkg = tmp_path / "noisy_upstream_pkg"; pkg.mkdir(); (pkg / "__init__.py").write_text("print('NOISE at import')\nFLAG = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    m = AT._module("noisy_upstream_pkg")
    cap = capsys.readouterr()
    assert m is not None and m.FLAG is True and "NOISE" not in cap.out and "NOISE at import" in cap.err

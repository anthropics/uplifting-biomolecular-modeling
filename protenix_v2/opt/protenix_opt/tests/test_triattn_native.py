"""Lever triattn_native (protenix_opt 0.3.51): the block core's Tier-2 triangle attention through the shared core's provider opt_core.kernels.triattn
BY THE TIER WORD ALONE — registry row, README rows (the cc-9.0 and cc-8.0 fast rows pre-export the MODE's tier word PTX_T_ATT=fast, big names
big), env.sh's case on the word (fast | big -> the provider slot, no alias words, no kit floor), the adapter's source contract (word from the
mode, prefer=None, no row-name pin, no kit kernel behind it), the marker, the LEVER evidence off the module's report(), the ablation word and the
import-time refusal by name on a CUDA-less host."""
import os
import re
import sys
import types
from unittest import mock

from protenix_opt import modes, registry, report, stack

NATIVE, K2B = "triattn_native", "k2b_flash_triattention"
FPF = stack.kit_home()
SRC = os.path.join(FPF, "src", "ptx_native_core.py")


def _src():
    with open(SRC, encoding="utf-8") as fh:
        return fh.read()


def test_registry_row():
    lv = registry.LEVERS[NATIVE]
    assert lv.tier == registry.TOLERANCE and lv.probe == "marker" and lv.extra and lv.row_dependent
    assert lv.env_keys == ("PTX_T_ATT",) and lv.words == (("PTX_T_ATT", "fast"), ("PTX_T_ATT", "big")), "the mode's own tier words; no alias"
    assert lv.conditional == ("PTX_BLK_ATT", "PTX_BLK_ATT_PROVIDER_MIN_TOKENS"), "the lever has no kit floor variable"
    assert registry.LEVERS[K2B].replaced_by == (NATIVE,) and NATIVE in registry.TRIATT_STATEMENT_LEVERS
    for gone in ("triattn_cuda", "triatt_headsplit_exact", "triattn_exact"):
        assert gone not in registry.LEVERS, f"{gone}: not a lever of this kit (the shared core's rows serve that op)"
    assert NATIVE in modes.MODES["fast"] and NATIVE in modes.MODES["big"] and NATIVE in modes.big_levers() and NATIVE not in modes.MODES["exact"]
    assert report.STRATEGY_IDS[NATIVE] == "F1.flash_triatt" and report.IMPL[NATIVE] == ("opt_core.kernels.triattn", "core")
    assert stack.MARKERS[NATIVE] == ("BLK", ("BLK:2(pro+PROVIDER(ptx_native_core:attn",))
    assert "prefer" not in lv.description.lower() or "no row preference" in lv.description


def test_rows_pre_export_the_tier_word_on_the_cards_with_cells():
    for key, row in modes.README_ROWS.items():
        want = "fast" if key.split("|")[0] in ("9.0", "8.0") else None
        assert row["fast"]["pre"].get("PTX_T_ATT") == want, key
        assert "PTX_T_ATT" not in row["exact"]["pre"] and "PTX_T_ATT" not in row["exact"]["post"], key
    for key in ("9.0|3.7", "9.0|3.3", "9.0|*", "8.0|3.7", "8.0|*"):
        assert modes.readme_row(key, "big")["pre"].get("PTX_T_ATT") == "big", key
    for key in ("10.0|3.7", "10.3|3.7", "other"):
        assert "PTX_T_ATT" not in modes.readme_row(key, "big")["pre"], f"{key}: a card without provider cells keeps env.sh's default (k2b)"
    for words in (list(r["fast"]["pre"].values()) for r in modes.README_ROWS.values()):
        assert "native" not in words and "core" not in words and "triattn_cuda" not in words, "no alias / row words for tri-attention"


def test_env_sh_case():
    env = open(os.path.join(FPF, "env.sh"), encoding="utf-8").read()
    case = env[env.index('case "${PTX_T_ATT:-k2b}" in'):env.index("esac", env.index('case "${PTX_T_ATT:-k2b}" in'))]
    assert re.search(r"^\s*fast\|big\) export PTX_BLK_ATT=ptx_native_core:attn PTX_BLK_ATT_PROVIDER_MIN_TOKENS=0 ;;", case, re.M), case
    assert re.search(r"^\s*k2b\) : ;;", case, re.M)
    assert "native)" not in case and "core|" not in case and "triattn_cuda" not in case and "PTX_NATIVE_MIN_TOKENS" not in env
    assert "PTX_TRIATT_HEADSPLIT" not in env, "the head split is the shared core's exact_headsplit row: no kit switch"
    assert "unknown PTX_T_ATT=${PTX_T_ATT}: fast | big | k2b" in case


def test_provider_module_contract():
    src = _src()
    assert re.search(r'^WORD = os\.environ\.get\("PTX_T_ATT", "fast"\)', src, re.M), "the word is the MODE's tier word"
    assert re.search(r"^PREFER = None", src, re.M), "no row preference"
    assert "word=WORD, prefer=None" in src and "prefer=(" not in src.replace("prefer=(row", ""), "select()/triangle_attention() take the tier word alone"
    for pin in ('word="triattn_native"', "word='triattn_native'", 'word="cuda_sm90a"', 'word="k2b"', "protenix_fpf_triattn_cuda", "MIN_TOKENS = int("):
        assert pin not in src, f"row-name pin / kit floor / kit kernel in the adapter: {pin}"
    assert 'PROVIDER = {"name": "core"' in src and "def attn(" in src and "def report(" in src
    assert "fallback_rows" in src and "Refusal" in src, "a per-call refusal by name is served by the row the provider names (counted)"
    assert re.search(r"^MIN_CORE = \(0, 5, 114, 0\)", src, re.M), "the face with the big tier word"
    assert '"bias_only" if m5 is None else "mask_bias"' in src, "the call form is the kit's own (nomask -> bias_only)"


def test_import_is_refused_by_name_without_cuda():
    """On this CPU host the import raises CoreAttnUnavailable('cuda: ...'): the slot records it as BLK_ATT:provider ptx_native_core:attn unavailable(...)."""
    sys.modules.pop("ptx_native_core", None)
    sys.path.insert(0, os.path.join(FPF, "src"))
    try:
        import torch  # noqa: F401
    except Exception:
        import pytest; pytest.skip("torch not importable here")
    try:
        with mock.patch("torch.cuda.is_available", return_value=False):
            try:
                import ptx_native_core  # noqa: F401
                raise AssertionError("import must refuse by name without CUDA")
            except RuntimeError as e:
                assert type(e).__name__ == "CoreAttnUnavailable" and str(e).startswith("cuda:"), repr(e)
    finally:
        sys.modules.pop("ptx_native_core", None); sys.path.remove(os.path.join(FPF, "src"))


def test_marker_classification_and_the_unavailable_form():
    on_line = "BLK:2(pro+PROVIDER(ptx_native_core:attn,tier2,min_tokens=0,below=k2b)+epi+fusion_transition,chunked=k2b)"
    assert stack.MARKERS[NATIVE][1][0] in on_line
    bad = "BLK_ATT:provider ptx_native_core:attn unavailable(CoreAttnUnavailable('select: ...')) -> cueq"
    assert "unavailable(" in bad and stack.MARKERS[NATIVE][1][0] not in bad


def test_lever_evidence_off_the_module_report():
    fake = types.ModuleType("ptx_native_core")
    fake.report = lambda: {"installed": True, "word": "big", "prefer": None, "bind": "tier:big", "opt_core": "0.5.114.0", "key": "torch2.13.0+cu130-x", "native_pkg": "v11",
                           "counts": {"calls": 10, "native": 8, "other_row": 2, "refused": 1}, "rows": {"triattn_native": 8, "cuda_sm90a": 2}, "fallback_rows": {"cuda_sm90a": 1},
                           "forms": {"bias_only": 10}, "refused_kinds": {"tokens": 1}, "sizes": {"512_1023": 10}, "pkg_fallbacks": {}}
    with mock.patch.dict(sys.modules, {"ptx_native_core": fake}):
        pairs = dict(report.native_evidence())
    assert pairs["word"] == "big" and pairs["bind"] == "tier:big" and pairs["prefer"] == "none" and pairs["min_tokens"] == 0
    assert pairs["rows"] == "cuda_sm90a:2,triattn_native:8" and pairs["fallback_rows"] == "cuda_sm90a:1" and pairs["forms"] == "bias_only:10"
    assert pairs["calls"] == 10 and pairs["native"] == 8 and pairs["other_row"] == 2 and pairs["refused"] == "tokens:1" and pairs["core"] == "0.5.114.0"
    for gone in ("tc_below", "tc_capture", "tc_refused", "tc_aside", "aside", "pkg", "exts", "byte_gate"):
        assert gone not in pairs, gone
    with mock.patch.dict(sys.modules, {}):
        sys.modules.pop("ptx_native_core", None)
        assert dict(report.native_evidence())["installed"] == 0


def test_ablation_word():
    from protenix_opt import ablation
    row = modes.README_ROWS["9.0|3.7"]
    assert ablation.restored_words([NATIVE], row["fast"]) == {}, "the displaced lever (k2b) has no PTX_T_ATT word: env.sh's default k2b IS that lever"

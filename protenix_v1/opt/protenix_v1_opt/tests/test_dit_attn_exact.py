"""dit_attn_exact — the shared core's bit-exact DiT attention kernel as this kit's exact-arm lever: the vendored package (files, manifest, the
switch word it reads), the arm membership (exact only), the strategy id, the card step-aside classification, and the LEVER evidence off a
stubbed account (kernel-served = on; card without a prebuilt = skipped/card_off, exit 0; failed install = fallback, exit 3). No GPU."""
import json
import os
import re

import pytest

from protenix_v1_opt import kit as K, modes as M, report as R
from protenix_v1_opt.tests._ditfast_account import DF_OK as _DF_OK

LEVER = "dit_attn_exact"
def _core_pkg(sub):
    import importlib.util
    spec = importlib.util.find_spec("opt_core")
    return os.path.join(os.path.dirname(spec.origin), "kernels", "apb", sub)


PKG_DIR = _core_pkg("dit_exact")                      # the shared core's carried dit_exact (row word dit_exact), bound by word since kit 0.2.21 


def test_membership_strategy_and_registry():
    assert LEVER in M.resolve("exact").levers and LEVER not in M.resolve("fast").levers
    assert M.LEVERS[LEVER]["class"] == "exact" and R.STRATEGY_IDS[LEVER] == "LOCAL.protenix_v1.dit_attn_exact"
    assert R.lever_impl(LEVER) == ("opt_core/kernels/apb/dit_exact", "core")
    _, levers = K.lever_grammar()
    assert LEVER in levers


def test_the_vendored_package_is_the_sibling_kits():
    for rel in ("__init__.py", "build_prebuilt.py", "LICENSE_NOTE.md", "csrc/dit_attn_exact.cu", "prebuilt/torch2.13.0-cu130-sm90/dit_attn_exact.so",
                "prebuilt/torch2.13.0-cu130-sm90/manifest.json", "prebuilt/torch2.13.0-cu130-sm90/PROVENANCE.md"):
        assert os.path.isfile(os.path.join(PKG_DIR, rel)), rel
    man = json.load(open(os.path.join(PKG_DIR, "prebuilt", "torch2.13.0-cu130-sm90", "manifest.json")))
    assert (man["kernel_version"], man["torch"], man["cuda"], man["arch"], man["loadcheck_cases"], len(man["loadcheck_digests"])) == ("7", "2.13.0+cu130", "13.0", ["sm90"], 3, 3)
    assert man["so_sha256"] == "f95aea477b48cd92c960b761c581b816c93a84cf24b6545706490cc25054c9ed"
    src = open(os.path.join(PKG_DIR, "__init__.py"), encoding="utf-8").read()
    assert sorted(set(re.findall(r"""environ(?:\.get)?\s*[\[(]\s*["']([A-Z][A-Z0-9_]+)["']""", src))) == ["PTX_DIT_ATTN_EXACT"]     # the unit reads its switch and nothing else
    assert 'raise RuntimeError(f"{NAME}: no prebuilt for stack' in src and "not in the prebuilt arch list" in src            # the refusals the kit classifies as card_off
    import ast
    lv = ast.parse(open(os.path.join(K.kit_home(), K.LEVERS_RELPATH)).read())
    consts = {n.targets[0].id: n.value.value for n in lv.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
    assert consts.get("DIT_ATTN_EXACT_ENV") == "PTX_DIT_ATTN_EXACT"


def _lv():
    import importlib.util, sys
    p = os.path.join(K.kit_home(), K.LEVERS_RELPATH)
    src = open(p).read()
    i = src.index("def _dit_card_off("); j = src.index("\ndef ", i + 10)
    g = {}; exec(src[i:j], g)
    return g["_dit_card_off"]


def test_card_off_classification():
    card_off = _lv()
    assert card_off("dit_attn_exact: no prebuilt for stack torch2.13.0-cu130-sm80 (expected /x/prebuilt/torch2.13.0-cu130-sm80/manifest.json)")
    assert card_off("dit_attn_exact: device cc (8, 0) not in the prebuilt arch list ['sm90']") and card_off("dit_attn_exact: no CUDA device")
    assert not card_off("dit_attn_exact: sha256 of dit_attn_exact.so does not match the manifest")
    assert not card_off("dit_attn_exact: load-time load-time check: output not bit-identical to torch SDPA in this process ({})")


ACCOUNT = {"cfg": {"trimul": "exact"}, "counts": {"trimul": {"exact": 40}, "triattn": {"gblock": 960}, "transition": {"xtr:C=128": 480}, "dit": {}, "atom": {}},
           "sampler": {"graphs": True, "prep": {"on": True, "parts": "rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec", "poison": "ok", "stats": {}, "aside": None}, "hoist_installed": True, "sampler": {"replays": 9}, "hoist": {"hits": 1}},
           "keep_pool": {"installed": True, "skipped_total": 3, "passed_total": 1, "errors": 0}, "summary_hostidx": {"installed": True, "samples": 5, "delegated": 0, "errors": 0}, "lazy_init": {"installed": True, "constructs": 1, "lazy_construct_s": 3.9, "patched": 12}, "ditfast": _DF_OK, "templ": {"template_dedupe": {"installed": True, "calls": 1, "evaluated": 2, "reused": 2, "stock": {}}, "tmpl_triatt": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul_exact": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_xtr": {"on": True, "routed_total": 4, "stock": {}, "fallback": {}}, "tmpl_pairfused": {"on": True, "calls": 8}}}


@pytest.mark.parametrize("dx, state, partial", [
    ({"installed": True, "calls": 1220, "routes": {"kernel": 1200, "no_bias": 20}}, "state=on impl=opt_core/kernels/apb/dit_exact origin=core strategy=LOCAL.protenix_v1.dit_attn_exact served=1200 gated=20 gated_by=no_bias:20", []),
    ({"installed": False, "card_off": "dit_attn_exact: no prebuilt for stack torch2.13.0-cu130-sm80 (expected …)", "calls": 0, "routes": {}},
     "state=skipped reason=card_off impl=opt_core/kernels/apb/dit_exact origin=core strategy=LOCAL.protenix_v1.dit_attn_exact served=0 card=dit_attn_exact:_no_prebuilt_for_stack_torch2.13.0-cu130-sm80_(expected_…)", []),
    ({"installed": False, "error": "dit_attn_exact: sha256 of dit_attn_exact.so does not match the manifest", "calls": 0, "routes": {}},
     "state=skipped reason=fallback impl=opt_core/kernels/apb/dit_exact origin=core strategy=LOCAL.protenix_v1.dit_attn_exact served=0 fallback=1 fallback_by=dit_attn_exact_installed:", ["dit_attn_exact"]),
])
def test_lever_evidence_and_line(dx, state, partial):
    res = M.resolve("exact")
    rep = {"mode": "exact", "levers": dict(ACCOUNT, dit_attn_exact=dx), "items": [{"N_token": 705}], "det_report": None}
    v = R.verdict(rep, res.trimul, res.levers, allow_partial=False)
    assert v["partial"] == partial and (v["exit_code"] == (R.EXIT_NOT_ACTIVE if partial else R.EXIT_OK))
    line = [l for l in R.lever_lines(v["evidence"], rep) if l.endswith(" lever=dit_attn_exact")]
    assert len(line) == 1 and line[0].startswith("[protenix-v1-opt] LEVER name=LOCAL.protenix_v1.dit_attn_exact " + state[: state.index(" impl=") ]), line
    assert state.split(" impl=")[0] in line[0] and (" card=" in line[0]) == ("card_off" in state)


def test_ablation_names_it_under_exact_only(monkeypatch):
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", LEVER)
    res = M.resolve("exact")
    assert LEVER not in res.levers and res.ablated == (LEVER,) and res.arm == "exact+gblock+triexact+xtr+sg+hoist+keep_pool+summary_hostidx+lazy_init+template_dedupe+sampler_prep+atom_attn_exact+tmpl_triatt+tmpl_xtr+tmpl_trimul_exact"
    with pytest.raises(RuntimeError, match=r"\['dit_attn_exact'\] not a lever of mode fast"):
        M.resolve("fast")


def test_card_off_words_are_the_packages_own():
    """The step-aside rule matches the vendored package's refusal words by substring: pin it against every `raise RuntimeError(f"{NAME}: …")`
    template in opt_core/kernels/apb/dit_exact/__init__.py — exactly the no-CUDA / no-prebuilt-for-stack / cc-not-in-arch-list refusals are a
    card/stack without a build (card_off, exit 0); every other refusal (torch/cuda mismatch of an existing prebuilt, digests, module version,
    load-time check) is a failed install (fallback, partial). A reworded package fails here, not silently at run time."""
    import re
    card_off = _lv()
    src = open(os.path.join(PKG_DIR, "__init__.py")).read()
    templates = re.findall(r'raise RuntimeError\(f"\{NAME\}: ([^"]*)"\)', src)
    assert len(templates) >= 8, templates
    rendered = [re.sub(r"\{[^}]*\}", "X", t) for t in templates]                     # every f-field replaced by a placeholder: the fixed words decide
    hits = [r for r in rendered if card_off("dit_attn_exact: " + r)]
    assert sorted(hits) == sorted(["no CUDA device", "no prebuilt for stack X (expected X)", "device cc X not in the prebuilt arch list X"]), hits
    assert all(not card_off("dit_attn_exact: " + r) for r in rendered if r not in hits)

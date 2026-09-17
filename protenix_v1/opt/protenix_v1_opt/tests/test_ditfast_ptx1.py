"""cond_dedupe / dit_fused / dit_lowp / atom_fused / atom_attn_exact — the carried fused diffusion-sampler package as this kit's arm words
(adapter ptxfpf/ditfast_ptx1.py over lib/protenix_fpf_ditfast) and atom_attn_exact by the shared apb face's exact TIER WORD (a fake face here: the
class contract only — the word passed, the stock-answer step-aside by name, the kernel-row install; never the core's winner): the vendored files, the arm membership per
tier, the strategy ids, the adapter's dependency / card / slot rules BY NAME (no GPU: the install functions are stubbed), and the LEVER evidence
off a stubbed account — a fused stack re-bases ditattn's census (subsumed_by), a card without a cell is skipped/card_off (exit 0), a word the
package refused by name is skipped/aside (exit 0), a dependency the caller withheld is a fallback (exit 3)."""
import os
import re
import sys
import types

import pytest

from protenix_v1_opt import kit as K, modes as M, report as R

from .conftest import KIT

PTXFPF = os.path.join(KIT, "ptxfpf")
WORDS = ("cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "atom_attn_exact")
FAST_WORDS = ("cond_dedupe", "dit_fused", "dit_lowp", "atom_fused")
LIB = os.path.join(K.kit_home(), "lib")


def _adapter():
    """ptxfpf/ditfast_ptx1.py imported fresh off the kit directory (importable without torch: the card is read lazily)."""
    sys.path.insert(0, PTXFPF)
    try:
        sys.modules.pop("ditfast_ptx1", None)
        import ditfast_ptx1 as mod
    finally:
        sys.path.pop(0)
    return mod


def test_membership_per_tier_strategy_and_registry():
    fast, exact, big = M.resolve("fast").levers, M.resolve("exact").levers, M.KIT_MODES["big"].arm.split("+")
    for w in FAST_WORDS:
        assert w in fast and w in big and w not in exact, w
        assert M.LEVERS[w]["class"] == "tolerance" and R.lever_impl(w) == ("ptxfpf/ditfast_ptx1.py", "kit")
    assert "atom_attn_exact" in exact and "atom_attn_exact" not in fast and "atom_attn_exact" not in big
    assert M.LEVERS["atom_attn_exact"]["class"] == "exact"
    assert [R.STRATEGY_IDS[w] for w in WORDS] == ["LOCAL.protenix_v1.cond_dedupe", "LOCAL.protenix_v1.dit_fused", "F4.autocast_policy", "LOCAL.protenix_v1.atom_fused", "LOCAL.protenix_v1.atom_attn_exact"]
    _, levers = K.lever_grammar()
    assert all(w in levers for w in WORDS)
    from protenix_v1_opt import stock_pred
    assert {"ditfast_ptx1", "protenix_fpf_ditfast"} <= set(stock_pred.KIT_MODULES) and "protenix_fpf_atom_attn_exact" not in stock_pred.KIT_MODULES


def test_the_vendored_packages_are_the_sibling_kits():
    for rel in ("protenix_fpf_ditfast/__init__.py", "protenix_fpf_ditfast/_plumbing.py", "protenix_fpf_ditfast/cond_dedupe.py", "protenix_fpf_ditfast/dit_fast.py",
                "protenix_fpf_ditfast/atom_fast.py", "protenix_fpf_ditfast/vectors.py",
                "protenix_fpf_ditfast/vectors.json", "protenix_fpf_ditfast/CELLS.json", "protenix_fpf_ditfast/NOTICE", "protenix_fpf_ditfast/README.md"):
        assert os.path.isfile(os.path.join(LIB, rel)), rel
    assert not os.path.exists(os.path.join(LIB, "protenix_fpf_atom_attn_exact"))    # kit 0.2.39: atom_attn_exact binds the shared apb face by the exact tier word; no vendored copy
    src = {n: open(os.path.join(LIB, "protenix_fpf_ditfast", n), encoding="utf-8").read() for n in ("dit_fast.py", "atom_fast.py", "cond_dedupe.py")}
    env = lambda t: sorted(set(re.findall(r"""environ(?:\.get)?\s*[\[(]\s*[\"']([A-Z][A-Z0-9_]+)[\"']""", t)))
    assert env(src["dit_fast.py"]) == ["PTX_DIT_ATTN", "PTX_DIT_ATTN_FP16", "PTX_DIT_FAST", "PTX_DIT_LOWP"] and env(src["atom_fast.py"]) == ["PTX_ATOM_ATTN", "PTX_ATOM_FAST"] and env(src["cond_dedupe.py"]) == []
    D = _adapter()                                                # the adapter's env words are the package's
    assert D.ENV == {"dit_attn": "PTX_DIT_ATTN", "dit_attn_fp16": "PTX_DIT_ATTN_FP16", "dit_fast": "PTX_DIT_FAST", "dit_lowp": "PTX_DIT_LOWP",
                     "atom_attn": "PTX_ATOM_ATTN", "atom_fast": "PTX_ATOM_FAST"}
    assert D.LEVERS == WORDS and D.LOWP_WORD == "fp16" and all(D.SERVED_CC[w] == ("9.0",) for w in FAST_WORDS) and "atom_attn_exact" not in D.SERVED_CC
    assert D.ATOM_FACE["word"] == "exact" and D.ATOM_FACE["cell"] == ("atom", 4, 32) and D.ATOM_FACE["dtype"] == "fp32"   # the TIER word and the cell the face is asked
    assert 'raise LeverRefused(f"{NAME}: requires lever dit_attn' in src["dit_fast.py"] and "t.stride(-1) == 0" in src["cond_dedupe.py"]


class _Sel(tuple):
    """A fake opt_core.kernels.apb Selection: row / cls / word (the fields the adapter reads)."""
    def __new__(cls, row, klass, word):
        o = tuple.__new__(cls, (row, klass, word)); o.row, o.cls, o.word = row, klass, word; return o


def _fake_face(answer="stock"):
    """A fake shared apb face: select() answers the EXACT word at the atom cell with a stock-class row (`sdpa_gather`, today's answer on every card),
    a kernel row (`atom_exact`, class exact — what a registered byte-vouched row would read) or raises; records the calls."""
    F = types.ModuleType("fake_apb_face"); F.calls = []
    F.STOCK_ROWS = ("sdpa", "sdpa_gather", "torch_module"); F.TIER_WORDS = ("fast", "exact", "faithful", "big")
    F.cell_word = lambda kind, heads=None, head_dim=None, **k: "%s_h%sd%sw32x128" % (kind, heads, head_dim)
    F.stack_word = lambda device=None: "H100:torch2.13.0+cu130/3.7.1/cueq0.11.1"
    def select(cc, dtype, cell, n_tokens, *, word, samples=1, **kw):
        F.calls.append(dict(cc=cc, dtype=dtype, cell=cell, n_tokens=n_tokens, word=word, samples=samples, **kw))
        if isinstance(answer, BaseException):
            raise answer
        return _Sel("sdpa_gather", "stock", word) if answer == "stock" else _Sel("atom_exact", "exact", word)
    F.select = select
    F.describe = lambda sel: "apb row=%s word=%s cell=9.0|fp32|atom_h4d32w32x128|S5|N<=400|graph|fwd class=%s x_stock=1" % (sel.row, sel.word, sel.cls)
    F.atom_attention = lambda *a, **k: (None, None)
    return F


def _fresh_adapter(monkeypatch, cc="9.0", apb=None, installs=None, face="stock", atom_sites=6, op=None):
    """ditfast_ptx1 with a pinned card, a fake apb_ptx1 account, stub package installs (name -> callable(model) or an exception to raise) and a
    fake shared apb face for atom_attn_exact (face = "stock" | "kernel" | None (no face) | an exception select raises); the kernel-row branch's
    module wrapping is stubbed to `atom_sites` modules (no torch here)."""
    D = _adapter()
    monkeypatch.setattr(D, "apb_face", (lambda: None) if face is None else (lambda F=_fake_face(face): F))
    monkeypatch.setattr(D, "_wrap_atom_sites", lambda model, APB, cc_: atom_sites)
    D.ATOM_COUNTS.clear()
    monkeypatch.setattr(D, "CARD", {"cc": cc, "name": None if cc is None else "sm_" + cc.replace(".", "")})
    if apb is not None:
        fake = types.ModuleType("apb_ptx1")
        fake.STATE = {"dit": {"installed_on": 24 if apb.get("ditattn") else 0, "fp16": bool(apb.get("ditattnfp16"))}, "atom": {"installed_on": 6 if apb.get("atomattn") else 0}}
        monkeypatch.setitem(sys.modules, "apb_ptx1", fake)
    else:
        monkeypatch.delitem(sys.modules, "apb_ptx1", raising=False)
    installs = dict(installs or {})

    class LeverRefused(RuntimeError):
        pass

    def stub(word):
        what = installs.get(word, {"installed": True})
        def fn(model):
            if isinstance(what, BaseException):
                raise what
            if what == "refuse":
                raise LeverRefused(f"{word}: cuBLAS numerics for full chunks (256) not tf32-sequential")
            rep = dict(what)
            if word == "dit_fused":
                rep.setdefault("lowp", os.environ.get("PTX_DIT_LOWP", "off")); rep.setdefault("blocks", 24); rep.setdefault("act", "torch.float16" if rep["lowp"] == "fp16" else "torch.float32")
            return rep
        return fn
    pk = types.ModuleType("protenix_fpf_ditfast"); pk.__version__ = "0.9.0"
    pk.install_cond_dedupe, pk.install_dit_fast, pk.install_atom_fast = stub("cond_dedupe"), stub("dit_fused"), stub("atom_fused")
    pk.report = lambda: {"dit_fused": {"stack_calls": 3, "bias_slots": "hoist"}, "atom_fused": {"stack_calls": {"enc": 3, "dec": 3}}}
    monkeypatch.setitem(sys.modules, "protenix_fpf_ditfast", pk); monkeypatch.delitem(sys.modules, "protenix_fpf_atom_attn_exact", raising=False)
    for v in D.ENV.values():                                     # absent at entry AND removed at teardown whatever apply() exported (setenv records the prior state)
        monkeypatch.setenv(v, "-"); monkeypatch.delenv(v)
    return D


def test_fast_words_engage_after_the_apb_levers_and_set_the_packages_env_words(monkeypatch):
    D = _fresh_adapter(monkeypatch, apb={"ditattn": True, "ditattnfp16": True, "atomattn": True})
    acc = D.apply(object(), cond_dedupe=True, dit_fused=True, dit_lowp=True, atom_fused=True)
    w = acc["words"]
    assert [w[x]["state"] for x in FAST_WORDS] == ["on", "on", "on", "on"] and w["atom_attn_exact"]["state"] == "off"
    assert (os.environ["PTX_DIT_ATTN"], os.environ["PTX_DIT_ATTN_FP16"], os.environ["PTX_DIT_FAST"], os.environ["PTX_DIT_LOWP"], os.environ["PTX_ATOM_ATTN"], os.environ["PTX_ATOM_FAST"]) == ("1", "1", "1", "fp16", "1", "1")
    assert w["dit_fused"]["served"] == 3 and w["dit_lowp"]["served"] == 3 and w["atom_fused"]["served"] == 6 and w["dit_lowp"]["facts"]["word"] == "fp16"
    assert w["dit_fused"]["facts"]["lowp"] == "fp16" and w["dit_fused"]["facts"]["blocks"] == 24


def test_dit_lowp_off_word_and_riders_without_their_base_are_named(monkeypatch):
    D = _fresh_adapter(monkeypatch, apb={"ditattn": True, "atomattn": False})
    w = D.apply(object(), dit_fused=True, atom_fused=True)["words"]                        # dit_lowp not in the arm: the package's word is off; atomattn not engaged: atom_fused installs nothing
    assert os.environ["PTX_DIT_LOWP"] == "off" and w["dit_fused"]["state"] == "on" and w["dit_lowp"]["state"] == "off"
    assert w["atom_fused"]["state"] == "skipped" and w["atom_fused"]["reason"] == "requires:atomattn" and w["atom_fused"]["fallback"] == {"requires": "requires:atomattn(not engaged)"}
    D = _fresh_adapter(monkeypatch, apb={"ditattn": False})
    w = D.apply(object(), dit_fused=True, dit_lowp=True)["words"]                          # MODEL_OPT_LEVERS_OFF=ditattn with dit_fused / dit_lowp left in the arm
    assert w["dit_fused"]["reason"] == "requires:ditattn" and w["dit_lowp"]["state"] == "skipped" and w["dit_lowp"]["reason"] == "rides_dit_fused" and "PTX_DIT_FAST" not in os.environ
    D = _fresh_adapter(monkeypatch, apb={"ditattn": True})
    w = D.apply(object(), dit_lowp=True)["words"]                                          # dit_lowp alone: rides dit_fused, named, installs nothing
    assert w["dit_lowp"]["state"] == "skipped" and w["dit_lowp"]["reason"] == "rides_dit_fused" and w["dit_fused"]["state"] == "off"


def test_a_card_without_a_cell_steps_aside_by_name_for_every_word(monkeypatch):
    D = _fresh_adapter(monkeypatch, cc="8.0", apb={"ditattn": True, "ditattnfp16": True, "atomattn": True})
    w = D.apply(object(), cond_dedupe=True, dit_fused=True, dit_lowp=True, atom_fused=True, atom_attn_exact=True)["words"]
    assert all(w[x]["state"] == "skipped" and w[x]["reason"] == "card_off" and w[x]["card"].startswith("sm_80: no %s cell (cells: sm_90)" % x) for x in FAST_WORDS), w
    assert w["atom_attn_exact"]["reason"] == "aside" and w["atom_attn_exact"]["aside"] == "slot_served_by:atomattn"   # no card list: the slot rule, then the face (below)
    assert all(v not in os.environ for v in D.ENV.values())                                 # nothing installed, nothing exported
    D = _fresh_adapter(monkeypatch, cc=None, apb={"ditattn": True})
    assert D.apply(object(), cond_dedupe=True)["words"]["cond_dedupe"]["aside"] == "no_cuda"


def test_atom_attn_exact_asks_the_shared_face_by_the_exact_tier_word(monkeypatch):
    """The class contract of the binding (never the core's winner): the EXACT tier word at the atom cell (atom_h4d32w32x128, fp32, S 5) on the
    process's card; a stock-class answer installs nothing — skipped / aside=exact_word:<row> (the upstream op by name, exit 0) on 9.0 AND 8.0; a
    kernel row is installed at the atom-attention modules and reads `on`; no face / a face error are named / fallback; beside atomattn or
    atom_fused the slot rule steps aside first."""
    for cc in ("9.0", "8.0"):
        D = _fresh_adapter(monkeypatch, cc=cc, apb={}, face="stock")
        w = D.apply(object(), atom_attn_exact=True)["words"]["atom_attn_exact"]
        call = D.apb_face().calls[-1]
        assert (call["cc"], call["dtype"], call["cell"], call["word"], call["samples"]) == (cc, "fp32", "atom_h4d32w32x128", "exact", 5), call
        assert w["state"] == "skipped" and w["reason"] == "aside" and w["aside"] == "exact_word:sdpa_gather" and w["served"] == 0, w
        assert w["facts"]["row"] == "sdpa_gather" and w["facts"]["apb"].startswith("apbrow=sdpa_gather") and "PTX_ATOM_ATTN_EXACT" not in os.environ
        v, lines = _verdict({"atom_attn_exact": w}, mode="exact", cfg={"trimul": "exact"}, counts={"dit": {}, "atom": {}, "trimul": {"exact": 40}, "triattn": {"gblock": 960}, "transition": {"xtr:C=128": 480}},
                            dit_attn_exact={"installed": True, "calls": 72, "routes": {"kernel": 72}})       # accounted (never partial); the LEVER line names the face's answer
        assert "atom_attn_exact" not in v["partial"], v["partial"]
        assert " state=skipped reason=aside " in lines["atom_attn_exact"] and " aside=exact_word:sdpa_gather " in lines["atom_attn_exact"] and " row=sdpa_gather " in lines["atom_attn_exact"], lines["atom_attn_exact"]



def test_the_real_face_answers_the_exact_word_at_the_atom_cell_on_both_cards():
    """opt_core.kernels.apb on this tree (CPU: the table): select() by the EXACT tier word at the atom cell returns a Selection on 9.0 and 8.0 —
    the class of the answer is the core's (stock today: the word names sdpa_gather; a registered byte-vouched row later), never asserted here."""
    D = _adapter()
    APB = D.apb_face()
    if APB is None:
        pytest.skip("this opt_core has no kernels.apb face")
    for cc in ("9.0", "8.0"):
        sel, token = D.atom_exact_selection(APB, cc, D.ATOM_FACE["probe_tokens"])
        assert sel.word == "exact" and token.startswith("apb row=%s" % sel.row) and "atom_h4d32w32x128" in token, token
        assert D._is_stock(APB, sel) == (sel.row in APB.STOCK_ROWS)


ACCOUNT = {"cfg": {"trimul": "fast", "gflash": True, "gblock": False}, "counts": {"dit": {}, "atom": {}, "trimul": {"fast": 40}, "triattn": {"gflash": 480}, "transition": {"ttr:C=128": 40}, "core:tricuda": {"cuda_sm90a": 480}},
           "sampler": {"graphs": True, "prep": {"on": True, "parts": "rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec", "poison": "ok", "stats": {}, "aside": None}, "hoist_installed": True, "sampler": {"replays": 199}, "hoist": {"hits": 24}},
           "keep_pool": {"installed": True, "skipped_total": 3, "passed_total": 1, "errors": 0}, "summary_hostidx": {"installed": True, "samples": 5, "delegated": 0, "errors": 0},
           "lazy_init": {"installed": True, "constructs": 1, "lazy_construct_s": 3.9, "patched": 12},
           "trunk2": {"pf": {"engaged": True, "cell_key": "9.0", "installed_on": 52, "calls": 96}, "opm": {"engaged": True, "cell_key": "9.0", "installed_on": 4, "calls": 8}, "pwa": {"engaged": True, "cell_key": "9.0", "installed_on": 3, "calls": 6}}, "templ": {"template_dedupe": {"installed": True, "calls": 1, "evaluated": 2, "reused": 2, "stock": {}}, "tmpl_triatt": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul_exact": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_xtr": {"on": True, "routed_total": 4, "stock": {}, "fallback": {}}, "tmpl_pairfused": {"on": True, "calls": 8}},
           "apb": {"dit": {"engaged": True, "fp16": True, "opd": "fp16", "cell_key": "9.0", "installed_on": 24, "calls": 0, "named": None},
                   "atom": {"engaged": True, "opd": "tf32rn", "cell_key": "9.0", "installed_on": 6, "calls": 0, "named": None}}}
WORDS_ON = {"cond_dedupe": {"state": "on", "served": 2, "facts": {"rows_in": 10, "rows_computed": 2, "guard": "stride0"}},
            "dit_fused": {"state": "on", "served": 2, "facts": {"blocks": 24, "act": "torch.float16", "lowp": "fp16", "bias_slots": "hoist"}},
            "dit_lowp": {"state": "on", "served": 2, "facts": {"word": "fp16"}},
            "atom_fused": {"state": "on", "served": 4, "facts": {"enc_blocks": 3, "dec_blocks": 3}}}


def _verdict(words, mode="fast", **extra):
    res = M.resolve(mode)
    rep = {"mode": mode, "levers": dict(ACCOUNT, ditfast={"words": words}, **extra), "items": [{"N_token": 705}], "triattn_floor": {"exported": 300}, "det_report": None}
    v = R.verdict(rep, res.trimul, res.levers, allow_partial=False)
    return v, {l.rsplit(" lever=", 1)[-1]: l for l in R.lever_lines(v["evidence"], rep)}     # the memory mode adds its own rows (big.lever_rows) after the per-lever lines


def test_engaged_words_are_on_and_rebase_the_attention_levers_census():
    v, lines = _verdict(WORDS_ON)
    assert v["partial"] == [] and v["exit_code"] == R.EXIT_OK, v["partial_reason"] if "partial_reason" in v else v["partial"]
    assert lines["dit_fused"] == "[protenix-v1-opt] LEVER name=LOCAL.protenix_v1.dit_fused state=on impl=ptxfpf/ditfast_ptx1.py origin=kit strategy=LOCAL.protenix_v1.dit_fused served=2 act=torch.float16 bias_slots=hoist blocks=24 lowp=fp16 lever=dit_fused"
    assert lines["dit_lowp"] == "[protenix-v1-opt] LEVER name=F4.autocast_policy state=on impl=ptxfpf/ditfast_ptx1.py origin=kit strategy=F4.autocast_policy served=2 word=fp16 lever=dit_lowp"
    assert lines["cond_dedupe"].endswith(" served=2 guard=stride0 rows_computed=2 rows_in=10 lever=cond_dedupe") and " state=on " in lines["cond_dedupe"]
    assert " state=on " in lines["ditattn"] and " served=2 " in lines["ditattn"] and " subsumed_by=dit_fused " in lines["ditattn"]          # the fused stack's attention IS ditattn's kernel: census re-based, never served0
    assert " state=on " in lines["ditattnfp16"] and " subsumed_by=dit_fused " in lines["ditattnfp16"]
    assert " state=on " in lines["atomattn"] and " served=4 " in lines["atomattn"] and " subsumed_by=atom_fused " in lines["atomattn"]
    assert " state=off reason=not_in_mode:fast " in lines["atom_attn_exact"]


def test_card_off_and_aside_are_accounted_exit_0_a_withheld_dependency_is_partial():
    card = {w: {"state": "skipped", "reason": "card_off", "card": "sm_80: no %s cell (cells: sm_90)" % w, "served": 0} for w in FAST_WORDS}
    apb80 = {"dit": dict(ACCOUNT["apb"]["dit"], calls=4800), "atom": dict(ACCOUNT["apb"]["atom"], calls=1200)}
    v, lines = _verdict(card, apb=apb80, counts=dict(ACCOUNT["counts"], dit={"apb:fp16": 4800}, atom={"apb:tf32rn": 1200}))
    assert v["partial"] == [] and v["exit_code"] == R.EXIT_OK
    assert all(" state=skipped reason=card_off " in lines[w] and " card=sm_80:_no_%s_cell_(cells:_sm_90)" % w in lines[w] for w in FAST_WORDS), lines["dit_fused"]
    dep = dict(WORDS_ON, atom_fused={"state": "skipped", "reason": "requires:atomattn", "fallback": {"requires": "requires:atomattn(not engaged)"}, "served": 0})
    v, lines = _verdict(dep, counts=dict(ACCOUNT["counts"], atom={"apb:tf32rn": 1200}))
    assert v["partial"] == ["atom_fused"] and v["exit_code"] == R.EXIT_NOT_ACTIVE and " state=skipped reason=fallback " in lines["atom_fused"] and "fallback_by=requires:requires:atomattn(not_engaged)" in lines["atom_fused"]
    v, lines = _verdict({}, )                                                                    # no account of the words at all (adapter never ran): fallback, partial
    assert set(v["partial"]) >= set(FAST_WORDS)
    xw = {"atom_attn_exact": {"state": "skipped", "reason": "aside", "aside": "exact_word:sdpa_gather", "facts": {"row": "sdpa_gather"}, "served": 0}}   # the exact word named the stock statement: the upstream op by name, accounted
    v, lines = _verdict(xw, mode="exact", cfg={"trimul": "exact"}, counts={"dit": {}, "atom": {}, "trimul": {"exact": 40}, "triattn": {"gblock": 960}, "transition": {"xtr:C=128": 480}},
                        dit_attn_exact={"installed": True, "calls": 72, "routes": {"kernel": 72}})
    assert "atom_attn_exact" not in v["partial"] and " state=skipped reason=aside " in lines["atom_attn_exact"] and " aside=exact_word:sdpa_gather " in lines["atom_attn_exact"]


def test_ablation_rules_by_name(monkeypatch):
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "dit_fused,dit_lowp")
    res = M.resolve("fast")
    assert "dit_fused" not in res.levers and "dit_lowp" not in res.levers and "cond_dedupe" in res.levers and res.ablated == ("dit_fused", "dit_lowp")
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "atom_attn_exact")
    assert "atom_attn_exact" not in M.resolve("exact").levers
    with pytest.raises(RuntimeError):
        M.resolve("fast")                                                                     # atom_attn_exact is not a lever of fast: refused by name


def test_cond_dedupe_idle_by_design_at_one_sample_is_named_not_partial(monkeypatch):
    """N_sample == 1 (`--sample 1`, a reach probe): every DiffusionConditioning call carries one sample row — nothing to de-duplicate BY DESIGN.
    The word reads skipped / aside=n_sample:1 (accounted, exit 0) in fast and in big; served 0 with multi-sample rows on the stock path stays PARTIAL."""
    one = {"installed": True, "census": {"dedupe_calls": 0, "stock_path_calls": 4, "rows_in": 4, "rows_computed": 4}}      # 4 denoiser calls x 1 sample row
    D = _fresh_adapter(monkeypatch, apb={"ditattn": True, "ditattnfp16": True, "atomattn": True}, installs={"cond_dedupe": one})
    w = D.apply(object(), cond_dedupe=True, dit_fused=True, dit_lowp=True, atom_fused=True)["words"]
    cd = w["cond_dedupe"]
    assert (cd["state"], cd["reason"], cd["aside"], cd["served"], cd["gated"]) == ("skipped", "aside", "n_sample:1", 0, {"stock_path": 4}) and cd["engaged"] is True
    for mode in ("fast", "big"):
        v, lines = _verdict(dict(WORDS_ON, cond_dedupe=cd), mode=mode)
        assert v["partial"] == [] and v["exit_code"] == R.EXIT_OK, (mode, v["partial_reason"])
        assert " state=skipped reason=aside " in lines["cond_dedupe"] and " served=0 " in lines["cond_dedupe"] and " gated=4 gated_by=stock_path:4 " in lines["cond_dedupe"] \
            and " aside=n_sample:1 " in lines["cond_dedupe"], lines["cond_dedupe"]
    five = {"installed": True, "census": {"dedupe_calls": 0, "stock_path_calls": 4, "rows_in": 20, "rows_computed": 20}}   # 5 sample rows per call took the stock path: NOT engaged
    D = _fresh_adapter(monkeypatch, apb={"ditattn": True, "ditattnfp16": True, "atomattn": True}, installs={"cond_dedupe": five})
    cd = D.apply(object(), cond_dedupe=True, dit_fused=True, dit_lowp=True, atom_fused=True)["words"]["cond_dedupe"]
    assert (cd["state"], cd["aside"], cd["served"], cd["gated"]) == ("on", None, 0, {"stock_path": 4})
    for mode in ("fast", "big"):
        v, lines = _verdict(dict(WORDS_ON, cond_dedupe=cd), mode=mode)
        assert v["partial"] == ["cond_dedupe"] and v["exit_code"] == R.EXIT_NOT_ACTIVE and " state=skipped reason=served0 " in lines["cond_dedupe"], (mode, lines["cond_dedupe"])

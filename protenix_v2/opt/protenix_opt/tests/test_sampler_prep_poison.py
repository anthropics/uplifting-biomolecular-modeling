"""The graphed sampler's host-path lever (sampler_prep, infopt_graphs/protenix/sampler_prep.py): its single switch, and the pass rule of the once-per-
process stale-copy probe (poison_classify) — which captured-input classes must be read is decided from the live objects (per-step inputs; hoist slots
recorded this process), hoist kinds another sampler lever serves are `subsumed`, a consumer that maps non-finite operands to finite output still
counts as reading (`squash`), produced kinds outside the documentation lists are `unlisted` and still required.  CPU only (torch is stubbed when absent)."""
import importlib.util
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SP_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "flashpairformer", "src", "infopt_graphs", "protenix", "sampler_prep.py"))


@pytest.fixture(scope="module")
def sp():
    stub = None
    if importlib.util.find_spec("torch") is None:                     # the module imports torch for its device-side parts; the rule under test is pure Python
        stub = types.ModuleType("torch"); stub.Tensor = type("Tensor", (), {}); stub.cuda = types.SimpleNamespace()
        sys.modules["torch"] = stub
    try:
        spec = importlib.util.spec_from_file_location("ptx_sampler_prep_under_test", SP_PATH)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        yield mod
    finally:
        if stub is not None and sys.modules.get("torch") is stub:
            del sys.modules["torch"]


def test_single_switch(sp):
    """PTX_SAMPLER_PREP is the whole lever: all six parts or none; no other variable selects a subset or tunes a part."""
    assert sp.PrepConfig.from_env({}).on is False and sp.PrepConfig.from_env({"PTX_SAMPLER_PREP": "0"}).parts == ()
    on = sp.PrepConfig.from_env({"PTX_SAMPLER_PREP": "1", "PTX_SAMPLER_PREP_SET": "rot_async", "PTX_SAMPLER_PREP_RING": "8"})
    assert on.on and on.parts == sp.PARTS and on.describe() == "rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec"
    assert all(getattr(on, p) for p in sp.PARTS) and on.reach is False
    assert sp.PrepConfig.from_env({"PTX_SAMPLER_REACH": "1"}).reach is False           # reach rides on the lever, never alone
    assert sp.PrepConfig.from_env({"PTX_SAMPLER_PREP": "1", "PTX_SAMPLER_REACH": "1"}).reach is True
    src = open(SP_PATH).read()
    import re
    assert sorted(set(re.findall(r'(?:environ|env)\.get\("([A-Z_]+)"', src))) == ["PTX_SAMPLER_PREP", "PTX_SAMPLER_REACH"]


def test_class_key_folds_block_indices(sp):
    assert sp.poison_class_key("hoist.atomenc2.layernorm_kv.sig") == "hoist.atomenc#.layernorm_kv.sig"
    assert sp.poison_class_key("hoist.tok.bias17") == "hoist.tok.bias#"
    assert sp.poison_class_key("st.rot") == "st.rot" and sp.poison_class_key("cond.c_l") == "cond.c_l"


ALL_KINDS = None


def _present(sp, drop=(), extra=()):
    """Hoist slot names (block indices concrete) for every documented kind except `drop`, plus `extra`."""
    names = []
    for k in sp.POISON_KNOWN_HOIST:
        if k in drop:
            continue
        body = k[len("hoist."):]
        names += [body.replace("#", str(i)) for i in range(2)] if "#" in body else [body]
    return names + list(extra)


def test_recorded_slots_are_required_and_a_stale_copy_fails(sp):
    kinds = list(sp.POISON_KNOWN_HOIST)
    v = sp.poison_classify(read=list(sp.POISON_REQUIRED_STEP) + kinds[:-1], squash=[], unread=[kinds[-1]],
                           recorded={k: True for k in kinds}, hoist_bound=True, hoist_present=_present(sp))
    assert v["missing"] == [kinds[-1]] and v["subsumed"] == [] and v["unlisted"] == []
    v = sp.poison_classify(read=list(sp.POISON_REQUIRED_STEP)[1:] + kinds, squash=[], unread=[sp.POISON_REQUIRED_STEP[0]],
                           recorded={k: True for k in kinds}, hoist_bound=True, hoist_present=_present(sp))
    assert v["missing"] == [sp.POISON_REQUIRED_STEP[0]]                                # a per-step input is always required


def test_kinds_another_lever_serves_are_subsumed_not_failed(sp):
    """A fused attention / block path never enters the hoist's producer for some kinds: no slot exists, nothing to read, reported as subsumed."""
    served = ("hoist.atomenc#.layernorm_kv.sig", "hoist.atomenc#.layernorm_kv.lin", "hoist.atomdec#.layernorm_kv.sig", "hoist.atomdec#.layernorm_kv.lin")
    kinds = [k for k in sp.POISON_KNOWN_HOIST if k not in served]
    v = sp.poison_classify(read=list(sp.POISON_REQUIRED_STEP) + kinds, squash=[], unread=list(sp.POISON_EXPECTED_UNREAD),
                           recorded={k: True for k in kinds}, hoist_bound=True, hoist_present=_present(sp, drop=served))
    assert v["missing"] == [] and v["subsumed"] == sorted(served)
    # whole atom blocks served by another path (every atomenc/atomdec kind absent) + the token bias producers: all subsumed, none missing
    served = tuple(k for k in sp.POISON_KNOWN_HOIST if ".atomenc#." in k or ".atomdec#." in k or k == "hoist.tok.bias#")
    kinds = [k for k in sp.POISON_KNOWN_HOIST if k not in served]
    new = ["tok.fastbias0.float32", "tok.fastbias1.float32", "atomfast.enc.0.cond", "atomfast.dec.1.locbias"]
    new_kinds = sorted({sp.poison_class_key("hoist." + n) for n in new})
    v = sp.poison_classify(read=list(sp.POISON_REQUIRED_STEP) + kinds + new_kinds, squash=[], unread=[],
                           recorded={k: True for k in kinds + new_kinds}, hoist_bound=True, hoist_present=_present(sp, drop=served, extra=new))
    assert v["missing"] == [] and v["subsumed"] == sorted(served) and sorted(v["unlisted"]) == new_kinds
    # ... and a NEW kind that is recorded but not read is a failure like any other recorded slot
    v = sp.poison_classify(read=list(sp.POISON_REQUIRED_STEP) + kinds + new_kinds[1:], squash=[], unread=[new_kinds[0]],
                           recorded={k: True for k in kinds + new_kinds}, hoist_bound=True, hoist_present=_present(sp, drop=served, extra=new))
    assert v["missing"] == [new_kinds[0]]


def test_finite_squash_counts_as_read_and_expected_unread_is_exempt(sp):
    kinds = list(sp.POISON_KNOWN_HOIST)
    squashed = [k for k in kinds if "layernorm_kv" in k]
    v = sp.poison_classify(read=list(sp.POISON_REQUIRED_STEP) + [k for k in kinds if k not in squashed], squash=squashed,
                           unread=list(sp.POISON_EXPECTED_UNREAD), recorded={k: True for k in kinds}, hoist_bound=True, hoist_present=_present(sp))
    assert v["missing"] == [] and v["subsumed"] == []
    assert not any(n in v["required"] for n in sp.POISON_EXPECTED_UNREAD)
    # an unrecorded slot object (producer never entered this process) is not required even if present
    v = sp.poison_classify(read=list(sp.POISON_REQUIRED_STEP) + kinds[:-1], squash=[], unread=[kinds[-1]],
                           recorded={**{k: True for k in kinds[:-1]}, kinds[-1]: False}, hoist_bound=True, hoist_present=_present(sp))
    assert v["missing"] == []
    # no hoist bound: hoist kinds are neither required nor subsumed
    v = sp.poison_classify(read=list(sp.POISON_REQUIRED_STEP), squash=[], unread=["cond.s_inputs"], recorded={}, hoist_bound=False, hoist_present=())
    assert v["missing"] == [] and v["subsumed"] == []


def test_magnitude_rule_and_status_line(sp):
    assert sp.poison_moved(float("inf"), 0.0) and sp.poison_moved(50.0, 0.05) and sp.poison_moved(1.0, 0.0)
    assert not sp.poison_moved(0.9, 0.0) and not sp.poison_moved(4.0, 0.5)             # below 1.0 absolute / below 10 x replay noise
    sp.POISON_STATUS.update(ran=False, unlisted=[], subsumed=[], squash=[], failed=False)
    assert sp.poison_status() == "not-run"
    sp.POISON_STATUS.update(ran=True)
    assert sp.poison_status() == "ok"
    sp.POISON_STATUS.update(subsumed=["hoist.atomenc#.layernorm_kv.sig"], squash=["hoist.x"], unlisted=["hoist.tok.fastbias#.float#"])
    assert sp.poison_status() == "ok;subsumed:hoist.atomenc#.layernorm_kv.sig;squash:hoist.x;unlisted:hoist.tok.fastbias#.float#"
    sp.POISON_STATUS.update(failed=True)
    assert sp.poison_status().startswith("failed;subsumed:")
    sp.POISON_STATUS.update(ran=False, unlisted=[], subsumed=[], squash=[], failed=False)

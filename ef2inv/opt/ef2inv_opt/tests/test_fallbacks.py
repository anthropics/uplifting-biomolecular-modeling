"""One test per fallback site of the carried kit (evidence.py F1-F11; tests/README.md has the table): each FORCES the site's condition on
a stand-in and asserts the verdict evidence.py reaches — a named refusal (a lever that did not install or confirm without a declared reason:
exit 3), a named aside (a lever that stood aside BY ITS OWN DECLARED WORD or yielded to a user pair-stack switch: LEVER state word +
EVIDENCE fallback=, exit 0) or an event. Nothing here imports torch or the kit's lever modules."""
import sys
import types

import pytest

from .. import envproof, evidence as EV, modes, fastkit as FK
from . import _paths as P

from .. import _absent_sdk_stub
STANDIN = {"modal": _absent_sdk_stub}                     # the arm process's registered `modal` (stock_design.prepare_arm_process)
PREFIXES = P.PINS["stock_environment"]["must_be_absent_prefixes"]


class Cfg:
    def __init__(self, trimul, transition, einsum_out_dtype="bf16", checkpoint="ckpt:12"):
        self.trimul, self.transition, self.einsum_out_dtype, self.checkpoint = trimul, transition, einsum_out_dtype, checkpoint


class Eeg:
    def __init__(self, captures=1, replays=150, eager=0):
        self.stats = dict(captures=captures, replays=replays, eager=eager); self.entries = {1: 1}


def pool_stats(slots=2, n_slots=2, **kw):
    return dict(dict(slots=slots, n_slots=n_slots, captures=slots, recaptures=0, replays_fwd=296, replays_bwd=296, eager=0, steps=150, active=True, pool_bytes=9 * 2 ** 30, disabled_reason=None), **kw)


class Model:
    def __init__(self, cfg, eeg, pool, skip=True, kernels=True, grad_calls=48, mode_off=()):
        self._agk_cfg = cfg
        self._esmc = types.SimpleNamespace(_eeg=eeg) if "ef2_esmc_graph" not in mode_off else types.SimpleNamespace()   # mode_off: the levers the composition switches off by name (big: fastkit.BIG_MODE_OFF) are simply not installed
        self._mode_off = tuple(mode_off)
        self._pool_stats = pool
        sub = lambda: types.SimpleNamespace(_chunk_size=(64 if kernels else None))       # the fork's layout the census reads the pair-stack chunk from: fast / big set 64 on the inversion models, exact's pin is None
        self.folding_trunk = types.SimpleNamespace(blocks=[types.SimpleNamespace(tri_mul_out=sub(), tri_mul_in=sub(), pair_transition=sub())])
        if skip:
            self._agk_orig_model_forward = object()
        if True:                                                                          # exact: ef2_trimul's tile table on the stock cuEquivariance kernels; fast: its fused kernel (2 TriMuls per grad-mode block call)
            self.__dict__["_ef2_trimul_handle"] = types.SimpleNamespace(variant="fused" if kernels else "cueq_tiles", stats={"served": grad_calls if kernels else 0, "fallback": 4 if kernels else 0}, cueq_saved={} if kernels else {1: 1, 2: 2})
        if kernels:                                                                       # fast / big: the checkpointed blocks' no-save first pass (ef2_trimul_nosave) — served 0 here: every tri-mul of this fake went through the fused kernel
            self.__dict__["_ef2_nosave_handle"] = types.SimpleNamespace(on=True, word="fast", row="v4", reason="", stats={"served": 0, "inner": 0, "first_pass": 0, "recompute": 0, "refused_calls": 0}, refused={"tx_sm90a": "no_prebuilt:torch211_cu128_sm90"}, canary="pass:0.004")
        floor = cfg.checkpoint == "block"                                                 # big's plan as ef2_bwd_ckpt.plan(floor=True) records it; the budget rule's record otherwise (no reason word)
        self._bwd_ckpt = dict({"policy": cfg.checkpoint, "kept_per_pass": 0 if floor else 12, "n_blocks": 24, "copies": 2 if pool else 1, "kernels": "fused3" if kernels else "stock", "forced": False,
                               "rule": "floor" if floor else "budget", "floor": floor}, **({"reason": "memory_floor"} if floor else {}))
        self.confidence_head = types.SimpleNamespace(**({"_bc_orig_forward": object()} if kernels else {}))
        if "ef2_lazy_structure" not in mode_off:
            self._els_orig_forward = object()
        if "ef2_pairbias_attn" not in mode_off:
            self._pba_state = types.SimpleNamespace(stats={"served": 100, "computed": 5, "unscoped": 0, "fused_calls": 0, "fallback_grad": 0, "fallback_shape": 0}, variant="hoist")
        if "ef2_sampler_graph" not in mode_off:
            self._sg_state = types.SimpleNamespace(stats={"samples": 5, "captures": 5, "replays": 95, "eager": 0, "mismatch": 0, "capture_failed": 0})

    def modules(self):
        return []


class Agk:
    KERNELS_OK = True; _KD3_OK = True; _EXP = "libdevice"; _HAS_BMM_OUT_DTYPE = True

    @staticmethod
    def describe(cfg):
        return f"trimul={cfg.trimul} transition={cfg.transition} ckpt={cfg.checkpoint}"


class Guard:
    def summary(self):
        return {"diffs": 0}


FAKE_LEVERS = {                                             # the census readers evidence.collect_after imports by name (the k/ modules need torch; these stand in)
    "ef2_stepgraph": {"stats": lambda m: getattr(m, "_pool_stats", None)}, "ef2_trimul": {}, "ef2_bwd_ckpt": {},
    "ef2_lazy_structure": {"stats": lambda m: {"deferred": 130, "materialized": 20, "eager": 0}}, "ef2_pairbias_attn": {}, "ef2_sampler_graph": {},
    "ef2_esmc_rope": {"handle": lambda h: types.SimpleNamespace(stats={"served": 6, "fallback": {}}, modules=[1] * 80)}, "ef2_esmc_overlap": {},
    "ef2_loop_prep": {"stats": lambda BD: {"served": 150, "early": 149, "late": 1, "memo_hits": 20, "anchored": 1, "mismatch": 0, "fallback_anchor": 0, "fallback_signature": 0, "spliced": 148, "splice_off": 0}},
    "ef2_loop_pppl": {"stats": lambda BD: {"served": 150, "checked": 150, "first_read": 1}},
    "ef2_fused_ln": {"stats": lambda: {"served": 96, "fallback": {}, "installed": True, "channel_major": False}},
    "ef2_esmc_hoist": {"stats": lambda m: None if "ef2_esmc_hoist" in getattr(m, "_mode_off", ()) else {"served": 148, "full": 2, "anchored": 1, "captures": 1, "replays": 148, "entries": 1, "fallback": {}, "off": None, "live": "82/199"}},
    "ef2_t16_transition": {"describe": lambda: t16_describe("on", served=48)},
    "ef2_kd3_gemmswiglu": {"describe": lambda: gsw_describe("on", served=48)}}


def t16_describe(state, served=0, reason=None, fallback=None):
    """ef2_t16_transition.describe() as the lever reports itself: on (an H100: the carried cubin loaded, canary passed, every grad-mode transition
    served), stepped_aside (another card / stack: the reason named, K-D3's own forward served), off (never asked)."""
    kern = {"cubin": "2f600bd17b86", "cubin_source": "shipped", "regs": 168, "spill_bytes": 0, "smem": 230512, "arch": "sm_90a", "canary": "pass:102:bitwise:6a3438b92fec:max_abs=0.0312"} if state == "on" else {}
    return {"name": "ef2_t16_transition", "state": state, "reason": reason, "reason_text": (f"ef2_t16_transition: NVIDIA A100 is sm_80; the kernel is sm_90a only" if reason else None),
            "version": "transition_cute.1.0", "device": "NVIDIA_H100_80GB_HBM3" if state == "on" else "NVIDIA_A100-SXM4-80GB", "kernel": kern, "served": served, "fallback": dict(fallback or {}), "packs": 96 if state == "on" else 0}


def gsw_describe(state, served=0, reason=None, fallback=None):
    """ef2_kd3_gemmswiglu.describe() as the lever reports itself: on (an A100: the sm_80 entry bound as K-D3's forward hook, every transition served),
    stepped_aside (a card without an entry: the reason named, K-D3's cuBLAS projection + SwiGLU kernel served), off (never asked: exact, or the t16
    kernel runs K-D3's forward on this card)."""
    return {"name": "ef2_kd3_gemmswiglu", "state": state, "reason": reason, "reason_text": (f"ef2_kd3_gemmswiglu: no launch entry for this device ({reason})" if reason else None),
            "version": "w12swiglu.1", "device": "NVIDIA_A100-SXM4-80GB", "cc": "sm_80" if state == "on" else "sm_89", "tile": "128x64x64g8w8s3" if state == "on" else None,
            "served": served, "fallback": dict(fallback or {}), "installed": state == "on"}


@pytest.fixture(autouse=True)
def fake_levers(monkeypatch):
    for name, attrs in FAKE_LEVERS.items():
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)


def app_for(mode, n_tokens=195, eeg=None, pool_slots=2):
    kernels = mode != "exact"
    cfg = (Cfg(None, "refround_lean", checkpoint="block") if mode == "big" else Cfg(None, "refround_lean")) if kernels else Cfg(None, None)   # big: the plan's floor (every block checkpointed)
    eeg = eeg or Eeg()
    pool = pool_stats(pool_slots) if n_tokens <= 256 and mode != "big" else None   # exact / fast carry the trunk graph pool at or below fastkit.POOL_MAX_TOKENS; big carries none at any size
    moff = FK.mode_off(modes.MODES[mode].kit_switch)                                # big: the levers switched off by name are not installed on the models / the cookbook module
    ms = {"ESMFold2-Experimental-Fast": Model(cfg, eeg, pool, kernels=kernels, mode_off=moff), "ESMFold2-Experimental-Fast-Cutoff2025": Model(cfg, eeg, pool, kernels=kernels, mode_off=moff)}
    for m in ms.values():
        m.set_kernel_backend = m.set_chunk_size = (lambda v: None)
    lm = types.SimpleNamespace(esmc=types.SimpleNamespace())
    return types.SimpleNamespace(inversion_models=ms, hf_critic_models=critics_for(mode), esmc_model=lm, _fast_kit_tokens=n_tokens)


def critics_for(mode, n=2, tiles=True):
    """The hero critics as the arm leaves them right after load (fork-layout stand-ins, test_model_switches._fork_model): every kit mode folds
    them on the stock arm's pair — exact by its every-model pins, fast / big by their critic scope (set_chunk_size(None) +
    set_kernel_backend("cuequivariance")); the after-loop census reads that back (evidence.critic_census). On fast / big the first critic holds
    the process-wide tile table's handle (ef2_trimul cueq_tiles; exact's is on the first inversion model, Model kernels=False)."""
    from .test_model_switches import _fork_model
    out = {}
    for i in range(n):
        m = _fork_model(msa_encoder=bool(i)); m.set_chunk_size(None); m.set_kernel_backend("cuequivariance"); out[f"critic{i}"] = m
    if tiles and mode != "exact" and out:
        out["critic0"].__dict__["_ef2_trimul_handle"] = types.SimpleNamespace(variant="cueq_tiles", stats={"served": 0, "fallback": 0}, cueq_saved={1: 1, 2: 2, 3: 3, 4: 4})
    return out


def pppl_ok(_):
    return dict(captures=1, replays=150, eager=3, replay_ok=1, replay_fail=0, pool_gb=[], evictions=0, disabled_reason="", entries=1)


def counter_for(mode, grad_calls=48):
    c = EV.KernelCounter()
    c.block_calls["grad_bf16"] = grad_calls; c.block_calls["other"] = 4
    if mode != "exact":
        c.calls["transition_refround"] = grad_calls                      # the fused triangle multiplication is counted by its own handles (Model: grad_calls each, 2 models = 2 x grad_calls)
    return c


LOG = "2026 binder_design INFO fast kit: agk3, checkpoint policy ckpt:12 for 195 tokens, trunk graph pool 2 slots"      # `--mode fast` runs EF2_FAST_KIT=agk3 (the kit line prints the switch word); exact's / big's logs below
LOG_BY_MODE = {"fast": LOG, "exact": LOG.replace("agk3", "exact"), "big": "2026 binder_design INFO fast kit: big, checkpoint policy block for 195 tokens, trunk graph pool none"}
BD = types.SimpleNamespace(_D59_GUARD=Guard(), _ef2_loop_pppl=object(), _ef2_esmc_overlap=types.SimpleNamespace(stats={"served": 149, "computed": 150, "fallback": {"no_reference_yet": 1}}), _FEATURE_CACHE={})


def bd_for(mode):
    """The cookbook module's lever state on this arm: big's composition never enables the levers it switches off by name (fastkit.BIG_MODE_OFF)."""
    moff = FK.mode_off(modes.MODES[mode].kit_switch)
    return types.SimpleNamespace(_D59_GUARD=BD._D59_GUARD, _ef2_loop_pppl=(BD._ef2_loop_pppl if "ef2_loop_pppl" not in moff else None),
                                 _ef2_esmc_overlap=(BD._ef2_esmc_overlap if "ef2_esmc_overlap" not in moff else None), _FEATURE_CACHE={})


def pppl_for(mode):
    return (lambda _: None) if "ef2_pppl_graph" in FK.mode_off(modes.MODES[mode].kit_switch) else pppl_ok


def collect(mode="fast", **kw):
    app = kw.pop("app", None) or app_for(mode)
    log = kw.pop("log", LOG_BY_MODE[mode])
    return EV.collect_after(mode, kw.pop("BD", bd_for(mode)), app, kw.pop("counter", counter_for(mode)), log, agk=kw.pop("agk", Agk), n_tokens=kw.pop("n_tokens", 195),
                            pppl_stats_fn=kw.pop("pppl", pppl_for(mode)), n_steps=kw.pop("n_steps", 150), critic_switches=kw.pop("critic_switches", None),
                            user_switches=kw.pop("user", None), batch_size=kw.pop("batch_size", 1))


def lever(rep, name):
    return next(l for l in rep["levers"] if l["name"] == name)


def aside_reasons(rep, name):
    return [a["reason"] for a in rep["asides"] if a["name"] == name]



def test_clean_fast_arm_is_applied():
    rep = collect("fast")
    assert rep["applied"] and rep["refusals"] == [] and rep["fallback"] == [] and rep["fastkit_lines"][0]["mode"] == "agk3"


def test_clean_big_arm_is_applied_at_every_size():
    from .. import report as R
    for toks in (195, 431, 900):                                                    # big: the floor plan and no trunk graph pool at any size — nothing size-gated to refuse
        rep = collect("big", app=app_for("big", n_tokens=toks), n_tokens=toks, log=LOG_BY_MODE["big"].replace("195", str(toks)))
        assert rep["applied"] and rep["refusals"] == [] and rep["fallback"] == [] and rep["fastkit_lines"][0] == {"mode": "big", "ckpt": "block", "tokens": str(toks)}, (toks, rep["refusals"])
        assert R.lever_line(lever(rep, "ef2_bwd_ckpt")) == "LEVER name=ef2_bwd_ckpt state=on served=static fallback=none policy=block reason=memory_floor kept_per_pass=0/24 kernels=fused3 copies=1" and R.lever_line(lever(rep, "ef2_stepgraph")) == "LEVER name=ef2_stepgraph state=off served=static fallback=none rule=never"
        assert FK.BIG_MODE_OFF and rep["mode_off"] == list(FK.BIG_MODE_OFF) and all(R.lever_line(lever(rep, n)) == f"LEVER name={n} state=off served=static fallback=none reason=mode:big" for n in FK.BIG_MODE_OFF), [R.lever_line(lever(rep, n)) for n in FK.BIG_MODE_OFF]   # the levers big switches off by name: one word each
        assert ("lm_graphs=none" in R.evidence_line(rep, "big")) == ("ef2_pppl_graph" in FK.BIG_MODE_OFF and "ef2_esmc_graph" in FK.BIG_MODE_OFF) and "missing=none" in R.evidence_line(rep, "big")
    engaged = collect("big", app=app_for("fast"), BD=BD, pppl=pppl_ok)                 # a big arm on which fast's graph levers / overlap engaged all the same: not the composition — refused by name, lever by lever
    assert sorted(r.split(":")[0] for r in engaged["refusals"] if "switched off by the big composition" in r) == sorted(FK.BIG_MODE_OFF), engaged["refusals"]


def test_clean_exact_arm_is_applied():
    rep = collect("exact", log=LOG_BY_MODE["exact"])
    assert rep["applied"] and rep["refusals"] == []


def test_F1_pppl_graph_disabled_stands_aside_by_its_word_and_its_counts_derive_from_the_run():
    """The pPPL graph's DECLARED word (disabled_reason: a capture that raised / a replay off its eager run — the loss ran the stock eager stack
    from then on) is an aside: LEVER state word, EVIDENCE fallback= names it, no refusal, the run completes. Without a word its bookkeeping is
    checked against the run's own — replays = steps x LM calls per step (B x LM_MASK_PASSES rows in chunks of LM_LOSS_BATCH_SIZE), captures =
    the step's LM input shapes, eager = the ESMC forward graph's capture runs through the shared trunk ((n_warmup + 1) x its captures: 3 at
    one ESMC capture, 6 at two) — a count off that has no word and is refused; not installed is refused."""
    from .. import report as R
    rep = collect("fast", pppl=lambda _: dict(pppl_ok(None), disabled_reason="capture failed: RuntimeError: x", replays=20, eager=133))
    assert rep["refusals"] == [] and rep["applied"] and aside_reasons(rep, "ef2_pppl_graph") == ["disabled:capture_failed"], (rep["refusals"], rep["asides"])
    assert lever(rep, "ef2_pppl_graph")["state"] == "on" and lever(rep, "ef2_pppl_graph")["extra"]["aside"] == "disabled:capture_failed"   # it served 20 steps, then stood aside by its word
    assert R.lever_line(lever(rep, "ef2_pppl_graph")) == "LEVER name=ef2_pppl_graph state=on served=20 fallback=none captures=1 aside=disabled:capture_failed"
    assert rep["fallback"][0].startswith("ef2_pppl_graph stood aside by its declared word: disabled:capture_failed") and R.evidence_line(rep, "fast").startswith("EVIDENCE levers=applied fallback=ef2_pppl_graph missing=none refusals=0 ")
    rep = collect("fast", pppl=lambda _: dict(pppl_ok(None), disabled_reason="replay != eager at capture", replay_fail=1, replays=0, eager=153))
    assert rep["refusals"] == [] and R.lever_line(lever(rep, "ef2_pppl_graph")) == "LEVER name=ef2_pppl_graph state=stepped_aside served=0 fallback=none captures=1 reason=replay_fail:1"   # never replayed: stepped_aside reason=
    rep = collect("fast", pppl=lambda _: dict(pppl_ok(None), eager=4))                                          # one eager call beyond the ESMC graph's capture runs and NO word: a loss call fell through — refused
    assert any("F1 pPPL graph eager calls 4 != 3" in r for r in rep["refusals"]) and rep["fallback"] and not rep["applied"] and rep["asides"] == []
    rep = collect("fast", pppl=lambda _: None)
    assert any(r.startswith("F1 pPPL graph not installed") for r in rep["refusals"])
    rep = collect("fast", pppl=lambda _: dict(pppl_ok(None), replays=149), n_steps=150)
    assert any("replays 149 != 150 (150 design steps x 1 LM call(s) per step: B=1 x 4 masked passes in chunks of 128)" in r for r in rep["refusals"]), rep["refusals"]
    rep = collect("fast", pppl=lambda _: dict(pppl_ok(None), captures=2, replays=150, eager=3), n_steps=150)      # a second capture in a one-trajectory process
    assert any("captures 2 != 1" in r for r in rep["refusals"])
    rep = collect("fast", pppl=pppl_ok, n_steps=150)                                                            # B=1: one ESMC capture -> 3 eager runs, one LM call per step
    assert rep["applied"] and any(e.startswith("F1 pPPL graph captures=1 replays=150 eager=3 (expected 1 / 150 / 3: B=1") for e in rep["events"]), rep["events"]
    # B=2: the design folds (2 rows) and the hero critic folds (1 row) are two ESMC input shapes -> two ESMC captures -> 6 eager runs through the shared trunk; still one LM call per step
    # the derivation as a function of B, on synthetic stats (a kit arm runs B = 1 — the launcher accepts --batch-size > 1 on --mode off only —; the arithmetic is general):
    # B=2 -> the design rows and the critic's row are two ESMC shapes -> 2 captures -> 6 eager runs; one LM call per step up to B=32
    f1 = lambda rep: [r for r in rep["refusals"] if r.startswith("F1 ")]
    rep = collect("fast", app=app_for("fast", eeg=Eeg(captures=2), n_tokens=300), n_tokens=300, log=LOG.replace("195 tokens, trunk graph pool 2 slots", "300 tokens, trunk graph pool none above 256 tokens"), pppl=lambda _: dict(pppl_ok(None), eager=6), batch_size=2)   # (fast: big installs no pPPL / ESMC graph since 0.6.3)
    assert f1(rep) == [] and rep["batch_size"] == 2 and any("eager=6 (expected 1 / 150 / 6: B=2" in e for e in rep["events"]), (rep["refusals"], rep["events"])
    assert any("F1 pPPL graph eager calls 6 != 3" in r for r in f1(collect("fast", pppl=lambda _: dict(pppl_ok(None), eager=6), batch_size=2)))   # 6 eager against ONE ESMC capture is 3 too many, whatever B is
    assert not any(r.startswith("F1 ") or r.startswith("F2 ") for r in collect("big")["refusals"]) and lever(collect("big"), "ef2_pppl_graph")["state"] == "off"   # big: no pPPL graph to count, its line reads off by the mode
    assert any("F1 pPPL graph eager calls 3 != 6" in r for r in f1(collect("fast", app=app_for("fast", eeg=Eeg(captures=2), n_tokens=300), n_tokens=300, log=LOG.replace("195 tokens, trunk graph pool 2 slots", "300 tokens, trunk graph pool none above 256 tokens"), batch_size=2)))
    rep = collect("fast", app=app_for("fast", n_tokens=300), n_tokens=300, log=LOG.replace("195 tokens, trunk graph pool 2 slots", "300 tokens, trunk graph pool none above 256 tokens"), pppl=lambda _: dict(pppl_ok(None), captures=2, replays=300), batch_size=40)   # B=40: 160 masked rows = a chunk of 128 + one of 32 per step: 2 LM calls, 2 shapes
    assert f1(rep) == [] and any("captures=2 replays=300 eager=3 (expected 2 / 300 / 3: B=40" in e for e in rep["events"]), (rep["refusals"], rep["events"])
    assert any("replays 150 != 300 (150 design steps x 2 LM call(s) per step: B=40" in r for r in f1(collect("fast", pppl=lambda _: dict(pppl_ok(None), captures=2), batch_size=40)))
    bd = types.SimpleNamespace(**dict(vars(BD), LM_MASK_PASSES=4, LM_LOSS_BATCH_SIZE=8))                        # the cookbook's own constants are read off the module when it carries them
    assert any("x 1 LM call(s) per step: B=2 x 4 masked passes in chunks of 8)" in r for r in f1(collect("fast", BD=bd, pppl=lambda _: dict(pppl_ok(None), replays=10), batch_size=2)))
    app = app_for("fast", eeg=Eeg()); [setattr(m._esmc, "_eeg", None) for m in app.inversion_models.values()]
    rep = collect("fast", app=app, BD=types.SimpleNamespace(**dict(vars(BD), _LEVERS_OFF=("ef2_esmc_graph",), _KIT_ASIDE={})), pppl=lambda _: dict(pppl_ok(None), eager=40))
    assert not any(r.startswith("F1 ") for r in rep["refusals"]) and any("unjudged: no ESMC forward graph" in e for e in rep["events"])   # no ESMC graph on the trunk (ablated): the eager count has nothing to derive from and is not judged


def test_F2_esmc_graph_eager_is_its_word_and_never_replayed_without_a_word_is_refused():
    from .. import report as R
    rep = collect("fast", app=app_for("fast", eeg=Eeg(eager=2)), pppl=lambda _: dict(pppl_ok(None)))            # calls outside the graphable form ran the stock forward, COUNTED: an aside
    assert rep["refusals"] == [] and aside_reasons(rep, "ef2_esmc_graph") == ["eager:2"] and R.lever_line(lever(rep, "ef2_esmc_graph")) == "LEVER name=ef2_esmc_graph state=on served=150 fallback=eager:2 captures=1 aside=eager:2"
    assert "fallback=ef2_esmc_graph " in R.evidence_line(rep, "fast")
    rep = collect("fast", app=app_for("fast", eeg=Eeg(replays=0, eager=152)))                                  # every call eager by its word: stepped aside, named, not refused
    assert not any(r.startswith("F2 ") for r in rep["refusals"]) and R.lever_line(lever(rep, "ef2_esmc_graph")) == "LEVER name=ef2_esmc_graph state=stepped_aside served=0 fallback=eager:152 captures=1 reason=eager:152"
    rep = collect("fast", app=app_for("fast", eeg=Eeg(replays=0)))                                             # never replayed and no word: refused
    assert any(r == "F2 ESMC forward graph never replayed" for r in rep["refusals"]) and rep["asides"] == []
    app = app_for("fast"); [setattr(m._esmc, "_eeg", None) for m in app.inversion_models.values()]
    assert "F2 ESMC-6B forward graph not installed" in collect("fast", app=app)["refusals"]                     # not installed, no word: refused


def test_F3_bmm_out_dtype_before_and_after():
    class A(Agk):
        _HAS_BMM_OUT_DTYPE = False
    rep = collect("fast", agk=A)                                   # agk3 = bf16 einsum out: the flag is not exercised -> an event, not a refusal
    assert not any(r.startswith("F3") for r in rep["refusals"]) and any(e.startswith("F3 not exercised") for e in rep["events"])
    app = app_for("fast")
    for m in app.inversion_models.values():
        m._agk_cfg = Cfg(None, "refround_lean", einsum_out_dtype="fp32")
    rep = collect("fast", app=app, agk=A)
    assert any(r.startswith("F3 _HAS_BMM_OUT_DTYPE is False") for r in rep["refusals"])

    class Torch:                                   # a torch whose bmm has no out_dtype keyword (probed on CUDA, where the kit uses it)
        class cuda:
            class CUDAGraph:
                register_generator_state = None
            @staticmethod
            def is_available(): return True
        @staticmethod
        def zeros(*a, **k): return None
        bfloat16 = "bf16"; float32 = "f32"
        @staticmethod
        def bmm(a, b, out_dtype=None):
            raise TypeError("bmm() got an unexpected keyword argument 'out_dtype'")
    before = EV.probe_before("fast", agk=Agk, torch=Torch)
    assert before["bmm_out_dtype_supported"] is False and before["bmm_out_dtype_probe"] == "cuda" and any(e.startswith("F3 torch.bmm(out_dtype=) unsupported") for e in before["events"]) and not before["refusals"]
    Torch.cuda.is_available = staticmethod(lambda: False)
    before = EV.probe_before("fast", agk=Agk, torch=Torch)
    assert before["bmm_out_dtype_probe"] == "none" and before["bmm_out_dtype_supported"] is None


def test_F4_libdevice_and_kd3_before():
    class A(Agk):
        _EXP = "tl.exp"; _KD3_OK = True
    before = EV.probe_before("fast", agk=A)
    assert before["refusals"] == [] and any("F4 Triton exp resolved to 'tl.exp'" in e for e in before["events"])   # another Triton's intrinsic: named as an event, the kernels engage

    class B(Agk):
        _KD3_OK = False; _EXP = "unavailable: boom"
    before = EV.probe_before("fast", agk=B)
    assert any(r.startswith("F4 K-D3 kernels unavailable") for r in before["refusals"])
    assert EV.probe_before("exact", agk=B)["refusals"] == []          # exact uses no K-D3 kernel: not refused there


def test_F5_F6_env_knobs_stripped_and_proven_absent():
    from .. import launch
    env = launch.arm_env(modes.MODES["fast"], PREFIXES, base={"EF2_FAST_KIT_CKPT": "none", "EF2_FAST_KIT_D59": "restore", "EF2_FAST_KIT_PPPL": "0",
                                                              "EF2INV_OPT": "fast", "HF_HOME": "/data/hf", "PATH": "/bin"})
    assert env["EF2_FAST_KIT"] == "agk3" and "EF2_FAST_KIT_CKPT" not in env and "EF2_FAST_KIT_D59" not in env and "EF2_FAST_KIT_PPPL" not in env   # fast's own switch word
    assert "EF2INV_OPT" not in env and env["HF_HOME"] == "/data/hf"
    rec = envproof.prove(PREFIXES, {"EF2_FAST_KIT": "big"}, arm="big", environ={"EF2_FAST_KIT": "big"}, modules=STANDIN, path=[P.KDIR])
    assert rec["clean"], rec
    rec = envproof.prove(PREFIXES, {"EF2_FAST_KIT": "big"}, arm="big", environ={"EF2_FAST_KIT": "big", "EF2_FAST_KIT_CKPT": "none"}, modules=STANDIN, path=[P.KDIR])
    assert not rec["clean"] and "EF2_FAST_KIT_CKPT" in rec["forbidden_present"]
    rec = envproof.prove(PREFIXES, {"EF2_FAST_KIT": "big"}, arm="big", environ={"EF2_FAST_KIT": "exact"}, modules=STANDIN, path=[P.KDIR])
    assert not rec["clean"] and rec["forbidden_present"] == ["EF2_FAST_KIT=exact!=big"]


def test_F7_second_complex_size_refused():
    rep = collect("fast", app=app_for("fast", n_tokens=209), n_tokens=195)
    assert any(r.startswith("F7 app._fast_kit_tokens=209 != n_tokens=195") for r in rep["refusals"])


def test_F8_trunk_pool_words_stand_aside_and_a_missing_or_bypassed_pool_is_refused():
    """The trunk graph pool's DECLARED words — a capture that failed and disabled it, grad-mode passes it ran eagerly and counted, a slot
    re-captured or never captured after a step — are asides (the stock eager trunk served there, named); a pool that is missing where the
    size rule wants one, or bypassed by a rebound forward, has no word: refused."""
    from .. import report as R
    rep = collect("fast", app=app_for("fast", pool_slots=1))
    assert not any(r.startswith("F8") for r in rep["refusals"]) and aside_reasons(rep, "ef2_stepgraph") == ["slots:1/2", "slots:1/2"]   # one aside per inversion model: 1/2 slots after 150 steps
    assert R.lever_line(lever(rep, "ef2_stepgraph")) == "LEVER name=ef2_stepgraph state=on served=296 fallback=none slots=1/2 captures=1 pool_gib=9.0 rule=<=256 aside=slots:1/2"
    assert rep["applied"] and rep["fallback"] == ["ef2_stepgraph stood aside by its declared word: slots:1/2 (ESMFold2-Experimental-Fast: trunk graph pool 0 grad-mode pass(es) ran the trunk eagerly, 1/2 slots after 150 step(s), recaptures 0 — the stock eager trunk served those passes); "
                                              "slots:1/2 (ESMFold2-Experimental-Fast-Cutoff2025: trunk graph pool 0 grad-mode pass(es) ran the trunk eagerly, 1/2 slots after 150 step(s), recaptures 0 — the stock eager trunk served those passes)"], rep["fallback"]
    app = app_for("fast")
    for m in app.inversion_models.values():
        m._pool_stats = None
    rep = collect("fast", app=app)
    assert any("no trunk graph pool installed" in r for r in rep["refusals"])
    app = app_for("fast")
    for m in app.inversion_models.values():
        m._pool_stats = pool_stats(active=False, eager=2, disabled_reason="RuntimeError: capture failed: OOM", replays_fwd=0)
    rep = collect("fast", app=app)
    words = " | ".join(rep["refusals"])
    assert "trunk graph pool bypassed" in words and "disabled" not in words and "eager" not in words, words       # the rebound forward has no word: refused; the pool's own words are asides
    assert aside_reasons(rep, "ef2_stepgraph") == ["disabled:RuntimeError,eager:2"] * 2 and R.lever_line(lever(rep, "ef2_stepgraph")) == "LEVER name=ef2_stepgraph state=stepped_aside served=0 fallback=eager:2 slots=2/2 captures=2 pool_gib=9.0 rule=<=256 reason=disabled:RuntimeError,eager:2"
    app = app_for("fast")
    for m in app.inversion_models.values():
        m._pool_stats = pool_stats(recaptures=3)
    rep = collect("fast", app=app)
    assert rep["refusals"] == [] and aside_reasons(rep, "ef2_stepgraph") == ["slots:2/2,recaptures:3"] * 2
    big = app_for("fast", n_tokens=431)                                                 # above POOL_MAX_TOKENS: no pool, no refusal; a pool there is refused
    assert collect("fast", app=big, n_tokens=431, log=LOG.replace("195 tokens, trunk graph pool 2 slots", "431 tokens, trunk graph pool none above 256 tokens"))["refusals"] == []
    for m in big.inversion_models.values():
        m._pool_stats = pool_stats()
    assert any("installed a trunk graph pool" in r for r in collect("fast", app=big, n_tokens=431)["refusals"])

    class Torch:
        class cuda:
            class CUDAGraph:
                pass
        bfloat16 = "bf16"; float32 = "f32"
        @staticmethod
        def zeros(*a, **k): return None
        @staticmethod
        def bmm(a, b, out_dtype=None): return None
    before = EV.probe_before("exact", agk=Agk, torch=Torch)
    assert any(r.startswith("F8 torch.cuda.CUDAGraph.register_generator_state missing") for r in before["refusals"])


def test_F9_kernel_accounting():
    c = counter_for("fast"); c.calls["transition_refround"] = 40         # 8 grad-mode blocks took the stock path
    rep = collect("fast", counter=c)
    assert any(r.startswith("F9 kernel calls") for r in rep["refusals"]) and rep["kernels"]["applied"] is False
    rep = collect("fast")
    assert rep["kernels"]["applied"] and rep["kernels"]["kernel_calls"] == {"trimul_with_residual_frozen": 0, "transition_refround": 48, "ef2_trimul_fused": 96, "ef2_trimul_nosave": 0}
    assert "F9 no-grad / non-bf16 block calls on the stock path by design: 4" in rep["events"]
    app = app_for("fast")
    for m in app.inversion_models.values():
        m.__dict__["_ef2_trimul_handle"].stats["served"] = 45                          # 3 block calls' triangle multiplications took the stock path under grad (45 + 45 of 96)
    rep = collect("fast", app=app)
    assert any(r.startswith("F9 kernel calls") for r in rep["refusals"]) and rep["kernels"]["expected"]["ef2_trimul_fused"] == 96
    # the wired alternative (agk's K-A2, trimul="bmm2") is counted at agk's entry point (2 TriMuls per block call) when a composition selects it
    c = counter_for("fast"); c.calls["trimul_with_residual_frozen"] = 96
    ks = c.summary("bmm2", "refround_lean", [None, None])
    assert ks["applied"] and ks["kernel_calls"]["trimul_with_residual_frozen"] == 96 == ks["expected"]["trimul_with_residual_frozen"] and ks["expected"]["ef2_trimul_fused"] == 0
    assert collect("exact", log=LOG_BY_MODE["exact"])["kernels"]["applied"]      # exact expects no kernel calls
    c = counter_for("exact"); c.calls["transition_refround"] = 2
    assert any(r.startswith("F9 exact arm engaged fast-class kernels") for r in collect("exact", log=LOG_BY_MODE["exact"], counter=c)["refusals"])


def test_F9_t16_transition_forward_census(monkeypatch):
    """fast / big ask K-D3's forward on the carried t16 kernel (fastkit.COMPOSITION transition_fwd): live -> LEVER state=on with the kernel's
    words and served == the grad-mode transition calls (else F9 refused); a card / stack it cannot serve -> state=stepped_aside reason=… as an
    EVENT (K-D3's forward served: named, never refused); the module absent or never engaged on an arm that asks it -> refused; exact -> off;
    ablated by name (MODEL_OPT_LEVERS_OFF) -> state=ablated, nothing refused."""
    lv = lambda rep: next(l for l in rep["levers"] if l["name"] == "ef2_t16_transition")
    rep = collect("exact", log=LOG_BY_MODE["exact"])
    assert lv(rep) == {"name": "ef2_t16_transition", "state": "off", "served": None, "fallback": {}, "extra": {}} and rep["t16_transition"] is None
    rep = collect("fast")
    served = lv(rep)["served"]
    assert rep["applied"] and lv(rep)["state"] == "on" and served == 48 and lv(rep)["extra"]["cubin"] == "2f600bd17b86" and lv(rep)["extra"]["canary"].startswith("pass:102:bitwise")
    assert lv(collect("big", n_tokens=900))["state"] == "on"
    monkeypatch.setattr(sys.modules["ef2_t16_transition"], "describe", lambda: t16_describe("on", served=40, fallback={"shape": 8}))
    rep = collect("fast")
    assert any(r.startswith("F9 ef2_t16_transition live but served 40 of 48") for r in rep["refusals"]) and lv(rep)["fallback"] == {"shape": 8}
    monkeypatch.setattr(sys.modules["ef2_t16_transition"], "describe", lambda: t16_describe("stepped_aside", reason="not_sm_90:sm_80", fallback={"not_live": 48}))
    rep = collect("fast")
    assert rep["applied"] and lv(rep)["state"] == "stepped_aside" and lv(rep)["extra"]["reason"] == "not_sm_90:sm_80" and lv(rep)["served"] == 0
    assert any(e.startswith("ef2_t16_transition stepped aside on this card / stack (not_sm_90:sm_80") for e in rep["events"]) and not any("t16" in r for r in rep["refusals"])
    monkeypatch.setattr(sys.modules["ef2_t16_transition"], "describe", lambda: t16_describe("off"))
    assert any(r.startswith("F9 ef2_t16_transition") and "never engaged" in r for r in collect("fast")["refusals"])
    import types as _types
    rep = collect("fast", BD=_types.SimpleNamespace(**dict(vars(BD), _LEVERS_OFF=("ef2_t16_transition",), _KIT_ASIDE={})))   # ablated by name: not expected, never refused
    assert lv(rep)["state"] == "ablated" and not any("t16" in r for r in rep["refusals"]) and rep["t16_transition"] is None
    monkeypatch.delitem(sys.modules, "ef2_t16_transition")
    monkeypatch.setattr("builtins.__import__", _refuse_import("ef2_t16_transition"))
    assert any(r.startswith("F9 ef2_t16_transition") and "not importable" in r for r in collect("fast")["refusals"])


def test_F9_kd3_gemmswiglu_forward_census(monkeypatch):
    """fast / big run K-D3 lean; where K-D3's OWN forward serves (the t16 kernel stepped aside: an A100) the kit asks its fused W12 + SwiGLU
    forward (fastkit.KD3_GEMMSWIGLU): live -> LEVER state=on tile=… cc=… and served == the grad-mode transition calls (else F9 refused); a card
    without an entry -> state=stepped_aside reason=cc_untuned:sm_NN as an EVENT (K-D3's cuBLAS chain served: named, never refused); the t16 kernel
    live on this card (an H100) -> state=off reason=t16_serves (not asked, named, never refused — read off t16's own record); the module absent or
    never engaged where asked -> refused; exact -> off; ablated by name -> state=ablated, nothing refused."""
    from .. import report as R
    lv = lambda rep: next(l for l in rep["levers"] if l["name"] == "ef2_kd3_gemmswiglu")
    rep = collect("exact", log=LOG_BY_MODE["exact"])
    assert lv(rep) == {"name": "ef2_kd3_gemmswiglu", "state": "off", "served": None, "fallback": {}, "extra": {}} and rep["kd3_gemmswiglu"] is None
    rep = collect("fast")                                                                                        # H100: t16 on -> not asked, by name
    assert rep["applied"] and rep["refusals"] == [] and lv(rep) == {"name": "ef2_kd3_gemmswiglu", "state": "off", "served": None, "fallback": {}, "extra": {"reason": "t16_serves"}}
    assert R.lever_line(lv(rep)) == "LEVER name=ef2_kd3_gemmswiglu state=off served=static fallback=none reason=t16_serves"
    monkeypatch.setattr(sys.modules["ef2_t16_transition"], "describe", lambda: t16_describe("stepped_aside", reason="not_sm_90:sm_80", fallback={"not_live": 48}))   # A100: K-D3's own forward serves
    for mode in ("fast", "big"):
        rep = collect(mode)
        assert rep["applied"] and rep["refusals"] == [] and lv(rep)["state"] == "on" and lv(rep)["served"] == 48 and lv(rep)["extra"] == {"kernel": "w12swiglu.1", "tile": "128x64x64g8w8s3", "cc": "sm_80"}
        assert R.lever_line(lv(rep)) == "LEVER name=ef2_kd3_gemmswiglu state=on served=48 fallback=none kernel=w12swiglu.1 tile=128x64x64g8w8s3 cc=sm_80"
    monkeypatch.setattr(sys.modules["ef2_kd3_gemmswiglu"], "describe", lambda: gsw_describe("on", served=40, fallback={"h": 8}))
    rep = collect("fast")
    assert any(r.startswith("F9 ef2_kd3_gemmswiglu live but served 40 of 48") for r in rep["refusals"]) and lv(rep)["fallback"] == {"h": 8}
    monkeypatch.setattr(sys.modules["ef2_kd3_gemmswiglu"], "describe", lambda: gsw_describe("stepped_aside", reason="cc_untuned:sm_89"))
    rep = collect("big")
    assert rep["applied"] and rep["refusals"] == [] and lv(rep)["state"] == "stepped_aside" and lv(rep)["extra"] == {"reason": "cc_untuned:sm_89", "cc": "sm_89"}
    assert any(e.startswith("ef2_kd3_gemmswiglu stepped aside on this card / stack (cc_untuned:sm_89") for e in rep["events"])
    assert R.lever_line(lv(rep)) == "LEVER name=ef2_kd3_gemmswiglu state=stepped_aside served=0 fallback=none reason=cc_untuned:sm_89 cc=sm_89"
    monkeypatch.setattr(sys.modules["ef2_kd3_gemmswiglu"], "describe", lambda: gsw_describe("off"))
    assert any(r.startswith("F9 ef2_kd3_gemmswiglu") and "never engaged" in r for r in collect("fast")["refusals"])
    monkeypatch.setattr(sys.modules["ef2_kd3_gemmswiglu"], "describe", lambda: gsw_describe("on", served=48))
    rep = collect("fast", BD=types.SimpleNamespace(**dict(vars(BD), _LEVERS_OFF=("ef2_kd3_gemmswiglu",), _KIT_ASIDE={})))   # ablated by name: not expected, never refused
    assert lv(rep)["state"] == "ablated" and not any("gemmswiglu" in r for r in rep["refusals"]) and rep["kd3_gemmswiglu"] is None
    monkeypatch.delitem(sys.modules, "ef2_kd3_gemmswiglu")
    monkeypatch.setattr("builtins.__import__", _refuse_import("ef2_kd3_gemmswiglu"))
    assert any(r.startswith("F9 ef2_kd3_gemmswiglu") and "not importable" in r for r in collect("fast")["refusals"])


def _refuse_import(name):
    import builtins
    real = builtins.__import__

    def imp(n, *a, **k):
        if n == name:
            raise ImportError(n)
        return real(n, *a, **k)
    return imp


def test_F10_guard_recorded_or_refused():
    rep = collect("fast")
    assert rep["d59_guard"] == {"diffs": 0}
    rep = collect("fast", BD=types.SimpleNamespace(_D59_GUARD=None))
    assert any(r.startswith("F10 no D59 state guard") for r in rep["refusals"])


def test_missing_activation_line_refused():
    rep = collect("fast", log="nothing here")
    assert any(r.startswith("missing: the kit printed no") for r in rep["refusals"])


def test_exact_arm_with_kernels_refused_and_fast_without_refused():
    app = app_for("exact")
    for m in app.inversion_models.values():
        m._agk_cfg = Cfg("bmm2", "refround_lean")
    rep = collect("exact", app=app, log=LOG_BY_MODE["exact"], counter=counter_for("exact"))
    assert any("exact arm trunk config is not the kit's (trimul=None transition=None)" in r for r in rep["refusals"])
    app = app_for("exact")
    for m in app.inversion_models.values():
        m.__dict__["_ef2_trimul_handle"].variant = "fused"
    assert any("exact arm carries ef2_trimul variant 'fused'" in r for r in collect("exact", app=app, log=LOG_BY_MODE["exact"], counter=counter_for("exact"))["refusals"])
    app = app_for("exact")
    for m in app.inversion_models.values():
        m.__dict__.pop("_ef2_trimul_handle")
    assert any("exact arm without the cuEquivariance tile table" in r for r in collect("exact", app=app, log=LOG_BY_MODE["exact"], counter=counter_for("exact"))["refusals"])
    app = app_for("fast")
    for m in app.inversion_models.values():
        m._agk_cfg = Cfg("bmm2", "refround_lean")
    rep = collect("fast", app=app)
    assert any("fast arm trunk config is not the kit's (trimul=None transition=refround_lean)" in r for r in rep["refusals"])
    app = app_for("fast")
    for m in app.inversion_models.values():
        m.__dict__.pop("_ef2_trimul_handle")
    assert any(r.startswith("F9 ESMFold2-Experimental-Fast: fast arm without the fused triangle multiplication") for r in collect("fast", app=app)["refusals"])


def test_F5_memory_plan_missing_or_forced_refused():
    app = app_for("fast")
    for m in app.inversion_models.values():
        m._bwd_ckpt = None
    assert any(r.startswith("F5 ESMFold2-Experimental-Fast: no memory plan installed") for r in collect("fast", app=app)["refusals"])
    app = app_for("fast")
    for m in app.inversion_models.values():
        m._bwd_ckpt["forced"] = True
    rep = collect("fast", app=app)
    assert any("the checkpoint policy was forced" in r for r in rep["refusals"]) and any(f.startswith("F5 ") for f in rep["fallback"])


def test_F11_loop_and_sampler_levers_that_switch_themselves_off_refused(monkeypatch):
    rep = collect("fast")
    names = [l["name"] for l in rep["levers"]]
    assert len(names) == 24 and names[:5] == ["ef2_bwd_ckpt", "trimul", "ef2_trimul_nosave", "cueq_tiles", "chunk"] and names[names.index("ef2_esmc_graph") + 1] == "ef2_esmc_hoist" and all(l["state"] == "on" for l in rep["levers"] if l["name"] != "ef2_kd3_gemmswiglu"), rep["levers"]
    assert names[names.index("ef2_t16_transition") + 1] == "ef2_kd3_gemmswiglu" and rep["levers"][names.index("ef2_kd3_gemmswiglu")] == {"name": "ef2_kd3_gemmswiglu", "state": "off", "served": None, "fallback": {}, "extra": {"reason": "t16_serves"}}   # the H100 rig: t16 runs K-D3's forward, the fused W12 + SwiGLU forward is not asked (named)
    assert len(collect("exact")["levers"]) == 22                                                     # exact: no `chunk` record (the pair-stack chunk there is the mode table's switch, the argv's / critic_switches' word)
    from .. import report as R
    prep_bad = types.ModuleType("ef2_loop_prep"); prep_bad.stats = lambda BD: {"served": 150, "mismatch": 2, "fallback_anchor": 2, "fallback_signature": 0, "splice_off": 1, "pin_failed": "cudaErrorNotSupported", "fallback_pageable": 12}
    monkeypatch.setitem(sys.modules, "ef2_loop_prep", prep_bad)                # the fold-entry lever's DECLARED words (its feature pass switched off, its splice switched off, pinned staging refused): asides, never refusals
    rep = collect("fast")
    assert not any("ef2_loop_prep" in r for r in rep["refusals"]) and aside_reasons(rep, "ef2_loop_prep") == ["mismatch:2,anchor:2", "splice_off:1", "pin_failed"], (rep["refusals"], rep["asides"])
    assert R.lever_line(lever(rep, "ef2_loop_prep")).endswith(" mismatch=2 splice_off=1 aside=mismatch:2,anchor:2,splice_off:1,pin_failed") and lever(rep, "ef2_loop_prep")["state"] == "on"   # it served the other steps: state=on … aside=<words>
    assert "fallback=ef2_loop_prep " in R.evidence_line(rep, "fast") and rep["applied"]
    prep_none = types.ModuleType("ef2_loop_prep"); prep_none.stats = lambda BD: {"served": 0, "mismatch": 1, "fallback_anchor": 0, "fallback_signature": 1, "splice_off": 0}
    monkeypatch.setitem(sys.modules, "ef2_loop_prep", prep_none)
    assert R.lever_line(lever(collect("fast"), "ef2_loop_prep")).startswith("LEVER name=ef2_loop_prep state=stepped_aside served=0 ") and R.lever_line(lever(collect("fast"), "ef2_loop_prep")).endswith(" reason=mismatch:1,signature:1")
    monkeypatch.setitem(sys.modules, "ef2_loop_prep", types.ModuleType("ef2_loop_prep")); sys.modules["ef2_loop_prep"].stats = FAKE_LEVERS["ef2_loop_prep"]["stats"]
    bd = types.SimpleNamespace(_D59_GUARD=Guard(), _ef2_loop_pppl=None, _ef2_esmc_overlap=types.SimpleNamespace(stats={"served": 0, "computed": 150, "fallback": {"early_failed:RuntimeError": 3}}), _FEATURE_CACHE={})
    rep = collect("fast", BD=bd)
    words = " | ".join(rep["refusals"])
    assert "F11 ef2_loop_pppl not installed" in words and "ef2_esmc_overlap" not in words                       # not installed (no word): refused; the early launch that failed SAID so: an aside
    assert aside_reasons(rep, "ef2_esmc_overlap") == ["early_failed:RuntimeError:3"] and R.lever_line(lever(rep, "ef2_esmc_overlap")) == "LEVER name=ef2_esmc_overlap state=stepped_aside served=0 fallback=early_failed:RuntimeError:3 computed=150 reason=early_failed:RuntimeError:3"
    app = app_for("fast")
    for m in app.inversion_models.values():
        m._sg_state.stats["capture_failed"] = 1; m._sg_state.last_failure = "OutOfMemoryError at capture"
    rep = collect("fast", app=app)                                              # the sampler graph's capture that failed after its retry: those samples ran the stock eager sampler, counted — an aside
    assert rep["refusals"] == [] and aside_reasons(rep, "ef2_sampler_graph") == ["capture_failed:1", "capture_failed:1"] and R.lever_line(lever(rep, "ef2_sampler_graph")) == "LEVER name=ef2_sampler_graph state=on served=95 fallback=capture_failed:1 captures=5 samples=5 aside=capture_failed:1"
    assert "last_failure=OutOfMemoryError at capture" in rep["fallback"][0]
    hoist_bad = types.ModuleType("ef2_esmc_hoist"); hoist_bad.stats = lambda m: {"served": 1, "full": 2, "anchored": 0, "fallback": {"anchor": 147, "form": 3}, "off": "anchor", "live": "82/199"}
    monkeypatch.setitem(sys.modules, "ef2_esmc_hoist", hoist_bad)              # the hoist switched itself off by its word (a failed anchor): the full forward served — an aside
    rep = collect("exact")
    assert not any("ef2_esmc_hoist" in r for r in rep["refusals"]) and aside_reasons(rep, "ef2_esmc_hoist") == ["off:anchor"] and R.lever_line(lever(rep, "ef2_esmc_hoist")) == "LEVER name=ef2_esmc_hoist state=on served=1 fallback=anchor:147,form:3 aside=off:anchor full=2 anchored=0 live=82/199", rep["levers"]
    hoist_inert = types.ModuleType("ef2_esmc_hoist"); hoist_inert.stats = lambda m: {"served": 0, "full": 150, "anchored": 0, "fallback": {}, "off": None}
    monkeypatch.setitem(sys.modules, "ef2_esmc_hoist", hoist_inert)
    rep = collect("exact")
    assert not any("ef2_esmc_hoist" in r for r in rep["refusals"]) and R.lever_line(lever(rep, "ef2_esmc_hoist")) == "LEVER name=ef2_esmc_hoist state=stepped_aside served=0 fallback=none reason=never_served full=150 anchored=0", rep["levers"]
    monkeypatch.delitem(sys.modules, "ef2_esmc_hoist")
    assert "F11 ef2_esmc_hoist not installed on the shared ESMC trunk" in collect("exact")["refusals"]           # not installed, no word: refused
    hoist_aside = types.ModuleType("ef2_esmc_hoist"); hoist_aside.PROVEN_CC = {(9, 0): "sm_90"}
    hoist_aside.stats = lambda m: {"served": 0, "full": 0, "anchored": 0, "captures": 0, "fallback": {}, "off": None, "live": None, "cc": "sm_80", "stepped_aside": "cc_unproven:sm_80", "installed": False}
    monkeypatch.setitem(sys.modules, "ef2_esmc_hoist", hoist_aside)          # an A100: the lever stepped aside by name at enable — a LEVER line and an event, never a refusal
    rep = collect("exact")
    lv = next(l for l in rep["levers"] if l["name"] == "ef2_esmc_hoist")
    assert not any("ef2_esmc_hoist" in r for r in rep["refusals"]) and lv["state"] == "stepped_aside" and lv["served"] == 0 and lv["extra"]["reason"] == "cc_unproven:sm_80" and lv["extra"]["cc"] == "sm_80", (lv, rep["refusals"])
    assert any(e.startswith("F11 ef2_esmc_hoist stepped aside by name at enable: cc_unproven:sm_80") and "sm_90 only" in e for e in rep["events"]), rep["events"]
    from ef2inv_opt import report as R
    assert R.lever_line(lv) == "LEVER name=ef2_esmc_hoist state=stepped_aside served=0 fallback=none reason=cc_unproven:sm_80 cc=sm_80"


def test_stock_arm_kit_lines():
    assert EV.stock_lever_lines("2026 INFO fast kit: exact, checkpoint policy ckpt:12 for 195 tokens\nplain line") == ["2026 INFO fast kit: exact, checkpoint policy ckpt:12 for 195 tokens"]
    assert EV.stock_lever_lines("Step 5 loss 1.0\n") == []
    assert EV.fastkit_lines(LOG) == [{"mode": "agk3", "ckpt": "ckpt:12", "tokens": "195"}] and EV.fastkit_lines(LOG_BY_MODE["big"]) == [{"mode": "big", "ckpt": "block", "tokens": "195"}]   # the kit line prints the switch word given


def test_envproof_off_arm(tmp_path):
    rec = envproof.prove(PREFIXES, None, arm="off", environ={"HF_HOME": "/x", "PATH": "/bin"}, modules=dict(STANDIN, os=None), path=["/usr/lib"])
    assert rec["clean"] and rec["modal"] == P.SDK_STANDIN and rec["stub_dirs_on_path"] == [], rec
    rec = envproof.prove(PREFIXES, None, arm="off", environ={"EF2_FAST_KIT": "off"}, modules=dict(STANDIN), path=[])
    assert not rec["clean"] and rec["forbidden_present"] == ["EF2_FAST_KIT"]
    rec = envproof.prove(PREFIXES, None, arm="off", environ={}, modules=dict(STANDIN, ef2_autograd_kernels=None), path=[])
    assert "kit modules loaded ['ef2_autograd_kernels']" in rec["problems"]
    rec = envproof.prove(PREFIXES, None, arm="off", environ={}, modules=dict(STANDIN), path=[P.KDIR])
    assert any(p.startswith("kit dir on sys.path") for p in rec["problems"])
    rec = envproof.prove(PREFIXES, None, arm="off", environ={}, modules=dict(STANDIN, torch=None), path=[])
    assert any(p.startswith("upstream loaded before the proof") for p in rec["problems"])
    assert envproof.line(rec).startswith("ENV-CLEAN FAIL: ") and "arm=stock" in envproof.line(rec)
    # the modal rule: the stand-in registered, nothing else, and no stubs directory on the path
    rec = envproof.prove(PREFIXES, None, arm="off", environ={}, modules={}, path=[])
    assert "modal stand-in not registered (stock_design.prepare_arm_process)" in rec["problems"]
    foreign = type(sys)("modal"); foreign.__file__ = "/site-packages/modal/__init__.py"
    rec = envproof.prove(PREFIXES, None, arm="off", environ={}, modules={"modal": foreign}, path=[])
    assert any(p.startswith("modal is '/site-packages/modal/__init__.py', not the kit's stand-in") for p in rec["problems"])
    stubs = tmp_path / "stubs"; (stubs / "modal").mkdir(parents=True); (stubs / "modal" / "__init__.py").write_text("")
    rec = envproof.prove(PREFIXES, None, arm="off", environ={}, modules=dict(STANDIN), path=[str(stubs)])
    assert f"modal stubs dir on sys.path ['{stubs}']" in rec["problems"]
def test_every_prefix_the_kit_reads_is_forbidden():
    reads = P.PINS["stock_environment"]["kit_reads"]
    for var in reads:                                                        # entries ending in "_" are prefixes, the others exact names (launch.arm_env / envproof.forbidden)
        assert any((var.startswith(p) if p.endswith("_") else var == p) for p in PREFIXES), var


def test_cueq_tiles_expected_on_every_kit_arm_and_steps_aside_by_name_under_a_backend_override():
    """The process-wide tile table (ef2_trimul cueq_tiles): exact holds it on the first inversion model, fast / big on a hero critic; a kit arm
    whose critics fold on cuequivariance without it is refused by name; under a user backend override no fold model runs cuequivariance and
    the lever steps aside by name (an event, LEVER cueq_tiles state=off reason=…), never a refusal."""
    for mode in ("exact", "fast", "big"):
        rep = collect(mode, n_tokens=195)
        cq = next(l for l in rep["levers"] if l["name"] == "cueq_tiles")
        assert rep["refusals"] == [] and cq["state"] == "on" and cq["extra"]["holder"] == ("inversion_models.ESMFold2-Experimental-Fast" if mode == "exact" else "hf_critic_models.critic0") and cq["extra"]["entries"] == (2 if mode == "exact" else 4) and cq["extra"]["models"] == (2 if mode == "exact" else 1), (mode, rep["refusals"], cq)
    app = app_for("fast"); app.hf_critic_models = critics_for("fast", tiles=False)
    rep = collect("fast", app=app)
    assert rep["refusals"] == ["cueq_tiles: fast arm without the cuEquivariance tile table on any fold model (ef2_trimul cueq_tiles not installed)"], rep["refusals"]
    app = app_for("fast"); app.hf_critic_models = critics_for("fast", tiles=False)
    for m in app.hf_critic_models.values():
        m.set_kernel_backend("fused")
    bd = types.SimpleNamespace(**dict(vars(BD), _KIT_ASIDE={"cueq_tiles": "no PairUpdateBlock runs the cuequivariance backend"}))
    rep = EV.collect_after("fast", bd, app, counter_for("fast"), LOG, agk=Agk, n_tokens=195, pppl_stats_fn=pppl_ok, n_steps=150, critic_switches={"chunk_size": "shipped", "kernel_backend": "fused", "scope": {"kernel_backend": "all"}})
    cq = next(l for l in rep["levers"] if l["name"] == "cueq_tiles")
    assert rep["refusals"] == [] and cq["state"] == "off" and cq["extra"]["reason"] == "no PairUpdateBlock runs the cuequivariance backend" and any(e.startswith("cueq_tiles stepped aside") for e in rep["events"]), (rep["refusals"], rep["events"])


def _user_backend_app(mode, backend):
    """The app as a `--kernel-backend <backend>` run of `mode` leaves it: EVERY model on the user's backend (stock's setter at load), so no hero
    critic holds the tile table; on exact the first inversion model holds none either (fastkit._enable_cueq_tiles stood it aside by name)."""
    app = app_for(mode); app.hf_critic_models = critics_for(mode, tiles=False)
    for m in app.hf_critic_models.values():
        m.set_kernel_backend(backend)
    if mode == "exact":
        for m in app.inversion_models.values():
            m.__dict__.pop("_ef2_trimul_handle")
    return app


def test_a_user_kernel_backend_on_big_stands_the_tile_table_aside_by_name_and_the_fused_kernel_serves():
    """`--mode fast|big --kernel-backend <v>`: the user's backend is every model's; the kit's fused triangle multiplication is an instance-level
    install that serves the inversion models' grad-mode calls whatever backend they carry (the user's backend reaches the no-grad folds, as the
    NOTE line says) — so `trimul` reads on as ever; the hero critics leave cuequivariance, the tile table has no holder and steps aside BY NAME
    (LEVER cueq_tiles state=stepped_aside reason=user_kernel_backend:<v>); nothing is refused."""
    from .. import report as R
    for kb, word in ((None, "None"), ("fused", "fused")):
        bd = types.SimpleNamespace(**dict(vars(bd_for("big")), _KIT_ASIDE={"cueq_tiles": f"user_kernel_backend:{word}"}, _USER_SWITCHES={"kernel_backend": kb}))
        rep = collect("big", BD=bd, app=_user_backend_app("big", kb), critic_switches=modes.MODES["big"].critic_switches("shipped", kb))
        assert rep["refusals"] == [] and rep["applied"] and rep["user_switches"] == {"kernel_backend": word} and rep["fallback"] == [], (kb, rep["refusals"])
        assert R.lever_line(lever(rep, "cueq_tiles")) == f"LEVER name=cueq_tiles state=stepped_aside served=static fallback=none reason=user_kernel_backend:{word}"
        assert R.lever_line(lever(rep, "trimul")) == "LEVER name=trimul state=on served=48 fallback=stock_path:4 kernel=fused" and any(e.startswith(f"cueq_tiles stepped aside by name: the user's --kernel-backend {word} is every model's") for e in rep["events"])
    rep = collect("big", BD=types.SimpleNamespace(**dict(vars(bd_for("big")), _USER_SWITCHES={"kernel_backend": "cuequivariance"})), critic_switches=modes.MODES["big"].critic_switches("shipped", "cuequivariance"))
    assert rep["refusals"] == [] and lever(rep, "cueq_tiles")["state"] == "on"                                   # the user's cuequivariance: the critics hold the table as ever


def test_exact_accepts_the_users_pair_stack_switches_and_names_what_steps_aside():
    """`--mode exact --kernel-backend None|fused`: every model on the user's backend, as on the stock arm with that flag; the cuEquivariance tile
    table has nothing to serve and reads state=stepped_aside reason=user_kernel_backend:<v>, `trimul` reads kernel=stock with the same reason,
    every other exact lever on, no refusal. `--mode exact --chunk-size N`: the pair stack chunked at N as stock's, LEVER chunk state=user size=N,
    the memory plan's row worded unchunked-priced, no refusal; the default exact census is untouched (no chunk record, tile table on)."""
    from .. import report as R
    for kb, word in ((None, "None"), ("fused", "fused")):
        bd = types.SimpleNamespace(**dict(vars(BD), _KIT_ASIDE={"cueq_tiles": f"user_kernel_backend:{word}"}, _USER_SWITCHES={"kernel_backend": kb}))
        rep = collect("exact", BD=bd, app=_user_backend_app("exact", kb), counter=counter_for("exact"), critic_switches=modes.MODES["exact"].critic_switches(None, kb))
        assert rep["refusals"] == [] and rep["applied"] and rep["fallback"] == [], (kb, rep["refusals"])
        assert R.lever_line(lever(rep, "cueq_tiles")) == f"LEVER name=cueq_tiles state=stepped_aside served=static fallback=none reason=user_kernel_backend:{word}"
        assert R.lever_line(lever(rep, "trimul")) == f"LEVER name=trimul state=off served=static fallback=none kernel=stock reason=user_kernel_backend:{word}"
        states = {l["name"]: l["state"] for l in rep["levers"]}
        assert states["ef2_bwd_ckpt"] == states["ef2_loop_prep"] == states["ef2_esmc_graph"] == states["ef2_pppl_graph"] == "on" and "chunk" not in states
    app = app_for("exact")                                                                                       # --chunk-size 64: every model chunked at 64 (stock's setter), read back off the trunk modules
    for m in app.inversion_models.values():
        for blk in m.folding_trunk.blocks:
            blk.tri_mul_out._chunk_size = blk.tri_mul_in._chunk_size = blk.pair_transition._chunk_size = 64
    for m in app.hf_critic_models.values():
        m.set_chunk_size(64)
    bd = types.SimpleNamespace(**dict(vars(BD), _USER_SWITCHES={"chunk_size": 64}))
    rep = collect("exact", BD=bd, app=app, counter=counter_for("exact"), critic_switches=modes.MODES["exact"].critic_switches(64, "cuequivariance"))
    assert rep["refusals"] == [] and rep["applied"] and rep["fallback"] == [] and rep["user_switches"] == {"chunk_size": "64"}, rep["refusals"]
    assert R.lever_line(lever(rep, "chunk")) == "LEVER name=chunk state=user served=static fallback=none size=64 models=2" and lever(rep, "cueq_tiles")["state"] == "on"
    assert R.lever_line(lever(rep, "ef2_bwd_ckpt")) == "LEVER name=ef2_bwd_ckpt state=on served=static fallback=none policy=ckpt:12 kept_per_pass=12/24 kernels=stock copies=2 plan_row=stock(unchunked-priced)"
    assert any(r.endswith("not the user's --chunk-size 64 ({'trunk.tri_mul_out': None, 'trunk.tri_mul_in': None, 'trunk.pair_transition': None})") or "not the user's --chunk-size 64" in r
               for r in collect("exact", BD=bd, app=app_for("exact"), counter=counter_for("exact"))["refusals"])   # a user chunk the models do not carry after the loop: refused by name (the setter did not hold)
    rep = collect("exact", counter=counter_for("exact"))                                                          # the default exact census: untouched
    assert rep["refusals"] == [] and "chunk" not in {l["name"] for l in rep["levers"]} and "plan_row" not in lever(rep, "ef2_bwd_ckpt")["extra"] and rep["user_switches"] == {}


def test_a_user_chunk_on_big_is_recorded_as_the_users_and_only_an_unexplained_chunk_is_refused():
    """`--mode fast|big --chunk-size none|N`: the kit's 64 yielded to the user's value at enable by name; the census reads the user's value
    back (LEVER chunk state=user size=<v> yielded=64) and refuses nothing; WITHOUT a user switch a pair stack off the mode's 64 is refused as ever."""
    from .. import report as R
    app = app_for("big")
    for m in app.inversion_models.values():
        for blk in m.folding_trunk.blocks:
            blk.tri_mul_out._chunk_size = blk.tri_mul_in._chunk_size = blk.pair_transition._chunk_size = None
    bd = types.SimpleNamespace(**dict(vars(bd_for("big")), _KIT_ASIDE={"chunk": "user_chunk_size:none"}, _USER_SWITCHES={"chunk_size": None}))
    rep = collect("big", BD=bd, app=app, critic_switches=modes.MODES["big"].critic_switches(None, "shipped"))
    assert rep["refusals"] == [] and rep["applied"] and rep["fallback"] == [] and R.lever_line(lever(rep, "chunk")) == "LEVER name=chunk state=user served=static fallback=none size=none models=2 yielded=64", (rep["refusals"], lever(rep, "chunk"))
    assert any(e.startswith("chunk: the user's --chunk-size none is every model's; the kit's 64 yielded to it by name (user_chunk_size:none)") for e in rep["events"]) and "chunk=None" in R.evidence_line(rep, "big")
    rep = collect("big", app=app)                                                                              # no user switch: the pair stack unchunked on a big arm is refused by name, as before
    assert any("big arm runs the pair stack" not in r and "pair-stack chunk is None, not the mode's 64" in r for r in rep["refusals"]), rep["refusals"]
    app128 = app_for("fast")                                                                                     # a fast arm under a user --chunk-size 128 (fast's plan and pool; big's would be its floor and none)
    for m in app128.inversion_models.values():
        for blk in m.folding_trunk.blocks:
            blk.tri_mul_out._chunk_size = blk.tri_mul_in._chunk_size = blk.pair_transition._chunk_size = 128
    for m in app128.hf_critic_models.values():
        m.set_chunk_size(128)
    rep = collect("fast", BD=types.SimpleNamespace(**dict(vars(BD), _USER_SWITCHES={"chunk_size": 128})), app=app128, critic_switches=modes.MODES["big"].critic_switches(128, "shipped"))
    assert rep["refusals"] == [] and R.lever_line(lever(rep, "chunk")) == "LEVER name=chunk state=user served=static fallback=none size=128 models=2 yielded=64"
    assert any("pair-stack chunk is 64, not the user's --chunk-size 128" in r for r in collect("fast", BD=types.SimpleNamespace(**dict(vars(BD), _USER_SWITCHES={"chunk_size": 128})))["refusals"])   # the user's value not on the models after the loop: refused (undeclared)


def test_the_ablation_word_reads_state_ablated_never_a_refusal_and_a_lever_that_failed_still_refuses(monkeypatch):
    """MODEL_OPT_LEVERS_OFF (fastkit.resolve_levers_off -> BD._LEVERS_OFF): a lever switched off by name is not expected after the loop — its record
    reads state=ablated, the report lists it under `ablated`, the EVIDENCE line names it, applied stays True; a lever NOT named that failed to
    engage is refused exactly as before; a lever named off that engaged anyway is refused by name."""
    from .. import report as R
    off = ("ef2_loop_prep", "ef2_loop_pppl", "ef2_esmc_overlap", "cueq_tiles", "chunk", "ef2_esmc_hoist")
    bd = types.SimpleNamespace(_D59_GUARD=Guard(), _ef2_loop_pppl=None, _ef2_esmc_overlap=None, _FEATURE_CACHE={}, _LEVERS_OFF=off, _KIT_ASIDE={})
    monkeypatch.setitem(sys.modules, "ef2_loop_prep", types.ModuleType("ef2_loop_prep")); sys.modules["ef2_loop_prep"].stats = lambda BD: {}
    monkeypatch.delitem(sys.modules, "ef2_esmc_hoist")
    app = app_for("fast"); app.hf_critic_models = critics_for("fast", tiles=False)
    for m in app.inversion_models.values():
        for blk in m.folding_trunk.blocks:
            blk.tri_mul_out._chunk_size = blk.tri_mul_in._chunk_size = blk.pair_transition._chunk_size = None     # `chunk` ablated: the kit set the pair stack unchunked
    rep = collect("fast", BD=bd, app=app)
    assert rep["refusals"] == [] and rep["applied"] and rep["ablated"] == list(off), rep["refusals"]
    states = {l["name"]: l["state"] for l in rep["levers"]}
    assert all(states[n] == "ablated" for n in off) and all(v == "on" for n, v in states.items() if n not in off and n != "ef2_kd3_gemmswiglu") and states["ef2_kd3_gemmswiglu"] == "off", states   # t16 on (the H100 rig): K-D3's fused forward not asked
    ev = R.evidence_line(rep, "fast")
    assert ev.endswith(" ablated=" + ",".join(off)) and "levers=applied" in ev and "chunk=None" in ev and R.lever_line(rep["levers"][3]) == "LEVER name=cueq_tiles state=ablated served=static fallback=none"
    rep = collect("fast", BD=types.SimpleNamespace(**dict(vars(bd), _LEVERS_OFF=("chunk",))), app=app)          # the loop levers and the hoist NOT named this time: their absence is refused by name as ever
    words = " | ".join(rep["refusals"])
    assert "F11 ef2_loop_prep not installed" in words and "F11 ef2_loop_pppl not installed" in words and "F11 ef2_esmc_hoist not installed" in words and "cueq_tiles: fast arm without" in words and not rep["applied"], words
    rep = collect("fast", BD=types.SimpleNamespace(**dict(vars(BD), _LEVERS_OFF=("ef2_lazy_structure",), _KIT_ASIDE={})))   # named off yet engaged: refused by name
    assert any(r.startswith("ef2_lazy_structure: ablated by MODEL_OPT_LEVERS_OFF yet engaged after the loop") for r in rep["refusals"]), rep["refusals"]
    assert "ablated=" not in R.evidence_line(collect("fast"), "fast") and "ablated=" not in R.active_line(modes.MODES["fast"], {"ablated": []})   # a plain run of a mode carries no such word
    assert " ablated=chunk line=" in R.active_line(modes.MODES["fast"], {"ablated": ["chunk"]})


def test_the_trunk_pool_size_rule_words():
    """fastkit.trunk_pool_wanted / pool_word: the mode carries the pool AND the complex is at most POOL_MAX_TOKENS tokens."""
    from .. import fastkit as FK
    assert FK.trunk_pool_wanted("agk3", 195) is True and FK.trunk_pool_wanted("agk3", 256) is True and FK.trunk_pool_wanted("agk3", 257) is False and FK.trunk_pool_wanted("exact", 195) is True
    assert FK.pool_word("agk3", 195) == "2 slots" and FK.pool_word("agk3", 300) == "none above 256 tokens" and FK.pool_word("agk3", 195, True) == "none (ablated)"
    assert FK.trunk_pool_wanted("big", 195) is False and FK.trunk_pool_wanted("big", 257) is False and FK.pool_word("big", 195) == FK.pool_word("big", 300) == FK.pool_word("big", 195, True) == "none"   # big carries no pool at any size

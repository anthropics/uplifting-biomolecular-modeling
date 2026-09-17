"""`big` and `fast`: the two tier-2 modes (distinct compositions; no mode is an alias of another).
`fast` (EF2_FAST_KIT=agk3) = the fused kernels, the pair stack chunked at 64, the memory plan FILLING the card, the trunk graph pool at or below
fastkit.POOL_MAX_TOKENS tokens and stepped aside by name above; `big` (EF2_FAST_KIT=big) = fast MINUS the two levers with a measured
memory cost — the memory plan pinned to its floor (policy block, reason memory_floor) and no trunk graph pool at any size. ACTIVE / EVIDENCE /
LEVER lines print those facts; the after-loop census refuses a wrong composition by name (a fill plan on big, a floor plan on fast); a CUDA
out-of-memory arm exits by name (no fallback). patches.py's other patch (the upstream bug-0002 fix, applied on every mode) is tested at the
bottom: its off-record."""
import logging
import os
import sys
import types

from .. import evidence as EV, fastkit as FK, launch, modes, patches as PT, report as R, stock_design as SD
from . import _paths as P


def test_big_is_fast_minus_the_levers_with_a_memory_cost_and_fast_is_a_mode_of_its_own():
    m, f = modes.MODES["big"], modes.MODES["fast"]
    assert (m.kit_switch, m.is_kit, m.tier) == ("big", True, "2") and (f.kit_switch, f.is_kit, f.tier) == ("agk3", True, "2") and modes.DEFAULT_MODE == "fast" and m is not f
    assert modes.ALIASES == {} and modes.resolve("fast") is f and modes.resolve("big") is m and modes.MODE_NAMES == ["off", "exact", "fast", "big"]   # no alias: two tier-2 rows
    assert modes.alias_word("fast") is None and modes.alias_word("big") is None and modes.alias_word("exact") is None
    for word in ("memory plan pinned to its floor", "policy block", "reason=memory_floor", "no trunk graph pool at any size", "off by name (reason=mode:big)", "the stock arm's", "fused kernels"):
        assert word in m.what, word
    for word in ("chunked pair stack", "memory plan filling the card", "trunk graph pool at or below 256 tokens", "fused kernels", "recommended"):
        assert word in f.what, word
    assert any("NO trunk" in l for l in m.levers) and any("memory_floor" in l for l in m.levers) and any("set_chunk_size(64)" in l for l in m.levers) and not any("stepped aside by name above" in l for l in m.levers)
    assert any("ef2_stepgraph" in l and "stepped aside by name above" in l for l in f.levers) and any("rule budget" in l for l in f.levers) and not any("NO trunk" in l for l in f.levers)
    assert m.levers[:9] == f.levers[:9] == list(modes.TIER2_KERNEL_LEVERS)                              # the tier-2 kernel levers are shared word for word
    fa, bi = FK.COMPOSITION["agk3"], FK.COMPOSITION["big"]
    assert (fa["ckpt"], fa["pool"], bi["ckpt"], bi["pool"]) == ("budget", True, "floor", False) and bi == {**fa, "ckpt": "floor", "pool": False, "mode_off": FK.BIG_MODE_OFF} and bi["chunk"] == fa["chunk"] == FK.BIG_CHUNK_SIZE == 64   # big = fast minus exactly {the plan's fill, the trunk graph pool, the levers named in BIG_MODE_OFF}
    assert fa["mode_off"] == () == FK.COMPOSITION["exact"]["mode_off"] and FK.mode_off("agk3") == () == FK.mode_off("exact") and FK.mode_off("big") == FK.BIG_MODE_OFF
    assert FK.BIG_MODE_OFF == ("ef2_esmc_overlap", "ef2_pairbias_attn", "ef2_sampler_graph", "ef2_esmc_graph", "ef2_esmc_hoist", "ef2_pppl_graph") and set(FK.BIG_MODE_OFF) <= set(FK.MODE_OFF_LEVERS)   # each keeps memory resident (CHANGES.md §big); the kernels, nosave, t16, fused LN, bf16 head, chunk and the loop levers stay
    assert [n for n in FK.levers_of("agk3") if n in FK.BIG_MODE_OFF] == list(FK.BIG_MODE_OFF)                         # the tuple is in census (enable) order
    assert FK.COMPOSITION["exact"]["ckpt"] == "budget" and FK.CKPT_RULES == ("budget", "floor") and FK.ckpt_floor("big") and not FK.ckpt_floor("agk3") and not FK.ckpt_floor("exact")
    assert (bi["trimul"], bi["transition"], bi["nosave_fwd"], bi["bf16_confidence"], FK.COMPOSITION["exact"]["trimul"], FK.COMPOSITION["exact"]["transition"]) == ("fused", "refround_lean", "fast", True, "cueq_tiles", None)   # nosave stays in big (every block is checkpointed: it serves them all)
    assert all(c["trimul"] in FK.TRIMUL_KERNELS for c in FK.COMPOSITION.values()) and set(FK.COMPOSITION) == set(FK.KIT_SWITCHES) == {"exact", "agk3", "big"}
    assert FK.levers_of("big") == [n for n in FK.levers_of("agk3") if n != "ef2_stepgraph" and n not in FK.BIG_MODE_OFF] and FK.levers_of(m) == [n for n in FK.levers_of(f) if n != "ef2_stepgraph" and n not in FK.BIG_MODE_OFF] and "ef2_stepgraph" in FK.levers_of("agk3")
    assert FK.levers_of(f) == FK.levers_of("agk3") + ["critic_switches"] and FK.levers_of(m)[-1] == "critic_switches"   # a Mode adds its critic-scope lever to the switch word's census names
    assert launch.arm_env(f, ["EF2_"], base={"PATH": "/usr/bin"})["EF2_FAST_KIT"] == "agk3"          # fast's arm carries its own switch word
    env = launch.arm_env(m, ["EF2_", "EF2INV_", "ESMFOLD2_"], base={"PATH": "/usr/bin", "EF2_FAST_KIT_CKPT": "none"})
    assert env["EF2_FAST_KIT"] == "big" and "EF2_FAST_KIT_CKPT" not in env and env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True" and "TORCHINDUCTOR_SHAPE_PADDING" not in env and "TORCHINDUCTOR_DETERMINISTIC" not in env
    mine = launch.arm_env(m, ["EF2_"], base={"PATH": "/usr/bin", "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256"}, det=1)
    assert mine["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:256" and (mine["TORCHINDUCTOR_SHAPE_PADDING"], mine["TORCHINDUCTOR_DETERMINISTIC"]) == ("0", "1")          # the operator's allocator setting is honoured; the det pins ride every det arm
    off = launch.arm_env(modes.MODES["off"], ["EF2_"], base={"PATH": "/usr/bin"}, det=1)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in off and "EF2_FAST_KIT" not in off and (off["TORCHINDUCTOR_SHAPE_PADDING"], off["TORCHINDUCTOR_DETERMINISTIC"]) == ("0", "1") and "TORCHINDUCTOR_FORCE_SHAPE_PAD" not in off   # stock's allocator untouched; the det pins on both arms
    assert launch.env_words(mine, 1) == " det_pins=TORCHINDUCTOR_SHAPE_PADDING=0,TORCHINDUCTOR_DETERMINISTIC=1 allocator=max_split_size_mb:256" and launch.env_words(off, 0) == " det_pins=none allocator=unset"


def _fake_kit(calls):
    """The kit's lever modules as call recorders (name, kwargs) in enable order; nothing of torch."""
    def rec(name, ret=None):
        def f(*a, **k):
            calls.append((name, k)); return ret(*a, **k) if callable(ret) else ret
        return f
    mods = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        mods[name] = m; return m
    mod("ef2_autograd_kernels", enable=rec("agk.enable"), enable_skip_unused_confidence=rec("skip_conf"), describe=lambda cfg: str(cfg))
    mod("ef2_state_guard", StateGuard=lambda **kw: ("guard", kw))
    def trimul_enable(model, variant="fused"):
        calls.append(("trimul", {"variant": variant})); model.__dict__["_ef2_trimul_handle"] = types.SimpleNamespace(variant=variant, stats={"served": 0, "fallback": 0}, cueq_saved={})
    mod("ef2_trimul", enable=trimul_enable, describe=lambda h: "stock" if h is None else h.variant)
    mod("ef2_fused_ln", enable=rec("fused_ln"), stats=lambda: {"served": 0, "fallback": {}, "installed": True, "channel_major": False})
    def bc_enable(m, n_tokens, **kw):
        calls.append(("bwd_ckpt", dict(kw, n_tokens=n_tokens)))
        m._bwd_ckpt = ({"policy": "block", "kept_per_pass": 0, "n_blocks": 24, "copies": kw.get("copies"), "kernels": "agk3", "floor": True, "reason": "memory_floor"} if kw.get("floor")
                       else {"policy": "ckpt:6" if n_tokens > 300 else "none", "kept_per_pass": 18, "n_blocks": 24, "copies": kw.get("copies"), "kernels": "agk3", "floor": False})   # the planner's two rules, as plan() records them
        return m._bwd_ckpt
    mod("ef2_bwd_ckpt", enable=bc_enable, describe=lambda m: f"LEVER name=ef2_bwd_ckpt state=on policy={m._bwd_ckpt['policy']}", kernels_of=lambda m: "fused3" if m.__dict__.get("_ef2_trimul_handle") else "stock",
        LM_GRAPH_POOL_BYTES=2.4 * 2 ** 30, TRANSIENT_KEPT_EQUIV=1.0, KEPT_BYTES_PER_POS={"stock": 11.2 * 1024, "agk3": 7.1 * 1024, "fused3": 4.8 * 1024}, TRANSIENT_CONST_BYTES=2 ** 30)
    mod("ef2_stepgraph", enable=rec("stepgraph"), disable=rec("stepgraph.disable"), stats=lambda m: None)
    mod("ef2_esmc_graph", enable=rec("esmc_graph"), release=rec("esmc_graph.release"))
    mod("ef2_esmc_hoist", enable=rec("esmc_hoist"), release=rec("esmc_hoist.release"), stats=lambda m: {"served": 3, "full": 1, "anchored": 1, "captures": 1, "fallback": {}, "off": None}, POOL_BYTES=0.5 * 2 ** 30,
        cc_route=lambda cc=None: (True, "sm_90"), PROVEN_CC={(9, 0): "sm_90"})
    mod("ef2_pppl_graph", enable=rec("pppl_graph"), release=rec("pppl_graph.release"), stats=lambda lm: None)
    mod("ef2_esmc_rope", enable=rec("rope"), handle=lambda h: None)
    mod("ef2_esmc_overlap", enable=rec("overlap"))
    def prep_enable(BD, **kw):
        calls.append(("loop_prep", kw)); inner = BD.fold_and_get_distogram
        BD.fold_and_get_distogram = lambda *a, **k: inner(*a, **k)               # replaces the module attribute, as the lever does
    mod("ef2_loop_prep", enable=prep_enable, release=rec("loop_prep.release"), stats=lambda BD: {})
    mod("ef2_loop_pppl", enable=rec("loop_pppl"), stats=lambda BD: {})
    mod("ef2_lazy_structure", enable=rec("lazy"), stats=lambda m: None)
    mod("ef2_pairbias_attn", enable=rec("pairbias"))
    mod("ef2_sampler_graph", enable=rec("sampler_graph"))
    mod("ef2_bf16_confidence", enable=rec("bf16_conf"))

    def t16_engage():
        calls.append(("t16.engage", {})); return {"name": "ef2_t16_transition", "state": "on", "reason": None, "reason_text": None, "version": "transition_cute.1.0", "device": "H100",
                                                    "kernel": {"cubin": "2f600bd17b86", "canary": "pass:102:bitwise:6a3438b92fec:max_abs=0.0312"}, "served": 0, "fallback": {}, "packs": 0}
    mod("ef2_t16_transition", engage=t16_engage, describe=lambda: dict(t16_engage(), served=0), live=lambda: True)
    def gsw_engage(cc=None, cfg=None):
        calls.append(("gemmswiglu.engage", {})); return {"name": "ef2_kd3_gemmswiglu", "state": "on", "reason": None, "reason_text": None, "version": "w12swiglu.1", "device": "A100",
                                                           "cc": "sm_80", "tile": "128x64x64g8w8s3", "served": 0, "fallback": {}, "installed": True}
    mod("ef2_kd3_gemmswiglu", engage=gsw_engage, describe=lambda: gsw_engage(), live=lambda: True, NAME="ef2_kd3_gemmswiglu")
    def nosave_enable(model, word="fast", n_tokens=None):                                            # the kit-enable step tells the lever the design's size (its LEVER line's tier_fast= is sized)
        calls.append(("nosave", {"word": word, "n_tokens": n_tokens})); h = types.SimpleNamespace(on=True, word=word, row="v4", reason="", stats={"served": 0}, refused={}, tier_fast="v4"); model.__dict__["_ef2_nosave_handle"] = h; return h
    mod("ef2_trimul_nosave", enable=nosave_enable, describe=lambda h: f"LEVER name=ef2_trimul_nosave state={'on' if h.on else 'stepped_aside'}", extra_bytes=lambda m, n: 0.0, NAME="ef2_trimul_nosave")
    return mods


class _Model:
    def __init__(self):
        self.chunk_set = []; self._esmc = types.SimpleNamespace()

    def set_chunk_size(self, v):
        self.chunk_set.append(v)

    def set_kernel_backend(self, v):
        pass


def _fake_cookbook(name="binder_design"):
    """A stock-shaped cookbook module: the names fastkit.install hooks (prepare_esmfold2_tensors, fold_and_get_distogram,
    ESMFold2Design.design / no _enable_fast_kit of its own), torch + logger; nothing of the kit on it."""
    BD = types.ModuleType(name)
    BD.os = __import__("os"); BD.logger = logging.getLogger("fake_cookbook"); BD.torch = types.SimpleNamespace(is_tensor=lambda v: False, cuda=types.SimpleNamespace(get_device_properties=lambda i: types.SimpleNamespace(total_memory=80 * 2**30)))
    BD.TARGET_SEQUENCES = {"pd-l1": "ACDE|FGH"}; BD.BINDER_PROMPT_FACTORIES = {"minibinder": types.SimpleNamespace(sample=lambda seed: "X" * (10 + seed))}
    BD.prepare_esmfold2_tensors = lambda input, max_tokens=None, max_atoms=None, max_seqs=16384, pad_to_max_seqs=False, seed=None, use_vectorized_msa_assembly=True: {"n": [len(p.sequence) for p in input.sequences], "max_atoms": max_atoms}
    BD.folds = []
    BD.fold_and_get_distogram = lambda model, *a, **kw: BD.folds.append(kw) or {"iptm": None}

    class ESMFold2Design:
        def __init__(self):
            self.inversion_models = {"a": _Model(), "b": _Model()}; self.hf_critic_models = {"c": _Model()}
            self.hf_critic_models["c"]._esmc = self.inversion_models["a"]._esmc                      # a critic sharing an inversion model's ESMC trunk: roped once
            self.esmc_model = types.SimpleNamespace(esmc=types.SimpleNamespace())

        def design(self, target_name, binder_name, target_sequence=None, binder_sequence=None, is_antibody=None, seed=0, batch_size=1, target_hotspot_ids=None, epitope_contact_distance=12.0):
            for step in range(3):
                BD.fold_and_get_distogram(self.inversion_models["a"], num_loops=1, calculate_confidence=False)
            BD.fold_and_get_distogram(self.inversion_models["a"], num_loops=3, calculate_confidence=True)      # a hero critic
            return ["SEQ"], {}, []
    BD.ESMFold2Design = ESMFold2Design
    return BD


def test_fast_and_big_compositions_are_exactly_the_named_levers(monkeypatch):
    for kit in ("agk3", "big"):                                                                   # fast's switch word, big's
        pooled = kit == "agk3"
        calls = []
        for name, mod in _fake_kit(calls).items():
            monkeypatch.setitem(sys.modules, name, mod)
        BD = _fake_cookbook()
        rec = FK.install(BD, kit)
        assert rec["applied"] is True and rec["name"] == "design_kit" and rec["kit"] == kit and len(rec["function_sha256"]) == 64 and rec["hooks"] == list(FK.HOOKS)
        assert FK.install(BD, kit) == rec and BD.FAST_KIT == kit                                    # idempotent per module object
        app = BD.ESMFold2Design(); app._enable_fast_kit(200)                                       # a small complex (<= POOL_MAX_TOKENS): the trunk graph pool rides fast, one per inversion model; big carries none
        names = [n for n, _ in calls]
        enables = [kw for n, kw in calls if n == "agk.enable"]
        assert len(enables) == 2 and all(kw == {"trimul": None, "transition": "refround_lean", "checkpoint": "block", "transition_fwd": "t16"} for kw in enables), kit   # the tier-2 kernels, K-D3's forward on the t16 kernel where the card serves it — both modes
        assert [kw for n, kw in calls if n == "trimul"] == [{"variant": "cueq_tiles"}] + [{"variant": "fused"}] * 2 and names.count("fused_ln") == 1 and names.count("stepgraph") == (2 if pooled else 0), kit   # the tile table through the hero critic, then the fused kernel per inversion model; the pool on both inversion models at 200 tokens (fast), on none (big)
        assert app.hf_critic_models["c"].__dict__["_ef2_trimul_handle"].variant == "cueq_tiles" and all(m.__dict__["_ef2_trimul_handle"].variant == "fused" for m in app.inversion_models.values())
        assert names.count("skip_conf") == 2 and names.count("bf16_conf") == 2 and all(m.chunk_set == [FK.BIG_CHUNK_SIZE] for m in app.inversion_models.values()), kit
        assert [kw for n, kw in calls if n == "nosave"] == [{"word": "fast", "n_tokens": 200}] * 2 and names.index("nosave") > names.index("trimul") and all(m.__dict__["_ef2_nosave_handle"].row == "v4" for m in app.inversion_models.values()), kit   # the no-save first pass per inversion model, over the fused kernel — kept in big (every block is checkpointed there)
        moff = FK.mode_off(kit)
        for lever_name, call, n_on in (("ef2_esmc_graph", "esmc_graph", 1), ("ef2_pppl_graph", "pppl_graph", 1), ("ef2_esmc_hoist", "esmc_hoist", 1), ("ef2_pairbias_attn", "pairbias", 3), ("ef2_sampler_graph", "sampler_graph", 3), ("ef2_esmc_overlap", "overlap", 1)):
            assert names.count(call) == (0 if lever_name in moff else n_on), (kit, lever_name, names.count(call))   # fast: the LM graphs, the hoist and the overlap ride it, the sampler levers reach the critic too; big never calls what it switches off by name
        assert (kit == "big") == bool(moff) and names.count("loop_prep") == 1 == names.count("loop_pppl"), kit                                     # the loop levers stay on both
        bc = [kw for n, kw in calls if n == "bwd_ckpt"]
        assert [kw["copies"] for kw in bc] == ([2, 2] if pooled else [1, 1]) and [kw["floor"] for kw in bc] == [not pooled] * 2 and app._fast_kit_tokens == 200 and BD._D59_GUARD is not None, kit   # the planner counts the pool's per-model copies (fast) and takes the composition's rule: floor on big, budget on fast — planned either way
        assert all(m._bwd_ckpt["policy"] == ("none" if pooled else "block") for m in app.inversion_models.values()), kit
    assert FK.trunk_pool_wanted("agk3", 200) and FK.trunk_pool_wanted("agk3", FK.POOL_MAX_TOKENS) and not FK.trunk_pool_wanted("agk3", FK.POOL_MAX_TOKENS + 1)
    assert not FK.trunk_pool_wanted("big", 200) and not FK.trunk_pool_wanted("big", 100) and not FK.trunk_pool_wanted("big", 431)   # big: no pool at any size
    assert FK.pool_word("agk3", 200) == "2 slots" and FK.pool_word("agk3", 431) == "none above 256 tokens" and FK.pool_word("agk3", 200, True) == "none (ablated)"
    assert FK.pool_word("big", 200) == FK.pool_word("big", 431) == FK.pool_word("big", 200, True) == "none"                # one word at every size (nothing to ablate)
    # above the gate: fast installs no pool either (stepped aside by name) — the reach regime, the plan's fill decides
    calls2 = []
    for name, mod in _fake_kit(calls2).items():
        monkeypatch.setitem(sys.modules, name, mod)
    BD2 = _fake_cookbook(); FK.install(BD2, "agk3"); app2 = BD2.ESMFold2Design(); app2._enable_fast_kit(431)
    assert [n for n, _ in calls2].count("stepgraph") == 0 and [(kw["copies"], kw["floor"]) for n, kw in calls2 if n == "bwd_ckpt"] == [(1, False), (1, False)] and app2._fast_kit_tokens == 431


def test_active_and_evidence_lines_print_the_composition():
    line = R.active_line(modes.MODES["big"], {"esm": "3.4.0", "torch": "2.11.0", "device": "H100", "cache": "shared"})
    assert line.startswith("ACTIVE mode=big route=subprocess tier=2 ") and "kit_switch=EF2_FAST_KIT=big" in line and "memory plan pinned to its floor" in line and "no trunk graph pool at any size" in line and "alias=" not in line
    fline = R.active_line(modes.resolve("fast"), {"esm": "3.4.0", "torch": "2.11.0", "device": "H100", "cache": "shared", "alias": modes.alias_word("fast")})
    assert fline.startswith("ACTIVE mode=fast route=subprocess tier=2 ") and "kit_switch=EF2_FAST_KIT=agk3" in fline and "trunk graph pool at or below 256 tokens" in fline and " alias=" not in fline   # fast's own row: no alias token
    assert R.alias_token(None) == "" and R.alias_token("fast") == " alias=fast"                                                     # the grammar's optional token stays defined (modes.ALIASES is empty)
    after = {"applied": True, "refusals": [], "events": ["F9 ..."], "kernels": {"kernel_calls": {"ef2_trimul_fused": 480, "transition_refround": 240, "trimul_with_residual_frozen": 0}},
             "models": [{"agk": "trimul=None transition=refround_lean ckpt=block", "trimul": {"variant": "fused", "served": 480, "fallback": 8}, "trimul_kernel": "fused", "bwd_ckpt": {"policy": "block", "reason": "memory_floor"},
                         "trunk_graph_pool": None, "chunk_size": 64, "esmc_graph": {"captures": 1}}], "pppl_graph": {"captures": 1}}
    ev = R.evidence_line(after, "big")
    assert "levers=applied" in ev and "trunk_graphs=none" in ev and "lm_graphs=esmc,pppl" in ev and "chunk=64" in ev and "kernels=trimul=None,transition=refround_lean,ckpt=block trimul=fused" in ev
    assert "kernel_calls=ef2_trimul_fused:480,transition_refround:240,trimul_with_residual_frozen:0 policy=block" in ev
    assert "trimul=bmm2" in R.evidence_line(dict(after, models=[dict(after["models"][0], trimul_kernel="bmm2")]), "big")      # the wired alternative's word
    fast_after = dict(after, models=[dict(after["models"][0], agk="trimul=None transition=refround_lean ckpt=ckpt:20", bwd_ckpt={"policy": "ckpt:20"}, trunk_graph_pool={"slots": 2, "n_slots": 2}, chunk_size=64)])
    evf = R.evidence_line(fast_after, "fast")
    assert "trunk_graphs=2/2" in evf and "chunk=64" in evf and "lm_graphs=esmc,pppl" in evf and "policy=ckpt:20" in evf and "ablated=" not in evf
    assert R.lever_line(EV.lever_record("ef2_bwd_ckpt", "on", None, None, policy="block", reason="memory_floor", kept_per_pass="0/24", kernels="fused3")) == "LEVER name=ef2_bwd_ckpt state=on served=static fallback=none policy=block reason=memory_floor kept_per_pass=0/24 kernels=fused3"
    assert R.lever_line(EV.lever_record("ef2_bwd_ckpt", "on", None, None, policy="ckpt:19", reason=None, kept_per_pass="5/24")) == "LEVER name=ef2_bwd_ckpt state=on served=static fallback=none policy=ckpt:19 kept_per_pass=5/24"   # the budget rule's line carries no reason token (None is not printed)
    lv = R.lever_line(EV.lever_record("trimul", "on", 480, {"stock_path": 8}, kernel="fused"))
    assert lv == "LEVER name=trimul state=on served=480 fallback=stock_path:8 kernel=fused" and R.lever_line(EV.lever_record("trimul", "on", 480, None, kernel="bmm2")) == "LEVER name=trimul state=on served=480 fallback=none kernel=bmm2"
    # the fused path's gated-GEMM launch table names itself on the lever line: the card's entry (sm_90) or the vendored launches by name
    assert R.lever_line(EV.lever_record("trimul", "on", 480, {"stock_path": 8}, kernel="fused", entries=None, reason=None, gemm="sm_90")) == "LEVER name=trimul state=on served=480 fallback=stock_path:8 kernel=fused gemm=sm_90"
    assert R.lever_line(EV.lever_record("trimul", "on", 480, None, kernel="fused", gemm="vendored:cc_untuned:sm_80")).endswith("kernel=fused gemm=vendored:cc_untuned:sm_80")
    assert R.lever_line(EV.lever_record("ef2_bf16_confidence", "off")) == "LEVER name=ef2_bf16_confidence state=off served=static fallback=none"


def _critics(chunk=None, backend="cuequivariance", n=2, tiles=True):
    """The app's hero critics (hf_critic_models): fork-layout stand-ins (test_model_switches._fork_model) as the arm leaves them right after load —
    fast / big call set_chunk_size(None) + set_kernel_backend("cuequivariance") on them (the stock arm's pair) and write the process-wide tile
    table through the first of them (ef2_trimul cueq_tiles: its handle); chunk=64 / backend="fused" = a critic left on the loader's own choice."""
    from .test_model_switches import _fork_model
    out = {}
    for i in range(n):
        m = _fork_model(msa_encoder=bool(i))
        m.set_chunk_size(chunk); m.set_kernel_backend(backend)
        out[f"c{i}"] = m
    if tiles and out:
        out["c0"].__dict__["_ef2_trimul_handle"] = types.SimpleNamespace(variant="cueq_tiles", stats={"served": 0, "fallback": 0}, cueq_saved={1: 1, 2: 2, 3: 3, 4: 4})
    return out


def _census_kit(monkeypatch, pool_stats=None, lazy=True, prep=None, rope_served=4, hoist=True):
    """Fake lever modules for the after-loop census readers (evidence.collect_after imports them by name). hoist=False: the target-chain hoist not
    installed (big switches it off by name; a card outside its proven set steps it aside instead — test_fallbacks)."""
    fake = {"ef2_stepgraph": {"stats": lambda m: pool_stats}, "ef2_trimul": {}, "ef2_bwd_ckpt": {},
            "ef2_lazy_structure": {"stats": lambda m: ({"deferred": 10, "materialized": 2, "eager": 0} if lazy else None)},
            "ef2_pairbias_attn": {}, "ef2_sampler_graph": {},
            "ef2_esmc_rope": {"handle": lambda h: types.SimpleNamespace(stats={"served": rope_served, "fallback": {}}, modules=[1, 2])},
            "ef2_esmc_overlap": {}, "ef2_loop_prep": {"stats": lambda BD: ({"served": 10, "early": 9, "late": 1, "memo_hits": 2, "anchored": 1, "mismatch": 0, "fallback_anchor": 0, "fallback_signature": 0} if prep is None else prep)},
            "ef2_loop_pppl": {"stats": lambda BD: {"served": 10, "checked": 10, "first_read": 1}}, "ef2_fused_ln": {"stats": lambda: {"served": 96, "fallback": {}, "installed": True, "channel_major": False}},
            "ef2_esmc_hoist": {"stats": lambda m: ({"served": 8, "full": 1, "anchored": 1, "captures": 1, "fallback": {}, "off": None, "live": "82/199"} if hoist else None)},
            "ef2_t16_transition": {"describe": lambda: {"name": "ef2_t16_transition", "state": "on", "reason": None, "reason_text": None, "version": "transition_cute.1.0", "device": "NVIDIA_H100_80GB_HBM3",
                                                        "kernel": {"cubin": "2f600bd17b86", "regs": 168, "smem": 230512, "arch": "sm_90a", "canary": "pass:102:bitwise:6a3438b92fec:max_abs=0.0312"},
                                                        "served": 96, "fallback": {}, "packs": 48}},
            "ef2_kd3_gemmswiglu": {"describe": lambda: {"name": "ef2_kd3_gemmswiglu", "state": "off", "reason": None, "reason_text": None, "version": "w12swiglu.1", "device": "NVIDIA_H100_80GB_HBM3",
                                                        "cc": "sm_90", "tile": None, "served": 0, "fallback": {}, "installed": False}}}
    for name, attrs in fake.items():
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)


def _big_model(cfg, chunk, **kw):
    """An inversion model as big's composition leaves it: the levers big switches off by name (fastkit.BIG_MODE_OFF) not installed."""
    moff = FK.BIG_MODE_OFF
    return _census_model(cfg, chunk, sampler=("ef2_sampler_graph" not in moff), pairbias=("ef2_pairbias_attn" not in moff), esmc=("ef2_esmc_graph" not in moff), **kw)


PPPL_ON = lambda lm: {"captures": 1, "replays": 15, "eager": 3, "entries": 1}          # the pPPL graph's stats as collect_after reads them (fast / exact); big's composition installs none (None)
PPPL_BIG = (lambda lm: None) if "ef2_pppl_graph" in FK.BIG_MODE_OFF else PPPL_ON


def _census_model(cfg, chunk, trimul="fused", plan=None, bf16=True, lazy=True, sampler=True, nosave="on", pairbias=None, esmc=True):
    class Trunk:                                                            # the fork's layout: set_chunk_size writes blocks[i].{tri_mul_out,tri_mul_in,pair_transition}._chunk_size
        def __init__(self, chunk):
            sub = lambda: types.SimpleNamespace(_chunk_size=chunk)
            self.blocks = [types.SimpleNamespace(tri_mul_out=sub(), tri_mul_in=sub(), pair_transition=sub())]
    pairbias = sampler if pairbias is None else pairbias
    m = types.SimpleNamespace(_agk_cfg=cfg, folding_trunk=Trunk(chunk), _esmc=(types.SimpleNamespace(_eeg=types.SimpleNamespace(stats={"captures": 1, "replays": 10, "eager": 0})) if esmc else types.SimpleNamespace()),
                              _agk_orig_model_forward=object(), confidence_head=types.SimpleNamespace(**({"_bc_orig_forward": 1} if bf16 else {})),
                              _bwd_ckpt=(dict(FLOOR_PLAN) if plan is None else plan))
    if trimul:
        m.__dict__["_ef2_trimul_handle"] = types.SimpleNamespace(variant=trimul, stats={"served": 192, "fallback": 8}, cueq_saved={})
    if nosave:
        m.__dict__["_ef2_nosave_handle"] = types.SimpleNamespace(on=(nosave == "on"), word="fast", row=("v4" if nosave == "on" else None), reason=("" if nosave == "on" else "cc_no_gain:sm_80"),
                                                                stats={"served": 76, "inner": 20, "first_pass": 38, "recompute": 38, "refused_calls": 0}, refused={"tx_sm90a": "no_prebuilt:torch211_cu128_sm90"}, canary="pass:0.004", tier_fast="v4")
    if lazy:
        m._els_orig_forward = object()
    if pairbias:
        m._pba_state = types.SimpleNamespace(stats={"served": 100, "computed": 5, "unscoped": 0, "fused_calls": 0, "fallback_grad": 0, "fallback_shape": 0}, variant="hoist")
    if sampler:
        m._sg_state = types.SimpleNamespace(stats={"samples": 5, "captures": 5, "replays": 95, "eager": 0, "mismatch": 0, "capture_failed": 0})
    return m


FLOOR_PLAN = {"policy": "block", "kept_per_pass": 0, "n_blocks": 24, "copies": 1, "kernels": "fused3", "rule": "floor", "floor": True, "reason": "memory_floor"}   # big's plan as ef2_bwd_ckpt.plan(floor=True) records it
FILL_PLAN = {"policy": "ckpt:20", "kept_per_pass": 4, "n_blocks": 24, "copies": 1, "kernels": "fused3", "rule": "budget", "floor": False}                           # fast's / exact's (the budget rule: no reason word)


class _Counter:                                                             # evidence.KernelCounter's summary shape
    def __init__(self, calls):
        self.calls = calls

    def summary(self, trimul, transition, trimul_handles=(), nosave_handles=()):
        hs = [h for h in trimul_handles if h is not None and h.variant == "fused"]; fused = bool(hs)
        ns = sum(h.stats["served"] for h in nosave_handles if h is not None and h.on)
        exp = {"trimul_with_residual_frozen": 0, "transition_refround": 96 if transition else 0, "ef2_trimul_fused": (192 + 76 - ns) if fused else 0}   # 96 block calls x 2 tri-muls; the no-save first passes' 76 went to the provider row
        calls = dict(self.calls, ef2_trimul_fused=sum(h.stats["served"] for h in hs), ef2_trimul_nosave=ns)
        wanted = {k: v for k, v in exp.items() if v}
        return {"kernel_calls": calls, "expected": exp, "applied": bool(wanted) and all(calls[k] == v for k, v in wanted.items()), "block_calls": {"grad_bf16": 96, "other": 4}}


def test_after_loop_census_refuses_a_wrong_big_composition_by_name(monkeypatch):
    _census_kit(monkeypatch, hoist=("ef2_esmc_hoist" not in FK.BIG_MODE_OFF))
    cfg_ok = types.SimpleNamespace(trimul=None, transition="refround_lean", checkpoint="block", einsum_out_dtype="bf16")
    agk = types.SimpleNamespace(describe=lambda c: f"trimul={c.trimul} transition={c.transition} ckpt={c.checkpoint}", _HAS_BMM_OUT_DTYPE=None)
    BD_FAST = types.SimpleNamespace(_D59_GUARD=types.SimpleNamespace(summary=lambda: {"diffs": 0}), _ef2_loop_pppl=object(), _ef2_esmc_overlap=types.SimpleNamespace(stats={"served": 9, "computed": 10, "fallback": {"no_reference_yet": 1}}), _FEATURE_CACHE={})
    BD = types.SimpleNamespace(**{**BD_FAST.__dict__, "_ef2_esmc_overlap": (None if "ef2_esmc_overlap" in FK.BIG_MODE_OFF else BD_FAST._ef2_esmc_overlap)})   # big's cookbook module: the overlap never enabled
    log = "x fast kit: big, checkpoint policy block for 600 tokens, trunk graph pool none"
    engaged = _Counter({"trimul_with_residual_frozen": 0, "transition_refround": 96})
    good = types.SimpleNamespace(inversion_models={"a": _big_model(cfg_ok, 64)}, hf_critic_models=_critics(), _fast_kit_tokens=600, esmc_model=types.SimpleNamespace(esmc=types.SimpleNamespace()))
    rep = EV.collect_after("big", BD, good, engaged, log, agk=agk, n_tokens=600, pppl_stats_fn=PPPL_BIG, n_steps=15)
    assert rep["refusals"] == [], rep["refusals"]
    assert rep["mode_off"] == list(FK.BIG_MODE_OFF) and all(R.lever_line(next(l for l in rep["levers"] if l["name"] == n)) == f"LEVER name={n} state=off served=static fallback=none reason=mode:big" for n in FK.BIG_MODE_OFF)   # one word per lever big switches off by name
    full = types.SimpleNamespace(inversion_models={"a": _census_model(cfg_ok, 64)}, hf_critic_models=_critics(), _fast_kit_tokens=600, esmc_model=types.SimpleNamespace(esmc=types.SimpleNamespace()))
    rep_full = EV.collect_after("big", BD_FAST, full, engaged, log, agk=agk, n_tokens=600, pppl_stats_fn=PPPL_ON, n_steps=15)   # fast's graph levers / overlap engaged on a big arm: not the composition — refused by name, one refusal per lever
    assert sorted(r.split(":")[0] for r in rep_full["refusals"] if "switched off by the big composition" in r) == sorted(set(FK.BIG_MODE_OFF) - {"ef2_esmc_hoist"}) and not rep_full["applied"], rep_full["refusals"]   # (the hoist's fake module reads not-installed in this test: no line to refuse)
    assert rep["applied"] and rep["models"][0]["chunk_size"] == 64 and rep["models"][0]["chunk_attrs"] == {f"folding_trunk.blocks[0].{s}._chunk_size": 64 for s in ("tri_mul_out", "tri_mul_in", "pair_transition")}
    names = [l["name"] for l in rep["levers"]]
    assert names == ["ef2_bwd_ckpt", "trimul", "ef2_trimul_nosave", "cueq_tiles", "chunk", "agk_transition", "ef2_t16_transition", "ef2_kd3_gemmswiglu", "ef2_fused_ln", "ef2_stepgraph", "skip_unused_confidence", "ef2_lazy_structure", "ef2_bf16_confidence", "ef2_pairbias_attn",
                     "ef2_sampler_graph", "ef2_esmc_graph", "ef2_esmc_hoist", "ef2_pppl_graph", "ef2_esmc_rope", "ef2_loop_prep", "ef2_loop_pppl", "ef2_esmc_overlap", "featurisation_cache", "critic_switches"]
    tl = R.lever_line(rep["levers"][1])
    assert R.lever_line(rep["levers"][0]) == "LEVER name=ef2_bwd_ckpt state=on served=static fallback=none policy=block reason=memory_floor kept_per_pass=0/24 kernels=fused3 copies=1"   # big's planned floor, its reason named
    assert R.lever_line(next(l for l in rep["levers"] if l["name"] == "ef2_stepgraph")) == "LEVER name=ef2_stepgraph state=off served=static fallback=none rule=never"   # big carries no trunk graph pool at any size: the composition's word
    assert R.lever_line(next(l for l in rep["levers"] if l["name"] == "ef2_t16_transition")) == "LEVER name=ef2_t16_transition state=on served=96 fallback=none kernel=transition_cute.1.0 device=NVIDIA_H100_80GB_HBM3 cubin=2f600bd17b86 regs=168 smem=230512 arch=sm_90a canary=pass:102:bitwise:6a3438b92fec:max_abs=0.0312"
    assert tl == "LEVER name=trimul state=on served=192 fallback=stock_path:8 kernel=fused" and "trimul=fused" in R.evidence_line(rep, "big") and ("lm_graphs=none" if "ef2_pppl_graph" in FK.BIG_MODE_OFF else "lm_graphs=esmc,pppl") in R.evidence_line(rep, "big")
    assert R.lever_line(next(l for l in rep["levers"] if l["name"] == "ef2_kd3_gemmswiglu")) == "LEVER name=ef2_kd3_gemmswiglu state=off served=static fallback=none reason=t16_serves"   # sm_90: the t16 kernel runs K-D3's forward, the fused W12 + SwiGLU forward is not asked (named)
    assert R.lever_line(rep["levers"][2]) == "LEVER name=ef2_trimul_nosave state=on served=76 fallback=inner:20 word=fast row=v4 first_pass=38 recompute=38 refused=tx_sm90a:no_prebuilt:torch211_cu128_sm90 canary=pass:0.004 tier_fast=v4", R.lever_line(rep["levers"][2])   # tier_fast=: what the provider's tier word resolves to on the stack, next to the row the lever asked by name
    assert R.lever_line(rep["levers"][3]) == "LEVER name=cueq_tiles state=on served=static fallback=none entries=4 holder=hf_critic_models.c0 models=1" and R.lever_line(rep["levers"][4]) == "LEVER name=chunk state=on served=static fallback=none size=64 models=1"
    assert R.lever_line(next(l for l in rep_full["levers"] if l["name"] == "ef2_esmc_overlap")).startswith("LEVER name=ef2_esmc_overlap state=o")   # (the overlap's by-design counter words are exercised on fast in test_fallbacks)
    cl = R.lever_line(rep["levers"][-1])                                                                  # the hero critics on the stock arm's pair, read back from the fork's attributes after their folds
    assert cl.startswith("LEVER name=critic_switches state=on served=static fallback=none chunk_size=none kernel_backend=cuequivariance scope=chunk_size:critics,kernel_backend:critics models=2 readback=") and int(cl.rsplit("readback=", 1)[1]) > 0, cl
    # every declared fact refused by name when wrong: exact-branch config, a trunk pool above the gate, unchunked, no plan, kernels idle, cueq_tiles, no bf16 head, a prep defect
    _census_kit(monkeypatch, pool_stats={"slots": 2, "n_slots": 2, "captures": 2, "recaptures": 0, "replays_fwd": 10, "eager": 0, "active": True, "pool_bytes": 2 ** 30, "disabled_reason": None},
                prep={"served": 10, "mismatch": 1, "fallback_anchor": 1, "fallback_signature": 0})
    bad = types.SimpleNamespace(inversion_models={"a": _census_model(types.SimpleNamespace(trimul=None, transition=None, checkpoint="ckpt:22", einsum_out_dtype="bf16"), None, trimul="cueq_tiles", plan={}, bf16=False)},
                                hf_critic_models=_critics(chunk=64, backend="fused", tiles=False), _fast_kit_tokens=600, esmc_model=None)
    bad.inversion_models["a"]._bwd_ckpt = None
    idle = _Counter({"trimul_with_residual_frozen": 0, "transition_refround": 0})
    rep = EV.collect_after("big", BD, bad, idle, log, agk=agk, n_tokens=600, pppl_stats_fn=PPPL_BIG, n_steps=15)
    words = " | ".join(rep["refusals"])
    for w in ("trunk config is not the kit's", "F9 a: big arm without the fused triangle multiplication", "pair-stack chunk is None",
              "F5 a: no memory plan installed", "installed a trunk graph pool", "bf16 confidence head absent",
              "critic switches (big arm: hero critics chunk_size=none kernel_backend=cuequivariance) not in effect after the loop: hf_critic_models:c0 folding_trunk.blocks.0 _kernel_backend='fused' expected='cuequivariance'"):
        assert w in words, (w, words)
    assert "ef2_loop_prep" not in words and [x["reason"] for x in rep["asides"] if x["name"] == "ef2_loop_prep"] == ["mismatch:1,anchor:1"]   # a prep defect is the lever's declared word: an aside (named, exit 0 on its own), never a refusal
    assert not rep["applied"] and any(f.startswith("F5 ") for f in rep["fallback"]) and any(f.startswith("F11 ") for f in rep["fallback"])
    _census_kit(monkeypatch, hoist=("ef2_esmc_hoist" not in FK.BIG_MODE_OFF))                          # an app holding no hero critic cannot prove the critic switches: refused by name, never assumed
    nocrit = types.SimpleNamespace(inversion_models={"a": _big_model(cfg_ok, 64)}, _fast_kit_tokens=600, esmc_model=types.SimpleNamespace(esmc=types.SimpleNamespace()))
    rep = EV.collect_after("big", BD, nocrit, engaged, log, agk=agk, n_tokens=600, pppl_stats_fn=PPPL_BIG, n_steps=15)
    assert rep["refusals"] == ["cueq_tiles: big arm without the cuEquivariance tile table on any fold model (ef2_trimul cueq_tiles not installed)",
                               "critic switches (big arm: hero critics chunk_size=none kernel_backend=cuequivariance): the loaded app holds no hero critic under hf_critic_models to read them back from"], rep["refusals"]
    # F5, the plan's rule per composition: a FILL plan on big is refused by name (its axis is peak memory), a FLOOR plan on fast is refused by name (fast fills the budget), a forced policy anywhere
    _census_kit(monkeypatch, hoist=("ef2_esmc_hoist" not in FK.BIG_MODE_OFF))
    lm = types.SimpleNamespace(esmc=types.SimpleNamespace())
    cfg_fill = types.SimpleNamespace(trimul=None, transition="refround_lean", checkpoint="ckpt:20", einsum_out_dtype="bf16")
    filled = types.SimpleNamespace(inversion_models={"a": _big_model(cfg_fill, 64, plan=dict(FILL_PLAN))}, hf_critic_models=_critics(), _fast_kit_tokens=600, esmc_model=lm)
    rep = EV.collect_after("big", BD, filled, engaged, "fast kit: big, checkpoint policy ckpt:20 for 600 tokens, trunk graph pool none", agk=agk, n_tokens=600, pppl_stats_fn=PPPL_BIG, n_steps=15)
    assert [r for r in rep["refusals"] if r.startswith("F5 ")] == ["F5 a: big arm's memory plan is policy=ckpt:20 reason=None kept_per_pass=4, not the mode's floor (policy=block reason=memory_floor: every pair block checkpointed)"], rep["refusals"]
    forced = types.SimpleNamespace(inversion_models={"a": _big_model(cfg_ok, 64, plan=dict(FLOOR_PLAN, forced=True))}, hf_critic_models=_critics(), _fast_kit_tokens=600, esmc_model=lm)
    rep = EV.collect_after("big", BD, forced, engaged, log, agk=agk, n_tokens=600, pppl_stats_fn=PPPL_BIG, n_steps=15)
    assert [r for r in rep["refusals"] if r.startswith("F5 ")] == ["F5 a: the checkpoint policy was forced (block), not planned"], rep["refusals"]   # a hand-forced `block` is not the planned floor
    _census_kit(monkeypatch)                                                                            # fast's arms below: every lever module installed (the hoist too)
    floored = types.SimpleNamespace(inversion_models={"a": _census_model(cfg_ok, 64, plan=dict(FLOOR_PLAN))}, hf_critic_models=_critics(), _fast_kit_tokens=600, esmc_model=lm)
    rep = EV.collect_after("fast", BD_FAST, floored, engaged, "fast kit: agk3, checkpoint policy block for 600 tokens, trunk graph pool none above 256 tokens", agk=agk, n_tokens=600, pppl_stats_fn=lambda lm: {"captures": 1, "replays": 15, "eager": 3}, n_steps=15)
    assert [r for r in rep["refusals"] if r.startswith("F5 ")] == ["F5 a: fast arm's memory plan was pinned to the floor (policy=block reason=memory_floor); the mode's plan fills the budget"], rep["refusals"]
    # the trunk graph pool per composition: fast above POOL_MAX_TOKENS carries none and its LEVER line says why (the size rule), at or below it a missing pool is refused (F8); big carries none at any size (rule=never) and an installed pool is refused by name
    fast_big = types.SimpleNamespace(inversion_models={"a": _census_model(cfg_fill, 64, plan=dict(FILL_PLAN))}, hf_critic_models=_critics(), _fast_kit_tokens=600, esmc_model=lm)
    rep = EV.collect_after("fast", BD_FAST, fast_big, engaged, "fast kit: agk3, checkpoint policy ckpt:20 for 600 tokens, trunk graph pool none above 256 tokens", agk=agk, n_tokens=600, pppl_stats_fn=lambda lm: {"captures": 1, "replays": 15, "eager": 3}, n_steps=15)
    assert rep["refusals"] == [], rep["refusals"]
    assert R.lever_line(next(l for l in rep["levers"] if l["name"] == "ef2_stepgraph")) == "LEVER name=ef2_stepgraph state=off served=static fallback=none rule=<=256"
    assert R.lever_line(rep["levers"][0]) == "LEVER name=ef2_bwd_ckpt state=on served=static fallback=none policy=ckpt:20 kept_per_pass=4/24 kernels=fused3 copies=1"   # fast's fill plan: no reason token
    fast_small = types.SimpleNamespace(inversion_models={"a": _census_model(cfg_fill, 64, plan=dict(FILL_PLAN))}, hf_critic_models=_critics(), _fast_kit_tokens=200, esmc_model=lm)
    rep = EV.collect_after("fast", BD_FAST, fast_small, engaged, "fast kit: agk3, checkpoint policy none for 200 tokens, trunk graph pool 2 slots", agk=agk, n_tokens=200, pppl_stats_fn=lambda lm: {"captures": 1, "replays": 15, "eager": 3}, n_steps=15)
    assert any(r.startswith("F8 a: no trunk graph pool installed") for r in rep["refusals"]), rep["refusals"]
    _census_kit(monkeypatch, hoist=("ef2_esmc_hoist" not in FK.BIG_MODE_OFF))
    for toks in (200, 600):                                                                             # big: no pool wanted at any size, none installed = no refusal
        b = types.SimpleNamespace(inversion_models={"a": _big_model(cfg_ok, 64)}, hf_critic_models=_critics(), _fast_kit_tokens=toks, esmc_model=lm)
        rep = EV.collect_after("big", BD, b, engaged, f"fast kit: big, checkpoint policy block for {toks} tokens, trunk graph pool none", agk=agk, n_tokens=toks, pppl_stats_fn=PPPL_BIG, n_steps=15)
        assert rep["refusals"] == [] and R.lever_line(next(l for l in rep["levers"] if l["name"] == "ef2_stepgraph")) == "LEVER name=ef2_stepgraph state=off served=static fallback=none rule=never", (toks, rep["refusals"])
    _census_kit(monkeypatch, pool_stats={"slots": 2, "n_slots": 2, "captures": 2, "recaptures": 0, "replays_fwd": 10, "eager": 0, "active": True, "pool_bytes": 2 ** 30, "disabled_reason": None, "steps": 5})
    pooled = types.SimpleNamespace(inversion_models={"a": _census_model(cfg_fill, 64, plan={"policy": "none", "kept_per_pass": 24, "n_blocks": 24, "copies": 2, "kernels": "fused3", "rule": "budget", "floor": False})}, hf_critic_models=_critics(), _fast_kit_tokens=200, esmc_model=lm)
    rep = EV.collect_after("fast", BD_FAST, pooled, engaged, "fast kit: agk3, checkpoint policy none for 200 tokens, trunk graph pool 2 slots", agk=agk, n_tokens=200, pppl_stats_fn=lambda lm: {"captures": 1, "replays": 15, "eager": 3}, n_steps=15)
    assert rep["refusals"] == [], rep["refusals"]
    assert R.lever_line(next(l for l in rep["levers"] if l["name"] == "ef2_stepgraph")) == "LEVER name=ef2_stepgraph state=on served=10 fallback=none slots=2/2 captures=2 pool_gib=1.0 rule=<=256" and "trunk_graphs=2/2" in R.evidence_line(rep, "fast")
    _census_kit(monkeypatch, hoist=("ef2_esmc_hoist" not in FK.BIG_MODE_OFF), pool_stats={"slots": 2, "n_slots": 2, "captures": 2, "recaptures": 0, "replays_fwd": 10, "eager": 0, "active": True, "pool_bytes": 2 ** 30, "disabled_reason": None, "steps": 5})
    bpooled = types.SimpleNamespace(inversion_models={"a": _big_model(cfg_ok, 64, plan=dict(FLOOR_PLAN, copies=2))}, hf_critic_models=_critics(), _fast_kit_tokens=200, esmc_model=lm)
    rep = EV.collect_after("big", BD, bpooled, engaged, "fast kit: big, checkpoint policy block for 200 tokens, trunk graph pool none", agk=agk, n_tokens=200, pppl_stats_fn=PPPL_BIG, n_steps=15)
    assert any("installed a trunk graph pool" in r for r in rep["refusals"]), rep["refusals"]          # a pool on big is not the composition: refused by name


def test_a_kit_arm_refuses_a_module_installed_for_another_composition(monkeypatch):
    """fastkit.install is per module object and per composition: a second install for a different EF2_FAST_KIT value is refused by name,
    an unknown value is refused by name, and the stock arm's record has the same keys with nothing applied."""
    import pytest
    for name, mod in _fake_kit([]).items():
        monkeypatch.setitem(sys.modules, name, mod)
    BD = _fake_cookbook()
    FK.install(BD, "big")
    with pytest.raises(RuntimeError, match="carries kit 'big', asked 'agk3'"):
        FK.install(BD, "agk3")                                                       # fast's composition on a module carrying big's: refused by name
    with pytest.raises(ValueError, match="is not one of"):
        FK.install(_fake_cookbook(), "off")
    off = FK.record_off()
    assert off["applied"] is False and set(off) == {"name", "applied", "kit", "hooks", "module", "source", "levers_off", "user_switches"} and off["name"] == FK.NAME


def test_out_of_memory_exits_by_name_no_fallback():
    err = 'Traceback ...\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 738.00 MiB. GPU 0 has a total capacity of 79.19 GiB'
    assert SD.is_cuda_oom(err) and not SD.is_cuda_oom("ValueError: bad fasta")
    line = SD.oom_line("fast", 600, 0)
    assert line.startswith("[ef2inv-opt fast] OUT OF MEMORY (CUDA) mode=fast tokens=600 after 0 design fold(s). No fallback applied") and "ef2_stepgraph" not in line   # above the gate nothing of the pool is resident: no hint
    small = SD.oom_line("fast", 200, 3)                                                               # at or below it the pool's resident activations are named with the ablation word that runs without them
    assert "trunk graph pool" in small and "MODEL_OPT_LEVERS_OFF=ef2_stepgraph" in small and "No fallback applied; exit 4." in small and "--mode" not in small
    for toks in (200, 600):                                                                           # big carries no pool at any size: the plain line, no hint (nothing resident to name)
        b = SD.oom_line("big", toks, 1)
        assert b.startswith(f"[ef2inv-opt big] OUT OF MEMORY (CUDA) mode=big tokens={toks} after 1 design fold(s). No fallback applied") and "trunk graph pool" not in b, b


# ---- the upstream-bug-0002 patch (patches.py): applied unconditionally (no switch), named by its source and its function text's sha256
def test_patch_is_unconditional_and_named():
    import inspect
    assert list(inspect.signature(PT.apply).parameters) == []                       # no on/off: the unpatched file cannot run on the pinned stack
    assert PT.NAME == "upstream_bug_0002_transition_addmm_dtype" and len(PT.text_sha256()) == 64
    assert PT.FIX_SOURCE.startswith("STOCK.md 'Stock exception'") and "Stock exception" in open(os.path.join(P.ROOT, "STOCK.md"), encoding="utf-8").read() and "torch.addmm" in PT.FUNCTION_TEXT

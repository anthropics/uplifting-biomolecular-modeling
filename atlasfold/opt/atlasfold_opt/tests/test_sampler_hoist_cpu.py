"""CPU tests of lever sampler_hoist (no GPU, no weights): a small stock DiffusionHead with random (non-degenerate) parameters runs its own sample()
roll-out eagerly and under the installed lever with the same RNG seed — the sampled coordinates are torch.equal (the hoists are bit-identical by
construction), per hoist and all together, with and without relpos_lazy's LazyFeat slots, with one and two denoiser chunks per step; the stand-in
protocol (in-place fold == the stock sum, foreign ops raise by name); the env grammar; the disabled / train guards; registry + installers coverage."""
import os

import pytest
import torch

from atlasfold_opt.hooks import sampler_hoist as SH


# --------------------------------------------------------------------------------------------------------------- a small stock head + batch
def _head(seed=1):
    from atlasfold.model.network.diffusion_head import DiffusionHead
    torch.manual_seed(seed)
    head = DiffusionHead(channel_a=32, channel_atom=16, channel_cond=16, channel_s=24, channel_z=8, num_heads=2, num_blocks=3,
                         num_atom_heads=2, num_atom_blocks=1).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():                                   # stock inits zero several output linears: randomise everything so every statement matters
        for p in head.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.25)
    return head


def _batch(B=2, L=20, seed=2, lazy=False):
    from atlasfold.model.network.rel_pos_encoding import RelativePositionEncoding, AtomRelativePositionEncoding
    from atlasfold.model import model as M
    g = torch.Generator().manual_seed(seed)
    aat = torch.randint(0, 21, (B, L), generator=g)
    seq_mask = torch.ones(B, L, dtype=torch.bool); seq_mask[-1, L - 3:] = False          # padded tokens in the last item
    atom14 = torch.rand(B, L, 14, generator=g) > 0.3; atom14[..., :4] = True; atom14 &= seq_mask[..., None]
    asym = torch.zeros(B, L, dtype=torch.long); asym[:, L // 2:] = 1                     # two chains
    batch = {"res_idx": torch.arange(L).unsqueeze(0).repeat(B, 1), "asym_id": asym, "entity_id": torch.zeros(B, L, dtype=torch.long),
             "sym_id": asym.clone(), "seq_mask": seq_mask, "aatype": torch.nn.functional.one_hot(aat, 21).float(), "atom14_mask": atom14,
             "aatype_int": aat}

    class Fake:                                             # the producers + the stock method, as AtlasFold carries them (relpos_lazy patches the method)
        seq_rel_pos_encoding = RelativePositionEncoding(r_max=32, s_max=2); atom_rel_pos_encoding = AtomRelativePositionEncoding(max_r=4)
        compute_rel_pos_encoding = M.AtlasFold.compute_rel_pos_encoding
    Fake().compute_rel_pos_encoding(batch)
    s = torch.randn(B, L, 24, generator=g); z = torch.randn(B, L, L, 8, generator=g)
    return batch, s, z


def _sample(head, batch, s, z, num_samples=2, steps=3, seed=11, grad=False):
    from atlasfold.model.network.diffusion_head import SamplingConfig
    torch.manual_seed(seed)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        return head.sample(dict(batch), s, z, num_samples=num_samples, config=SamplingConfig(num_steps=steps, chunk_size=5))


def _sites():
    from atlasfold.model.network.diffusion_head import DiffusionHead
    from atlasfold.model.network.diffusion_transformer import PairConditioning, AtomTransformerStack, SingleConditioning
    from atlasfold.model.network.atom_attention import AtomEncoder, AtomAttentionStack
    from atlasfold.model.network.primitives.linear import LinearNoBias
    from atlasfold.model import model as M
    from atlasfold.model.network.attention import Attention
    return [(DiffusionHead, "sample"), (PairConditioning, "forward"), (AtomTransformerStack, "forward"), (SingleConditioning, "forward"),
            (AtomEncoder, "forward"), (AtomAttentionStack, "forward"), (LinearNoBias, "forward"), (M.AtlasFold, "compute_rel_pos_encoding"), (Attention, "forward")]


def _innermost(f):
    seen = 0
    while seen < 16 and (hasattr(f, "__wrapped_stock__") or hasattr(f, "__wrapped__")):
        f = getattr(f, "__wrapped_stock__", None) or getattr(f, "__wrapped__"); seen += 1
    return f


@pytest.fixture(autouse=True)
def stock_sites():
    """Hermetic per test whatever ran before in the session: every patch site starts at its innermost (stock) callable and is put back afterwards."""
    saved = [(cls, attr, cls.__dict__.get(attr, None), getattr(cls, attr)) for cls, attr in _sites()]
    for cls, attr, own, cur in saved:
        inner = _innermost(cur)
        if inner is not cur:
            setattr(cls, attr, inner)
    yield
    for cls, attr, own, cur in saved:
        if own is None:
            if attr in cls.__dict__:
                delattr(cls, attr)
        else:
            setattr(cls, attr, own)


@pytest.fixture
def lever(monkeypatch, stock_sites):
    """install(exact) with the env of the test; restores every class attribute it patched and the instance memos afterwards."""
    installed = []

    def _install(hoists=None, switch=None):
        if hoists is not None:
            monkeypatch.setenv(SH.ENV_HOISTS, hoists)
        else:
            monkeypatch.delenv(SH.ENV_HOISTS, raising=False)
        if switch is not None:
            monkeypatch.setenv(SH.ENV_SWITCH, switch)
        else:
            monkeypatch.delenv(SH.ENV_SWITCH, raising=False)
        ins = SH.install("exact", "atlasfold-opt", {})
        installed.append(ins)
        return ins
    yield _install
    for ins in installed:
        if ins.applied:
            ins.facts["restore"]()
    for cls, attr in _sites()[:6]:
        assert "sampler_hoist" not in getattr(getattr(cls, attr), "__qualname__", ""), f"{cls.__name__}.{attr} left patched by sampler_hoist"


# --------------------------------------------------------------------------------------------------------------- bitwise: all hoists, per hoist, chunks, LazyFeat
def test_all_hoists_bitwise_vs_stock_and_accounting(lever):
    head = _head(); batch, s, z = _batch()
    x0 = _sample(head, batch, s, z)
    ins = lever()
    assert ins.applied and tuple(ins.facts["hoists"]) == SH.HOISTS, ins.facts.get("off")
    x1 = _sample(head, batch, s, z)
    assert torch.equal(x0, x1)
    led = ins.facts["ledger"]
    assert led.served == 1 and led.get("rollouts") == 1
    assert led.get("dit_folds") == 3 and led.get("dit_hits") == 3 * (3 - 1)          # 3 DiT blocks folded at the first call, hit at the 2 later steps
    assert led.get("atom_builds") == 1 + 2 + 2                                        # encoder consts + 2 AtomAttentionStack + 2 AtomTransformerStack builds
    assert led.get("memo_hits") > 0 and led.get("memo_miss") > 0 and led.get("scond_hits") == 2
    assert led.get("relpos_mats") == 0                                                # plain tensors in the batch: nothing to replay
    assert ins.gates[0]().ok
    line = ins.lines[0]()
    assert "sampler_hoist" in line and "hoists=dit_bias+atom_c+atom_pair+atom_win+atom_cond+atom_bias+scond" in line and 'state=on' in line.replace('"', '')
    SH.remove_memos(head)


@pytest.mark.parametrize("hoists", ["dit_bias", "atom_c", "atom_pair", "scond", "atom_c,atom_pair,atom_win", "atom_c,atom_pair,atom_win,atom_cond",
                                    "atom_c,atom_pair,atom_win,atom_bias", "-dit_bias", "-scond,-atom_cond"])
def test_each_hoist_bitwise_vs_stock(lever, hoists):
    head = _head(3); batch, s, z = _batch(B=1, L=16, seed=4)
    x0 = _sample(head, batch, s, z, num_samples=3)
    ins = lever(hoists=hoists)
    assert ins.applied and set(ins.facts["hoists"]) == set(SH.parse_hoists(hoists))
    x1 = _sample(head, batch, s, z, num_samples=3)
    assert torch.equal(x0, x1), hoists
    assert ins.facts["ledger"].served == 1
    SH.remove_memos(head)


def test_two_chunks_per_step_bitwise(lever):
    """num_samples 7 > chunk 5: two denoiser calls per step with N=5 and N=2 share every held tensor (the mask bias does not depend on N)."""
    head = _head(5); batch, s, z = _batch(B=1, L=12, seed=6)
    x0 = _sample(head, batch, s, z, num_samples=7, steps=2)
    ins = lever()
    x1 = _sample(head, batch, s, z, num_samples=7, steps=2)
    assert torch.equal(x0, x1)
    led = ins.facts["ledger"]
    assert led.get("dit_folds") == 3 and led.get("dit_hits") == 3 * (2 * 2 - 1)
    SH.remove_memos(head)


def test_composes_with_relpos_lazy(lever):
    """relpos_lazy's LazyFeat slots: the atom one-hot producer replays once per stack per roll-out inside the atom_pair hoist (2), not per step;
    the lever's PairConditioning wrapper composes over relpos_lazy's (installed first, as in the mode rows)."""
    from atlasfold_opt.hooks import relpos as R
    from atlasfold.model import model as M
    from atlasfold.model.network.primitives import linear as LIN
    from atlasfold.model.network import diffusion_transformer as DT
    head = _head(7)
    rins = R.install("exact", "atlasfold-opt", {})
    ins = None
    try:
        assert rins.applied
        batch, s, z = _batch(B=2, L=20, seed=8)                                          # built through the PATCHED method: LazyFeat slots
        assert isinstance(batch["atom_rel_pos"], R.LazyFeat) and isinstance(batch["seq_rel_pos"], R.LazyFeat)
        x0 = _sample(head, batch, s, z)
        ins = lever()
        assert DT.PairConditioning.forward.__wrapped_stock__.__qualname__.startswith("install.<locals>") or hasattr(DT.PairConditioning.forward.__wrapped_stock__, "__wrapped_stock__")
        x1 = _sample(head, batch, s, z)
        assert torch.equal(x0, x1)
        assert ins.facts["ledger"].get("relpos_mats") == 2 and ins.facts["ledger"].get("dit_folds") == 3
    finally:
        if ins is not None:
            ins.facts["restore"](); ins.applied = False                                   # unwind ours first (it sits over relpos_lazy's PairConditioning wrapper)
        M.AtlasFold.compute_rel_pos_encoding = getattr(M.AtlasFold.compute_rel_pos_encoding, "__wrapped_stock__", M.AtlasFold.compute_rel_pos_encoding)
        if "forward" in LIN.LinearNoBias.__dict__ and getattr(LIN.LinearNoBias.forward, "__wrapped_stock__", None):
            delattr(LIN.LinearNoBias, "forward")
        DT.PairConditioning.forward = getattr(DT.PairConditioning.forward, "__wrapped_stock__", DT.PairConditioning.forward)
        SH.remove_memos(head)


# --------------------------------------------------------------------------------------------------------------- guards
def test_switch_off_runs_stock_counted_disabled(lever):
    head = _head(9); batch, s, z = _batch(B=1, L=8, seed=10)
    x0 = _sample(head, batch, s, z)
    ins = lever(switch="0")
    assert ins.applied
    x1 = _sample(head, batch, s, z)
    assert torch.equal(x0, x1)
    led = ins.facts["ledger"]
    assert led.served == 0 and led.fallbacks == {"disabled": 1}
    assert ins.gates[0]().ok                                                               # disabled is an expected word: the ablation run's gates pass
    assert "AFO_SAMPLER_HOIST=0" in ins.lines[0]()


def test_grad_enabled_rollout_steps_aside(lever):
    head = _head(11); batch, s, z = _batch(B=1, L=8, seed=12)
    x0 = _sample(head, batch, s, z, grad=True)
    ins = lever()
    x1 = _sample(head, batch, s, z, grad=True)
    assert torch.equal(x0, x1)
    led = ins.facts["ledger"]
    assert led.served == 0 and led.fallbacks == {"train": 1} and not ins.gates[0]().ok   # train is NOT expected: a promised lever that never served refuses


def test_source_digest_mismatch_turns_hoist_off(monkeypatch):
    monkeypatch.setitem(SH.SOURCE_SHA256, "SingleConditioning.forward", ("0" * 64,))
    ok, off, seen = SH.source_check(SH.HOISTS)
    assert "scond" not in ok and off["scond"] == "source:SingleConditioning.forward" and len(seen["SingleConditioning.forward"]) == 64
    monkeypatch.setitem(SH.SOURCE_SHA256, "AtomEncoder.forward", ("0" * 64,))
    ok, off, _ = SH.source_check(SH.HOISTS)
    assert off["atom_c"] == "source:AtomEncoder.forward" and off["atom_win"] == "needs:atom_c+atom_pair" and "atom_cond" not in ok and "dit_bias" in ok


# --------------------------------------------------------------------------------------------------------------- the stand-in protocol
def test_block_bias_fold_is_the_stock_sum_in_place_and_by_reference():
    torch.manual_seed(0)
    n, B, H, L = 3, 2, 4, 6
    raw = torch.randn(n, B, H, L, L); ref = raw.clone()
    mask = torch.rand(B, L) > 0.3
    hp = SH.HoistedPairBias(raw)
    assert hp.unsqueeze(2) is hp and len(hp) == n and tuple(hp.shape) == (n, B, 1, H, L, L)
    for i in range(n):
        bias = ((~mask.unsqueeze(1).unsqueeze(-2)).to(torch.float32) * -1e9).unsqueeze(-3)       # attention.py L94-95 on DiffusionModule's mask.unsqueeze(1)
        stock = bias + ref.unsqueeze(2)[i].to(torch.float32)
        got = bias + hp[i].to(torch.float32)                                                     # the statement, on the stand-in
        assert torch.equal(got, stock) and got.data_ptr() == raw.unsqueeze(2)[i].data_ptr()     # folded INTO the slice: no new memory
        again = bias.clone() + hp[i].to(torch.float32)
        assert again is got                                                                      # later steps: the held tensor by reference
    assert hp.folds_inplace == n and hp.hits == n and hp.folds_oop == 0
    hp.release()
    with pytest.raises(SH.HoistError):
        hp[0] + bias                                          # a closed roll-out's stand-in refuses by name
    # a dtype unlike the operand's: the out-of-place stock statement (cast), still the stock value
    raw2 = torch.randn(n, B, H, L, L); hp2 = SH.HoistedPairBias(raw2)
    bias16 = ((~mask.unsqueeze(1).unsqueeze(-2)).to(torch.bfloat16) * -1e9).unsqueeze(-3)
    got = bias16 + hp2[0].to(torch.bfloat16)
    assert torch.equal(got, bias16 + raw2.unsqueeze(2)[0].to(torch.bfloat16)) and hp2.folds_oop == 1
    # torch.add spelling reaches the same fold
    hp3 = SH.HoistedPairBias(ref.clone())
    assert torch.equal(torch.add(bias, hp3[1].to(torch.float32)), bias + ref.unsqueeze(2)[1])


def test_stand_in_refuses_foreign_ops_by_name():
    raw = torch.randn(2, 1, 2, 4, 4); hp = SH.HoistedPairBias(raw)
    with pytest.raises(SH.HoistError):
        hp.unsqueeze(0)
    with pytest.raises(SH.HoistError):
        torch.cat([hp, hp])
    with pytest.raises(SH.HoistError):
        hp["x"]
    b = hp[0]
    with pytest.raises(SH.HoistError):
        torch.mul(torch.ones(1, 1, 1, 1, 4), b)
    with pytest.raises(SH.HoistError):                       # logits-shaped operand (the use_high_precision `attn += pair_bias` form) is not a mask bias: never folded
        torch.randn(1, 3, 2, 4, 4) + b
    with pytest.raises(SH.HoistError):
        b.to("meta")
    first = torch.zeros(1, 1, 1, 1, 4) + b.to(torch.float32)
    with pytest.raises(SH.HoistError):                       # a mask bias of another shape after the fold
        torch.zeros(1, 1, 1, 4, 4) + b
    assert first.shape == (1, 1, 2, 4, 4)


# --------------------------------------------------------------------------------------------------------------- env grammar, registry, installers
def test_parse_hoists_grammar_and_dependencies():
    assert SH.parse_hoists(None) == SH.HOISTS and SH.parse_hoists("all") == SH.HOISTS and SH.parse_hoists("") == SH.HOISTS
    assert SH.parse_hoists("dit_bias") == ("dit_bias",)
    assert SH.parse_hoists("atom_cond") == ()                                     # needs atom_win (needs atom_c + atom_pair): closes to nothing alone
    assert SH.parse_hoists("atom_c,atom_pair,atom_win,atom_cond") == ("atom_c", "atom_pair", "atom_win", "atom_cond")
    assert SH.parse_hoists("-atom_c") == ("dit_bias", "atom_pair", "scond")       # its dependants go with it
    assert SH.parse_hoists("-dit_bias") == tuple(h for h in SH.HOISTS if h != "dit_bias")
    with pytest.raises(ValueError):
        SH.parse_hoists("bogus")


def test_registry_modes_installers_cover_the_lever():
    from atlasfold_opt import modes, registry
    from atlasfold_opt.hooks import installers
    assert "sampler_hoist" in modes.MODES[modes.EXACT] and "sampler_hoist" in modes.MODES[modes.FAST]      # R3: exact ⊂ fast
    row = registry.LEVERS["sampler_hoist"]
    assert row["cls"] == "exact" and row["module"] == "atlasfold_opt.hooks.sampler_hoist" and tuple(row["expected"]) == SH.EXPECTED == ("disabled", "atom_bias:atom_sdpa") and row["what"]
    assert installers()["sampler_hoist"] is SH.install
    ex, fa = modes.MODES[modes.EXACT], modes.MODES[modes.FAST]
    assert ex.index("relpos_lazy") < ex.index("sampler_hoist") < ex.index("lever_report")
    for later in ("atom_sdpa", "atom_tf32", "lever_report"):                         # fast: BEFORE the levers that wrap AtomAttentionStack.forward / re-state the atom attention
        if later in fa:                                                              # (atom_tf32's TF32 window must enclose the hoisted prologue; see the module docstring)
            assert fa.index("sampler_hoist") < fa.index(later), later
    for earlier in ("sampler_hostsync",):
        if earlier in fa:
            assert fa.index(earlier) < fa.index("sampler_hoist"), earlier


def test_attention_wrapper_words():
    """dit_bias composes under dit_apb's and atom_sdpa's Attention.forward wrappers; atom_bias only under dit_apb's (atom_sdpa re-states the atom
    statement) — anything else is off by the wrapper's lever name."""
    from atlasfold.model.network import attention as ATT
    from atlasfold_opt.hooks import rebind
    stock = ATT.Attention.forward
    assert SH._attention_forward_known(ATT.Attention) is None

    def w1(self, *a, **k): return stock(self, *a, **k)
    w1.__qualname__ = "Attention.forward[atlasfold_opt:atom_sdpa]"
    rebind(ATT.Attention, "forward", w1, stock)
    try:
        assert SH._attention_forward_known(ATT.Attention, SH.DIT_BIAS_WRAPPERS) is None
        assert SH._attention_forward_known(ATT.Attention, SH.ATOM_BIAS_WRAPPERS) == "atom_sdpa"

        def w2(self, *a, **k): return w1(self, *a, **k)
        w2.__qualname__ = "Attention.forward[atlasfold_opt:someone_else]"
        rebind(ATT.Attention, "forward", w2, w1)
        assert SH._attention_forward_known(ATT.Attention) == "someone_else"
    finally:
        ATT.Attention.forward = stock


def test_denoiser_graph_call_key_pins_the_stand_in_by_reference():
    """denoiser_graph keys a call by the addresses of its tensors and the id of non-tensor objects it then holds: the hoisted pair bias is such an
    object — same object every step of a roll-out -> same key -> replay; held in refs for the graph's lifetime."""
    from atlasfold_opt.hooks import denoiser_graph as DG
    raw = torch.randn(2, 1, 2, 4, 4); hp = SH.HoistedPairBias(raw)
    batch = {"seq_mask": torch.ones(1, 4, dtype=torch.bool)}
    r, sc = torch.randn(1, 2, 4, 14, 3), torch.randn(1, 2, 4, 8)
    refs1, refs2 = [], []
    k1 = DG.call_key(batch, r, sc, hp, refs1); k2 = DG.call_key(batch, r.clone(), sc.clone(), hp, refs2)
    assert any(o is hp for o in refs1)
    assert k1[3] == k2[3]                                                          # the pair-bias part of the key: the same object


def test_held_tensors_are_freed_by_refcount_when_the_rollout_returns(lever):
    """No reference cycle may keep the pair-bias storage (or any held tensor) alive past DiffusionHead.sample: with the cyclic GC disabled, a
    weak reference to PairConditioning's output is dead as soon as sample() returns (the leak this guards against showed as +1.3 GB peak at
    1,280 tokens: the confidence head ran with the previous roll-out's pair bias still allocated)."""
    import gc, weakref
    from atlasfold.model.network.diffusion_transformer import PairConditioning
    head = _head(13); batch, s, z = _batch(B=1, L=12, seed=14)
    ins = lever()
    inner = PairConditioning.forward
    seen = []

    def spy(self, b, zz):
        out = inner(self, b, zz)
        seen.append(weakref.ref(out.raw6 if isinstance(out, SH.HoistedPairBias) else out))
        return out
    PairConditioning.forward = spy
    gc.disable()
    try:
        x = _sample(head, batch, s, z)
        assert len(seen) == 1 and seen[0]() is None, "pair-bias storage still referenced after the roll-out"
    finally:
        gc.enable()
        PairConditioning.forward = inner
        SH.remove_memos(head)
    assert x.shape[-2:] == (14, 3)


def test_step_aside_words_expected_structural_miss_refused(lever):
    """atom_bias stepping aside under atom_sdpa is a declared composition (expected word, gate ok); a hoist off for an unforeseen reason is an
    unexpected fallback the gate refuses by name."""
    from opt_core.counters import Ledger
    head = _head(31); batch, s, z = _batch(B=1, L=8, seed=32)
    ins = lever()
    led = ins.facts["ledger"]
    ro = SH.Roll(head, SH.HOISTS, led)
    ro._off("atom_bias", "atom_sdpa"); ro.pair = SH.HoistedPairBias(torch.zeros(3, 1, 2, 4, 4)); ro.close()
    assert led.fallbacks == {"atom_bias:atom_sdpa": 1} and ins.gates[0]().ok
    ro2 = SH.Roll(head, SH.HOISTS, led); ro2._off("dit_bias", "dit_blocks"); ro2.pair = SH.HoistedPairBias(torch.zeros(3, 1, 2, 4, 4)); ro2.close()
    assert not ins.gates[0]().ok
    SH.remove_memos(head)

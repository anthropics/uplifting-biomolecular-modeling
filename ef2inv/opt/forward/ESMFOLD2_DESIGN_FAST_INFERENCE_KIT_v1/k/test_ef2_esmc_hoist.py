"""Unit tests for ef2_esmc_hoist (the Biohub transformers fork; random-init ESMC stacks, no checkpoint; the layout tests need no GPU).

    pytest -q k/test_ef2_esmc_hoist.py        or, without pytest:   python k/run_tests_nopytest.py test_ef2_esmc_hoist

The class under test is EXACT: on a design-like trajectory (target fixed, binder re-drawn every step) every one of the n_layers+1 hidden
states the hoisted forward returns must be tensor-equal (torch.equal) to the trunk's own forward, eager and CUDA-graph replayed, alone and
stacked over ef2_esmc_graph (the design kit's order), with and without the loop's bf16 autocast; a changed target is a FULL call; the
by-design inner calls (grad, form, layout) and the self-switch-off (anchor, structure) are counted by name and return the inner result;
disable hands back the forward that was under the lever. Per card: the exact claim is proven on `PROVEN_CC` (sm_90) — on any other card the lever
steps aside by name (registered, nothing installed) and the bitwise tests skip by name.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import pytest
except ImportError:
    pytest = None
import numpy as np
import torch
import ef2_esmc_hoist as eh

def skipif(cond, reason=""):
    def deco(f):
        f._skip = bool(cond) or getattr(f, "_skip", False)
        return pytest.mark.skipif(cond, reason=reason)(f) if pytest else f
    return deco


CC = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
cuda = skipif(CC is None, reason="CUDA required")
# the exact claim is per card: proven (and shipped) on eh.PROVEN_CC; any other card steps aside by name — test_an_unproven_compute_capability_… covers it there
proven = skipif(CC is None or not eh.cc_route(CC)[0], reason=f"ef2_esmc_hoist steps aside on this card ({'no CUDA' if CC is None else eh.cc_word(CC)}): its reduced pass is proven bitwise on {sorted(eh.PROVEN_CC)} only")
BOS, PAD, EOS = 0, 1, 2


def _lm_row(chains):
    ids = [BOS]
    for i, ch in enumerate(chains):
        ids += list(ch)
        ids += [EOS, BOS] if i < len(chains) - 1 else [EOS]
    return ids


def _lm_inputs(rows, device="cpu", pad_to=None):
    L = max(len(r) for r in rows) if pad_to is None else pad_to
    ids = np.full((len(rows), L), PAD, dtype=np.int64)
    for b, r in enumerate(rows):
        ids[b, : len(r)] = r
    seq = np.cumsum(ids == BOS, axis=1) - 1
    seq[ids == PAD] = -1
    return torch.from_numpy(ids).to(device), torch.from_numpy(seq).to(device)


# ------------------------------------------------------------------------------------------------------------------ host-side layout
def test_live_block_layouts():
    rng = np.random.default_rng(0)
    tgt = rng.integers(4, 24, 30); b1 = rng.integers(4, 24, 12); b2 = rng.integers(4, 24, 12)
    ids1, seq1 = (t.numpy() for t in _lm_inputs([_lm_row([tgt, b1])]))
    ids2, seq2 = (t.numpy() for t in _lm_inputs([_lm_row([tgt, b2])]))
    b0, k1 = eh.live_block(ids1, seq1); b0b, k2 = eh.live_block(ids2, seq2)
    assert b0 == 32 == b0b and k1 == k2, (b0, b0b)                       # [BOS] 30 [EOS] -> the binder's BOS is column 32; the key ignores the live rows' ids
    ids3, seq3 = (t.numpy() for t in _lm_inputs([_lm_row([rng.integers(4, 24, 30), b1])]))
    assert eh.live_block(ids3, seq3)[0] == 32 and eh.live_block(ids3, seq3)[1] != k1     # another target: another key
    ids4, seq4 = (t.numpy() for t in _lm_inputs([_lm_row([tgt[:10], tgt[10:], b1])]))   # a two-chain target: the live block is the last chain
    assert eh.live_block(ids4, seq4)[0] == 1 + 10 + 2 + 20 + 1
    assert eh.live_block(*(t.numpy() for t in _lm_inputs([_lm_row([b1])]))) == (None, "layout")                     # one chain: nothing frozen
    assert eh.live_block(*(t.numpy() for t in _lm_inputs([_lm_row([tgt, b1]), _lm_row([tgt, b1[:8]])]))) == (None, "layout")   # padding
    assert eh.live_block(*(t.numpy() for t in _lm_inputs([_lm_row([tgt, b1]), _lm_row([tgt[:28], rng.integers(4, 24, 14)])]))) == (None, "layout")  # b0 differs
    both = _lm_inputs([_lm_row([tgt, b1]), _lm_row([tgt, b2])])
    assert eh.live_block(*(t.numpy() for t in both))[0] == 32                                                          # a batch of two designs of one target


def test_structure_problem_names_what_differs():
    class Fake(torch.nn.Module):
        pass
    assert eh.structure_problem(Fake()).startswith("type:")


def test_cc_route_names_the_card_and_the_proven_table():
    assert eh.cc_route((9, 0)) == (True, "sm_90") and eh.cc_route((8, 0)) == (False, "sm_80") and eh.cc_route((12, 0)) == (False, "sm_120")
    assert (9, 0) in eh.PROVEN_CC and (8, 0) not in eh.PROVEN_CC        # sm_80: TE's cuBLASLt GEMMs are not row-count invariant — the lever steps aside there


# ------------------------------------------------------------------------------------------------------------------ GPU: the exact claim
def _stack(d_model=1280, n_heads=20, n_layers=2, seed=1):
    import transformers.models.esmc.modeling_esmc as ME
    cfg = ME.ESMCConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, vocab_size=64)
    torch.manual_seed(seed)
    return ME.ESMCModel(cfg).to("cuda", torch.bfloat16).eval().requires_grad_(False)


def _trajectory(T=100, Lb=80, steps=5, seed=0, B=1, new_target_at=None):
    """LM inputs of a design: the target fixed, the binder re-drawn every step (a new target at `new_target_at`), batch B."""
    rng = np.random.default_rng(seed)
    tgt = rng.integers(4, 24, T)
    out = []
    for s in range(steps):
        if new_target_at is not None and s == new_target_at:
            tgt = rng.integers(4, 24, T)
        out.append(_lm_inputs([_lm_row([tgt, rng.integers(4, 24, Lb)]) for _ in range(B)], device="cuda"))
    return out


def _run(esmc, traj, autocast=True):
    outs = []
    for ids, seq in traj:
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=autocast, dtype=torch.bfloat16):
            hs = esmc(input_ids=ids, sequence_id=seq, output_hidden_states=True).hidden_states
            outs.append(hs.clone())
    return outs


def _equal(ref, got, label):
    assert len(ref) == len(got), label
    for s, (a, b) in enumerate(zip(ref, got)):
        assert a.shape == b.shape and a.dtype == b.dtype, (label, s, a.shape, b.shape, a.dtype, b.dtype)
        if not torch.equal(a, b):
            d = (a.float() - b.float()).abs()
            bad = [i for i in range(a.shape[0]) if not torch.equal(a[i], b[i])]
            raise AssertionError(f"{label}: step {s} hidden states differ at layers {bad}, max|d|={d.max().item():.3e}")


@cuda
@proven
def test_design_trajectory_bitwise_eager_graphed_and_over_esmc_graph():
    import ef2_esmc_graph as eeg
    esmc = _stack()
    traj = _trajectory(steps=6, new_target_at=4)                          # steps 0-3 one target, 4-5 another: FULL at 0 and 4
    ref = _run(esmc, traj)
    assert all(torch.equal(a, b) for a, b in zip(ref, _run(esmc, traj))), "the stock forward is not run-to-run deterministic here — comparison void"
    for graphs in (False, True):
        h = eh.enable(esmc, graphs=graphs)
        try:
            got = _run(esmc, traj)
            st = eh.stats(esmc)
        finally:
            eh.disable(esmc)
        assert "forward" not in vars(esmc) and eh.handle(esmc) is None
        _equal(ref, got, f"hoist graphs={graphs}")
        assert st["full"] == 2 and st["served"] == 4 and st["anchored"] == 1 and not st["fallback"] and st["off"] is None, st
        assert st["captures"] == (1 if graphs else 0) and st["replays"] == (4 if graphs else 0) and st["live"] == "82/184", st
    holder = torch.nn.Module(); holder._esmc = esmc                        # the design kit's order: the ESMC graph first, the hoist over it
    g = eeg.enable(holder); h = eh.enable(holder)
    try:
        assert esmc.forward is h and h.inner is g
        got = _run(esmc, traj)
        st, gs = eh.stats(holder), dict(g.stats)
    finally:
        eh.disable(holder)
        assert esmc.forward is g                                           # disable hands back the graph wrapper
        eeg.disable(holder)
    _equal(ref, got, "hoist over ef2_esmc_graph")
    assert st["full"] == 2 and st["served"] == 4 and gs["eager"] == 0 and gs["replays"] == 2 + 1, (st, gs)   # the graph serves the FULL calls and the one anchor


@cuda
@proven
def test_no_autocast_and_a_batch_of_two_designs_bitwise():
    esmc = _stack(d_model=640, n_heads=10, n_layers=3, seed=2)
    for B, autocast in ((1, False), (2, True)):
        traj = _trajectory(T=40, Lb=24, steps=4, seed=3, B=B)
        ref = _run(esmc, traj, autocast=autocast)
        eh.enable(esmc)
        try:
            got = _run(esmc, traj, autocast=autocast); st = eh.stats(esmc)
        finally:
            eh.disable(esmc)
        _equal(ref, got, f"B={B} autocast={autocast}")
        assert st["served"] == 3 and st["full"] == 1 and not st["fallback"], st


@cuda
@proven
def test_by_design_inner_calls_and_switch_off_by_name():
    esmc = _stack(d_model=640, n_heads=10, n_layers=2, seed=4)
    traj = _trajectory(T=40, Lb=24, steps=3, seed=5)
    ref = _run(esmc, traj)
    h = eh.enable(esmc)
    try:
        ids, seq = traj[0]
        esmc(input_ids=ids, sequence_id=seq, output_hidden_states=True)   # grad mode on (the default): the inner forward, counted
        with torch.inference_mode():
            esmc(input_ids=ids, sequence_id=seq)                           # another argument set: form
            pid, pseq = _lm_inputs([_lm_row([[5] * 40, [6] * 24]), _lm_row([[5] * 40, [6] * 20])], device="cuda")
            esmc(input_ids=pid, sequence_id=pseq, output_hidden_states=True)   # padding: layout
        st = eh.stats(esmc)
        assert st["fallback"] == {"grad": 1, "form": 1, "layout": 1} and st["served"] == 0 and st["full"] == 0, st
        # the anchor: a reduced forward that disagrees switches the lever off by name and the caller gets the inner result
        orig = h._reduced

        def wrong(ent):
            orig(ent); ent.hs[0, :, ent.b0:] += 1
        h._reduced = wrong
        got = _run(esmc, traj)
        st = eh.stats(esmc)
    finally:
        eh.disable(esmc)
    _equal(ref, got, "anchor failure returns the inner result")
    assert st["off"] == "anchor" and st["fallback"].get("anchor") == 2 and st["served"] == 0 and st["full"] == 1, st   # call 1 FULL, call 2 the failed anchor, call 3 inner by name
    orig_sp = eh.structure_problem                                         # a trunk form this file does not compute (here: pretended): never hoisted, named
    eh.structure_problem = lambda esmc: "sae"
    try:
        h2 = eh.enable(esmc)
        got = _run(esmc, traj[:1])
        assert h2.off == "structure:sae" and eh.stats(esmc)["fallback"] == {"structure:sae": 1} and torch.equal(got[0], ref[0]), eh.stats(esmc)
    finally:
        eh.disable(esmc); eh.structure_problem = orig_sp


@cuda
def test_an_unproven_compute_capability_steps_aside_by_name_and_installs_nothing():
    esmc = _stack(d_model=640, n_heads=10, n_layers=2, seed=8)
    traj = _trajectory(T=40, Lb=24, steps=3, seed=9)
    ref = _run(esmc, traj)
    h = eh.enable(esmc, cc=(None if not eh.cc_route(CC)[0] else (8, 0)))   # an unproven card (this one, probed, if it is one; else as on an A100): registered, NOT installed
    try:
        sm = eh.cc_word(CC) if not eh.cc_route(CC)[0] else "sm_80"
        assert h.stepped_aside == f"cc_unproven:{sm}" and not h.installed and "forward" not in vars(esmc) and eh.handle(esmc) is h
        got = _run(esmc, traj)                                             # the trunk's own forward ran: nothing of the lever
        st = eh.stats(esmc)
        assert st["stepped_aside"] == f"cc_unproven:{sm}" and st["cc"] == sm and st["served"] == 0 and st["full"] == 0 and not st["fallback"] and st["installed"] is False, st
        assert eh.enable(esmc) is h                                        # idempotent: the same (aside) handle
    finally:
        eh.disable(esmc)
    assert eh.handle(esmc) is None and "forward" not in vars(esmc)
    _equal(ref, got, "stepped aside = the stock forward")
    h = eh.enable(esmc, cc=(9, 0))                                         # a listed capability installs (whatever this box is); the anchor guards the unexpected
    try:
        assert h.installed and h.stepped_aside is None and vars(esmc).get("forward") is h and eh.stats(esmc)["cc"] == "sm_90"
    finally:
        eh.disable(esmc)


@cuda
@proven
def test_enable_idempotent_spellings_release_and_disable():
    esmc = _stack(d_model=640, n_heads=10, n_layers=2, seed=6)
    lm = torch.nn.Module(); lm.esmc = esmc
    fold = torch.nn.Module(); fold._esmc = esmc
    h = eh.enable(esmc)
    assert eh.enable(lm) is h and eh.enable(fold) is h and eh.handle(fold) is h
    traj = _trajectory(T=40, Lb=24, steps=3, seed=7)
    ref = None
    try:
        got = _run(esmc, traj)
        eh.release(fold)                                                   # remembered states dropped: the next call is FULL again
        got2 = _run(esmc, traj[2:])
        st = eh.stats(lm)
    finally:
        eh.disable(lm)
    assert "forward" not in vars(esmc)
    ref = _run(esmc, traj)
    _equal(ref, got, "before release"); _equal(ref[2:], got2, "after release")
    assert st["full"] == 2 and st["served"] == 2, st


if __name__ == "__main__":
    n_fail = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if getattr(fn, "_skip", False):
                print("SKIP", name); continue
            try:
                fn(); print("PASS", name, flush=True)
            except Exception as e:  # noqa: BLE001
                n_fail += 1; print("FAIL", name, type(e).__name__, str(e)[:300], flush=True)
    print("RESULT", "OK" if not n_fail else f"{n_fail} failed")
    sys.exit(1 if n_fail else 0)

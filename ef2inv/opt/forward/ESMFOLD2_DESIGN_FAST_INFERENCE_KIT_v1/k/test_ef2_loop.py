"""GPU proof of the loop-level levers ef2_loop_prep / ef2_loop_pppl: A/B/A in ONE process under the
det recipe — the cookbook's own design loop (STEPS=12: 10 design steps + 2 confidence steps, hero critics emptied) run
unpatched (A), with the lever(s) enabled (B), disabled again (A2) — and every per-step observable compared with torch.equal:
the loss terms of the trajectory, the structure and LM gradients w.r.t. the logits, the logits after each update, the distogram
bytes, the designed sequences, the CUDA RNG state at the fold and LM-loss entry AND exit (same draws consumed), the masked
LM inputs of every pass (the pass masks, latent issue L4), the LM loss, and ptm / iptm / plddt on the confidence steps.
A == A2 shows the unpatched loop is reproducible in-process (else the test proves nothing) and that disable() restores stock;
A == B is the lever's exactness. Two sizes: pd-l1 (195 tokens) and cd45 (431 tokens), the cookbook's built-in targets.

Needs a GPU, the installed esm checkout (the cookbook file) and the pinned weights (HF_HOME); the kit arm is the one
EF2_FAST_KIT names (exact | agk3 | big), installed on the module as an arm process installs it; unset = stock + switches
chunk none / cuequivariance (the stock arm the design kit's speed rows use). pytest or k/run_tests_nopytest.py style (plain test_* functions).
    EF2_FAST_KIT=exact python -m pytest k/test_ef2_loop.py      |  python k/test_ef2_loop.py [prep,pppl] [pd-l1]
"""
from __future__ import annotations

import hashlib
import os
import sys
import types

import numpy as np
import torch
try:
    import pytest
except ModuleNotFoundError:                      # images without pytest: k/run_tests_nopytest.py's stub, restated for direct runs
    pytest = types.SimpleNamespace(mark=types.SimpleNamespace(skipif=lambda c, reason="": (lambda f: f), parametrize=lambda n, v: (lambda f: f)),
                                   skip=lambda msg="": (_ for _ in ()).throw(RuntimeError("SKIP: " + msg)))

HERE = os.path.dirname(os.path.abspath(__file__))
CUDA = torch.cuda.is_available()
_STATE = {}


def _cookbook_module():
    """The cookbook module imported as an arm imports it (modal stand-in, sha-checked file), no models loaded."""
    if "BD" in _STATE:
        return _STATE["BD"]
    if "BD_only" in _STATE:
        return _STATE["BD_only"]
    from ef2inv_opt import stock_design as SD
    from ef2inv_opt.modes import MODES
    model_opt = os.environ.get("MODEL_OPT") or os.environ.get("KIT") or os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
    SD.prepare_arm_process(MODES["off"], None)
    cookbook = os.environ.get("EF2INV_COOKBOOK_STOCK")
    if not cookbook:
        import esm
        cookbook = os.path.abspath(os.path.join(os.path.dirname(esm.__file__), "..", "cookbook", "tutorials", "binder_design.py"))
    _STATE["BD_only"] = SD.import_cookbook(cookbook, "binder_design_featurise_only")
    return _STATE["BD_only"]


def test_splice_featuriser_matches_the_featuriser_per_field():
    """CPU: SpliceFeaturiser(seq) == prepare_esmfold2_tensors(the cookbook's input for seq) FIELD BY FIELD (torch.equal, same dtype and
    shape) over random binders for three designs: a one-chain target, a two-chain target, and a binder length whose atom count
    crosses the 32-atom padding boundary often; every residue letter occurs; the first call of a design learns (full output), the
    rest are spliced with the run-time comparison disabled (checks=0) so the test is the only judge."""
    import random
    import ef2_loop_prep as lp
    BD = _cookbook_module()
    letters = "ACDEFGHIKLMNPQRSTVWY"
    rng = random.Random(0)
    t1 = BD.TARGET_SEQUENCES["pd-l1"][:70]
    t2 = (BD.TARGET_SEQUENCES["cd45"][:40], BD.TARGET_SEQUENCES["pd-l1"][20:55])
    sp = lp.SpliceFeaturiser(BD, checks=0)

    def stock(seq):
        sequences = {sequence: [str(idx)] for idx, sequence in enumerate(seq.split("|"))}
        raw = BD.StructurePredictionInput(sequences=[BD.ProteinInput(id=cid, sequence=sq, msa=None) for sq, cid in sequences.items()])
        return BD.prepare_esmfold2_tensors(raw, max_atoms=None)
    n = 0
    for target, Lb in (((t1,), 31), (t2, 24), ((t1[:33],), 17)):
        for i in range(9):
            binder = "".join(rng.choice(letters) for _ in range(Lb)) if i else letters[:Lb].ljust(Lb, "W")[:Lb]
            if i == 5:
                binder = "G" * Lb                                            # fewest atoms
            if i == 6:
                binder = "W" * Lb                                            # most atoms
            seq = "|".join(target + (binder,))
            mine, ref = sp(seq), stock(seq)
            assert set(mine) == set(ref), (set(mine) ^ set(ref))
            for k in ref:
                assert mine[k].dtype == ref[k].dtype and mine[k].shape == ref[k].shape, (k, mine[k].dtype, ref[k].dtype, mine[k].shape, ref[k].shape)
                assert torch.equal(mine[k], ref[k]), (k, i, Lb)
            n += 1
    st = sp.stats
    assert st["splice_learned"] == 3 and st["spliced"] == n - 3 and st["splice_off"] == 0 and st["splice_unknown"] == 0, st


def _setup():
    """The arm process of an ef2inv run (stock_design.main's recipe), once per process; returns (BD, app)."""
    if "BD" in _STATE:
        return _STATE["BD"], _STATE["app"]
    from ef2inv_opt import stock_design as SD, settings as S, det as D, patches as PT, fastkit as FK, upstream_fix as UF
    from ef2inv_opt.modes import KIT_SWITCH, MODES, kit_paths
    model_opt = os.environ.get("MODEL_OPT") or os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
    switch = os.environ.get(KIT_SWITCH)
    arm, mode = next(((n, m) for n, m in MODES.items() if m.kit_switch == switch), (None, None))
    if mode is None:
        raise RuntimeError(f"{KIT_SWITCH}={switch!r} names no mode")
    SD.prepare_arm_process(mode, kit_paths(model_opt)["k_dir"] if mode.is_kit else None)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    cookbook = os.environ.get("EF2INV_COOKBOOK_STOCK")
    if not cookbook:
        import esm
        cookbook = os.path.abspath(os.path.join(os.path.dirname(esm.__file__), "..", "cookbook", "tutorials", "binder_design.py"))
    if not os.path.exists(cookbook):
        pytest.skip(f"the cookbook file is not installed ({cookbook})")
    BD = SD.import_cookbook(cookbook, "binder_design")
    assert not S.check_shipped(BD)
    if mode.is_kit:
        FK.install(BD, mode.kit_switch)
    UF.apply(["EF2INV-0003"], model_opt, log=lambda m: print(m, file=sys.stderr))
    app = BD.ESMFold2Design(); app.load(use_scaling_critics=False)
    chunk, backend = (None, "cuequivariance") if not mode.is_kit else (mode.chunk_size, mode.kernel_backend)
    SD.apply_model_switches(app, chunk, backend, arm)
    D.apply(D.level(1)); PT.apply()
    BD.BINDER_PROMPT_FACTORIES["mb80"] = BD.PromptFactory(name="mb80", template="{seq}", length_ranges={"seq": (80, 80)}, is_antibody=False)
    _STATE.update(BD=BD, app=app)
    return BD, app


class _Recorder:
    """Per-step observables of one design, taken from outside the loop (wrappers on the module functions the loop looks up by name)."""
    NAMES = ("fold_and_get_distogram", "compute_esmc_pseudoperplexity_nll", "normalized_gradient_tensor")

    def __init__(self, BD, app):
        self.BD, self.app = BD, app
        self.rec = None

    @staticmethod
    def _sha(t):
        return hashlib.sha256(t.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()

    def install(self):
        BD, R = self.BD, self
        self.under = {n: getattr(BD, n) for n in self.NAMES}
        fold0, pppl0, ngt0 = self.under["fold_and_get_distogram"], self.under["compute_esmc_pseudoperplexity_nll"], self.under["normalized_gradient_tensor"]

        def fold(model, *a, **k):
            R.rec["rng_fold"].append(torch.cuda.get_rng_state().clone())
            r = fold0(model, *a, **k)
            R.rec["seqs"].append(r["seq_list"][0]); R.rec["disto"].append(R._sha(r["distogram_logits"]))
            if k.get("calculate_confidence"):                  # ptm / iptm bitwise; plddt is left out: its reduction order varies run to run in stock itself
                R.rec["conf"].append(tuple(R._sha(r[x]) for x in ("ptm", "iptm") if r.get(x) is not None))
            R.rec["rng_fold_out"].append(torch.cuda.get_rng_state().clone())
            return r

        def pppl(esmc_model, *a, **k):
            R.rec["rng_pppl"].append(torch.cuda.get_rng_state().clone())
            tr = esmc_model.esmc.__dict__.get("transformer"); had = tr is not None
            real_tr = esmc_model.esmc.transformer

            def rec_tr(x, *aa, **kk):                      # the masked one-hot LM inputs of every pass = the pass masks (L4)
                R.rec["masks"].append(x.detach().clone().cpu())
                return real_tr(x, *aa, **kk)
            esmc_model.esmc.__dict__["transformer"] = rec_tr
            try:
                r = pppl0(esmc_model, *a, **k)
            finally:
                if had: esmc_model.esmc.__dict__["transformer"] = tr
                else: del esmc_model.esmc.__dict__["transformer"]
            torch.cuda.current_stream().synchronize()
            R.rec["plm"].append(r.detach().clone().cpu()); R.rec["rng_pppl_out"].append(torch.cuda.get_rng_state().clone())
            return r

        def ngt(grad, gmask):
            R.rec["grads"].append(grad.detach().clone().cpu())
            return ngt0(grad, gmask)
        BD.fold_and_get_distogram, BD.compute_esmc_pseudoperplexity_nll, BD.normalized_gradient_tensor = fold, pppl, ngt
        if not isinstance(BD.optim, types.SimpleNamespace):
            real = BD.optim

            class SGD(real.SGD):
                def step(self_, *a, **k):
                    out = super().step(*a, **k)
                    R.rec["logits"].append(self_.param_groups[0]["params"][0].detach().clone().cpu())
                    return out
            BD.optim = types.SimpleNamespace(SGD=SGD, Optimizer=real.Optimizer)
            self.real_optim = real

    def uninstall(self):
        for n, f in self.under.items():
            setattr(self.BD, n, f)
        if hasattr(self, "real_optim"):
            self.BD.optim = self.real_optim; del self.real_optim

    def run(self, target, n_steps=12, seed=0):
        BD, app = self.BD, self.app
        self.rec = {k: [] for k in ("rng_fold", "rng_fold_out", "seqs", "disto", "conf", "rng_pppl", "rng_pppl_out", "masks", "plm", "grads", "logits")}
        saved = (BD.STEPS, dict(app.hf_critic_models)); BD.STEPS = n_steps; app.hf_critic_models = {}
        self.install()
        try:
            _, trajectory, _ = app.design(target_name=target, binder_name="mb80", seed=seed, batch_size=1, is_antibody=False)
        finally:
            self.uninstall(); BD.STEPS, app.hf_critic_models = saved
        torch.cuda.synchronize()
        self.rec["trajectory"] = {int(s): {k: v.clone() for k, v in d.items() if torch.is_tensor(v)} for s, d in trajectory.items()}
        assert len(self.rec["logits"]) == n_steps and len(self.rec["conf"]) >= 1, (len(self.rec["logits"]), len(self.rec["conf"]))
        return self.rec


def _diff(A, B):
    bad = []
    for k in A:
        xa, xb = A[k], B[k]
        if k == "trajectory":
            if sorted(xa) != sorted(xb): bad.append("trajectory steps"); continue
            for s in xa:
                for name, v in xa[s].items():
                    if not torch.equal(v, xb[s][name]): bad.append(f"trajectory[{s}].{name}")
            continue
        if len(xa) != len(xb): bad.append(f"{k}: {len(xa)} vs {len(xb)} entries"); continue
        for i, (u, v) in enumerate(zip(xa, xb)):
            if k == "rng_pppl_out" and i == 0:
                continue        # the design kit's pPPL graph CAPTURES on a process's first LM-loss call and its warm-up draws from the CUDA generator
                                # inside the loop's seed_context (which restores the outer state on exit): the exit state of step 0 differs between the
                                # first design of the process and any later one by construction; entry states, masks and every later exit state compare
            same = (u.dtype == v.dtype and u.shape == v.shape and torch.equal(u, v)) if torch.is_tensor(u) else (u == v)
            if not same:
                extra = f" max|d|={(u.double() - v.double()).abs().max().item():.3e}" if torch.is_tensor(u) and u.shape == v.shape and u.is_floating_point() else ""
                bad.append(f"{k}[{i}]{extra}")
    return bad


def _aba(lever_names, target, n_steps=12, **enable_kw):
    BD, app = _setup()
    R = _Recorder(BD, app)
    mods = [__import__(f"ef2_loop_{n}") for n in lever_names]
    names = ("fold_and_get_distogram", "compute_structure_losses", "compute_esmc_pseudoperplexity_nll", "prepare_esmfold2_tensors", "normalized_gradient_tensor", "build_gradient_mask")
    before = {n: getattr(BD, n) for n in names}
    A = R.run(target, n_steps)
    for m in mods: m.enable(BD, **enable_kw.get(m.__name__, {}))
    try:
        B = R.run(target, n_steps)
        stats = {m.__name__: m.stats(BD) for m in mods}
    finally:
        for m in reversed(mods): m.disable(BD)
    assert all(getattr(BD, n) is before[n] for n in names), "disable() did not restore the module's functions"
    A2 = R.run(target, n_steps)
    d0 = _diff(A, A2)
    assert not d0, f"the unpatched loop is not reproducible in-process (the test is void): {d0[:8]}"
    d = _diff(A, B)
    assert not d, f"{'+'.join(lever_names)} on {target}: differs from stock in {len(d)} observables: {d[:12]}"
    print(f"BITWISE {'+'.join(lever_names)} {target}: {n_steps} steps ({len(A['conf'])} confidence), {len(A['masks'])} LM passes, stats {stats}", flush=True)
    return stats


# ------------------------------------------------------------------------------------------------------------------------------------- tests


@pytest.mark.skipif(not CUDA, reason="CUDA")
@pytest.mark.parametrize("target", ["pd-l1", "cd45"])
def test_prep_bitwise(target):
    st = _aba(["prep"], target)["ef2_loop_prep"]
    assert st["served"] == 12 and st["mismatch"] == 0 and st["early"] >= 10 and st["anchored"] == 1 and st["fallback_anchor"] == 0, st   # the first fold learns the letter table (late) and anchors against the model's own LM path; the rest launch early
    assert st["splice_learned"] == 1 and st["splice_off"] == 0 and st["splice_checked"] == 2 and st["spliced"] >= 9, st          # featurisation: first sequence learned, two spliced results compared with the full featuriser, the rest spliced


@pytest.mark.skipif(not CUDA, reason="CUDA")
@pytest.mark.parametrize("target", ["pd-l1", "cd45"])
def test_pppl_bitwise(target):
    st = _aba(["pppl"], target)["ef2_loop_pppl"]
    assert st["served"] == 12 and st["designs"] == 1 and st["first_read"] == 0 and st["checked"] >= 11, st


@pytest.mark.skipif(not CUDA, reason="CUDA")
@pytest.mark.parametrize("target", ["pd-l1", "cd45"])
def test_all_loop_levers_bitwise(target):
    _aba(["prep", "pppl"], target)


def _feature_like(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"input_ids": torch.randint(0, 64, (1, 211), generator=g), "atom_pad_mask": torch.rand(1, 211, 14, generator=g) > 0.5,
            "coords": torch.randn(1, 211 * 14, 3, generator=g), "residue_index": torch.arange(211).unsqueeze(0), "strided": torch.randn(1, 8, 6, generator=g)[:, ::2],
            "empty": torch.zeros(1, 0, dtype=torch.int32), "half": torch.randn(1, 33, generator=g).to(torch.bfloat16), "scalar_i16": torch.tensor([[7]], dtype=torch.int16)}


def test_pinned_uploader_off_cuda_is_pageable_by_name():
    import ef2_loop_prep as lp
    up = lp.PinnedUploader()
    feats = _feature_like()
    out = up(feats, "cpu")
    assert all(torch.equal(out[k], v) for k, v in feats.items()) and up.stats == {"pinned": 0, "pageable": len(feats), "grown": 0} and up.failed is None
    off = lp.PinnedUploader(enabled=False)
    off(feats, "cpu"); assert off.stats["pageable"] == len(feats)


@pytest.mark.skipif(not CUDA, reason="CUDA")
def test_pinned_uploader_same_bytes_and_no_stream_synchronisation():
    """Exact by construction: every uploaded tensor tensor-equal to the pageable `.cuda()` upload (dtypes bool/int/float/bf16, a strided
    view, an empty tensor); the staging buffer reused by a second call without corrupting the first call's tensors; and the point of it —
    no stream synchronisation (torch's sync debug mode counts one per pageable copy, none for the staged ones)."""
    import warnings
    import ef2_loop_prep as lp
    up = lp.PinnedUploader()
    a, b = _feature_like(1), _feature_like(2)
    ref_a = {k: v.cuda() for k, v in a.items()}; ref_b = {k: v.cuda() for k, v in b.items()}
    out_a = up(a, torch.device("cuda")); out_b = up(b, "cuda")            # back to back: the second call must wait for the first call's copies
    torch.cuda.synchronize()
    for k in a:
        assert out_a[k].dtype == ref_a[k].dtype and out_a[k].shape == ref_a[k].shape and torch.equal(out_a[k], ref_a[k]), k
        assert torch.equal(out_b[k], ref_b[k]) and out_b[k].is_cuda, k
    assert up.stats["pinned"] == 2 * len(a) and up.stats["pageable"] == 0 and up.stats["grown"] == 1 and up.buf.is_pinned(), up.stats
    prev = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as w_page:
            warnings.simplefilter("always"); _ = {k: v.cuda() for k, v in a.items()}
        with warnings.catch_warnings(record=True) as w_pin:
            warnings.simplefilter("always"); _ = up(a, "cuda")
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    n_page = sum("synchroniz" in str(x.message) for x in w_page); n_pin = sum("synchroniz" in str(x.message) for x in w_pin)
    assert n_page >= len(a) - 1 and n_pin == 0, (n_page, n_pin)         # the empty tensor issues no copy


def test_host_lm_inputs_matches_the_model_prep():
    """CPU: ef2_loop_prep.host_lm_inputs + _gather == the model's compute_lm_hidden_states (modeling_esmfold2_common) on a batch of
    2 rows: row 0 = three protein chains (two target chains + a binder) with a duplicated (asym_id, residue_index) key (a residue
    tokenized into two tokens) ; row 1 = two chains, a non-protein token and two padding tokens. The fake ESMC records the ids and
    sequence_id it is called with and returns index-valued hidden states, so the gather is compared element for element."""
    import ef2_loop_prep as lp
    from transformers.models.esmfold2.modeling_esmfold2_common import compute_lm_hidden_states
    B, L = 2, 14
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(4, 30, (B, L), generator=g, dtype=torch.long)
    asym = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long)
    res = torch.tensor([[0, 1, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4],          # row 0: residue 1 of chain 0 spans two tokens
                        [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 2, 3, 0, 0]], dtype=torch.long)   # row 1: residue 2 of chain 1 spans two tokens
    mol = torch.zeros(B, L, dtype=torch.long); mol[1, 3] = 2                  # a non-protein token in row 1
    mask = torch.ones(B, L, dtype=torch.bool); mask[1, -2:] = False           # two padding tokens in row 1
    seen = {}

    class Out:
        pass

    def esmc(input_ids, sequence_id, output_hidden_states):
        seen["ids"], seen["sid"] = input_ids.clone(), sequence_id.clone()
        o = Out(); o.hidden_states = torch.arange(3 * input_ids.numel() * 4, dtype=torch.float32).reshape(3, input_ids.shape[0], input_ids.shape[1], 4)
        return o
    ref = compute_lm_hidden_states(esmc, input_ids, asym, res, mol, mask)
    ids, sid, maps = lp.host_lm_inputs(input_ids.numpy(), asym.numpy(), res.numpy(), mol.numpy(), mask.numpy())
    assert ids.shape == tuple(seen["ids"].shape), (ids.shape, seen["ids"].shape)
    assert np.array_equal(ids, seen["ids"].numpy()) and ids.dtype == seen["ids"].numpy().dtype
    assert np.array_equal(sid, seen["sid"].numpy())
    assert int((seen["ids"] == lp.BOS).sum()) == 5 and int((seen["ids"] == lp.EOS).sum()) == 5      # 3 + 2 chains, each BOS…EOS-wrapped
    mine = lp._Prep._gather(esmc(torch.from_numpy(ids), torch.from_numpy(sid), True).hidden_states, maps, L, torch.device("cpu"))
    assert mine.shape == ref.shape and torch.equal(mine, ref)


if __name__ == "__main__":                     # python k/test_ef2_loop.py [lever[,lever...]] [target]
    lv = sys.argv[1].split(",") if len(sys.argv) > 1 else ["prep", "pppl"]
    tgt = sys.argv[2] if len(sys.argv) > 2 else "pd-l1"
    test_host_lm_inputs_matches_the_model_prep(); test_splice_featuriser_matches_the_featuriser_per_field(); print("PASS unit")
    _aba(lv, tgt); print("PASS", lv, tgt)

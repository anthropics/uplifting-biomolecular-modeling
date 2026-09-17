"""HAZARDS 29 — ef2_atom's fused fold state across a whole-generation reset in the MIDDLE of a fold (katom0.7).

The fused atom path (lever `af`) hands three per-fold device tables (seqlen / tstart / tok_flat) to kernels that a captured step graph replays by
ADDRESS. The registry discipline (ef2_atom._static): a buffer a graph may read lives in _STATIC, is refreshed IN PLACE by the next fold's eager
step 0, and leaves only through clear_static() together with every graph of the generation. katom0.6 cached the _static() results inside
STATE['fold']['fused'] across calls; a reset at step 1 (ef2_opt's step-graph sampler LRU reset, `sampler_budget`) emptied the registry after step 0
had filled it, the eager warm-up before the next capture did NOT re-register the three tables (the cached dict still 'had' them), the graph baked
in buffers only that dict referenced, and the next fold's fold.clear() freed them under the live graph (the sg × af crash: cusolver gesvd
INTERNAL_ERROR / illegal address at 800 tokens, `linalg.svd failed to converge` at 400).

These CPU tests drive _fused_fold_state through exactly that sequence and hold the invariant: every tensor the returned fold state hands to a
kernel IS a registry entry at the moment it is handed out, and a following fold refreshes the same addresses in place."""
import os, sys, unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "fast_inference", "driver"))
A = None                                                    # the REAL driver module, loaded in setUpModule and unloaded in tearDownModule (report's exit tally reads sys.modules)


def setUpModule():
    global A
    import importlib.util
    spec = importlib.util.spec_from_file_location("ef2_atom", os.path.join(DRIVER, "ef2_atom.py"))
    A = importlib.util.module_from_spec(spec)
    sys.modules["ef2_atom"] = A
    spec.loader.exec_module(A)


def tearDownModule():
    sys.modules.pop("ef2_atom", None)


class _Enc:
    """stands in for the atom encoder module: _fused_fold_state keys the registry by id(enc)."""


def _fold_inputs(Bp=1, L=5, atoms_per_tok=3, pad=2, hd=8):
    """mask_exp [Bp, N] (valid prefix), attention_params (cos, sin [Bp, N, hd]), atom_to_token_exp [Bp, N] (sorted token ids), n_tokens."""
    n_valid = L * atoms_per_tok
    N = n_valid + pad
    mask = torch.zeros(Bp, N, dtype=torch.bool); mask[:, :n_valid] = True
    tok = torch.zeros(Bp, N, dtype=torch.long)
    tok[:, :n_valid] = torch.arange(L).repeat_interleave(atoms_per_tok)[None, :]
    cos = torch.randn(Bp, N, hd); sin = torch.randn(Bp, N, hd)
    return mask, (cos, sin), tok, L


def _graph_read(fs):
    """the tensors of a fused fold state that kernels (hence a captured graph) read by address."""
    return {k: fs[k] for k in ("seqlen", "tstart", "tok", "cos", "sin")}


class TestFusedFoldStateAcrossMidFoldReset(unittest.TestCase):
    def setUp(self):
        A.clear_static(); A.STATE.pop("fold", None); A.STATS.clear()
        self.enc = _Enc()

    def _step0(self, inputs):
        """the fold's eager step 0: the encoder bumps the fold epoch, _fold_context clears the fold state, the fused state is built (new_fold)."""
        mask, ap, tok, L = inputs
        A._FOLD["n"] += 1
        A.STATE.setdefault("fold", {}).clear()
        return A._fused_fold_state(self.enc, None, ap, mask, tok, L, True)

    def _later_step(self, inputs):
        mask, ap, tok, L = inputs
        return A._fused_fold_state(self.enc, None, ap, mask, tok, L, False)

    def assertAllRegistered(self, fs):
        reg = {id(t) for t in A._STATIC.values()}
        for k, t in _graph_read(fs).items():
            self.assertIn(id(t), reg, f"{k!r} handed to a kernel is not a registry (_STATIC) entry — a graph capturing it reads a buffer nothing refreshes / keeps")

    def test_values(self):
        inp = _fold_inputs()
        fs = self._step0(inp)
        mask, (cos, sin), tok, L = inp
        self.assertTrue(fs["row_prefix"]); self.assertTrue(fs["tok_sorted"])
        self.assertEqual((fs["Bp"], fs["N"], fs["M"], fs["L"]), (1, mask.shape[1], mask.shape[1], L))
        self.assertTrue(torch.equal(fs["seqlen"], mask.sum(-1).to(torch.int32)))
        self.assertEqual(tuple(fs["tstart"].shape), (1, L + 1))
        self.assertEqual(int(fs["tstart"][0, -1]), int(mask.sum()))                      # prefix ends at the valid-atom count
        self.assertTrue(torch.equal(fs["tok"], tok.reshape(-1)))
        self.assertTrue(torch.equal(fs["cos"], cos.reshape(mask.shape[1], -1)))
        self.assertNotIn("_src", fs)                                                    # sources never leave the fold state
        self.assertAllRegistered(fs)

    def test_same_fold_later_steps_hit(self):
        inp = _fold_inputs()
        fs0 = self._step0(inp)
        h0 = A.STATS["static_hits"]
        fs1 = self._later_step(inp)
        for k in _graph_read(fs0):
            self.assertEqual(fs1[k].data_ptr(), fs0[k].data_ptr(), k)
        self.assertEqual(A.STATS["static_hits"] - h0, 5)                               # 5 registry reads, no rebuild, no host read
        self.assertEqual(A.STATS["a5_static_reregistered"], 0)

    def test_mid_fold_reset_then_warmup_reregisters(self):
        """THE HAZARD: step 0 fills the registry; a generation reset at step 1 empties it; the capture helper's eager warm-up (same fold, new_fold
        False) must hand out REGISTRY buffers (re-registered from the kept sources), never the pre-reset ones."""
        inp = _fold_inputs()
        fs0 = self._step0(inp)
        pre = {k: t.data_ptr() for k, t in _graph_read(fs0).items()}
        A.clear_static()                                                                 # ef2_opt.clear_graphs(reason='sampler_budget') -> chained clear_static()
        self.assertEqual(len(A._STATIC), 0)
        fs_w = self._later_step(inp)                                                     # the eager warm-up before the capture
        self.assertAllRegistered(fs_w)
        self.assertEqual(A.STATS["a5_static_reregistered"], 1)
        for k, t in _graph_read(fs_w).items():                                           # values intact (from the kept step-0 sources; no host read needed)
            self.assertTrue(torch.equal(t, fs0[k]), k)
        del fs0                                                                          # katom0.6: the pre-reset buffers' last owner was the fold dict; here they may go
        fs_c = self._later_step(inp)                                                     # the capture call: same addresses as the warm-up registered
        for k, t in _graph_read(fs_c).items():
            self.assertEqual(t.data_ptr(), fs_w[k].data_ptr(), k)
        _ = pre

    def test_next_fold_refreshes_reregistered_buffers_in_place(self):
        """after the reset + warm-up + capture of fold k, fold k+1 of the same signature must refresh the SAME addresses (the captured graph of
        fold k replays on them) — katom0.6 allocated new ones here and freed the graph's."""
        inp = _fold_inputs()
        self._step0(inp)
        A.clear_static()
        fs_cap = self._later_step(inp)                                                   # what fold k's graph captured
        cap_ptrs = {k: t.data_ptr() for k, t in _graph_read(fs_cap).items()}
        inp2 = _fold_inputs()                                                            # fold k+1: same shapes (same signature), new values
        r0 = A.STATS["static_refresh"]; a0 = A.STATS["static_allocs"]
        fs_next = self._step0(inp2)
        self.assertAllRegistered(fs_next)
        for k, t in _graph_read(fs_next).items():
            self.assertEqual(t.data_ptr(), cap_ptrs[k], f"{k!r}: fold k+1 must refresh the captured buffer in place, not re-allocate")
        self.assertEqual(A.STATS["static_allocs"] - a0, 0)
        self.assertEqual(A.STATS["static_refresh"] - r0, 5)
        mask2, (cos2, _sin2), tok2, L2 = inp2
        self.assertTrue(torch.equal(fs_next["cos"], cos2.reshape(mask2.shape[1], -1)))  # and hold fold k+1's values

    def test_reset_before_step0_is_the_ordinary_path(self):
        """a reset BETWEEN folds (new_input / new_shape: before the sampler's step 0) never needed re-registration: step 0 allocates afresh."""
        inp = _fold_inputs()
        self._step0(inp)
        A.clear_static()
        fs = self._step0(_fold_inputs())
        self.assertAllRegistered(fs)
        self.assertEqual(A.STATS["a5_static_reregistered"], 0)

    def test_two_signatures_share_shape_equal_tables(self):
        """two items of one token count and different atom counts share the [Bp] / [Bp, L+1] tables (one registry key) and own their [M] tables;
        alternating folds refresh the shared ones in place and keep both atom-count keyed sets registered (one generation holds both graphs)."""
        a = _fold_inputs(atoms_per_tok=3, pad=2)                                         # N = 17
        b = _fold_inputs(atoms_per_tok=3, pad=5)                                         # N = 20, same L
        fa = self._step0(a); fb = self._step0(b)
        self.assertEqual(fa["seqlen"].data_ptr(), fb["seqlen"].data_ptr())              # shared key, refreshed in place
        self.assertEqual(fa["tstart"].data_ptr(), fb["tstart"].data_ptr())
        self.assertNotEqual(fa["tok"].data_ptr(), fb["tok"].data_ptr())                 # [M]-keyed: distinct
        fa2 = self._step0(a)
        self.assertEqual(fa2["tok"].data_ptr(), fa["tok"].data_ptr())                   # a's graph still reads a registered, refreshed buffer
        self.assertAllRegistered(fa2)


if __name__ == "__main__":
    unittest.main()

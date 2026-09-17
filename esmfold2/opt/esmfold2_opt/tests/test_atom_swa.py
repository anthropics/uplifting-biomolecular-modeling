"""The atom_swa sub-lever (esmfold2_opt.atom_swa: ESMFold2's sliding-window atom attention evaluated on key bands instead of the dense
[B, N, N] mask of SWA3DRoPEAttention.forward's fallback, modeling_esmfold2_common.py:613-631 @ef32577f). CPU, torch required (skipped by
name without it):

* ATTENDED SET — for random validity patterns (all valid, trailing padding, interior invalid runs, B = 2 rows with different patterns) and
  several (N, half_window, q_block, key_align), the band of every query block contains every key the dense mask allows and the block's mask
  equals the dense mask's block, bit for bit (:func:`atom_swa.assert_band_covers`);
* NUMERICS — banded == the dense stock statements (a verbatim copy of :620-630 in this file, the independent reference) on random q/k/v:
  with the default key_align (512 = the torch-CPU flash kernel's key tile) ``torch.equal`` in fp32 AND bf16 (the dtype the stock forward
  computes in), every case; with band edges off the tile (key_align 128 / 32 / 1) fp32 max|diff| <= 1e-5 and bf16 within 1 ulp at the
  output scale (the online-softmax rescaling order), ``torch.equal`` counts reported;
* MEMORY — a TorchDispatchMode records the numel of every tensor an op returns during the banded call at N = 4096 atoms: none reaches
  B·N² (the dense statement allocates three), and the largest is bounded by the block formula;
* INSTALL RULE — n_gpu = 1 installs nothing and reads nothing off the model; n_gpu > 1 installs the class attribute and ``dense`` is refused
  by name; n_gpu = 1 installs nothing (one GPU; nothing read, nothing patched); a bad spelling / non-positive block size is refused by name; a forward whose
  source moved is refused by name (source guard) with nothing patched; uninstall restores the stock attribute (the same object); an INSTANCE
  ``forward`` marked ``_ef2opt_swa`` (the fast line's U1 lever rebinds one on every instance, and ``nn.Module.__call__`` serves an instance
  attribute before the class patch) is rebound to the banded forward — the banded forward is what RUNS on that instance and U1's never does —
  reported as ``atom_forward=bandedxN atom_rebound=K``, restored by uninstall; an instance forward without the mark is refused by name,
  nothing patched;
* WHOLE MODULE — with the transformers fork of stock/PINS.json importable (or its source tree under stock/src), the REAL
  ``SWA3DRoPEAttention`` runs stock (dense fallback) and installed (banded) on the same input: ``torch.equal``; and the stock forward's source
  digest equals :data:`atom_swa.FORWARD_SOURCE_SHA256` (the text the banded forward mirrors).
"""
import importlib.util
import os
import sys
import types
import unittest

try:
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.diffusion_loop.source_guard import source_sha256
except ImportError as _e:                                                   # an older / absent core: named skip
    raise unittest.SkipTest(f"producer_missing:opt_core.mem.rowpair ({_e}): the atom_swa tests need opt_core >= 0.4.2")
from esmfold2_opt import atom_swa as AA


def _torch_or_skip():
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"torch is not importable here ({type(e).__name__}): the atom_swa numerics run where torch is installed")
    return sys.modules["torch"]


STOCK_RELPATH = os.path.join("stock", "src", "transformers", "src", "transformers", "models", "esmfold2", "modeling_esmfold2_common.py")


def _stock_common_or_skip():
    """The stock module that defines SWA3DRoPEAttention: the installed transformers fork when importable, else the fork's source file under
    the kit's stock/src tree loaded stand-alone (its one relative import, configuration_esmfold2, is stubbed: the atom attention reads no
    config) — else a named skip."""
    _torch_or_skip()
    try:
        from transformers.models.esmfold2 import modeling_esmfold2_common as CMN
        return CMN
    except Exception:  # noqa: BLE001
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("EF2_STOCK_COMMON", ""),
             os.path.join(os.environ.get("MODEL_OPT", ""), STOCK_RELPATH) if os.environ.get("MODEL_OPT") else "",
             os.path.normpath(os.path.join(here, "..", "..", "..", STOCK_RELPATH))]
    path = next((c for c in cands if c and os.path.isfile(c)), None)
    if path is None:
        raise unittest.SkipTest(f"the transformers fork is not importable and {STOCK_RELPATH} is not under the kit (set MODEL_OPT or EF2_STOCK_COMMON): skipped by name")
    pkg = "_ef2stock_atom_swa"
    if pkg + ".modeling_esmfold2_common" in sys.modules:
        return sys.modules[pkg + ".modeling_esmfold2_common"]
    parent = types.ModuleType(pkg); parent.__path__ = [os.path.dirname(path)]
    cfg = types.ModuleType(pkg + ".configuration_esmfold2"); cfg.ESMFold2Config = type("ESMFold2Config", (), {})
    sys.modules[pkg] = parent; sys.modules[pkg + ".configuration_esmfold2"] = cfg
    spec = importlib.util.spec_from_file_location(pkg + ".modeling_esmfold2_common", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001
        sys.modules.pop(spec.name, None)
        raise unittest.SkipTest(f"{path} did not load stand-alone ({type(e).__name__}: {e}): skipped by name")
    return mod


def _dense_reference(q, k, v, valid, hw, scale):
    """modeling_esmfold2_common.py:620-630 verbatim (the stock dense fallback before its `* valid`), the independent reference."""
    import torch
    import torch.nn.functional as F
    B, N = valid.shape
    rank = torch.cumsum(valid, dim=1) - 1
    within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= hw
    allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
    allowed |= torch.eye(N, dtype=torch.bool, device=q.device)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed.unsqueeze(1), scale=scale,
    ).transpose(1, 2)


def _valid_patterns(torch, B, N, gen):
    """all valid; trailing padding per row; interior invalid runs; (B=2) rows with different lengths."""
    out = {}
    out["all_valid"] = torch.ones(B, N, dtype=torch.bool)
    v = torch.ones(B, N, dtype=torch.bool)
    for b in range(B):
        v[b, N - (7 + 13 * b):] = False
    out["trailing_pad"] = v
    v = torch.rand(B, N, generator=gen) > 0.15
    v[:, 0] = True
    out["interior_holes"] = v
    v = torch.ones(B, N, dtype=torch.bool)
    v[0, N // 2:] = False
    out["ragged_rows"] = v
    return out


class AttendedSet(unittest.TestCase):
    """The band of every query block covers the dense mask's allowed keys and reproduces its block bit for bit."""

    def test_band_covers_dense_mask(self):
        torch = _torch_or_skip()
        gen = torch.Generator().manual_seed(0)
        seen = 0
        for (B, N) in ((1, 257), (2, 300), (1, 1031)):
            for name, valid in _valid_patterns(torch, B, N, gen).items():
                for hw in (2, 17, 64):
                    for q_block, ka in ((64, 1), (64, 128), (256, 128), (100, 32), (4096, 128)):
                        with self.subTest(B=B, N=N, valid=name, hw=hw, q_block=q_block, key_align=ka):
                            maxw = AA.assert_band_covers(valid, hw, q_block=q_block, key_align=ka)
                            self.assertGreaterEqual(maxw, min(N, q_block))
                            if name == "all_valid" and q_block < N:
                                self.assertLessEqual(maxw, AA.band_width_bound(q_block, hw, ka))
                            seen += 1
        self.assertGreaterEqual(seen, 100)

    def test_diagonal_only_rows_for_invalid_queries(self):
        torch = _torch_or_skip()
        valid = torch.tensor([[True, True, False, False, True, False]])
        rank = torch.cumsum(valid, 1) - 1
        (i0, i1, klo, khi), = AA.band_plan(rank, 1, 64, 1)
        blk = AA.allowed_block(rank, valid, i0, i1, klo, khi, 1)[0]
        self.assertEqual(blk[2].nonzero().flatten().tolist(), [2])            # invalid query: itself only (the `| eye` term)
        self.assertEqual(blk[3].nonzero().flatten().tolist(), [3])
        self.assertEqual(blk[1].nonzero().flatten().tolist(), [0, 1, 4])      # valid query rank 1: ranks 0..2 that are valid -> atoms 0, 1, 4


class BandedEqualsDense(unittest.TestCase):
    """banded_window_attention == the dense stock statements on random inputs."""

    def _cases(self, torch, dtype):
        gen = torch.Generator().manual_seed(1)
        H, hd = 4, 32
        for (B, N) in ((1, 257), (2, 300), (1, 777), (2, 1500)):
            pats = _valid_patterns(torch, B, N, gen)
            for name in ("all_valid", "trailing_pad", "interior_holes", "ragged_rows"):
                valid = pats[name]
                q = torch.randn(B, N, H, hd, generator=gen).to(dtype)
                k = torch.randn(B, N, H, hd, generator=gen).to(dtype)
                v = torch.randn(B, N, H, hd, generator=gen).to(dtype)
                for hw in (3, 64):
                    for q_block, ka in ((64, AA.KEY_ALIGN), (256, AA.KEY_ALIGN), (64, 128), (128, 1), (100, 32)):
                        yield dict(B=B, N=N, valid=name, hw=hw, q_block=q_block, key_align=ka), q, k, v, valid, hw

    def test_fp32_within_1e5_and_report_equal(self):
        torch = _torch_or_skip()
        scale = 32 ** -0.5
        n_equal = n = 0
        worst = 0.0
        for desc, q, k, v, valid, hw in self._cases(torch, torch.float32):
            with self.subTest(**desc):
                ref = _dense_reference(q, k, v, valid, hw, scale) * valid[..., None, None]
                got = AA.banded_window_attention(q, k, v, valid, hw, scale, q_block=desc["q_block"], key_align=desc["key_align"]) * valid[..., None, None]
                diff = float((ref - got).abs().max())
                worst = max(worst, diff)
                n += 1; n_equal += int(torch.equal(ref, got))
                self.assertLessEqual(diff, 1e-5, f"fp32 max|diff| {diff:.3e}")
                if desc["key_align"] == AA.KEY_ALIGN:
                    if ref.is_cuda:                                        # bitwise is the GPU-image claim; CPU sgemm is not M-invariant (1e-5 class there, asserted above)
                        self.assertTrue(torch.equal(ref, got), f"fp32 not bitwise at key_align={AA.KEY_ALIGN} (max|diff| {diff:.3e})")
        print(f"[atom_swa fp32] cases={n} torch.equal={n_equal}/{n} max|diff|={worst:.3e}")

    def test_bf16_equal_at_default_align_and_1ulp_off_tile(self):
        torch = _torch_or_skip()
        scale = 32 ** -0.5
        n_eq_def = n_def = n_eq_off = n_off = 0
        worst_off = 0.0
        for desc, q, k, v, valid, hw in self._cases(torch, torch.bfloat16):
            with self.subTest(**desc):
                ref = _dense_reference(q, k, v, valid, hw, scale) * valid[..., None, None]
                got = AA.banded_window_attention(q, k, v, valid, hw, scale, q_block=desc["q_block"], key_align=desc["key_align"]) * valid[..., None, None]
                eq = torch.equal(ref, got)
                if desc["key_align"] == AA.KEY_ALIGN:
                    n_def += 1; n_eq_def += int(eq)
                    if ref.is_cuda:                                        # bitwise is the GPU-image claim; on CPU the 1-ulp class below is the assertion
                        self.assertTrue(eq, f"bf16 not bitwise at key_align={AA.KEY_ALIGN} (max|diff| {float((ref.float() - got.float()).abs().max()):.3e})")
                    else:
                        d = float((ref.float() - got.float()).abs().max()); u = float(torch.finfo(torch.bfloat16).eps * ref.float().abs().max())
                        self.assertLessEqual(d, u, f"bf16 CPU diff {d:.3e} > 1 ulp {u:.3e} at key_align={AA.KEY_ALIGN}")
                else:
                    diff = float((ref.float() - got.float()).abs().max()); worst_off = max(worst_off, diff); n_off += 1; n_eq_off += int(eq)
                    ulp = 2.0 ** -7 * float(ref.float().abs().max())                # one bf16 ulp at the output's scale
                    self.assertLessEqual(diff, ulp, f"bf16 off-tile diff {diff:.3e} > 1 ulp {ulp:.3e}")
        print(f"[atom_swa bf16] key_align={AA.KEY_ALIGN}: torch.equal={n_eq_def}/{n_def}; off-tile aligns: torch.equal={n_eq_off}/{n_off} max|diff|={worst_off:.3e}")
        self.assertGreater(n_def, 0); self.assertGreater(n_off, 0)


class NumelGuard(unittest.TestCase):
    """No tensor with B·N² elements is created by the banded call (the dense statement creates three)."""

    def test_no_dense_sized_tensor(self):
        torch = _torch_or_skip()
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_flatten

        class MaxNumel(TorchDispatchMode):
            def __init__(self):
                super().__init__(); self.max = 0; self.argmax = None
            def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                for t in tree_flatten(out)[0]:
                    if isinstance(t, torch.Tensor) and t.numel() > self.max:
                        self.max, self.argmax = int(t.numel()), str(func)
                return out

        B, N, H, hd, hw, q_block, ka = 1, 4096, 4, 32, 64, 256, AA.KEY_ALIGN
        gen = torch.Generator().manual_seed(2)
        q = torch.randn(B, N, H, hd, generator=gen); k = torch.randn(B, N, H, hd, generator=gen); v = torch.randn(B, N, H, hd, generator=gen)
        valid = torch.ones(B, N, dtype=torch.bool); valid[:, N - 100:] = False
        with MaxNumel() as mode:
            AA.banded_window_attention(q, k, v, valid, hw, hd ** -0.5, q_block=q_block, key_align=ka)
        W = AA.band_width_bound(q_block, hw, ka, pad=100)
        bound = B * H * q_block * W                                                   # the largest per-block object: [B, H, q_block, W] scores / weights
        print(f"[atom_swa numel] N={N} max numel={mode.max} ({mode.argmax}) bound={bound} B*N^2={B * N * N}")
        self.assertLess(mode.max, B * N * N)
        self.assertLessEqual(mode.max, bound)
        with MaxNumel() as dense_mode:                                                # the dense statement, same input: allocates >= B*N^2
            _dense_reference(q, k, v, valid, hw, hd ** -0.5)
        self.assertGreaterEqual(dense_mode.max, B * N * N)

    def test_formulas(self):
        self.assertEqual(AA.dense_transient_bytes(59_000), 17 * 59_000 ** 2)          # ≈ 59.2 GB at ≈ 7.8k tokens
        self.assertEqual(AA.banded_transient_bytes(2048, 3198), 17 * 2048 * 3198)     # ≈ 0.11 GB per block at the defaults
        self.assertEqual(AA.band_width_bound(2048, 64, 512), 2048 + 128 + 1022)
        self.assertEqual(AA.band_width_bound(AA.Q_BLOCK, 64, AA.KEY_ALIGN), 3198)


class _Host:
    """A minimal `model`: modules() yields the SWA instances (install walks nothing else)."""
    def __init__(self, *mods):
        self._mods = list(mods)
    def modules(self):
        for m in self._mods:
            yield from m.modules()


def _pin_sdpa(case: unittest.TestCase) -> None:
    """The banded lever replaces upstream's SDPA fallback: the CPU tests below pin FLASH_ATTN_AVAILABLE to False for their duration whatever the
    box (on the pinned image it is True and install() declines by name — InstallRule.test_flash_present_is_named_not_patched sets it so)."""
    try:
        CMN = _stock_common_or_skip()
    except unittest.SkipTest:
        return
    old = CMN.FLASH_ATTN_AVAILABLE
    CMN.FLASH_ATTN_AVAILABLE = False
    case.addCleanup(setattr, CMN, "FLASH_ATTN_AVAILABLE", old)


class InstallRule(unittest.TestCase):

    def setUp(self):
        _pin_sdpa(self)

    def tearDown(self):
        AA.uninstall()

    def test_p1_installs_nothing_and_reads_nothing(self):
        class Untouchable:
            def __getattr__(self, name):
                raise AssertionError(f"install(P=1) read model.{name}")
        rec = AA.install(Untouchable(), 1, environ={})
        self.assertFalse(rec["installed"]); self.assertEqual(rec["path"], "dense_stock"); self.assertEqual(len(AA.PATCHES), 0)
        self.assertEqual(AA.census_fields(rec)["atom_swa"], "dense_stock")

    def test_resolve_words(self):
        """banded under n_gpu > 1 (the one implementation: no switch, no dense path), dense (= nothing installed) at n_gpu = 1."""
        self.assertEqual(AA.resolve(1), "dense")
        for P in (2, 4, 8):
            self.assertEqual(AA.resolve(P), "banded")
        with self.assertRaises(RowpairRefused):
            AA.resolve(0)
        self.assertFalse(hasattr(AA, "ENV_KIND")); self.assertFalse(hasattr(AA, "NOTE_P1"))
        with self.assertRaises(RowpairRefused):
            AA.settings({AA.ENV_QBLOCK: "0"})
        with self.assertRaises(RowpairRefused):
            AA.settings({AA.ENV_KALIGN: "x"})
        self.assertEqual(AA.settings({}), {"q_block": AA.Q_BLOCK, "key_align": AA.KEY_ALIGN})
        self.assertEqual(AA.settings({AA.ENV_QBLOCK: "512"}, key_align=64), {"q_block": 512, "key_align": 64})


    def test_gate_table_names_every_gate(self):
        names = [g[0] for g in AA.GATES]
        self.assertEqual(names, ["atom_count", "flash_attn_present", "valid_from_indices", "instance_forward"])   # no switch gate: banded is the one implementation under n_gpu > 1
        self.assertTrue(all(len(g) == 5 and all(g) for g in AA.GATES))

    def test_moved_source_is_refused_nothing_patched(self):
        torch = _torch_or_skip()
        fake = types.ModuleType("_fake_cmn_atom_swa")
        fake.FLASH_ATTN_AVAILABLE = False; fake.qk_norm = lambda x: x; fake.apply_rotary_emb_3d = lambda x, c, s: x
        src = ("import torch\nclass SWA3DRoPEAttention(torch.nn.Module):\n    def __init__(self):\n        super().__init__(); self.half_window = 4; self.n_heads = 1; self.head_dim = 4; self.scale = 0.5\n"
               "        self.Wqkv = torch.nn.Linear(4, 12); self.gate_proj = torch.nn.Linear(4, 4); self.out_proj = torch.nn.Linear(4, 4)\n"
               "    def forward(self, x, attention_params):\n        return x\n")
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src); path = f.name
        spec = importlib.util.spec_from_file_location("_fake_cmn_atom_swa", path)
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
        mod.FLASH_ATTN_AVAILABLE = False; mod.qk_norm = fake.qk_norm; mod.apply_rotary_emb_3d = fake.apply_rotary_emb_3d
        m = mod.SWA3DRoPEAttention()
        stock_forward = mod.SWA3DRoPEAttention.forward
        with self.assertRaises(RowpairRefused) as cm:
            AA.install(_Host(m), 2, environ={})
        self.assertIn("refusing to patch", str(cm.exception.reason))
        self.assertIs(mod.SWA3DRoPEAttention.forward, stock_forward); self.assertEqual(len(AA.PATCHES), 0)
        os.unlink(path)

    def test_install_p2_patches_class_and_uninstall_restores(self):
        CMN = _stock_common_or_skip()
        cls = CMN.SWA3DRoPEAttention
        stock_forward = cls.forward
        m = cls(128, 4, half_window=8)
        rec = AA.install(_Host(m), 2, environ={AA.ENV_QBLOCK: "64"})
        self.assertTrue(rec["installed"]); self.assertEqual(rec["path"], "banded"); self.assertEqual(rec["q_block"], 64)
        self.assertEqual(rec["half_window"], [8]); self.assertEqual(rec["modules"], 1); self.assertEqual(rec["sites"], ["SWA3DRoPEAttention.forward"])
        self.assertIs(cls.forward, AA._forward_banded)
        f = AA.census_fields()
        self.assertEqual((f["atom_swa"], f["atom_qblock"], f["atom_kalign"], f["atom_window"], f["atom_modules"]), ("banded", 64, AA.KEY_ALIGN, "16", 1))
        with self.assertRaises(RowpairRefused):                                      # a second install is refused by name, not stacked
            AA.install(_Host(m), 2, environ={})
        self.assertEqual(AA.uninstall(), ["SWA3DRoPEAttention.forward"])
        self.assertIs(cls.forward, stock_forward)
        rec0 = AA.install(_Host(m), 1, environ={})                                     # without the switch nothing is touched at n_gpu = 1 (unchanged)
        self.assertFalse(rec0["installed"]); self.assertIs(cls.forward, stock_forward)

    def test_u1_instance_forward_is_rebound_reported_and_restored(self):
        """The fast line's U1 lever (ef2_opt.install_swa_mask_cache) sets ``m.forward = MethodType(_swa_forward_cached, m)`` and
        ``m._ef2opt_swa = True`` on every instance; ``nn.Module.__call__`` serves that instance attribute before any class attribute, so a
        class patch alone never runs there. install() rebinds the marked instance forward to the banded forward: the banded forward RUNS on
        the instance, U1's does not; the census says what ``__call__`` resolves; uninstall serves U1's bound method again."""
        CMN = _stock_common_or_skip()
        torch = sys.modules["torch"]
        cls = CMN.SWA3DRoPEAttention
        stock_forward = cls.forward
        torch.manual_seed(5)
        m_u1, m_plain = cls(128, 4, half_window=8).eval(), cls(128, 4, half_window=8).eval()
        calls = {"u1": 0, "banded": 0}
        def _swa_forward_cached(self, x, attention_params):                           # stands for U1's instance forward (dense memoized masks), named as ef2_opt names it
            calls["u1"] += 1
            return stock_forward(self, x, attention_params)
        m_u1.forward = types.MethodType(_swa_forward_cached, m_u1); m_u1._ef2opt_swa = True     # the two statements of ef2_opt.install_swa_mask_cache
        gen = torch.Generator().manual_seed(2)
        B, N = 1, 300
        x = torch.randn(B, N, 128, generator=gen)
        cos, sin = CMN.build_3d_rope(torch.randn(B, N, 3, generator=gen) * 5, torch.arange(N).repeat(B, 1) // 3, m_u1.head_dim)
        with torch.no_grad():
            ref = m_u1(x, (cos, sin))                                                   # U1's forward runs (the instance attribute is served)
        self.assertEqual(calls["u1"], 1)
        banded = AA.banded_window_attention
        def counted(*a, **k):
            calls["banded"] += 1
            return banded(*a, **k)
        AA.banded_window_attention = counted
        try:
            rec = AA.install(_Host(m_u1, m_plain), 2, environ={AA.ENV_QBLOCK: "128"})
            self.assertTrue(rec["installed"]); self.assertEqual(rec["path"], "banded")
            with torch.no_grad():
                got = m_u1(x, (cos, sin))
            self.assertEqual(calls["u1"], 1, "U1's instance forward ran under the installed lever: the banded class patch is shadowed on this instance")
            self.assertEqual(calls["banded"], 1, "the banded forward did not run on the U1-marked instance")
            self.assertTrue(torch.equal(ref, got))
            self.assertEqual(rec["rebound"], 1); self.assertEqual(rec["forward_resolved"], {"banded": 2})
            self.assertEqual(rec["sites"], ["SWA3DRoPEAttention.forward", "SWA3DRoPEAttention().forward x1"])
            f = AA.census_fields()
            self.assertEqual((f["atom_forward"], f["atom_rebound"]), ("bandedx2", 1))
            self.assertIs(vars(m_u1)["forward"].__func__, AA._forward_banded)         # the instance attribute now IS the banded forward, bound to the instance
            self.assertEqual(AA.U1_MARK, "_ef2opt_swa")
        finally:
            AA.banded_window_attention = banded
            names = AA.uninstall()
        self.assertEqual(sorted(names), sorted(["SWA3DRoPEAttention().forward", "SWA3DRoPEAttention.forward"]))
        self.assertIs(cls.forward, stock_forward)
        self.assertIs(vars(m_u1)["forward"].__func__, _swa_forward_cached)             # U1's bound method is served again
        self.assertEqual(AA.forward_resolution([m_u1, m_plain]), {"instance:_swa_forward_cached": 1, "stock": 1})
        with torch.no_grad():
            m_u1(x, (cos, sin))
        self.assertEqual(calls["u1"], 2)

    def test_foreign_instance_forward_is_refused_nothing_patched(self):
        CMN = _stock_common_or_skip()
        cls = CMN.SWA3DRoPEAttention
        stock_forward = cls.forward
        m_f, m_plain = cls(128, 4, half_window=8), cls(128, 4, half_window=8)
        def some_other_forward(self, x, attention_params):
            return x
        m_f.forward = types.MethodType(some_other_forward, m_f)                       # no _ef2opt_swa mark: a lever this module does not know
        with self.assertRaises(RowpairRefused) as cm:
            AA.install(_Host(m_f, m_plain), 2, environ={})
        self.assertEqual(cm.exception.lever, AA.LEVER)
        self.assertIn("instance-level forward", str(cm.exception.reason)); self.assertIn("some_other_forward", str(cm.exception.reason))
        self.assertIs(cls.forward, stock_forward); self.assertEqual(len(AA.PATCHES), 0); self.assertEqual(len(AA.INSTANCE_PATCHES), 0)
        self.assertIs(vars(m_f)["forward"].__func__, some_other_forward)
        self.assertFalse(AA.STATE.get("installed"))

    def test_flash_present_is_named_not_patched(self):
        CMN = _stock_common_or_skip()
        cls = CMN.SWA3DRoPEAttention
        m = cls(128, 4, half_window=8)
        old = CMN.FLASH_ATTN_AVAILABLE
        try:
            CMN.FLASH_ATTN_AVAILABLE = True
            rec = AA.install(_Host(m), 2, environ={})
        finally:
            CMN.FLASH_ATTN_AVAILABLE = old
        self.assertFalse(rec["installed"]); self.assertEqual(rec["path"], "stock_flash"); self.assertEqual(len(AA.PATCHES), 0)


class WholeModule(unittest.TestCase):
    """The REAL SWA3DRoPEAttention: stock dense fallback vs the installed banded forward on one input."""

    def setUp(self):
        _pin_sdpa(self)

    def tearDown(self):
        AA.uninstall()

    def test_stock_forward_source_pin(self):
        CMN = _stock_common_or_skip()
        self.assertIn(source_sha256(CMN.SWA3DRoPEAttention.forward), AA.FORWARD_SOURCE_SHA256)

    def test_forward_equal_stock(self):
        CMN = _stock_common_or_skip()
        torch = sys.modules["torch"]
        self.assertFalse(CMN.FLASH_ATTN_AVAILABLE, "this test compares against the SDPA fallback (setUp pins upstream's flag to False on any box)")
        gen = torch.Generator().manual_seed(3)
        n_equal = n = 0; worst = 0.0
        for (B, N, hw) in ((1, 300, 8), (2, 257, 64), (3, 520, 5), (1, 1100, 64)):
            torch.manual_seed(11)
            m = CMN.SWA3DRoPEAttention(128, 4, half_window=hw).eval()
            x = torch.randn(B, N, 128, generator=gen)
            ref_pos = torch.randn(B, N, 3, generator=gen) * 5
            uid = torch.arange(N).repeat(B, 1) // 3
            cos, sin = CMN.build_3d_rope(ref_pos, uid, m.head_dim)
            mask = torch.ones(B, N, dtype=torch.bool); mask[0, N - 37:] = False
            if B > 1:
                mask[1, 100:140] = False
            indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
            for params in ((cos, sin, indices, None, None), (cos, sin)):
                with torch.no_grad():
                    ref = m(x, params)                                                # class attribute = stock forward (dense fallback)
                    AA.install(_Host(m), 2, environ={AA.ENV_QBLOCK: "128"})
                    try:
                        got = m(x, params)
                    finally:
                        AA.uninstall()
                with self.subTest(B=B, N=N, hw=hw, indices=len(params) > 2):
                    diff = float((ref - got).abs().max()); worst = max(worst, diff); n += 1; n_equal += int(torch.equal(ref, got))
                    self.assertTrue(torch.equal(ref, got), f"module forward not bitwise equal (max|diff| {diff:.3e})")
        print(f"[atom_swa module] cases={n} torch.equal={n_equal}/{n} max|diff|={worst:.3e}")


if __name__ == "__main__":
    unittest.main()

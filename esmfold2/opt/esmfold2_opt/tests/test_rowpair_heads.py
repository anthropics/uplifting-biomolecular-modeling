"""The Mode-S row statements (rowpair_heads) against the stock modules they re-issue, on P = 2 CPU ranks (gloo, opt_core's launcher): the
relative-position rows, the LM pair rows, the confidence head on rows (incl. the gathered TM-map scalars), the diffusion token transformer
with local query rows, and the conditioning rows — each compared with the dense stock forward computed in the same processes (fp32,
max |diff| bounds; torch.equal reported). Plus the structural rules: the source anchors of every re-issued statement exist in the stock text,
the n_gpu > 1 line has NO size gate that switches to a replicated path (R-TP-3), and P = 1 installs nothing."""
from __future__ import annotations

import os
import re
import sys
import unittest

from esmfold2_opt import rowpair, rowpair_heads as H

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def _torch_or_skip():
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"torch is not importable here ({type(e).__name__}): run on a CPU or GPU box with torch")
    return sys.modules["torch"]


def _stock_or_skip():
    _torch_or_skip()
    try:
        from transformers.models.esmfold2 import modeling_esmfold2 as MOD, modeling_esmfold2_common as CMN  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"the transformers fork (stock/PINS.json) is not importable here ({type(e).__name__}: {e}): skipped by name")
    return CMN, MOD


def _mp(P, entry, *args):
    from opt_core.mem.rowpair import launch as RL
    return RL.run_sharded(P, entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=120, run_timeout_s=1200)


# ================================================================================================ structural rules (no torch needed)
class SourceAnchors(unittest.TestCase):
    """Every re-issued statement is pinned to the stock text: if upstream edits one, this fails by name (re-issue it, do not patch around)."""

    def test_anchors_exist_in_stock_sources(self):
        stock = os.path.join(PKG, "..", "..", "stock", "src", "transformers", "src", "transformers", "models", "esmfold2")
        if not os.path.isdir(stock):
            raise unittest.SkipTest("stock/src not checked out here")
        M = open(os.path.join(stock, "modeling_esmfold2.py")).read()
        C = open(os.path.join(stock, "modeling_esmfold2_common.py")).read()
        missing = [a for a in H.CONF_SOURCE_ANCHORS if a not in M] + [a for a in H.RELPOS_SOURCE_ANCHORS + H.LM_SOURCE_ANCHORS if a not in C]
        missing += [a for a in (H.DM_RETURN_ANCHOR, 'if inference_cache is not None and "z" in inference_cache:') if a not in C]
        missing += [a for a in ("distogram_logits = self.distogram_head(z + z.transpose(-2, -3))",) if a not in M]
        self.assertEqual(missing, [], f"anchors absent from the stock text: {missing}")


class NoSizeGates(unittest.TestCase):
    """R-TP-3: under --mode big --n_gpu P nothing switches to a replicated path by size. The adapter's modules carry no threshold that selects
    a replicated statement; the gate names of the GATE TABLE (FORBIDDEN below) are ABSENT from the kit (grep-able), and
    the RNG rows path is chosen by device capability, never by N."""
    FORBIDDEN = ("EF2_TP_MODE", "EF2_TP_E_MAX", "E_MAX", "EF2_TP_CONF_FINISH", "replicate_below", "REPLICATE_BELOW", "GATHER_BELOW", "CONF_TP_ABOVE")

    def test_no_inherited_size_gate_names(self):
        for fn in ("rowpair.py", "rowpair_heads.py"):
            txt = open(os.path.join(PKG, fn)).read()
            for name in self.FORBIDDEN:
                self.assertNotIn(name, txt, f"{fn} mentions gate {name}")

    def test_no_token_threshold_selects_a_path(self):
        txt = open(os.path.join(PKG, "rowpair_heads.py")).read()
        self.assertEqual(re.findall(r"if\s+N\s*[<>]=?\s*\d", txt), [])                      # no `if N < k:` style switch in the row statements
        self.assertIn('return "philox_rows" if', txt)                                        # the one mode choice is device capability, named


class P1InstallsNothing(unittest.TestCase):
    def test_p1(self):
        class Untouchable:
            def __getattr__(self, name):
                raise AssertionError(f"install(P=1) read model.{name}")
        rep = rowpair.install(Untouchable(), 1)
        self.assertFalse(rep["installed"]); self.assertEqual(rep["sharding"], "none"); self.assertEqual(rowpair.PATCHES.names(), [])


# ================================================================================================ the row statements vs the stock modules (P = 2, gloo)
def _cfg_small():
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    return ESMFold2Config()                                             # d_pair 256 / d_single 384: small N keeps the CPU cost low; real widths keep the statements honest


def _perturb(mod, seed):
    import torch
    with torch.no_grad():
        for p in mod.parameters():
            p.add_(torch.randn(p.shape, generator=torch.Generator().manual_seed(seed + p.numel() % 101)) * 0.05)


def _entry_relpos_lm(N: int, seed: int):
    import torch
    from opt_core.mem.rowpair import dist as RD, shard as RS
    CMN, MOD = _stock_or_skip()
    P, rank = RD.world()
    lay = RD.require_sharded(RD.Layout.auto(N, P, rank), "test")
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    enc = CMN.ResIdxAsymIdSymIdEntityIdEncoding(n_relative_residx_bins=32, n_relative_chain_bins=2, d_pair=256)
    _perturb(enc, seed)
    residue_index = torch.arange(N).view(1, N); residue_index[:, N // 2:] -= N // 2
    asym_id = torch.zeros(1, N, dtype=torch.long); asym_id[:, N // 2:] = 1
    sym_id = torch.zeros(1, N, dtype=torch.long); entity_id = torch.zeros(1, N, dtype=torch.long)
    token_index = torch.arange(N).view(1, N)
    with torch.no_grad():
        full = enc(residue_index=residue_index, asym_id=asym_id, sym_id=sym_id, entity_id=entity_id, token_index=token_index)
        rows = H.relpos_rows(enc, lay, residue_index=residue_index, asym_id=asym_id, sym_id=sym_id, entity_id=entity_id, token_index=token_index)
    rel = RS.unshard_rows(rows.contiguous(), lay, dim=1)
    out = {"relpos_bitwise": bool(torch.equal(rel, full)), "relpos_diff": (rel - full).abs().max().item()}
    shim = CMN.LanguageModelShim(d_z=256, d_model=64, num_layers=4)
    _perturb(shim, seed + 1)
    x = torch.randn((1, N, 5, 64), generator=g)
    with torch.no_grad():
        full = shim(x)
        rows = H.lm_pair_rows(shim, lay, x)
    got = RS.unshard_rows(rows.contiguous(), lay, dim=1)
    out.update(lm_bitwise=bool(torch.equal(got, full)), lm_diff=(got - full).abs().max().item(), lm_shape=list(got.shape))
    return out


def _confidence_case(N: int, S: int, seed: int):
    """The confidence-rows entries' shared case: a perturbed stock ConfidenceHead (chunk 16), random z / relpos / bonds, S structure samples,
    the stock head's own dense forward as the reference. Returns (torch, CMN, P, rank, lay, head, rows_inputs, kw, ref)."""
    import torch
    from opt_core.mem.rowpair import dist as RD
    CMN, MOD = _stock_or_skip()
    P, rank = RD.world()
    lay = RD.require_sharded(RD.Layout.auto(N, P, rank), "test")
    cfg = _cfg_small()
    torch.manual_seed(seed)
    head = MOD.ConfidenceHead(cfg).eval()
    _perturb(head, seed)
    head.set_chunk_size(16)
    g = torch.Generator().manual_seed(seed + 3)
    d_in = head.s_inputs_norm.normalized_shape[0]
    n_at = 3 * N
    s_inputs = torch.randn((1, N, d_in), generator=g)
    z = torch.randn((1, N, N, 256), generator=g) * 0.5
    rel = torch.randn((1, N, N, 256), generator=g) * 0.1
    tb = torch.randn((1, N, N, 256), generator=g) * 0.1
    x_pred = torch.randn((1, S, n_at, 3), generator=g) * 6.0
    atom_to_token = (torch.arange(n_at) // 3).view(1, n_at)
    disto_idx = (torch.arange(N) * 3 + 1).view(1, N)
    tok_mask = torch.ones((1, N), dtype=torch.bool); tok_mask[:, N - 2:] = False
    atm_mask = torch.ones((1, n_at), dtype=torch.bool); atm_mask[:, n_at - 6:] = False
    asym = torch.zeros((1, N), dtype=torch.long); asym[:, N // 2:] = 1
    mol = torch.zeros((1, N), dtype=torch.long)
    kw = dict(s_inputs=s_inputs, x_pred=x_pred, distogram_atom_idx=disto_idx, token_attention_mask=tok_mask, atom_to_token=atom_to_token,
              atom_attention_mask=atm_mask, asym_id=asym, mol_type=mol, num_diffusion_samples=S)
    with torch.no_grad():
        ref = head(z=z, relative_position_encoding=rel, token_bonds_encoding=tb, **kw)
    rows = dict(z=z[:, lay.r0:lay.r1].contiguous(), relative_position_encoding=rel[:, lay.r0:lay.r1], token_bonds_encoding=tb[:, lay.r0:lay.r1])
    return torch, CMN, P, rank, lay, head, rows, kw, ref


def _with_rows(torch, CMN, P, rank, lay, fn):
    """``fn(run_trunk)`` inside the rank's sharded context (STATE / CTX.layout / the TMB row dispatcher on the reference backend), restored after."""
    rowpair.STATE.update(P=P, rank=rank)
    rowpair.CTX.layout = lay

    def run_trunk(trunk, pair_rows, mask_rows):                         # the reference backend on CPU: the kit's einsum split via the TMB dispatcher
        with rowpair._sharded(lay, mask_rows.contiguous()):
            return trunk(pair_rows, pair_attention_mask=mask_rows)
    rowpair.PATCHES.replace(CMN.TriangleMultiplicativeBlock, "forward", rowpair._tmb_forward_dispatch)
    try:
        with torch.no_grad():
            return fn(run_trunk)
    finally:
        rowpair.PATCHES.restore(); rowpair.CTX.layout = None


def _entry_confidence(N: int, S: int, seed: int):
    """The confidence head on rows vs the stock head: z / relpos / bonds rows in, every output compared (pae gathered, logits rows)."""
    torch, CMN, P, rank, lay, head, rows, kw, ref = _confidence_case(N, S, seed)
    got = _with_rows(torch, CMN, P, rank, lay, lambda run_trunk: H.confidence_forward_rows(head, lay, run_trunk, **rows, **kw))
    out = {"rank": rank, "rows": (lay.r0, lay.r1)}
    for k in ("plddt", "complex_plddt", "complex_iplddt", "ptm", "iptm", "pair_chains_iptm", "plddt_logits", "resolved_logits"):
        out[k] = {"diff": (got[k].float() - ref[k].float()).abs().max().item(), "scale": ref[k].float().abs().max().item(),
                  "bitwise": bool(torch.equal(got[k], ref[k])), "shape_ok": tuple(got[k].shape) == tuple(ref[k].shape)}
    if rank == 0:                                                       # pae: the whole [S, N, N] on rank 0's host, None elsewhere
        out["pae"] = {"diff": (got["pae"] - ref["pae"].cpu()).abs().max().item(), "scale": ref["pae"].abs().max().item(),
                      "bitwise": bool(torch.equal(got["pae"], ref["pae"].cpu())), "shape_ok": tuple(got["pae"].shape) == tuple(ref["pae"].shape),
                      "device": str(got["pae"].device)}
    else:
        assert got["pae"] is None, type(got["pae"])
    for k in ("pae_logits", "pde_logits", "pde"):                       # rows: compare against the stock rows; the two logits keys are None when S > 1 (dropped by name)
        if got[k] is None:
            out[k] = {"dropped": True, "census": got["_rowpair"].get(k)}
            continue
        ref_rows = ref[k][:, lay.r0:lay.r1]
        out[k] = {"diff": (got[k] - ref_rows).abs().max().item(), "scale": ref[k].abs().max().item(),
                  "bitwise": bool(torch.equal(got[k], ref_rows)), "shape_ok": tuple(got[k].shape) == tuple(ref_rows.shape)}
    out["census"] = got["_rowpair"]
    return out


def _entry_confidence_loop(N: int, S: int, seed: int):
    """The per-sample confidence loop vs the batched statement: ``confidence_forward_rows`` (the statement once per sample when S > 1) and
    ``confidence_forward_rows_batched`` (one call at Bm = B·S) on the same rows, every key the loop keeps measured against the batched one
    (max |diff| and torch.equal); the two dropped logits keys reported as such. S == 1: the dispatcher IS the batched statement (no `samples`
    word in its census, identical outputs)."""
    torch, CMN, P, rank, lay, head, rows, kw, ref = _confidence_case(N, S, seed)
    got = _with_rows(torch, CMN, P, rank, lay, lambda run_trunk: H.confidence_forward_rows(head, lay, run_trunk, **rows, **kw))
    bat = _with_rows(torch, CMN, P, rank, lay, lambda run_trunk: H.confidence_forward_rows_batched(head, lay, run_trunk, **rows, **kw))
    out = {"rank": rank, "S": S, "keys_got": sorted(k for k in got if k != "_rowpair"), "keys_bat": sorted(k for k in bat if k != "_rowpair"),
           "samples": got["_rowpair"].get("samples"), "bat_samples": bat["_rowpair"].get("samples"), "conf_logits": got["_rowpair"].get("conf_logits"),
           "cmp": {}, "dropped": sorted(k for k in bat if k != "_rowpair" and bat[k] is not None and got.get(k) is None)}
    for k in out["keys_bat"]:
        g, b = got.get(k), bat[k]
        if b is None or g is None:
            out["cmp"][k] = {"equal": g is None and b is None, "diff": 0.0, "scale": 0.0, "shape_ok": True, "both_none": g is None and b is None}
            continue
        out["cmp"][k] = {"equal": bool(torch.equal(g, b)), "diff": (g.float() - b.float()).abs().max().item(), "scale": b.float().abs().max().item(),
                         "shape_ok": tuple(g.shape) == tuple(b.shape), "dtype_ok": g.dtype == b.dtype}
    out["vs_stock_pae_diff"] = (got["pae"] - ref["pae"].cpu()).abs().max().item() if rank == 0 else None
    return out


def _entry_token_transformer(N: int, S: int, seed: int):
    """The 12-block diffusion token transformer with local query rows vs the stock DiffusionTransformer (z = conditioned pair; s conditioning)."""
    import torch
    from opt_core.mem.rowpair import dist as RD
    CMN, MOD = _stock_or_skip()
    P, rank = RD.world()
    lay = RD.require_sharded(RD.Layout.auto(N, P, rank), "test")
    cfg = _cfg_small()
    torch.manual_seed(seed)
    sh = CMN.DiffusionStructureHead(cfg) if hasattr(CMN, "DiffusionStructureHead") else None
    dm = sh.diffusion_module if sh is not None else None
    if dm is None:
        return {"skip": "DiffusionStructureHead not constructible from the config here"}
    tt = dm.token_transformer.eval()
    _perturb(tt, seed)
    g = torch.Generator().manual_seed(seed + 5)
    d_model = tt.attn_blocks[0].d_model
    c_z = tt.attn_blocks[0].pair_norm.normalized_shape[0]
    a = torch.randn((S, N, d_model), generator=g)
    z = torch.randn((1, N, N, c_z), generator=g) * 0.5
    s = torch.randn((S, N, tt.attn_blocks[0].adaln.s_gate.in_features), generator=g) if hasattr(tt.attn_blocks[0], "adaln") else None
    mask = torch.ones((1, N), dtype=torch.bool); mask[:, N - 3:] = False
    with torch.no_grad():
        ref, _ = tt(a, s, z, beta=0.0, attention_mask=mask, num_diffusion_samples=S)
        box = {}
        from opt_core.mem.rowpair import diffusion as RDF
        box["sched"] = RDF.DiffusionSchedule.decide(lay, c_z=c_z, c_in=2 * c_z, c_cond=c_z, H=int(tt.attn_blocks[0].num_heads), S=S, n_blocks=len(tt.attn_blocks))
        box["bias_cache"] = RDF.PairBiasCache(enabled=box["sched"].bias_cache)
        got = H.token_transformer_rows(tt, lay, box, a.clone(), s, z[:, lay.r0:lay.r1].contiguous(), attention_mask=mask, num_diffusion_samples=S)
    return {"rank": rank, "diff": (got - ref).abs().max().item(), "scale": ref.abs().max().item(), "bitwise": bool(torch.equal(got, ref)),
            "shape_ok": tuple(got.shape) == tuple(ref.shape), "sched": dict(box["sched"].fields()), "bias_cache_hits": box["bias_cache"].hits}


def _entry_conditioning(N: int, seed: int):
    """pair_cond_rows over the stock DiffusionConditioning's z path vs the stock module's z (fresh cache)."""
    import torch
    from opt_core.mem.rowpair import dist as RD, shard as RS, diffusion as RDF
    CMN, MOD = _stock_or_skip()
    P, rank = RD.world()
    lay = RD.require_sharded(RD.Layout.auto(N, P, rank), "test")
    cfg = _cfg_small()
    torch.manual_seed(seed)
    sh = CMN.DiffusionStructureHead(cfg)
    cond = sh.diffusion_module.conditioning.eval()
    _perturb(cond, seed)
    g = torch.Generator().manual_seed(seed + 9)
    c_z = 256
    z = torch.randn((1, N, N, c_z), generator=g) * 0.5
    rel = torch.randn((1, N, N, c_z), generator=g) * 0.1
    s_inputs = torch.randn((1, N, cond.s_input_norm.normalized_shape[0]), generator=g)
    t = torch.full((1,), 3.0)
    with torch.no_grad():
        cache_ref = {}
        _s_ref, z_ref = cond(t_hat=t, s_inputs=s_inputs, s_trunk=None, z_trunk=z, relative_position_encoding=rel, sigma_data=16.0, num_diffusion_samples=1,
                             inference_cache=cache_ref)
        rel_rows = rel[:, lay.r0:lay.r1]

        def embed_fn(zb, g0, g1):
            i0, i1 = g0 - lay.r0, g1 - lay.r0
            return cond.z_proj(cond.z_input_norm(torch.cat([zb.float(), rel_rows[:, i0:i1].float()], dim=-1)))

        def tr(block):
            return lambda zb, g0, g1: block(zb)
        z_rows = RDF.pair_cond_rows(embed_fn, z[:, lay.r0:lay.r1].contiguous(), lay, c_out=int(cond.z_proj.out_features), rows=8,
                                    transitions=[tr(b) for b in cond.z_transitions])
    got = RS.unshard_rows(z_rows.contiguous(), lay, dim=1)
    return {"rank": rank, "diff": (got - z_ref).abs().max().item(), "scale": z_ref.abs().max().item(), "bitwise": bool(torch.equal(got, z_ref))}


class RowStatementsVsStock(unittest.TestCase):
    """P = 2 CPU ranks (gloo); every row statement compared with the stock module it re-issues, in fp32."""

    ROWS_TOL = 2e-4      # rows vs the dense stock statement per key: |diff| <= ROWS_TOL·max(1, |stock|) (fp32 CPU; row-blocked reductions)

    @classmethod
    def setUpClass(cls):
        _stock_or_skip()

    def test_relpos_and_lm_pair_rows_are_the_stock_rows(self):
        res = _mp(2, _entry_relpos_lm, 61, 3)
        print(f"[test] relpos rows: bitwise={res['relpos_bitwise']} diff={res['relpos_diff']:.2e}; lm pair rows: bitwise={res.get('lm_bitwise')} diff={res.get('lm_diff')}")
        self.assertTrue(res["relpos_bitwise"], res)
        if res.get("lm_bitwise") is not None:
            self.assertLessEqual(res["lm_diff"], 1e-5, res)

    def test_confidence_head_rows_match_the_stock_head(self):
        for S in (1, 2):
            res = _mp(2, _entry_confidence, 46, S, 5)
            worst = {k: (v["diff"], v["bitwise"]) for k, v in res.items() if isinstance(v, dict) and "diff" in v}
            print(f"[test] confidence rows N=46 S={S} P=2: {worst} census={res['census']}")
            for k, v in res.items():
                if isinstance(v, dict) and v.get("dropped"):
                    self.assertGreater(S, 1, (k, v)); self.assertIn(k, H.CONF_LOGITS_DROPPED); self.assertEqual(v["census"], "dropped", (k, v))
                elif isinstance(v, dict) and "diff" in v:
                    self.assertTrue(v["shape_ok"], (k, v))
                    self.assertLessEqual(v["diff"], self.ROWS_TOL * max(1.0, v["scale"]), (S, k, v))

    LOOP_TOL = 1e-6      # loop vs batched per key: |diff| <= LOOP_TOL·max(1, |batched|). The statements are per-row / per-sample, but their GEMMs / the
                         # stock `_categorical_mean` GEMV run on FLATTENED leading dims ([B·S·rows, ·] batched vs [B·rows, ·] per sample) and the BLAS
                         # picks its blocking by that leading size: fp32 summation grouping may differ at the ulp level (measured: plddt_per_atom /
                         # plddt_ca 1.2e-7, every other kept key bitwise on the CPU box; torch.equal per key is printed, not demanded, for S > 1)

    def test_confidence_per_sample_loop_equals_the_batched_statement(self):
        """num_diffusion_samples > 1 runs the confidence statement once per sample: at S in {2, 3} every key the loop keeps equals the batched
        statement's per-sample numbers within LOOP_TOL (torch.equal reported per key), `pae_logits` / `pde_logits` are the dropped keys (None,
        census conf_logits=dropped), the schedule census says loop:<S>; at S = 1 the dispatcher is the batched statement itself: every key
        bitwise, no `samples` word."""
        for S in (1, 2, 3):
            res = _mp(2, _entry_confidence_loop, 46, S, 5)
            worst = {k: (v["equal"], v["diff"]) for k, v in res["cmp"].items()}
            print(f"[test] confidence per-sample loop N=46 S={S} P=2: samples={res['samples']} dropped={res['dropped']} {worst}")
            self.assertEqual(res["keys_got"], res["keys_bat"], res)                       # the same output keys as the batched statement
            self.assertIsNone(res["bat_samples"])
            self.assertEqual(res["samples"], None if S == 1 else f"loop:{S}", res)
            self.assertEqual(res["dropped"], [] if S == 1 else sorted(H.CONF_LOGITS_DROPPED), res)
            self.assertEqual(res["conf_logits"], None if S == 1 else "dropped", res)
            for k, v in res["cmp"].items():
                self.assertTrue(v["shape_ok"], (S, k, v))
                self.assertTrue(v.get("dtype_ok", True), (S, k, v))
                if S == 1:
                    self.assertTrue(v["equal"], (S, k, v))                                # one statement, one call: bitwise
                else:
                    self.assertLessEqual(v["diff"], self.LOOP_TOL * max(1.0, v["scale"]), (S, k, v))
            self.assertLessEqual(res["vs_stock_pae_diff"], self.ROWS_TOL * 32.0, res)     # and the stock head's numbers (the rows tolerance above at the pae scale, 32 A)

    def test_token_transformer_local_queries_match_the_stock_transformer(self):
        for S in (1, 3):
            res = _mp(2, _entry_token_transformer, 40, S, 7)
            if "skip" in res:
                raise unittest.SkipTest(res["skip"])
            print(f"[test] token transformer rows N=40 S={S} P=2: diff={res['diff']:.2e} scale={res['scale']:.2f} bitwise={res['bitwise']} sched={res['sched']} cache_hits={res['bias_cache_hits']}")
            self.assertTrue(res["shape_ok"]); self.assertLessEqual(res["diff"], self.ROWS_TOL * max(1.0, res["scale"]), res)

    def test_pair_conditioning_rows_match_the_stock_conditioning(self):
        res = _mp(2, _entry_conditioning, 44, 9)
        print(f"[test] conditioning rows N=44 P=2: diff={res['diff']:.2e} bitwise={res['bitwise']}")
        self.assertLessEqual(res["diff"], 1e-4 * max(1.0, res["scale"]), res)


if __name__ == "__main__":
    unittest.main()

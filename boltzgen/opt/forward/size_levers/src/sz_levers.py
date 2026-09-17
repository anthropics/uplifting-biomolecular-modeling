# sz_levers.py — single-GPU size levers for BoltzGen 0.3.2 (environment-driven; imported by every kit mode)
#   SZ_TD_CHUNK=<rows>     : TokenDistanceModule feature construction row-chunked. Stock builds, every trunk pass and in fp32, a
#                            dense [B,4,N,N,161] tensor (backbone-atom distance gaussians + rel-pos encoding) -> Transition ->
#                            [N,N,512] -> cat distogram one-hot -> [N,N,550] -> a_proj. That transient is ~N^2 x 11 KB (78 GB
#                            single allocation at 5,500 tokens) and is the first thing that OOMs. Row-chunking is the same
#                            per-pair math (row-independent linear ops); only GEMM shapes change. bitwise.
#   oom trace (always on)  : a CUDA OOM raised inside Boltz.forward prints the boltzgen frames + a `[sz] {"event": "oom", ...}`
#                            site line and re-raises the same exception — evidence only (WHERE it happened), upstream's control
#                            flow unchanged: its own predict_step catches `RuntimeError` text-matching "out of
#                            memory", warns (`ran out of memory, skipping batch`), empty_cache()s and returns {"exception": True},
#                            exactly as it does without the levers. Armed in every kit mode (this module is on every kit mode's
#                            path; SZ_TD_CHUNK is set in big only). HOW MANY batches upstream skipped is the run's design census
#                            (`[boltzgen-opt] DESIGNS … oom_skipped=<k>`, counted from upstream's own line by the caller).
#
# STATS (fail-loud: a lever whose switch is set but that never actually fires is a NAMED event the caller gates the run on, never
# silence): `<name>_enabled` flips True the instant install() patches the class (boltzgen_opt's registry "stats" probe reads this at
# activation time, before any design has run); `td_calls` / `td_rows`
# accumulate DURING real forward passes (a per-design census a caller
# reads after a run — an activation-time flag does not by itself mean the lever fired on this design); `<name>_site` names the exact
# class.method install() patched, set the moment the patch lands (evidence beside the flag, not a substitute for it).
STATS = {
    "td_enabled": False, "td_calls": 0, "td_rows": None, "td_site": None,
    "oom_trace_enabled": False, "oom_trace_site": None, "oom_hit": False,
}
_INSTALLED = False   # install() is idempotent: boltzgen_opt's activation only imports this module (no separate apply step), so a second
                     # import (or an explicit second call) must not double-wrap an already-patched method.

import json, os, traceback
import torch


def _require(cls, attr, where):
    """Fail loud by name when the target symbol is absent — a version/shape mismatch upstream, never a silent no-op: setting a
    class attribute that doesn't already exist SUCCEEDS in Python (it creates a new one instead of patching), which would hide
    a stale hook target as an inert extra attribute rather than refusing."""
    if not hasattr(cls, attr):
        raise AttributeError(f"sz_levers: {where} has no {attr!r} — version/shape mismatch, refusing a silent no-op patch")


def _td_forward_chunked(CH, row_getter=None):
    from torch.nn.functional import one_hot
    def forward(self, z, feats, pair_mask, relative_position_encoding, use_kernels=False):
        STATS["td_calls"] += 1
        STATS["td_rows"] = CH
        token_distance_mask = feats["token_distance_mask"]
        token_coords = feats["center_coords"]
        rows = row_getter(z) if row_getter is not None else None
        with torch.autocast(device_type="cuda", enabled=False):
            dists = torch.cdist(token_coords, token_coords)
            boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1).to(dists.device)
            B, N = dists.shape[0], dists.shape[1]
            R0, R1 = (rows if rows is not None else (0, N))
            enc = self.token_distance_encoder if self.use_token_distance_feats else None
            if enc is not None:
                r = feats["coords"]; t2b = feats["token_to_bb4_atoms"]
                r_repr = torch.bmm(t2b.float().view(B, N * 4, -1), r.view(B, -1, 3)).reshape(B, N, 4, 3).permute(0, 2, 1, 3)  # [B,4,N,3]
            a_ij = None
            for i0 in range(R0, R1, CH):
                i1 = min(R1, i0 + CH)
                dg = (dists[:, i0:i1, :, None] > boundaries).sum(dim=-1).long()
                dg = one_hot(dg, num_classes=self.num_bins)
                if enc is not None:
                    d = (r_repr[:, :, i0:i1, None, :] - r_repr[:, :, None, :, :]).norm(dim=-1).unsqueeze(-1)  # [B,4,nb,N,1]
                    dgau = enc.distance_gaussian_smearing(d)
                    ro = i0 - R0 if relative_position_encoding.shape[1] != N else i0
                    rel = relative_position_encoding[:, ro:ro + (i1 - i0)].reshape(B, 1, i1 - i0, N, -1).expand(-1, 4, -1, -1, -1)
                    inp = torch.cat((dgau, d, rel), dim=-1)
                    feat = enc.distance_token_bias_trans(inp).permute(0, 2, 3, 4, 1).reshape(B, i1 - i0, N, -1)
                    a = torch.cat([dg, feat], dim=-1); del inp, feat, dgau, d
                else:
                    a = dg
                a = a * token_distance_mask[:, i0:i1].unsqueeze(-1)
                a = self.a_proj(a)
                if a_ij is None:
                    a_ij = torch.empty(B, R1 - R0, N, a.shape[-1], dtype=a.dtype, device=a.device)
                a_ij[:, i0 - R0:i1 - R0] = a
                del a, dg
        (B,) = a_ij.shape[:1]
        v = self.z_proj(self.z_norm(z)) + a_ij
        del a_ij
        v = v.view(B, *v.shape[1:])
        v = v + self.pairformer(v, pair_mask, use_kernels=use_kernels)
        v = self.v_norm(v)
        v = v.view(B, *v.shape[1:])
        u = self.u_proj(self.relu(v))
        return u
    return forward


def install():
    """Idempotent: a second call (a second import, or an explicit re-call) is a no-op that changes nothing and re-raises
    nothing — the guard is a flag checked first, not a try/except around an already-broken double-wrap. Each patch target is
    checked present (`_require`) before it is replaced; a target that is not there raises by name instead of silently
    creating a fresh, never-called attribute."""
    global _INSTALLED
    if _INSTALLED:
        return
    td = int(os.environ.get("SZ_TD_CHUNK", "0") or 0)
    if td > 0:
        from boltzgen.model.modules.trunk import TokenDistanceModule
        _require(TokenDistanceModule, "forward", "boltzgen.model.modules.trunk.TokenDistanceModule")
        TokenDistanceModule.forward = _td_forward_chunked(td)
        STATS["td_enabled"] = True
        STATS["td_site"] = "boltzgen.model.modules.trunk.TokenDistanceModule.forward"
        print(f"[levers] TokenDistanceModule row-chunked, rows/block={td}", flush=True)
    # the OOM trace: armed in every kit-mode process — evidence only, upstream's control flow unchanged
    import boltzgen.model.models.boltz as _bz
    _require(_bz.Boltz, "forward", "boltzgen.model.models.boltz.Boltz")
    _of = _bz.Boltz.forward
    def _fwd(self, *a, **k):
        try:
            return _of(self, *a, **k)
        except torch.cuda.OutOfMemoryError as e:
            tb = traceback.extract_tb(e.__traceback__)
            fr = [f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in tb if any(s in f.filename for s in ("boltzgen", "bg_", "xa_", "sz_", "cuequivariance"))]
            STATS["oom_hit"] = True; STATS["oom_events"] = STATS.get("oom_events", 0) + 1
            print("[sz] " + json.dumps({"event": "oom", "where": " <- ".join(fr[-6:][::-1]), "msg": str(e).split("\n")[0][:300],
                                        "max_alloc_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}), flush=True)
            raise                       # the same OutOfMemoryError, to upstream's own handler (predict_step: warn, skip the batch) — the line above is the evidence
    _bz.Boltz.forward = _fwd
    STATS["oom_trace_enabled"] = True
    STATS["oom_trace_site"] = "boltzgen.model.models.boltz.Boltz.forward"
    _INSTALLED = True


def report() -> dict:
    """The per-design census the caller's lever gate reads: a lever whose switch is set but whose STATS shows zero calls by
    the end of a design is a named event, never silence (the caller compares against which switches it exported)."""
    return dict(STATS)


# installs on import: boltzgen_opt's activation only does `importlib.import_module(m)` (no separate apply step, the same route as
# sitecustomize -> bg_hook), so this module patches itself in as the last statement below.
install()

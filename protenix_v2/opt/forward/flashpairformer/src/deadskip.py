"""
deadskip.py — checkpoint-keyed EXACT skip of numerically dead sub-modules.
probe()/install() distinguish a probe EXCEPTION (forward cannot run on this process, e.g. a lever Triton kernel OutOfResources on sm_120/sm_89 smem) from a
probe MISMATCH (ran, output != 0 => not dead on this stack); install() prints and records (install.last_report) the per-module reason; no all-N assert anywhere —
callers must NOT assert len(skipped) == len(manifest): print the report instead.

A module instance is 'near-dead' when EVERY parameter tensor has |w|max < THR (1e-30; Protenix v2 has ~41 such instances with |w| ~5e-37, none exactly 0).
It is SKIPPABLE only if a load-time numerical PROBE on THIS process shows its forward output is exactly 0 (bit pattern: all +0.0 or -0.0; and x + out == x bitwise)
for realistic random inputs at two sizes. The manifest records (checkpoint sha256, module path, class, probe result); the hook re-probes at load and never trusts an
index list alone (rule).

API
  manifest = build_manifest(model, ckpt_path, sizes=(64, 160), device="cuda")      # probe every near-dead instance; returns dict (json-serialisable)
  n = install(model, manifest, ckpt_path, reprobe=True)                            # wraps forward of every PASSED instance: returns zeros shaped like the stock output
  remove(model)                                                                    # restores stock forwards
Skipped forward returns torch.zeros of the stock OUTPUT shape/dtype (so `s = s + apb(...)`, `z = z + op(z)` and in-place `z += op(z)` all stay exact and shape-correct);
it does NOT try to elide the residual add itself (that is the block path's business; adding +0.0 is bitwise equality for every finite x incl. -0.0+0.0=+0.0 —
note: x=-0.0 would become +0.0; real activations contain no -0.0 (checked on the dumps) and torch.equal treats them equal anyway).
"""
from __future__ import annotations
import hashlib, json, os, re, time
import torch

THR = 1e-30
KINDS = ("attention_pair_bias", "single_transition", "pair_transition", "tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end",
         "outer_product_mean_msa", "msa_pair_weighted_averaging", "msa_transition", "transition_m")


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()


def near_dead_instances(model, thr=THR):
    """{module_path: (module, kind, max|w|)} for instances whose every parameter has |w|max < thr."""
    out = {}
    for name, mod in model.named_modules():
        kind = name.split(".")[-1]
        if kind not in KINDS: continue
        ps = [p for p in mod.parameters(recurse=True)]
        if not ps: continue
        mx = max(float(p.detach().float().abs().max()) for p in ps)
        if mx < thr: out[name] = (mod, kind, mx)
    return out


def _width(*cands):
    """first usable channel width from candidate LayerNorm modules / ints"""
    for c in cands:
        if c is None: continue
        if isinstance(c, int): return c
        ns = getattr(c, "normalized_shape", None)
        if ns is not None:
            return int(ns[0] if isinstance(ns, (tuple, list, torch.Size)) else ns)
        w = getattr(c, "weight", None)
        if w is not None: return int(w.shape[-1])
    raise AttributeError("width")


def _probe_inputs(kind, mod, N, device, dtype, gen):
    """realistic random inputs (magnitudes like the real trunk: s std~30, z std~30-60) for each kind; returns (args, kwargs, residual_base)"""
    def rnd(*shape, scale):
        return (torch.randn(*shape, generator=gen, device=device, dtype=torch.float32) * scale).to(dtype)
    if kind == "attention_pair_bias":
        c_a = _width(getattr(mod, "layernorm_a", None), getattr(getattr(mod, "attention", None), "c_q", None))
        c_z = _width(getattr(mod, "layernorm_z", None), getattr(getattr(mod, "linear_nobias_z", None), "in_features", None))
        s = rnd(N, c_a, scale=30.0); z = rnd(N, N, c_z, scale=40.0)
        return (), dict(a=s, s=None, z=z), s
    if kind in ("single_transition", "pair_transition", "msa_transition", "transition_m"):
        c = _width(getattr(mod, "layernorm1", None), getattr(mod, "c_in", None))
        x = rnd(N, c, scale=30.0) if kind == "single_transition" else rnd(N, N, c, scale=40.0)
        return (x,), {}, x
    if kind in ("tri_mul_out", "tri_mul_in"):
        c = _width(getattr(mod, "layer_norm_in", None), getattr(mod, "layernorm_in", None), getattr(mod, "c_z", None))
        z = rnd(N, N, c, scale=40.0); mask = torch.ones(N, N, device=device, dtype=dtype)
        return (z.clone(),), dict(mask=mask, inplace_safe=False, _add_with_inplace=False), z
    if kind in ("tri_att_start", "tri_att_end"):
        c = _width(getattr(mod, "layer_norm", None), getattr(mod, "c_in", None))
        z = rnd(N, N, c, scale=40.0); mask = torch.ones(N, N, device=device, dtype=dtype)
        return (z,), dict(mask=mask, inplace_safe=False, chunk_size=None), z
    if kind == "outer_product_mean_msa":
        m = rnd(8, N, _width(getattr(mod, "layer_norm", None), getattr(mod, "layernorm_m", None), getattr(mod, "c_m", None)), scale=10.0); return (m,), dict(mask=None, chunk_size=None, inplace_safe=False), None
    if kind == "msa_pair_weighted_averaging":
        m = rnd(8, N, _width(getattr(mod, "layernorm_m", None), getattr(mod, "c_m", None)), scale=10.0); z = rnd(N, N, _width(getattr(mod, "layernorm_z", None), getattr(mod, "c_z", None)), scale=40.0); return (m, z), {}, m
    raise KeyError(kind)


def probe(mod, kind, sizes=(64, 160), device="cuda", dtype=torch.bfloat16, seed=0, autocast=True):
    """Run the module on realistic random inputs; PASS iff output is exactly zero (all elements ==0) at every size AND base + out is bitwise base."""
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    rows = []; ok = True; n_exc = 0; n_mismatch = 0
    for N in sizes:
        try:
            args, kw, base = _probe_inputs(kind, mod, N, device, dtype, gen)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                try:
                    out = mod(*args, **kw)
                except TypeError:
                    kw2 = {k: v for k, v in kw.items() if k not in ("_add_with_inplace", "inplace_safe", "chunk_size")}
                    out = mod(*args, **kw2)
            if isinstance(out, (tuple, list)): out = out[0]
            absmax = float(out.float().abs().max()); allzero = bool((out == 0).all()); negzero = bool(torch.signbit(out).any())
            ident = None
            if base is not None and base.shape == out.shape:
                bsum = base + out.to(base.dtype)
                ident = bool(torch.equal(bsum.view(torch.int16) if bsum.dtype in (torch.bfloat16, torch.float16) else bsum, base.view(torch.int16) if base.dtype in (torch.bfloat16, torch.float16) else base))
            r = {"N": N, "out_shape": list(out.shape), "out_dtype": str(out.dtype), "absmax": absmax, "all_zero": allzero, "has_neg_zero": negzero, "base_plus_out_bitwise_base": ident}
            good = allzero and (ident in (True, None)); ok &= good; n_mismatch += (not good)
            r["status"] = "PASS" if good else "MISMATCH"
        except Exception as e:   # the module's (possibly lever-patched) forward could not RUN here -> says nothing about deadness
            r = {"N": N, "status": "EXCEPTION", "error": (type(e).__name__ + ": " + str(e))[:300]}; ok = False; n_exc += 1
        rows.append(r)
    # verdict vocabulary: PASS = output exactly 0 at every size; MISMATCH = ran but output != 0 (module NOT dead on this stack -> stock);
    # EXCEPTION = forward raised (e.g. a lever kernel that cannot launch on this GPU: Triton OutOfResources/smem) -> 'kernel cannot run here', NOT 'module not dead'.
    probe.last_status = "PASS" if ok else ("EXCEPTION" if n_exc and not n_mismatch else "MISMATCH")
    return ok, rows


def build_manifest(model, ckpt_path, sizes=(64, 160), device="cuda", thr=THR):
    t0 = time.time()
    inst = near_dead_instances(model, thr)
    man = {"format": "deadskip_manifest/v1", "created": time.strftime("%FT%TZ", time.gmtime()), "checkpoint_path": ckpt_path, "checkpoint_sha256": sha256_file(ckpt_path),
           "thr_absmax": thr, "probe_sizes": list(sizes), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__, "modules": []}
    for name, (mod, kind, mx) in sorted(inst.items()):
        ok, rows = probe(mod, kind, sizes=sizes, device=device)
        man["modules"].append({"path": name, "class": type(mod).__name__, "kind": kind, "param_absmax": mx, "n_params": sum(p.numel() for p in mod.parameters()),
                               "probe_pass": ok, "probe_status": getattr(probe, "last_status", None), "probe_rows": rows})
        print(f"[deadskip] {name:70s} {kind:22s} |w|max {mx:.2e} probe={'PASS' if ok else 'FAIL'} " + " ".join(f"N{r.get('N')}:absmax={r.get('absmax', 'ERR')}" for r in rows), flush=True)
    man["n_near_dead"] = len(inst); man["n_probe_pass"] = sum(m["probe_pass"] for m in man["modules"]); man["seconds"] = round(time.time() - t0, 1)
    return man


_ORIG = {}


def _zeros_forward(mod, kind):
    def fwd(*args, **kw):
        # output shape == residual base shape for all skippable kinds (APB: a; transitions: x; trimul/triatt: z); dtype follows autocast like stock (bf16 under the production amp config)
        if kind == "attention_pair_bias":
            base = kw.get("a", args[0] if args else None)
        elif kind == "msa_pair_weighted_averaging":
            base = args[0] if args else kw.get("m")
        else:
            base = args[0] if args else next(iter(kw.values()))
        dt = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else base.dtype
        return torch.zeros(base.shape, device=base.device, dtype=dt)
    return fwd


def install(model, manifest, ckpt_path=None, reprobe=True, sizes=None, device="cuda"):
    """Wrap forward of every manifest module whose probe passed (and re-passes now if reprobe). Returns list of skipped paths."""
    if ckpt_path is not None:
        sha = sha256_file(ckpt_path)
        if sha != manifest["checkpoint_sha256"]:      # another checkpoint than the manifest's: named, never a refusal — every module is re-checked on the
            reprobe = True                             # loaded weights below (|w|max against the threshold, then the zero-output probe) before it is skipped
            print(f"[deadskip] checkpoint sha256 {sha[:12]} is not the manifest's {manifest['checkpoint_sha256'][:12]}: every manifest module is re-probed on the loaded weights "
                  f"(skipped only where its weights are still below {manifest['thr_absmax']:g} and its output is exactly zero)", flush=True)
    mods = dict(model.named_modules()); skipped = []; report = []
    def stay(m, reason, detail=""):
        report.append({"path": m["path"], "action": "STOCK", "reason": reason, "detail": detail})
        print(f"[deadskip] {m['path']}: left on STOCK — {reason}{(' — ' + detail) if detail else ''}", flush=True)
    for m in manifest["modules"]:
        if not m["probe_pass"]:
            stay(m, "MANIFEST_PROBE_" + str(m.get("probe_status") or "FAIL")); continue
        mod = mods.get(m["path"])
        if mod is None or type(mod).__name__ != m["class"]:
            stay(m, "CLASS_MISMATCH", f"expected {m['class']}, found {type(mod).__name__ if mod is not None else None}"); continue
        mx = max(float(p.detach().float().abs().max()) for p in mod.parameters())
        if mx >= manifest["thr_absmax"]:
            stay(m, "WEIGHTS_CHANGED", f"|w|max {mx:.3g} >= thr {manifest['thr_absmax']:g} (not the manifest's checkpoint?)"); continue
        if reprobe:
            ok, rows = probe(mod, m["kind"], sizes=tuple(sizes or manifest["probe_sizes"]), device=device)
            if not ok:
                st = getattr(probe, "last_status", "FAIL")
                if st == "EXCEPTION":   # forward could not run on this process (e.g. lever kernel OutOfResources on this GPU) -> NOT a deadness verdict
                    stay(m, "LIVE_PROBE_EXCEPTION (kernel cannot run here; module deadness undetermined — fix/guard the patched forward, then re-install)",
                         "; ".join(r.get("error", "") for r in rows if r.get("status") == "EXCEPTION")[:300])
                else:
                    stay(m, "LIVE_PROBE_MISMATCH (module output != 0 on this stack -> not dead here)",
                         "; ".join(f"N{r.get('N')}: absmax={r.get('absmax')} negzero={r.get('has_neg_zero')} ident={r.get('base_plus_out_bitwise_base')}" for r in rows if r.get("status") == "MISMATCH")[:300])
                continue
        _ORIG[m["path"]] = mod.forward
        mod.forward = _zeros_forward(mod, m["kind"])
        skipped.append(m["path"]); report.append({"path": m["path"], "action": "SKIPPED"})
    n_pass = sum(1 for m in manifest["modules"] if m["probe_pass"])
    print(f"[deadskip] installed: {len(skipped)}/{n_pass} manifest-pass modules skipped (EXACT zeros); "
          f"{sum(1 for r in report if r['action'] == 'STOCK')} left on stock: " + ", ".join(f"{r['path']} [{r['reason'].split(' ')[0]}]" for r in report if r["action"] == "STOCK"), flush=True)
    install.last_report = report
    return skipped


def remove(model):
    mods = dict(model.named_modules())
    for path, f in list(_ORIG.items()):
        if path in mods: mods[path].forward = f
        _ORIG.pop(path, None)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.environ.get("FPF_SPEC_DIR") or os.path.dirname(os.path.abspath(__file__)))   # the directory holding the fpf package (this file's own src/ by default)
    import fpf.reference as R
    runner = R.build_runner(); model = runner.model.eval()
    ckpt = os.path.join(os.environ["PROTENIX_ROOT_DIR"], "checkpoint", "protenix-v2.pt")
    man = build_manifest(model, ckpt, sizes=(64, 160))
    out = os.environ.get("OUTJSON", "deadskip_manifest.json"); json.dump(man, open(out, "w"), indent=1)
    print(f"near-dead {man['n_near_dead']}, probe PASS {man['n_probe_pass']}; wrote {out}", flush=True)
    # op test of the hook on real dumps: trunk pairformer blocks with skipped modules vs stock, bitwise
    dumps = os.environ.get("FPF_DUMPS") or sys.exit("deadskip self-test: set FPF_DUMPS=<directory holding acts_<N>tok.pt activation dumps>"); res = []
    for N in (356, 705):
        d = torch.load(f"{dumps}/acts_{N}tok.pt", map_location="cpu", weights_only=False)
        for key in ("pf_c1_b0", "pf_c10_b47"):
            s = d[key]["s"].cuda(); z = d[key]["z"].cuda(); pm = torch.ones(N, N, device="cuda", dtype=z.dtype)
            blocks = sorted({int(re.search(r"^pairformer_stack\.blocks\.(\d+)\.", m["path"]).group(1)) for m in man["modules"] if m["probe_pass"] and re.search(r"^pairformer_stack\.blocks\.(\d+)\.", m["path"])})
            with torch.no_grad(), R.amp():
                ref = {b: model.pairformer_stack.blocks[b](s=s.clone(), z=z.clone(), pair_mask=pm) for b in blocks}
            skipped = install(model, man, ckpt, reprobe=True)
            try:
                with torch.no_grad(), R.amp():
                    got = {b: model.pairformer_stack.blocks[b](s=s.clone(), z=z.clone(), pair_mask=pm) for b in blocks}
            finally:
                remove(model)
            eq = {b: (bool(torch.equal(ref[b][0], got[b][0])) and bool(torch.equal(ref[b][0].view(torch.int16), got[b][0].view(torch.int16))), bool(torch.equal(ref[b][1].view(torch.int16), got[b][1].view(torch.int16)))) for b in blocks}
            res.append({"N": N, "key": key, "n_skipped_modules": len(skipped), "blocks_tested": blocks, "all_s_bitwise": all(v[0] for v in eq.values()), "all_z_bitwise": all(v[1] for v in eq.values()), "per_block": {str(b): v for b, v in eq.items()}})
            print(f"[hook-selftest] N={N} {key}: skipped {len(skipped)} modules; {len(blocks)} trunk blocks: s bitwise {res[-1]['all_s_bitwise']} z bitwise {res[-1]['all_z_bitwise']}", flush=True)
    # confidence-head pairformer blocks op test (random realistic inputs; c_s/c_z of the head)
    try:
        ch = model.confidence_head.pairformer_stack
        cblocks = sorted({int(re.search(r"^confidence_head\.pairformer_stack\.blocks\.(\d+)\.", m["path"]).group(1)) for m in man["modules"] if m["probe_pass"] and m["path"].startswith("confidence_head.pairformer_stack")})
        gen = torch.Generator(device="cuda"); gen.manual_seed(1)
        for N in (160, 356):
            b0 = ch.blocks[cblocks[0]]
            c_z = _width(getattr(b0.tri_mul_out, "layer_norm_in", None), getattr(b0.tri_mul_out, "c_z", None)); c_s = _width(getattr(b0.attention_pair_bias, "layernorm_a", None), b0.attention_pair_bias.attention.c_q) if getattr(b0, "attention_pair_bias", None) is not None else 0
            z = (torch.randn(N, N, c_z, generator=gen, device="cuda") * 40).to(torch.bfloat16); s = (torch.randn(N, c_s, generator=gen, device="cuda") * 30).to(torch.bfloat16) if c_s else None
            pm = torch.ones(N, N, device="cuda", dtype=z.dtype)
            with torch.no_grad(), R.amp():
                ref = {b: ch.blocks[b](s=None if s is None else s.clone(), z=z.clone(), pair_mask=pm) for b in cblocks}
            skipped = install(model, man, ckpt, reprobe=False)
            try:
                with torch.no_grad(), R.amp():
                    got = {b: ch.blocks[b](s=None if s is None else s.clone(), z=z.clone(), pair_mask=pm) for b in cblocks}
            finally:
                remove(model)
            def eqt(a, b):
                if a is None and b is None: return True
                return a.shape == b.shape and bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
            eq = {b: (eqt(ref[b][0], got[b][0]), eqt(ref[b][1], got[b][1])) for b in cblocks}
            res.append({"confidence_head": True, "N": N, "blocks_tested": cblocks, "all_s_bitwise": all(v[0] for v in eq.values()), "all_z_bitwise": all(v[1] for v in eq.values()), "per_block": {str(b): v for b, v in eq.items()}})
            print(f"[hook-selftest confidence_head] N={N}: blocks {cblocks}: s bitwise {res[-1]['all_s_bitwise']} z bitwise {res[-1]['all_z_bitwise']} per-block {eq}", flush=True)
    except Exception as e:
        res.append({"confidence_head_error": repr(e)[:300]}); print("[hook-selftest confidence_head] ERROR", repr(e)[:300], flush=True)
    man["hook_selftest"] = res; json.dump(man, open(out, "w"), indent=1); print("done", flush=True)

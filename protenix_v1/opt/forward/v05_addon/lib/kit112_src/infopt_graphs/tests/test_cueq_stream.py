"""Empirical stream/capture certificate for prebuilt binary ops (cuEquivariance triangle ops, Protenix fast LayerNorm):
  (1) side-stream: run the op on a NON-default stream vs the default stream on identical inputs -> bitwise?
  (2) capture: capture the op into a CUDA graph, replay, vs eager on the default stream -> bitwise?
  (3) replay twice -> bitwise (the replay is deterministic for these ops)?
A stream-less launch does not error: it silently reads stale memory (Protenix fast_layernorm case).  The test produces
fresh random inputs ON THE SIDE STREAM (so a kernel enqueued on the legacy stream would see unwritten/stale inputs with high
probability) and compares against the default-stream result.  Names the kernel path in the report row.
Run: python test_cueq_stream.py [--n 150 300] -> prints a JSON report (also written to cueq_stream_cert.json)."""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from infopt_graphs.core import capture_context


def _cert(name, fn, make_inputs, n_repeat=3):
    """fn(*inputs) -> tensor; make_inputs(device) -> tuple of tensors (drawn with torch.randn on the CURRENT stream)."""
    rep = {"op": name}
    torch.manual_seed(0)
    ins = make_inputs()
    torch.cuda.synchronize()
    ref = fn(*ins).clone(); torch.cuda.synchronize()
    # (1) side stream, inputs produced on the side stream
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        ins_s = tuple(x.clone() * 1.0 for x in ins)  # written on the side stream
        out_s = fn(*ins_s)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    rep["side_stream_bitwise"] = bool(torch.equal(out_s, ref))
    rep["side_stream_max_abs_delta"] = float((out_s.float() - ref.float()).abs().max())
    # (2) capture + replay (static inputs = copies; the graph reads them)
    st = tuple(x.clone() for x in ins)
    g = torch.cuda.CUDAGraph()
    try:
        # warm-up on a side stream (torch docs pattern)
        s2 = torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s2):
            for _ in range(2):
                _ = fn(*st)
        torch.cuda.current_stream().wait_stream(s2); torch.cuda.synchronize()
        with capture_context(g, stream=s2):
            out_g = fn(*st)
        g.replay(); torch.cuda.synchronize()
        r1 = out_g.clone()
        rep["graph_replay_bitwise_vs_eager"] = bool(torch.equal(r1, ref))
        rep["graph_replay_max_abs_delta"] = float((r1.float() - ref.float()).abs().max())
        rep["graph_replay_finite"] = bool(torch.isfinite(r1).all())
        # (3) replay repeatability + new inputs
        g.replay(); torch.cuda.synchronize(); rep["replay_repeat_bitwise"] = bool(torch.equal(out_g, r1))
        torch.manual_seed(1); new = make_inputs()
        for a, b in zip(st, new): a.copy_(b)
        ref2 = fn(*new).clone(); torch.cuda.synchronize()
        g.replay(); torch.cuda.synchronize()
        rep["graph_new_inputs_bitwise_vs_eager"] = bool(torch.equal(out_g, ref2))
        rep["graph_new_inputs_max_abs_delta"] = float((out_g.float() - ref2.float()).abs().max())
    except Exception as e:  # noqa
        rep["capture_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    # eager run-to-run floor of the op itself
    torch.cuda.synchronize(); r_a = fn(*ins).clone(); r_b = fn(*ins).clone(); torch.cuda.synchronize()
    rep["eager_run_to_run_bitwise"] = bool(torch.equal(r_a, r_b))
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, nargs="+", default=[150, 300]); ap.add_argument("--out", default="cueq_stream_cert.json")
    args = ap.parse_args()
    dev = "cuda"; reports = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "ops": []}
    try:
        import cuequivariance_torch as cuet
        reports["cuequivariance_torch"] = cuet.__version__
    except Exception as e:
        cuet = None; reports["cuequivariance_torch"] = f"MISSING {e!r}"
    for N in args.n:
        c = 128
        if cuet is not None and hasattr(cuet, "triangle_multiplicative_update"):
            from cuequivariance_torch import triangle_multiplicative_update as tmu
            torch.manual_seed(0)
            # Protenix call (protenix/model/triangular/triangular.py ~L496): x = z[None] [1, N, N, c] bf16 (autocast), mask ones [1, N, N],
            # p_in = cat(linear_a_p.weight, linear_b_p.weight) [2c, c], g_in = cat(a_g, b_g) [2c, c], p_out/g_out [c, c], LN weights fp32 params
            w = {"in": torch.randn(2 * c, c, device=dev) * 0.05, "gin": torch.randn(2 * c, c, device=dev) * 0.05,
                 "out": torch.randn(c, c, device=dev) * 0.05, "gout": torch.randn(c, c, device=dev) * 0.05}
            ln_w = {k: torch.ones(c, device=dev) for k in ("norm_in", "norm_out")}
            ln_b = {k: torch.zeros(c, device=dev) for k in ("norm_in", "norm_out")}
            for direction in ("outgoing", "incoming"):
                def fn(x, mask, direction=direction):
                    return tmu(x, direction=direction, mask=mask, norm_in_weight=ln_w["norm_in"], norm_in_bias=ln_b["norm_in"],
                               p_in_weight=w["in"], g_in_weight=w["gin"], norm_out_weight=ln_w["norm_out"], norm_out_bias=ln_b["norm_out"],
                               p_out_weight=w["out"], g_out_weight=w["gout"], eps=1e-5)
                mk = lambda: (torch.randn(1, N, N, c, device=dev, dtype=torch.bfloat16), torch.ones(1, N, N, device=dev, dtype=torch.bfloat16))
                r = _cert(f"cuet.triangle_multiplicative_update[{direction}] N={N} bf16", fn, mk); r["N"] = N; reports["ops"].append(r); print(json.dumps(r), flush=True)
        if cuet is not None and hasattr(cuet, "triangle_attention"):
            from cuequivariance_torch import triangle_attention as tatt
            # Protenix call (protenix/model/triangular/layers.py ~L463): q,k,v [B, N, H, N, D] bf16 (after transpose(-2,-3)),
            # bias = biases[1].float() [B, 1, H, N, N] fp32, mask = (biases[0] == 0).bool() [B, N, 1, 1, N]; returns a tuple, [0] used
            H, D = 4, 32
            def fn_att(q, k, v, bias, mask):
                out = tatt(q, k, v, bias, mask=mask, scale=D ** -0.5)
                return out[0] if isinstance(out, (tuple, list)) else out
            mk = lambda: (torch.randn(1, N, H, N, D, device=dev, dtype=torch.bfloat16), torch.randn(1, N, H, N, D, device=dev, dtype=torch.bfloat16),
                          torch.randn(1, N, H, N, D, device=dev, dtype=torch.bfloat16), torch.randn(1, 1, H, N, N, device=dev, dtype=torch.float32),
                          torch.ones(1, N, 1, 1, N, device=dev, dtype=torch.bool))
            try:
                r = _cert(f"cuet.triangle_attention N={N} bf16", fn_att, mk); r["N"] = N; reports["ops"].append(r); print(json.dumps(r), flush=True)
            except Exception as e:
                reports["ops"].append({"op": f"cuet.triangle_attention N={N}", "error": f"{type(e).__name__}: {str(e)[:300]}"})
        # torch SDPA (cuDNN/flash) fp32 as in the Protenix denoiser attention
        def fn_sdpa(q, k, v, b):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=b, scale=1.0)
        mk = lambda: (torch.randn(1, 16, N * 8, 24, device=dev), torch.randn(1, 16, N * 8, 24, device=dev), torch.randn(1, 16, N * 8, 24, device=dev),
                      torch.randn(1, 16, N * 8, N * 8, device=dev))
        r = _cert(f"F.scaled_dot_product_attention fp32 bias [1,16,{N*8},24]", fn_sdpa, mk); r["N"] = N; reports["ops"].append(r); print(json.dumps(r), flush=True)
    # Protenix fast LayerNorm: original vs stream-correct rebuild (if protenix present)
    try:
        import protenix.model.layer_norm.layer_norm as LN
        orig = LN.fast_layer_norm_cuda_v2
        def fn_ln(x, w, b):
            return orig.forward_with_both_affine(x, (x.shape[-1],), w, b, 1e-5)[0]
        mk = lambda: (torch.randn(1302, 128, device=dev) * 3, torch.randn(128, device=dev), torch.randn(128, device=dev))
        r = _cert("protenix fast_layer_norm_cuda_v2 ORIGINAL (stream-less launch)", fn_ln, mk); reports["ops"].append(r); print(json.dumps(r), flush=True)
        from infopt_graphs.protenix.fastln_stream import install_stream_correct_fastln
        rep = install_stream_correct_fastln(require_bitwise=False)
        reports["fastln_stream_fix"] = {k: rep.get(k) for k in ("installed", "build_s", "patched_launches", "side_stream")} | {"bitwise_all_equal": rep["bitwise"]["all_bitwise_equal"]}
        patched = LN.fast_layer_norm_cuda_v2
        def fn_ln2(x, w, b):
            return patched.forward_with_both_affine(x, (x.shape[-1],), w, b, 1e-5)[0]
        r = _cert("protenix fast_layer_norm_cuda_v2 STREAM-CORRECT rebuild", fn_ln2, mk); reports["ops"].append(r); print(json.dumps(r), flush=True)
    except Exception as e:
        reports["fastln"] = f"skipped: {type(e).__name__}: {str(e)[:200]}"
    json.dump(reports, open(args.out, "w"), indent=1)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()

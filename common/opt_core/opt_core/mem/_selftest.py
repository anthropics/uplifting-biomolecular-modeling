"""CPU self-test of opt_core.mem (``python -m opt_core.mem._selftest``): every primitive on small tensors, torch on CPU (the GPU numerics
class of a memory line is the kit's equality record, not this test). Exits 0 with one ``[opt_core.mem] SELFTEST ok ...`` line, non-zero
with the failing check named. Integer-valued float64 inputs make every matmul sum exact, so the row-chunked assembly is compared
BIT-EXACT to the full evaluation (the primitive's indexing is what is under test); layer_norm row-split is compared to 1e-12."""
from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from . import Ledger, MemLeverRefused, env_flag, env_int, evidence_line, parse_levers, row_blocks
    from .graph_gate import decide, parse_cap
    from . import torch_alloc as A
    from .budget import nbytes, rows_within

    checks = []

    def ok(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        if not cond:
            print(f"[opt_core.mem] SELFTEST FAIL {name} {detail}", file=sys.stderr)

    # ---- pure python
    ok("row_blocks", row_blocks(10, 4) == [(0, 4), (4, 8), (8, 10)] and row_blocks(3, 8) == [(0, 3)] and row_blocks(0, 2) == [])
    ok("parse_levers", parse_levers("trans+cond", ("trans", "cond", "free")) == ("trans", "cond") and parse_levers("", ("a",)) == ()
       and parse_levers("all", ("a", "b"), default=("b",)) == ("b",))
    try:
        parse_levers("nope", ("a",))
        ok("parse_levers_refuses_unknown", False)
    except MemLeverRefused as exc:
        ok("parse_levers_refuses_unknown", exc.lever == "nope")
    ok("env_int", env_int({"X": " 12 "}, "X", 3) == 12 and env_int({}, "X", 3) == 3)
    ok("env_flag", env_flag({"F": "on"}, "F") is True and env_flag({"F": "0"}, "F", True) is False and env_flag({}, "F", True) is True)
    line = evidence_line("kit-opt", "xl_trans", rows=256, calls=3, patched=["Transition.forward"], alloc=None)
    ok("evidence_line", line == "[kit-opt] LEVER name=xl_trans state=on rows=256 calls=3 patched=Transition.forward alloc=none", line)
    line2 = evidence_line("kit-opt", "xl_cond", "skipped", reason="n_tok<min_tokens", n_tok=100)
    ok("evidence_line_skipped", line2 == "[kit-opt] LEVER name=xl_cond state=skipped reason=n_tok<min_tokens n_tok=100", line2)
    try:
        evidence_line("kit-opt", "xl_cond", "skipped")
        ok("evidence_line_skipped_needs_reason", False)
    except ValueError:
        ok("evidence_line_skipped_needs_reason", True)
    ok("graph_gate", decide(900, 1536).fragment == "graph=capture" and decide(2000, 1536).fragment == "graph=eager:n_tok>1536"
       and decide(10, 0).fragment == "graph=off" and decide(10 ** 6, None).capture and parse_cap("", 7) == 7 and parse_cap("off", 7) == 0
       and parse_cap("always", 7) is None and parse_cap("2048", 7) == 2048 and decide(5, 9, name="arena").facts()["arena_cap"] == 9)
    dby = decide(900, 1536, need_bytes=3 * 2 ** 30, budget_bytes=2 * 2 ** 30)
    ok("graph_gate_bytes", dby.mode == "eager" and dby.reason == "bytes>budget" and dby.fragment == "graph=eager:bytes>budget"
       and dby.facts()["gate"] == "bytes>budget" and decide(900, 1536).mode == "capture" and decide(10, 0).mode == "off"
       and decide(2000, 1536).facts()["gate"] == "n_tok>cap")
    # ---- allocator policy (no CUDA needed: environ dicts + explicit cuda_initialized)
    ok("conf_for", A.conf_for("expandable") == "expandable_segments:True"
       and A.conf_for("capped", 0.5) == "expandable_segments:True,garbage_collection_threshold:0.5" and A.env_row(None) == {})
    env = {}
    f1 = A.export("expandable", env, cuda_initialized=False)
    f2 = A.export("expandable", env, cuda_initialized=False)
    ok("export", env == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"} and f1["alloc_export"] == "exported" and f2["alloc_export"] == "present")
    for name, kw, envd in (("export_refuses_other_conf", {}, {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}),
                           ("export_refuses_cuda_initialized", {"cuda_initialized": True}, {}),
                           ("export_refuses_graphs_on", {"graphs_on": True}, {})):
        try:
            kw.setdefault("cuda_initialized", False)
            A.export("expandable", envd, **kw)
            ok(name, False)
        except MemLeverRefused as exc:
            ok(name, exc.lever == "alloc", exc.reason[:60])
    ok("export_graphs_allowed", A.export("capped", {}, graphs_on=True, allow_with_graphs=True, cuda_initialized=False)["alloc"] == "capped")
    ok("budget", rows_within(10 * 2 ** 20, 2 ** 20, 64, multiple=4) == 8 and rows_within(2 ** 40, 2 ** 20, 64) == 64 and nbytes((2, 3), 4) == 24)
    try:
        rows_within(100, 2 ** 20, 64, floor=1)
        ok("budget_refuses", False)
    except MemLeverRefused:
        ok("budget_refuses", True)

    # ---- torch (CPU)
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"[opt_core.mem] SELFTEST torch-unavailable ({type(exc).__name__}): pure-python checks only", file=sys.stderr)
        torch = None
    if torch is not None:
        from .torch_rowchunk import FreeList, PatchSet, free_storage, is_oom, pair_ok, rowchunk_apply, rowchunk_concat, rowchunk_tensor
        from .torch_hostpair import HostPair, LayerNormGuard, PinPool, PinRefused, ln_rowsplit
        torch.manual_seed(0)
        B, N, C, H = 1, 24, 8, 32
        x = torch.randint(-3, 4, (B, N, N, C)).to(torch.float64)
        w1 = torch.randint(-2, 3, (H, C)).to(torch.float64)
        w2 = torch.randint(-2, 3, (C, H)).to(torch.float64)

        def transition(t):                      # integer-valued in, integer-valued out: every sum exact -> bit-exact comparable
            h = torch.relu(t @ w1.t())
            return h @ w2.t() + t

        full = transition(x)
        led = Ledger()
        got = rowchunk_tensor(transition, x, 5, ledger=led, key="trans_rowchunked_calls")
        ok("rowchunk_tensor_bitwise", torch.equal(full, got) and led.facts() == {"trans_rowchunked_calls": 1})
        got3 = rowchunk_tensor(transition, x[0], 7)                                   # [N, N, C] form, default row_dim=-3
        ok("rowchunk_tensor_3d", torch.equal(full[0], got3))
        got32 = rowchunk_apply(lambda r0, r1: transition(x[:, r0:r1]).to(torch.float32), N, 9, out_dtype=torch.float64)
        ok("rowchunk_apply_out_dtype", got32.dtype == torch.float64 and torch.equal(got32, full.to(torch.float32).to(torch.float64)))
        prods = [lambda r0, r1, k=k: (x[:, r0:r1] * (k + 1))[..., :3] for k in range(4)]
        cat_full = torch.cat([(x * (k + 1))[..., :3] for k in range(4)], dim=-1)
        led2 = Ledger()
        cat_got = rowchunk_concat(prods, N, 5, out_dtype=torch.float32, ledger=led2, key="cond_rowchunked_calls")
        ok("rowchunk_concat", torch.equal(cat_got, cat_full.to(torch.float32)) and led2.get("cond_rowchunked_calls") == 1
           and led2.get("cond_rowchunked_calls_dtype") == "float32_from_float64")
        for name, fn in (("refuses_wrong_rows", lambda: rowchunk_apply(lambda r0, r1: x[:, r0:r1 + 1], N, 5)),
                         ("refuses_non_tensor", lambda: rowchunk_apply(lambda r0, r1: None, N, 5)),
                         ("refuses_out_mismatch", lambda: rowchunk_apply(lambda r0, r1: x[:, r0:r1], N, 5, out=torch.empty(2, 3)))):
            try:
                fn()
                ok(name, False)
            except MemLeverRefused as exc:
                ok(name, exc.lever == "rowchunk", exc.reason[:50])
        ok("pair_ok", pair_ok(x, 16) and not pair_ok(x, 32) and not pair_ok(x[..., :5, :], 1) and not pair_ok(torch.zeros(3, 4), 1)
           and pair_ok(x[0], 16))
        xg = x.clone().requires_grad_(True)
        ok("pair_ok_grad_guard", not pair_ok(xg, 1))
        with torch.no_grad():
            ok("pair_ok_no_grad", pair_ok(xg, 1))
        ok("is_oom", is_oom(RuntimeError("CUDA out of memory. Tried to allocate")) and not is_oom(ValueError("x")))

        class Mod(object):
            def forward(self):
                return "stock"
        ps = PatchSet("xl_trans")
        orig = ps.replace(Mod, "forward", lambda self: "lever")
        ok("patchset", Mod().forward() == "lever" and orig(Mod()) == "stock" and ps.original(Mod, "forward") is orig
           and ps.names() == ["Mod.forward"])
        try:
            ps.replace(Mod, "forward", lambda self: "twice")
            ok("patchset_refuses_double", False)
        except MemLeverRefused:
            ok("patchset_refuses_double", True)
        try:
            ps.replace(Mod, "nope", 1)
            ok("patchset_refuses_missing", False)
        except MemLeverRefused:
            ok("patchset_refuses_missing", True)
        ps.restore()
        ok("patchset_restore", Mod().forward() == "stock" and len(ps) == 0)

        class Base(object):
            def forward(self):
                return "base"

        class Sub(Base):
            pass
        ps2 = PatchSet("xl_x")
        ps2.replace(Sub, "forward", lambda self: "patched")
        ok("patchset_inherited", Sub().forward() == "patched" and "forward" in vars(Sub))
        ps2.restore()
        ok("patchset_restore_inherited_delattr", "forward" not in vars(Sub) and Sub().forward() == "base")

        keep = torch.zeros(1000)
        dead = {"a": torch.zeros(2000), "b": [torch.zeros(10), keep[10:20]]}           # keep[10:20] aliases a protected tensor
        fl, led3 = FreeList(), Ledger()
        n_reg = fl.register(dead, protect=(keep,), ledger=led3)
        freed = fl.release("seam", led3)
        ok("freelist", n_reg == 2 and freed == (2000 + 10) * 4 and dead["a"].untyped_storage().nbytes() == 0
           and keep.untyped_storage().nbytes() == 4000 and led3.get("free_skipped_aliased") == 1 and led3.get("free_events") == 1
           and led3.get("free_tensors") == 2 and fl.release("again", led3) == 0)
        ok("free_storage", free_storage(torch.zeros(8)) == 32)
        pz = torch.zeros(4, 4); pi = torch.zeros(3, dtype=torch.int32)
        flp, ledp = FreeList("xl_free", poison=True), Ledger()
        flp.register([pz, pi]); nbp = flp.release("poison", ledp)
        ok("freelist_poison", nbp == 64 + 12 and pz.untyped_storage().nbytes() == 64 and bool(torch.isnan(pz).all())
           and int(pi[0]) == torch.iinfo(torch.int32).max and ledp.get("free_poisoned") == 2 and ledp.get("free_mode") == "poison")

        pool = PinPool(1 << 16, pin=False, lever="selftest")              # the ONE pinned pool, CPU line: budget 64 KiB, plain host tensors
        pa = pool.alloc((64, 128), torch.float32, "a")                     # 32 KiB
        pb = pool.alloc((64, 128), torch.float32, "b")                     # 64 KiB == budget
        try:
            pool.alloc((1,), torch.float32, "c")
            refused = None
        except PinRefused as r:
            refused = r.kind
        pool.release(pa)
        ok("pinpool_budget", tuple(pb.shape) == (64, 128) and refused == "pin_budget_exceeded" and pool.bytes_now == 1 << 15
           and pool.bytes_peak == 1 << 16 and pool.n_alloc == 2)
        lax = PinPool(1024, pin=False, pageable=True, lever="selftest")
        _buf, denied = lax.alloc_counted((1024,), torch.float32, "big")
        ok("pinpool_pageable_counted", denied == "pin_budget_exceeded" and lax.n_pageable == 1 and lax.events[0]["kind"] == denied and lax.bytes_now == 0)

        z = torch.randn(N, N, C, dtype=torch.float32)
        led4 = Ledger()
        hp = HostPair.from_tensor(z, rows=7, device="cpu", ledger=led4)
        ok("hostpair_roundtrip", torch.equal(hp.to_tensor(), z) and torch.equal(hp.rows(3, 9), z[3:9]) and torch.equal(hp.cols(2, 5), z[:, 2:5])
           and hp.nbytes == N * N * C * 4)
        hp.put_rows(0, 2, torch.ones(2, N, C))
        hp.put_cols(5, 6, torch.full((N, 1, C), 2.0))
        ok("hostpair_put", float(hp.t[1, 0, 0]) == 1.0 and float(hp.t[N - 1, 5, 0]) == 2.0 and led4.get("d2h_bytes") > 0 and led4.get("h2d_bytes") > 0)
        try:
            hp.put_rows(0, 2, torch.ones(3, N, C))
            ok("hostpair_refuses_shape", False)
        except MemLeverRefused:
            ok("hostpair_refuses_shape", True)

        F = torch.nn.functional
        stock_ln = F.layer_norm
        xl = torch.randn(64, 16, dtype=torch.float64)
        wln, bln = torch.randn(16, dtype=torch.float64), torch.randn(16, dtype=torch.float64)
        ref = stock_ln(xl, (16,), wln, bln, 1e-5)
        led5 = Ledger()
        split = ln_rowsplit(lambda blk: stock_ln(blk, (16,), wln, bln, 1e-5), xl, limit=100, ledger=led5)
        ok("ln_rowsplit", torch.allclose(split, ref, rtol=0, atol=1e-12) and led5.get("ln_guard_hits") == 1,
           f"bitwise={torch.equal(split, ref)}")
        g = LayerNormGuard(limit=100)
        ok("ln_guard_install", g.install() is True and F.layer_norm is not stock_ln and LayerNormGuard(limit=5).install() is False)
        y = torch.nn.LayerNorm(16, dtype=torch.float64, elementwise_affine=False)(xl)   # the stock module routes through the guard
        ok("ln_guard_module_path", g.hits == 1 and torch.allclose(y, stock_ln(xl, (16,), None, None, 1e-5), atol=1e-12, rtol=0), f"hits={g.hits}")
        try:
            ln_rowsplit(lambda blk: blk, xl, limit=100, dim=1)                          # dim 1 is the normalized dim of a [64,16] input
            ok("ln_rowsplit_refuses_normalized_dim", False)
        except MemLeverRefused as exc:
            ok("ln_rowsplit_refuses_normalized_dim", exc.lever == "ln_guard", exc.reason[:60])
        x3 = torch.randn(8, 8, 16, dtype=torch.float64)
        r3 = ln_rowsplit(lambda blk: stock_ln(blk, (8, 16)), x3, limit=100, normalized_ndim=2)   # splits dim 0 only
        ok("ln_rowsplit_normalized_ndim", torch.allclose(r3, stock_ln(x3, (8, 16)), atol=1e-12, rtol=0))
        ok("ln_guard_remove", g.remove() is True and F.layer_norm is stock_ln)
        eff = A.effective()
        ok("alloc_effective_cpu", eff["expandable"] in (False, None) and isinstance(eff["source"], str), str(eff))

    n_ok = sum(1 for _, c, _ in checks if c)
    bad = [n for n, c, _ in checks if not c]
    verdict = "ok" if not bad else "FAIL"
    print(f"[opt_core.mem] SELFTEST {verdict} checks={len(checks)} passed={n_ok} torch={'cpu' if torch is not None else 'absent'}"
          + (f" failed={','.join(bad)}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

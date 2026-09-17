"""GPU tests of fpf_trimul_v4's warm numerics probe and its named lever states (run on a CUDA box with Triton:
`python -m pytest common/opt_core/tests/gpu/test_trimul_v4_probe_gpu.py -q -s`; skipped by name elsewhere). Each test runs in a FRESH interpreter
(the loader decides once per process):
(1) the certified key of this box: the first served call of a shape class runs the probe ONCE — err vs fp64 at most 1.25 x (rms) / 2.5 x (max)
    the bf16-autocast statement's — prints its line, records COUNTS['probe'], and the FIRST CALL line carries `cell=<row key>`;
(2) a table without this capability's rows: the SAFE cell is ENGAGED and NAMED (`safe settings served (no_cell:<cc>|<mm>, …)`,
    `cell=safe(uncertified <cc>|<mm>)`) and the probe passes on it;
(3) a launch cell that cannot build: the SAFE cell serves by name; the SAFE cell unable to build too: `cell=none:<why>`, the lever off for the
    process, `generic` raising `none:<why>` with .cannot_run — the caller's mode refuses by name (no stock route)."""
import json
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch", reason="needs torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
try:
    import triton  # noqa: F401
except ImportError:
    pytest.skip("needs Triton", allow_module_level=True)

CORE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRELUDE = textwrap.dedent("""
    import json, os, sys, torch
    sys.path.insert(0, %r)
    from opt_core.kernels.fpf_trimul_v4 import generic as G, cells as CELLS, table as T, kernels as K
    dev = torch.device("cuda")
    def call(C=128, D=128, has_bias=False, f32=False, N=192, seed=1):
        g = torch.Generator(device="cpu").manual_seed(seed)
        r = lambda *s: torch.randn(*s, generator=g)
        raw = dict(ln_in_w=1 + 0.1 * r(C), ln_in_b=0.1 * r(C), w_ag=r(D, C) * C ** -0.5, w_ap=r(D, C) * C ** -0.5, w_bg=r(D, C) * C ** -0.5, w_bp=r(D, C) * C ** -0.5,
                   ln_out_w=1 + 0.1 * r(D), ln_out_b=0.1 * r(D), w_o=r(C, D) * D ** -0.5, w_og=r(C, C) * C ** -0.5)
        if has_bias:
            raw.update(b_ag=0.1 * r(D), b_ap=0.1 * r(D), b_bg=0.1 * r(D), b_bp=0.1 * r(D), b_o=0.1 * r(C), b_og=0.1 * r(C))
        raw = {k: v.to(dev) for k, v in raw.items()}
        z = r(N, N, C).to(dev, torch.float32 if f32 else torch.bfloat16)
        return G.trimul(z, None, direction="outgoing", weights=G.pack_weights(**raw))
""" % CORE)


def _run(body, env=None):
    code = PRELUDE + textwrap.dedent(body)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900, env={**os.environ, **(env or {})})
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr


def test_certified_key_probe_passes_once_and_first_call_names_the_cell():
    res, err = _run("""
        out = call(); out2 = call(seed=2); outb = call(has_bias=True); outf = call(C=256, D=256)
        info = CELLS.INFO[str(torch.device("cuda", torch.cuda.current_device()))] if CELLS.INFO else list(CELLS.INFO.values())[0]
        print(json.dumps({"probe": G.COUNTS["probe"], "served": G.COUNTS["generic_served"], "unsupported": G.COUNTS["unsupported"], "key": info["key"],
                          "served_by": info["served_by"], "word": CELLS.cell_word(out.device), "finite": bool(torch.isfinite(out.float()).all())}))
    """)
    assert res["served"] == 4 and res["unsupported"] == {} and res["finite"] and res["served_by"] in ("core", "kit-table")
    probes = {k: v for k, v in res["probe"].items() if k != "deferred"}
    assert len(probes) == 3, probes                                                  # (128,128,nobias,bf16), (128,128,bias,bf16), (256,256,nobias,bf16): once each
    for k, v in probes.items():
        assert v["ok"] and v["ratio_rms"] <= 1.25 and v["ratio_max"] <= 2.5, (k, v)
    assert res["word"] == "cell=%s" % res["key"] + ("" if res["served_by"] == "core" else ":" + res["served_by"])
    assert err.count("[fpf_trimul_v4.generic] warm probe (") == 3 and err.count("-> served") == 3
    assert "FIRST CALL served:" in err and res["word"] in err and err.count("[fpf_trimul_v4] cell on cuda") == 1


def test_a_capability_without_rows_engages_the_safe_cell_by_name(tmp_path):
    cc = "%d.%d" % torch.cuda.get_device_capability(0)
    from opt_core.kernels.fpf_trimul_v4 import table as T
    from opt_core.kernels import safe_settings as SAFE
    if SAFE.safe_row("pair_fused:trimul", cc) is None:
        pytest.skip("no SAFE cell for cc %s: that capability is refused by name (no-cell), covered by the CPU tests" % cc)
    table = {k: v for k, v in T.load_table().items() if not k.startswith(cc + "|")}
    p = tmp_path / "table.json"; p.write_text(json.dumps(table))
    res, err = _run("""
        CELLS.TABLE_PATH = %r
        out = call()
        info = list(CELLS.INFO.values())[0]
        print(json.dumps({"probe": G.COUNTS["probe"], "served": G.COUNTS["generic_served"], "word": CELLS.cell_word(out.device), "net": CELLS.NET.word(),
                          "served_by": info["served_by"], "finite": bool(torch.isfinite(out.float()).all())}))
    """ % str(p))
    mm = res["net"].split("|")[-1]
    assert res["served"] == 1 and res["served_by"] == "safe" and res["net"] == "safe:no_cell:%s|%s" % (cc, mm) and res["word"] == "cell=safe(uncertified %s|%s)" % (cc, mm)
    assert all(v["ok"] for k, v in res["probe"].items() if k != "deferred") and res["finite"]
    assert err.count("[opt_core/pair_fused:trimul] safe settings served (no_cell:%s|%s, cc %s, triton %s)" % (cc, mm, cc, mm)) == 1


def test_a_cell_that_cannot_build_serves_safe_then_the_lever_goes_off_by_name():
    res, err = _run("""
        UNBUILDABLE = {"BM": 1024, "BN": 512, "num_warps": 64, "num_stages": 9}          # no GPU has the registers / shared memory / threads for this tile: a triton BUILD failure
        CELLS._CFG_OVERRIDE = None
        orig = T.load_table()
        bad = dict(orig); cc = "%d.%d" % torch.cuda.get_device_capability(0)
        for k in list(T.rows(orig)):
            if k.startswith(cc + "|"):
                bad[k] = dict(orig[k], k1=dict(UNBUILDABLE), k3=dict(UNBUILDABLE), overrides={})
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "table.json"); open(p, "w").write(json.dumps(bad)); CELLS.TABLE_PATH = p
        out = call()                                                                     # the tuned cell fails to build -> the SAFE cell serves, named
        rec = {"after_build_failure": {"word": CELLS.cell_word(out.device), "net": CELLS.NET.word(), "served": G.COUNTS["generic_served"], "finite": bool(torch.isfinite(out.float()).all())}}
        CELLS.safe_cfg = lambda cc: {"k1": dict(UNBUILDABLE), "k3": dict(UNBUILDABLE), "overrides": {}}   # now the SAFE cell cannot build either (a fresh shape class forces a build)
        CELLS._CELL_CACHE.clear(); CELLS.NET.reset(); G._PROBES.clear()
        try:
            call(C=256, D=256)
            rec["lever_off"] = "NOT RAISED"
        except G.TrimulUnsupported as e:
            rec["lever_off"] = {"reason": e.reason, "cannot_run": e.cannot_run, "off_word": CELLS.off_word(), "word": CELLS.cell_word(out.device), "supported": G.supported(torch.zeros(128, 128, 128, dtype=torch.bfloat16, device=dev))}
        print(json.dumps(rec))
    """)
    a = res["after_build_failure"]
    assert a["served"] == 1 and a["finite"] and a["net"].startswith("safe:build_failed:") and a["word"] == "cell=safe(%s)" % a["net"][len("safe:"):]
    assert err.count("[opt_core/pair_fused:trimul] safe settings served (build_failed:") == 2, err[-3000:]   # once per net engagement: scenario 1, and scenario 2 (net reset) before its SAFE cell fails too
    b = res["lever_off"]
    assert b != "NOT RAISED" and b["reason"] == "none" and b["cannot_run"] and b["off_word"].startswith("none:") and b["word"] == "cell=" + b["off_word"]
    assert b["supported"] == [False, b["off_word"]] and err.count("[fpf_trimul_v4] cell=none: ") == 1, err[-3000:]

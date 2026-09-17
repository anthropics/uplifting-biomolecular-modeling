"""The fpf_pallas / pallas_glut carried kernels and their generic layer opt_core.kernels.fpf_pallas_serve.

CPU (always): the sums hold; the serve layer is stdlib at import; the tile tables answer per compute capability with a named fallback; the
served-shape predicate answers by NAME; the parameter builders produce the kernel layout. GPU (``-m gpu``, a CUDA jax with Pallas-Triton):
every fused block against its pure-jnp f32 reference at the pairformer shapes (pair C=128, H=4, D=32; template C=64, H=4, D=16) for both
equations / orientations, with the same-class rule (rel-rms error of the fused block <= 1.25 x the error of a bf16 op-by-op body,
run-to-run bit-exact deterministic, NaN-free), the named refusal of f32 activations on the triangle multiplication, and the f32 attention
route (``fpf_pallas_f32``: f32 activations and parameters, tf32 MMAs) against the same reference at HIGHEST precision with ITS same-class
rule (rel-rms of the fused f32 block <= 1.25 x the error of the pure-jnp f32 body at XLA's DEFAULT matmul precision — tf32-class on sm_80+,
what a stock f32 model runs there), bit-exact run-to-run, NaN-free on fully-masked lines, plus a kernel micro-timing print per shape.
"""
import importlib
import json
import os
import subprocess
import sys

import pytest

from opt_core import kernels
from opt_core.kernels import fpf_pallas_serve as S

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------------------------------------------------------ CPU
def test_the_two_kernels_are_carried_and_hold_their_sums():
    assert "fpf_pallas" in kernels.names() and "pallas_glut" in kernels.names()
    assert kernels.verify_carry("fpf_pallas") == [] and kernels.verify_carry("pallas_glut") == []
    doc = kernels.sums("fpf_pallas")
    assert set(doc["files"]) == {"__init__.py", "trimul_pallas.py", "triattn_pallas.py", "transition_pallas.py", "NOTICE"}   # the three kernels, the package marker and its attribution notice
    assert kernels.sums("pallas_glut")["kind"] == "module"


def test_serve_layer_is_stdlib_at_import():
    code = ("import sys; import opt_core.kernels.fpf_pallas_serve as S; "
            "bad = [m for m in ('jax', 'jaxlib', 'numpy', 'tokamax', 'haiku') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_tile_tables_answer_by_compute_capability_and_refuse_unknown_parts_by_name(monkeypatch, capsys):
    cc, t, own = S.tables_for("9.0")
    assert own and t["trimul"]["t1"] == 64 and t["attn_by_n"][1024]["bq"] == 128 and S.tiles_label("9.0") == "own:9.0"
    cc, t, own = S.tables_for("10.3")
    assert own and t["trimul_by_n"][256]["t1"] == 64
    monkeypatch.delenv(S.FALLBACK_ENV, raising=False)
    monkeypatch.setattr(S, "_SAFE_ANNOUNCED", {}); monkeypatch.setattr(S, "_SAFE_NET", None)
    for part, gen in (("8.6", "8.x"), ("8.9", "8.x"), ("10.9", "10.x"), ("9.5", "9.x")):   # no row of its own, a named generation: the SAFE rows, engaged, one line
        cc, t, own = S.tables_for(part)
        assert (cc, own) == (part, False) and t is S.GENERATION_SAFE_TABLES[gen] is S.SAFE_TABLE and S.tiles_label(part) == f"safe:{gen}"
    err = capsys.readouterr().err.splitlines()
    assert [l for l in err if "safe settings served" in l] == ["[opt_core/fpf_pallas] safe settings served (no_cell:tile_table, cc 8.6, triton ?)"]   # the shared helper's ONE line per process
    assert S.safe_net().on and S.safe_net().word() == "safe:no_cell:tile_table" and set(S._SAFE_ANNOUNCED) == {"8.6", "8.9", "10.9", "9.5"}
    S.tables_for("8.6"); assert "safe settings served" not in capsys.readouterr().err
    assert S.trimul_cfg(256, "8.6") == dict(t1=32, w1=4, s1=1, t2=32, w2=4, s2=1, ein="xla") == S.safe_row("8.6", "trimul")
    assert S.attn_cfg(1536, "8.6") == dict(t1=32, w1=4, t2=32, w2=4, bq=32, bk=32, wa=4, sa=1) == S.safe_row("8.9", "triattn") == S.safe_row("9.0", "attn")   # the accessor answers for a tuned part too
    assert S.trimul_cfg(448, "8.6", dtype="float32") == dict(t1=32, w1=4, hc1=32, t2=32, w2=4, hc2=16, ein="xla") == S.safe_row("8.6", "trimul", "float32")
    assert S.attn_cfg(448, "8.6", dtype="float32") == S.safe_row("10.9", "attn", dtype="float32") and S.safe_row("8.6", "attn", "float32")["vt"] == 1
    assert S.required_multiple("trimul", 256, "8.6") == 32 and S.kernel_n("triattn", 250, "8.6") == 256 and S.kernel_n("trimul", 200, "8.6") == 224
    for unknown in ("7.5", "12.0", "", "sm90"):                          # no row and no named generation: refused by name, never a silent sm_90 fallback
        with pytest.raises(S.Refusal) as e:
            S.tables_for(unknown)
        assert e.value.kind == S.NO_TILES and e.value.detail.startswith(f"no-tiles:{unknown or '?'}") and "--mode off" in e.value.detail, e.value.detail
        with pytest.raises(S.Refusal):
            S.safe_row(unknown, "trimul")
    with pytest.raises(S.Refusal):
        S.trimul_cfg(256, "7.5")
    monkeypatch.setenv(S.FALLBACK_ENV, "1")                             # the explicit opt-in: the 9.0 table, RECORDED as a fallback (a generation part keeps its safe rows)
    cc, t, own = S.tables_for("12.0")
    assert not own and t is S.TILE_TABLES["9.0"] and S.tiles_label("12.0") == "fallback:12.0->9.0" and S.tables_for("8.6")[1] is S.SAFE_TABLE
    cc, t, own = S.tables_for("8.0")                                     # A100: a table of its own
    assert own and S.tiles_label("8.0") == "own:8.0" and S.trimul_cfg(256, "8.0") == dict(t1=64, w1=4, s1=2, t2=64, w2=4, s2=2, ein="xla")
    assert S.attn_cfg(1536, "8.0") == dict(t1=64, w1=4, t2=64, w2=4, bq=128, bk=32, wa=4, sa=3) and S.attn_cfg(512, "8.0")["bq"] == 64
    assert S.trimul_cfg(256, "10.0") == dict(t1=128, w1=8, s1=2, t2=128, w2=8, s2=2, ein="xla")
    assert S.attn_cfg(1024, "9.0") == dict(t1=64, w1=4, t2=64, w2=4, bq=128, bk=32, wa=4, sa=3)
    assert S.attn_cfg(512, "9.0", overrides={"bq": 32})["bq"] == 32
    assert set(S.TILE_TABLES) == set(S.TILE_TABLES_SOURCE)


def test_generation_safe_rows_have_the_table_schema_and_fit_an_sm86_program():
    """The generation SAFE table carries every key of the sm_90 table; every tile divides 64; the largest single-MMA working set of every row is
    under the 99 KB of shared memory an sm_86 / sm_89 program may use (a necessary condition; the compile on such a part is the evidence)."""
    assert set(S.SAFE_TABLE) == set(S.TILE_TABLES["9.0"]) | {"trimul_f32", "trimul_f32_by_n", "attn_f32_default", "attn_f32_by_n"} == set(S.TILE_TABLES["9.0"])
    assert set(S.GENERATION_SAFE_TABLES) == {"8.x", "9.x", "10.x"} and all(t is S.SAFE_TABLE for t in S.GENERATION_SAFE_TABLES.values())
    for key in ("trimul", "attn_default", "trimul_f32", "attn_f32_default"):
        row = S.SAFE_TABLE[key]; f32 = "f32" in key; kind = "trimul" if key.startswith("trimul") else "attn"
        assert all(64 % row[k] == 0 for k in ("t1", "t2", "bq", "bk") if k in row), (key, row)
        for c in (128, 64):
            assert _largest_dot_working_set(kind, row, c=c, f32=f32) <= SM86_SHARED_BYTES, (key, c, _largest_dot_working_set(kind, row, c=c, f32=f32))
    assert S.generation("8.6") == "8.x" and S.generation("10.0") == "10.x" and S.generation("7.5") is None and S.generation("") is None and S.generation(None) is None


def _fake_jax_line(monkeypatch, version):
    """The serve layer reads the jax line from sys.modules['jax'].__version__ (jax_line): a stand-in module pins it without a GPU stack."""
    import types
    fake = types.ModuleType("jax"); fake.__version__ = version
    monkeypatch.setitem(sys.modules, "jax", fake); S._LINE_TABLES.clear()


def test_sm80_table_has_the_sm90_schema_and_pads_pair_sizes_to_64(monkeypatch):
    """The A100 table carries every row kind the H100 table does (bf16 and f32), its default rows use tiles that divide 64 (a kit's kernel_n on
    sm_80 equals its kernel_n on sm_90), and its by-N rows sit on pair sizes their own tiles divide."""
    t80, t90 = S.TILE_TABLES["8.0"], S.TILE_TABLES["9.0"]
    assert set(t90) <= set(t80) and set(t80) - set(t90) == {S.F32_BY_JAX_LINE} and set(S.TILE_TABLES) == set(S.TILE_TABLES_SOURCE)
    for line in ("0.10.2", "0.5.3", "0.7.0"):
        _fake_jax_line(monkeypatch, line)
        for kind in ("trimul", "triattn"):
            for dtype in ("bfloat16", "float32"):
                assert S.required_multiple(kind, cc="8.0", dtype=dtype) == 64 == S.required_multiple(kind, cc="9.0", dtype=dtype), (line, kind, dtype)
                assert S.kernel_n(kind, 1473, "8.0", dtype=dtype) == 1536 and S.kernel_n(kind, 574, "8.0", dtype=dtype) == 576
        served = S.tables_for("8.0")[1]
        for key in ("trimul_by_n", "attn_by_n", "trimul_f32_by_n", "attn_f32_by_n"):
            base = {"trimul_by_n": "trimul", "attn_by_n": "attn_default", "trimul_f32_by_n": "trimul_f32", "attn_f32_by_n": "attn_f32_default"}[key]
            for n, over in served[key].items():
                row = {**served[base], **over}
                assert all(n % row[k] == 0 for k in ("t1", "t2", "bq", "bk") if k in row), (line, key, n, row)


def test_sm80_f32_rows_follow_the_jax_line(monkeypatch):
    """On sm_80 the f32 triangle-multiplication rows are served per jax line (f32_by_jax_line): the 0.5 line's on jax 0.5.x, the base rows on 0.10.x
    and on a line the table names no rows for; the attention and bf16 rows and every other table are the same objects on every line."""
    _fake_jax_line(monkeypatch, "0.5.3")
    cc, t5, own = S.tables_for("8.0")
    assert own and S.F32_BY_JAX_LINE not in t5 and t5 is S.tables_for("8.0")[1]
    assert t5["trimul_f32"] == S.TILE_TABLES["8.0"][S.F32_BY_JAX_LINE]["0.5"]["trimul_f32"] == S.trimul_cfg(448, "8.0", dtype="float32") and t5["trimul_f32"]["hc1"] == 32
    assert t5["attn_f32_default"] is S.TILE_TABLES["8.0"]["attn_f32_default"] and t5["trimul"] is S.TILE_TABLES["8.0"]["trimul"]   # one attention row, one bf16 row set
    for line in ("0.10.2", "0.7.0", "none"):
        _fake_jax_line(monkeypatch, line)
        t = S.tables_for("8.0")[1]
        assert t is S.TILE_TABLES["8.0"] and S.attn_cfg(448, "8.0", dtype="float32")["hc1"] == 64 and S.trimul_cfg(448, "8.0", dtype="float32")["hc1"] == 64
        assert S.tables_for("9.0")[1] is S.TILE_TABLES["9.0"] and S.tables_for("10.0")[1] is S.TILE_TABLES["10.0"]


SM80_SHARED_BYTES = 163 * 1024                                               # sm_80: 163 KB of shared memory per thread block (opt-in maximum)
SM86_SHARED_BYTES = 99 * 1024                     # sm_86 / sm_89 (A10, A40, L4, L40, RTX 30/40): 100 KB of shared memory per SM, 99 KB opt-in per program


def _largest_dot_working_set(kind, row, c=128, num_head=4, f32=False):
    """Bytes of the operand tiles + accumulator of the largest single MMA a program of each fused kernel issues — a necessary (not sufficient)
    condition for a tile row to compile on a part; the compile on the part itself is the sweep's evidence."""
    eb = 4 if f32 else 2                                                     # operand element bytes (accumulators are f32)
    d = S.head_dim(c, num_head); hd = num_head * d
    if kind == "trimul":                                                      # prologue: x[t1,C] @ W[C,hc] (hc = C for bf16); epilogue: y[t2,C] @ Wo[C,hc2|C]
        h1 = row.get("hc1", c); h2 = row.get("hc2", c)
        return max(row["t1"] * c * eb + c * h1 * eb + row["t1"] * h1 * 4, row["t2"] * c * eb + c * h2 * eb + row["t2"] * h2 * 4)
    h1 = row.get("hc1", hd); h2 = row.get("hc2", c)                          # attention prologue: x[t1,C] @ Wq[C,hc1]; core: q[bq,D] k[bk,D] v[bk,D] s[bq,bk] o[bq,D]; epilogue
    pro = row["t1"] * c * eb + c * h1 * eb + row["t1"] * h1 * 4
    core = (row["bq"] * d + 2 * row["bk"] * d) * eb + row["bq"] * row["bk"] * 4 + row["bq"] * d * 4
    epi = row["t2"] * hd * eb + hd * h2 * eb + row["t2"] * h2 * 4
    return max(pro, core, epi)


def test_sm80_rows_fit_the_parts_shared_memory(monkeypatch):
    for line in ("0.10.2", "0.5.3"):
        _fake_jax_line(monkeypatch, line)
        t = S.tables_for("8.0")[1]
        rows = [("trimul", {**t["trimul"], **o}, False) for o in [{}] + list(t["trimul_by_n"].values())]
        rows += [("triattn", {**t["attn_default"], **o}, False) for o in [{}] + list(t["attn_by_n"].values())]
        rows += [("trimul", {**t["trimul_f32"], **o}, True) for o in [{}] + list(t["trimul_f32_by_n"].values())]
        rows += [("triattn", {**t["attn_f32_default"], **o}, True) for o in [{}] + list(t["attn_f32_by_n"].values())]
        for kind, row, f32 in rows:
            for c, h in ((128, 4), (64, 4)):
                need = _largest_dot_working_set(kind, row, c=c, num_head=h, f32=f32)
                assert need <= SM80_SHARED_BYTES, (line, kind, row, c, need)


def test_probe_names_no_tiles_or_backend(monkeypatch):
    monkeypatch.delenv(S.FALLBACK_ENV, raising=False)
    p = S.probe()
    assert set(p) >= {"ok", "kind", "detail", "jax", "backend", "cc", "own_table"}
    if not p["ok"]:
        assert p["kind"] in ("jax_missing", "pallas_missing", S.BACKEND_NOT_GPU, S.KERNEL_IMPORT_FAILED, S.NO_TILES)


@pytest.mark.parametrize("shape,dtype,mask,kind,h,expect", [
    ((256, 256, 128), "bfloat16", (256, 256), "trimul", None, None),
    ((256, 256, 64), "bfloat16", (256, 256), "triattn", 4, None),            # template stack: C=64 -> D=16
    ((256, 256, 128), "bfloat16", (256, 256), "triattn", 4, None),           # pair stack: D=32
    ((256, 256, 128), "float32", (256, 256), "trimul", None, None),               # f32 triangle multiplication: the f32 kernels (fpf_pallas_f32)
    ((448, 448, 64), "float32", (448, 448), "trimul", None, None),
    ((256, 256, 128), "float64", (256, 256), "trimul", None, S.DTYPE_NOT_SERVED),
    ((256, 256, 128), "float32", (256, 256), "triattn", 4, None),            # f32 attention: served by the f32 kernels (fpf_pallas_f32)
    ((256, 256, 64), "float32", (256, 256), "triattn", 4, None),
    ((448, 448, 128), "float32", (448, 448), "triattn", 4, None),
    ((250, 250, 128), "float32", (250, 250), "triattn", 4, None),            # padded to kernel_n inside, as the bf16 route
    ((256, 256, 128), "float16", (256, 256), "triattn", 4, S.DTYPE_NOT_SERVED),
    ((250, 250, 128), "bfloat16", (250, 250), "trimul", None, None),                     # any N: the blocks pad to kernel_n and slice back
    ((448, 448, 128), "bfloat16", (448, 448), "trimul", None, None),
    ((320, 320, 64), "bfloat16", (320, 320), "triattn", 4, None),
    ((96, 96, 128), "bfloat16", (96, 96), "triattn", 4, None),
    ((256, 128, 128), "bfloat16", (256, 128), "trimul", None, S.NOT_SQUARE),
    ((256, 256, 120), "bfloat16", (256, 256), "trimul", None, S.C_NOT_MULTIPLE),
    ((256, 256, 512), "bfloat16", (256, 256), "trimul", None, S.C_OUT_OF_RANGE),
    ((256, 256, 128), "bfloat16", None, "trimul", None, S.MASK_SHAPE),
    ((256, 256, 128), "bfloat16", (1, 256), "trimul", None, S.MASK_SHAPE),
    ((4, 256, 256, 128), "bfloat16", (256, 256), "trimul", None, S.RANK_NOT_3),
    ((256, 256, 128), "bfloat16", (256, 256), "triattn", 1, S.HEAD_DIM_UNSUPPORTED),   # D=128
    ((256, 256, 256), "bfloat16", (256, 256), "triattn", 16, None),                     # D=16, HD=256
    ((256, 256, 256), "bfloat16", (256, 256), "triattn", None, S.HEAD_DIM_UNSUPPORTED),
])
def test_served_reason_answers_by_name(shape, dtype, mask, kind, h, expect):
    assert S.served_reason(kind, shape, dtype, mask, num_head=h) == expect


def test_bias_presence_is_all_or_none_by_name():
    from opt_core.kernels import fpf_pallas_bias as B
    assert B.has_bias({}, "trimul") is False and B.has_bias({"bg": 1, "bo": 2}, "triattn") is True
    assert B.has_bias({"b_proj": 1}, "trimul") is None                                   # some but not all: the caller refuses by name
    with pytest.raises(S.Refusal) as e:
        S._kernel_pair_mask(None, "starting", "sideways")
    assert e.value.kind == S.UNKNOWN_KEY_MASK


def test_bias_module_is_stdlib_at_import():
    code = ("import sys; import opt_core.kernels.fpf_pallas_bias as B; "
            "bad = [m for m in ('jax', 'jaxlib', 'numpy', 'tokamax', 'haiku') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_required_multiple_follows_the_tile_table(monkeypatch):
    assert S.N_MULTIPLE == 64 and S.required_multiple("trimul", 448, "9.0") == 64 and S.required_multiple("triattn", 448, "9.0") == 64
    assert S.required_multiple("triattn", 1024, "9.0") == 128                      # the by-N row (bq=128) applies at N=1024 only
    assert S.required_multiple("trimul", 448, "10.0") == 128                       # t1=128 there
    assert S.served_reason("trimul", (448, 448, 128), "bfloat16", (448, 448), cc="10.0", pad=False) == S.N_NOT_MULTIPLE
    assert S.served_reason("trimul", (448, 448, 128), "bfloat16", (448, 448), cc="10.0") is None          # padded to 512 there
    assert S.served_reason("trimul", (448, 448, 128), "bfloat16", (448, 448), cc="9.0", pad=False) is None
    assert S.kernel_n("trimul", 400, "9.0") == 448 and S.kernel_n("triattn", 300, "9.0") == 320 and S.kernel_n("trimul", 448, "9.0") == 448
    assert S.kernel_n("trimul", 400, "10.0") == 512 and S.kernel_n("triattn", 1000, "9.0") == 1024      # 1024's own row (bq=128) divides 1024


def test_f32_module_is_stdlib_at_import():
    code = ("import sys; import opt_core.kernels.fpf_pallas_f32 as F; "
            "bad = [m for m in ('jax', 'jaxlib', 'numpy', 'tokamax', 'haiku') if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_f32_route_vocabulary_and_tiles(monkeypatch):
    from opt_core.kernels import fpf_pallas_f32 as F
    assert F.PRECISIONS == S.F32_PRECISIONS == ("tf32", "tf32x3", "ieee") and F.DEFAULT_PRECISION is None      # no default word on the f32 route
    for missing in ("std", None):
        with pytest.raises(S.Refusal) as e:
            S.f32_precision(missing)
        assert e.value.kind == S.PRECISION_REQUIRED == "precision_required" and "tf32" in str(e.value)
    assert S.f32_precision("tf32") == "tf32" and S.f32_precision("tf32x3") == "tf32x3" and S.f32_precision("ieee") == "ieee"
    assert S.jax_line("0.5.3") == "0.5" and S.jax_line("0.10.2") == "0.10" and S.jax_line("junk") == "none"          # the class table is keyed by jax line
    assert S.f32_class("tf32", "0.5.3") == "bf16op" and S.f32_class("tf32", "0.10.2") == "body" and S.f32_class("tf32", "0.12.0") == "body"
    assert S.f32_class("tf32", "0.7.0") == "unlisted:0.7" and S.f32_class("tf32x3", "0.5.3") == "f32" and S.f32_class("ieee", "0.10.2") == "f32"
    assert ("triattn", "ieee") in S.F32_LINES["0.5"]["unlaunchable"] and ("trimul", "ieee") not in S.F32_LINES["0.5"]["unlaunchable"] and S.PRECISION_UNLAUNCHABLE == "precision_unlaunchable"
    assert set(S.F32_LINES["0.5"]["ceiling"]) == {"triattn", "trimul"} and S.F32_LINES["0.10"]["ratio"] == 1.25
    assert "F32_PRECISIONS = (" not in open(S.__file__).read()                  # the tuple is the kernel module's (PEP 562), not re-typed
    assert S.DTYPE_NOT_BF16 == S.DTYPE_NOT_SERVED == "dtype_not_served"          # alias kept one release; served_reason answers dtype_not_served
    with pytest.raises(S.Refusal) as e:
        S.f32_precision("hi")                                            # the bf16 route's word means nothing on the f32 route: refused by name
    assert e.value.kind == S.UNKNOWN_PRECISION
    assert S.SERVED_DTYPES == {"trimul": ("bfloat16", "float32"), "triattn": ("bfloat16", "float32")}
    ct = S.trimul_cfg(256, "9.0", dtype="float32")
    assert set(ct) >= {"t1", "w1", "hc1", "t2", "w2", "hc2", "ein"} and ct["ein"] == "xla" and 64 % ct["t1"] == 0 and 64 % ct["t2"] == 0
    assert S.trimul_cfg(256, "9.0") == S.TILE_TABLES["9.0"]["trimul"] and S.required_multiple("trimul", 448, "9.0", dtype="float32") == 64
    c = S.attn_cfg(256, "9.0", dtype="float32")
    assert set(c) >= {"t1", "w1", "hc1", "t2", "w2", "hc2", "bq", "bk", "wa", "sa", "vt"}
    assert S.required_multiple("triattn", 448, "9.0", dtype="float32") == 64 and S.kernel_n("triattn", 400, "9.0", "float32") == 448
    assert S.kernel_n("triattn", 1408, "9.0", "float32") == 1408 and S.kernel_n("triattn", 1000, "9.0", "float32") % 64 == 0
    for tb in S.TILE_TABLES.values():                                     # every f32 row's tiles divide the pair sizes it answers for
        if "attn_f32_default" not in tb:
            continue
        rows = [(64, tb["attn_f32_default"])] + [(n, {**tb["attn_f32_default"], **r}) for n, r in tb.get("attn_f32_by_n", {}).items()]
        for n, row in rows:
            assert all(n % row[k] == 0 for k in ("t1", "t2", "bq", "bk")), (n, row)
    assert S.attn_cfg(256, "9.0") == S.TILE_TABLES["9.0"]["attn_default"]  # the bf16 rows are untouched by the dtype switch
    monkeypatch.delenv(S.FALLBACK_ENV, raising=False)
    for cc in ("10.0", "10.3"):                                           # a table without f32 rows: refused by name, never the bf16 tiles
        for fn in (S.attn_cfg, S.trimul_cfg):
            with pytest.raises(S.Refusal) as e:
                fn(256, cc, dtype="float32")
            assert e.value.kind == S.NO_TILES and e.value.detail.startswith(f"no-f32-tiles:{cc}"), e.value.detail


def test_equation_and_orientation_vocabulary():
    assert S.EQUATIONS["outgoing"] == "ikc,jkc->ijc" and S.EQUATIONS["incoming"] == "kjc,kic->ijc"
    assert S.ORIENTATIONS == {"starting": False, "ending": True, "per_row": False, "per_column": True}
    with pytest.raises(S.Refusal) as e:
        S.trimul_block(None, None, {}, equation="sideways")
    assert e.value.kind == "unknown_equation"
    with pytest.raises(S.Refusal) as e:
        S.tri_attn_block(None, None, {"H": 4}, orientation="diagonal")
    assert e.value.kind == "unknown_orientation"


jnp = None
try:  # the parameter builders and the reference need jax (CPU jax is enough); the blocks need a GPU
    import jax
    import jax.numpy as jnp  # noqa: F811
    HAVE_JAX = True
    GPU = jax.default_backend() == "gpu"
except Exception:  # noqa: BLE001
    HAVE_JAX, GPU = False, False
needs_jax = pytest.mark.skipif(not HAVE_JAX, reason="jax not importable")
gpu = pytest.mark.skipif(not GPU, reason="needs a CUDA jax backend (Pallas-Triton lowering)")


@needs_jax
def test_param_builders_produce_the_kernel_layout():
    C, H, D = 64, 4, 16
    key = jax.random.PRNGKey(0)
    ks = jax.random.split(key, 8)
    w = lambda k, shape: jax.random.normal(k, shape, jnp.float32).astype(jnp.bfloat16)  # noqa: E731
    q_w, k_w, v_w, g_w = (w(ks[i], (C, H, D)) for i in range(4))
    b_w, o_w = w(ks[4], (C, H)), w(ks[5], (H, D, C))
    kp = S.attn_params(ln_scale=jnp.ones((C,)), ln_offset=jnp.zeros((C,)), q_w=q_w, k_w=k_w, v_w=v_w, bias_w=b_w, gate_w=g_w, out_w=o_w)
    assert kp["wq_t"].shape == (C, H * D) and kp["wo"].shape == (H * D, C) and kp["wb16"].shape == (C, 16) and kp["H"] == H and kp["D"] == D
    assert bool(jnp.all(kp["wb16"][:, H:] == 0)) and bool(jnp.all(kp["wb16"][:, :H] == b_w))
    # the AlphaFold 3-layout tree converter agrees with the math-layout builder
    tree = {"act_norm": {"scale": jnp.ones((C,)), "offset": jnp.zeros((C,))}, "pair_bias_projection": {"weights": b_w},
            "q_projection": {"weights": jnp.transpose(q_w, (1, 2, 0))}, "k_projection": {"weights": jnp.transpose(k_w, (1, 2, 0))},
            "v_projection": {"weights": v_w}, "gating_query": {"weights": g_w.reshape(C, H * D).T}, "output_projection": {"weights": o_w.reshape(H * D, C)}}
    if GPU or S.probe(require_gpu=False)["kind"] is None:
        kt = S.attn_params_from_tree(tree)
        for name in ("wq_t", "wk_t", "wv2", "wb16", "wg_t", "wo"):
            assert bool(jnp.all(kt[name] == kp[name])), name
    tp = S.trimul_params(ln_in_scale=jnp.ones((C,)), ln_in_offset=jnp.zeros((C,)), left_w=jnp.full((C, C), 1.0), right_w=jnp.full((C, C), 2.0),
                         left_gate_w=jnp.full((C, C), 3.0), right_gate_w=jnp.full((C, C), 4.0), ln_c_scale=jnp.ones((C,)), ln_c_offset=jnp.zeros((C,)),
                         out_w=jnp.eye(C), gate_w=jnp.eye(C))
    assert tp["w_proj"].shape == (C, 2 * C) and bool(jnp.all(tp["w_proj"][:, 0::2] == 1.0)) and bool(jnp.all(tp["w_proj"][:, 1::2] == 2.0))
    assert bool(jnp.all(tp["w_gate"][:, 0::2] == 3.0)) and bool(jnp.all(tp["w_gate"][:, 1::2] == 4.0)) and "b_proj" not in tp
    tb = S.trimul_params(ln_in_scale=jnp.ones((C,)), ln_in_offset=jnp.zeros((C,)), left_w=jnp.eye(C), right_w=jnp.eye(C), left_gate_w=jnp.eye(C),
                         right_gate_w=jnp.eye(C), ln_c_scale=jnp.ones((C,)), ln_c_offset=jnp.zeros((C,)), out_w=jnp.eye(C), gate_w=jnp.eye(C),
                         left_b=jnp.full((C,), 1.0), right_b=jnp.full((C,), 2.0), left_gate_b=jnp.full((C,), 3.0), right_gate_b=jnp.full((C,), 4.0),
                         out_b=jnp.full((C,), 5.0), gate_b=jnp.full((C,), 6.0))
    assert tb["b_proj"].shape == (2 * C,) and bool(jnp.all(tb["b_proj"][0::2] == 1.0)) and bool(jnp.all(tb["b_proj"][1::2] == 2.0))
    assert bool(jnp.all(tb["b_gate"][0::2] == 3.0)) and bool(jnp.all(tb["b_gate"][1::2] == 4.0)) and bool(jnp.all(tb["b_out"] == 5.0))
    with pytest.raises(S.Refusal) as e:
        S.trimul_params(ln_in_scale=jnp.ones((C,)), ln_in_offset=jnp.zeros((C,)), left_w=jnp.eye(C), right_w=jnp.eye(C), left_gate_w=jnp.eye(C),
                        right_gate_w=jnp.eye(C), ln_c_scale=jnp.ones((C,)), ln_c_offset=jnp.zeros((C,)), out_w=jnp.eye(C), gate_w=jnp.eye(C), left_b=jnp.ones((C,)))
    assert e.value.kind == S.BIAS_KEYS_INCOMPLETE
    kb = S.attn_params(ln_scale=jnp.ones((C,)), ln_offset=jnp.zeros((C,)), q_w=q_w, k_w=k_w, v_w=v_w, bias_w=b_w, gate_w=g_w, out_w=o_w,
                       gate_b=jnp.ones((H, D)), out_b=jnp.zeros((C,)))
    assert kb["bg"].shape == (H * D,) and kb["bo"].shape == (C,) and kb["bg"].dtype == jnp.float32
    kf = S.attn_params(ln_scale=jnp.ones((C,)), ln_offset=jnp.zeros((C,)), q_w=q_w.astype(jnp.float32), k_w=k_w, v_w=v_w, bias_w=b_w, gate_w=g_w, out_w=o_w,
                       weights_dtype=jnp.float32)
    assert kf["wq_t"].dtype == jnp.float32 and kf["wo"].dtype == jnp.float32 and kp["wq_t"].dtype == jnp.bfloat16


# ------------------------------------------------------------------------------------------------------------------ GPU parity
def _rel_rms(a, b):
    a = a.astype(jnp.float32); b = b.astype(jnp.float32)
    return float(jnp.sqrt(jnp.mean(jnp.square(a - b))) / (jnp.sqrt(jnp.mean(jnp.square(b))) + 1e-30))


def _inputs(n, c, seed=0, pad=32):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    act = jax.random.normal(k1, (n, n, c), jnp.float32).astype(jnp.bfloat16)
    valid = jnp.arange(n) < (n - pad)                                     # a padded tail, as a bucketed input
    mask = (valid[:, None] & valid[None, :]).astype(jnp.bfloat16)
    return act, mask, k2


def _trimul_params(c, key, bias, dtype=None):
    dtype = dtype or jnp.bfloat16
    ks = jax.random.split(key, 10)
    sc = 1.0 / (c ** 0.5)
    w = lambda k, shape: (jax.random.normal(k, shape, jnp.float32) * sc).astype(dtype)  # noqa: E731
    p = dict(ln_in_scale=jnp.ones((c,)), ln_in_offset=jnp.zeros((c,)), w_proj=w(ks[0], (c, 2 * c)), w_gate=w(ks[1], (c, 2 * c)),
             ln_c_scale=jnp.ones((c,)), ln_c_offset=jnp.zeros((c,)), w_out=w(ks[2], (c, c)), w_gl=w(ks[3], (c, c)))
    if bias:                                                                     # the AlphaFold 2 parameterisation: N(0, 0.3) projection biases, gates near 1
        b = lambda k, shape, m=0.0: (m + 0.3 * jax.random.normal(k, shape, jnp.float32)).astype(dtype)  # noqa: E731
        p.update(b_proj=b(ks[4], (2 * c,)), b_gate=b(ks[5], (2 * c,), 1.0), b_out=b(ks[6], (c,)), b_gl=b(ks[7], (c,), 1.0))
    return p


def _trimul_case_f32(n, c, bias):
    """f32 activations (full mantissa), the padded-tail mask of ``_inputs``, f32 parameters (+ the AlphaFold 2 biases when ``bias``)."""
    _, mask, key = _inputs(n, c)
    ks = jax.random.split(key, 12)
    act = jax.random.normal(ks[11], (n, n, c), jnp.float32)
    return act, mask, _trimul_params(c, key, bias, dtype=jnp.float32)


@gpu
@pytest.mark.parametrize("equation", ["outgoing", "incoming"])
def test_house_trimul_reference_equals_the_carried_replica(equation):
    """The serve layer's pure-jnp reference (bias-capable) against the carried op-by-op replica of the stock module (bias-free)."""
    n, c = 256, 128
    act, mask, key = _inputs(n, c)
    p = _trimul_params(c, key, bias=False)
    K = S.kernels()[0]
    # f32: the carried replica keeps the stock module's bf16 LayerNorm-output cast in every compute dtype, the house reference rounds nothing
    # (the stricter reference) -> they differ by one bf16 rounding of the LN output (~1.7e-3 rel), not by f32 noise; bf16: op-by-op chains agree.
    for cd, tol in ((jnp.float32, 4e-3), (jnp.bfloat16, 2.5e-2)):
        mine = S.trimul_reference(act, mask, p, equation=equation, compute_dtype=cd, precision=jax.lax.Precision.HIGHEST)
        theirs = K.triangle_multiplication_reference(act, mask, p, equation=S.EQUATIONS[equation], compute_dtype=cd, precision=jax.lax.Precision.HIGHEST)
        e = _rel_rms(mine, theirs)
        print(f"house vs carried trimul reference {equation} {cd.__name__ if hasattr(cd, '__name__') else cd}: rel_rms {e:.3e}")
        assert e <= tol, (cd, e)


@gpu
@pytest.mark.parametrize("n,c", [(256, 128), (256, 64), (512, 128), (448, 128), (320, 64), (400, 128), (300, 64)])   # 400/300: padded to 448/320 inside
@pytest.mark.parametrize("equation", ["outgoing", "incoming"])
@pytest.mark.parametrize("bias", [False, True])
def test_trimul_block_matches_its_reference_in_class(n, c, equation, bias):
    act, mask, key = _inputs(n, c)
    p = _trimul_params(c, key, bias)
    ref = S.trimul_reference(act, mask, p, equation=equation, compute_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST)
    stock_like = S.trimul_reference(act, mask, p, equation=equation, compute_dtype=jnp.bfloat16)
    out = jax.block_until_ready(S.trimul_block(act, mask, p, equation=equation))
    out2 = jax.block_until_ready(S.trimul_block(act, mask, p, equation=equation))
    assert out.shape == act.shape and out.dtype == act.dtype
    assert not bool(jnp.any(jnp.isnan(out.astype(jnp.float32))))
    assert bool(jnp.all(out == out2)), "run-to-run bitwise"
    e_fused, e_stock = _rel_rms(out, ref), _rel_rms(stock_like, ref)
    print(f"trimul {equation} bias={bias} N={n} (kernel_n {S.kernel_n('trimul', n, '9.0')}) C={c}: rel_rms fused {e_fused:.3e} stock-like bf16 {e_stock:.3e}")
    assert e_fused <= 1.25 * e_stock + 1e-6, (e_fused, e_stock)


def _attn_case(n, c, h, variant, dtype=None):
    """Inputs, an [N,N] pair mask (padded tail; ASYMMETRIC holes for the af2 variant so the key-mask convention is exercised), math-layout
    weights ``mw`` (+ gating/output biases for af2), LayerNorm ``ln``, the kernel dict ``kp`` and the call flags for one attention case.
    ``dtype=jnp.float32``: f32 activations and f32 parameters (the f32 route); default bf16."""
    dtype = dtype or jnp.bfloat16
    ending_bias_transposed = variant != "af3_official_bias"
    bias, key_mask = variant == "af2", ("by_line" if variant == "af2" else "by_column")
    d = S.head_dim(c, h)
    act, pmask, key = _inputs(n, c, seed=1)
    ks = jax.random.split(key, 10)
    if dtype != jnp.bfloat16:
        act = jax.random.normal(ks[8], (n, n, c), jnp.float32).astype(dtype)   # full-mantissa activations, not bf16-representable ones
    if bias:
        pmask = pmask * (jax.random.uniform(ks[9], (n, n)) > 0.1).astype(pmask.dtype)
    sc = 1.0 / (c ** 0.5)
    w = lambda k, shape, s=sc: (jax.random.normal(k, shape, jnp.float32) * s).astype(dtype)  # noqa: E731
    mw = dict(q_w=w(ks[0], (c, h, d)), k_w=w(ks[1], (c, h, d)), v_w=w(ks[2], (c, h, d)), bias_w=w(ks[3], (c, h)), gate_w=w(ks[4], (c, h, d)),
              out_w=w(ks[5], (h, d, c), 1.0 / ((h * d) ** 0.5)))
    if bias:
        mw.update(gate_b=(1.0 + 0.3 * jax.random.normal(ks[6], (h, d))).astype(dtype), out_b=(0.3 * jax.random.normal(ks[7], (c,))).astype(dtype))
    ln = dict(ln_scale=jnp.ones((c,)), ln_offset=jnp.zeros((c,)))
    kp = S.attn_params(**ln, **mw, weights_dtype=(None if dtype == jnp.bfloat16 else dtype))
    assert ("bg" in kp) == bias
    flags = dict(ending_bias_transposed=ending_bias_transposed, key_mask=key_mask)
    return act, pmask, ln, mw, kp, flags


@gpu
@pytest.mark.parametrize("n,c,h", [(256, 128, 4), (256, 64, 4), (512, 128, 4), (448, 128, 4), (320, 64, 4), (400, 128, 4), (300, 64, 4)])
@pytest.mark.parametrize("orientation", ["starting", "ending"])
@pytest.mark.parametrize("variant", ["af3", "af3_official_bias", "af2"])      # af3: OF3-port bias convention; af2: +bias epilogue, by_line key mask, asymmetric pair mask
def test_tri_attn_block_matches_its_reference_in_class(n, c, h, orientation, variant):
    if orientation == "starting" and variant == "af3_official_bias":
        pytest.skip("the flag only concerns the ending node")
    d = S.head_dim(c, h)
    act, pmask, ln, mw, kp, flags = _attn_case(n, c, h, variant)
    ref = S.tri_attn_reference(act, pmask, **ln, **mw, orientation=orientation, **flags, precision=jax.lax.Precision.HIGHEST)
    out = jax.block_until_ready(S.tri_attn_block(act, pmask, kp, orientation=orientation, **flags))
    out2 = jax.block_until_ready(S.tri_attn_block(act, pmask, kp, orientation=orientation, **flags))
    assert out.shape == act.shape and out.dtype == act.dtype
    assert not bool(jnp.any(jnp.isnan(out.astype(jnp.float32))))
    assert bool(jnp.all(out == out2)), "run-to-run bitwise"
    live = n - 32                                                           # _inputs masks the last 32 rows/cols entirely (a padded tail)
    e = _rel_rms(out[:live, :live], ref[:live, :live])                     # pixels whose attended line has valid keys
    e_dead = _rel_rms(out[live:, live:], ref[live:, live:]) if n % S.required_multiple("triattn", n, "9.0") else None
    # a bf16 block's floor against an f32 reference: outputs rounded once to bf16 (rel ~4e-3) plus bf16 q/k/v/p operands; the stock op-by-op
    # bf16 body measures 1.5-2.5e-2 on these inputs (the kit's parity launcher); the fused block must sit at or below that class.
    # Lines whose keys are ALL masked average uniformly over kernel_n keys in the padded kernel vs N in the reference: junk in both, reported
    # separately (e_dead) and not asserted — the only place per-call padding is visible.
    print(f"triattn {orientation} variant={variant} N={n} (kernel_n {S.kernel_n('triattn', n, '9.0')}) C={c} H={h} D={d}: rel_rms fused vs f32 ref {e:.3e} (live lines)"
          + ("" if e_dead is None else f"; fully-masked lines {e_dead:.3e} (not asserted)"))
    assert e <= 2.5e-2, e


F32_CASES = [(256, 128, 4), (256, 64, 4), (512, 128, 4), (448, 128, 4), (400, 128, 4)]   # 448 = 7 x 64 (a tile multiple); 400 pads to 448 inside


def _assert_f32_class(kind, e_fused, e_stock):
    """The f32 route's same-class rule PER JAX LINE (S.F32_LINES): 'body' -> e_fused <= ratio x the XLA-default body's error; 'bf16op' ->
    e_fused under the line's measured ceiling for `kind`; no row -> fail by name (measure this line and add a row; never a widened band)."""
    row, cls = S.f32_line_row(), S.f32_class("tf32")
    if cls == "body":
        assert e_fused <= row["ratio"] * e_stock + 1e-6, (S.jax_line(), cls, e_fused, e_stock)
    elif cls == "bf16op":
        assert e_fused <= row["ceiling"][kind], (S.jax_line(), cls, e_fused, row["ceiling"][kind])
    else:
        pytest.fail("no F32_LINES row for jax line %s (f32_class=%s): the f32 route's class is unmeasured here — measure and add a row" % (S.jax_line(), cls))

@gpu
@pytest.mark.parametrize("n,c,h", F32_CASES)
@pytest.mark.parametrize("orientation", ["starting", "ending"])
@pytest.mark.parametrize("variant", ["af3", "af2"])                # af3: no biases, by_column key mask; af2: gating+output biases, by_line key mask, asymmetric pair mask
def test_tri_attn_block_f32_matches_its_reference_in_class(n, c, h, orientation, variant):
    """The f32 route (f32 activations AND parameters; tf32 MMAs, nothing rounded to bf16) against the pure-jnp f32 body at HIGHEST precision.
    Same-class rule: the fused block's rel-rms error <= 1.25 x that of the same pure-jnp body at XLA's DEFAULT matmul precision (the
    'stock-like' comparator: an f32 model's GEMMs run tf32-class at default precision on sm_80+ parts), bit-exact run-to-run, f32 out, NaN-free
    (the padded tail lines are fully masked: uniform average, as a stock softmax). Live lines asserted; fully-masked lines reported (they
    average over kernel_n keys in a padded kernel vs N in the reference)."""
    d = S.head_dim(c, h)
    act, pmask, ln, mw, kp, flags = _attn_case(n, c, h, variant, dtype=jnp.float32)
    assert act.dtype == jnp.float32 and kp["wq_t"].dtype == jnp.float32
    ref = S.tri_attn_reference(act, pmask, **ln, **mw, orientation=orientation, **flags, precision=jax.lax.Precision.HIGHEST)
    stock_like = S.tri_attn_reference(act, pmask, **ln, **mw, orientation=orientation, **flags, precision=None)
    out = jax.block_until_ready(S.tri_attn_block(act, pmask, kp, orientation=orientation, **flags, precision="tf32"))
    out2 = jax.block_until_ready(S.tri_attn_block(act, pmask, kp, orientation=orientation, **flags, precision="tf32"))
    assert out.shape == act.shape and out.dtype == jnp.float32
    assert not bool(jnp.any(jnp.isnan(out)))
    assert bool(jnp.all(out == out2)), "run-to-run bitwise"
    live = n - 32
    e_fused, e_stock = _rel_rms(out[:live, :live], ref[:live, :live]), _rel_rms(stock_like[:live, :live], ref[:live, :live])
    e_dead = _rel_rms(out[live:, live:], ref[live:, live:])
    amax = float(jnp.max(jnp.abs(out[:live, :live] - ref[:live, :live])))
    print(f"triattn f32 {orientation} variant={variant} N={n} (kernel_n {S.kernel_n('triattn', n, '9.0', 'float32')}) C={c} H={h} D={d}: "
          f"rel_rms fused-tf32 vs HIGHEST ref {e_fused:.3e} | stock-like (jnp body, DEFAULT precision) vs ref {e_stock:.3e} | fused max_abs {amax:.3e} | "
          f"fully-masked lines {e_dead:.3e} (not asserted)")
    print(f"    f32_class={S.f32_class('tf32')} (jax line {S.jax_line()}); fused/body ratio {e_fused / max(e_stock, 1e-12):.2f}")
    _assert_f32_class("triattn", e_fused, e_stock)
    assert e_fused <= 6e-3, e_fused                                    # the tf32-word ceiling on any line, whatever the comparator does


@gpu
@pytest.mark.parametrize("precision,ceiling", [("tf32", 5e-3), ("tf32x3", 5e-5), ("ieee", 2e-5)])
def test_tri_attn_block_f32_precision_words_name_their_class(precision, ceiling):
    """Each MMA operand-precision word lowers and lands in its class against the HIGHEST reference (tf32 ~1e-3; tf32x3 / ieee ~f32)."""
    n, c, h = 256, 128, 4
    act, pmask, ln, mw, kp, flags = _attn_case(n, c, h, "af2", dtype=jnp.float32)
    ref = S.tri_attn_reference(act, pmask, **ln, **mw, orientation="ending", **flags, precision=jax.lax.Precision.HIGHEST)
    row = S.f32_line_row() or {}
    if ("triattn", precision) in row.get("unlaunchable", {}):              # a word this jax line cannot launch is REFUSED BY NAME before any launch
        with pytest.raises(S.Refusal) as ei:
            S.tri_attn_block(act, pmask, kp, orientation="ending", **flags, precision=precision)
        assert ei.value.kind == S.PRECISION_UNLAUNCHABLE, ei.value.kind
        print(f"triattn f32 precision={precision}: refused by name on jax {S.jax_line()} ({ei.value.kind})")
        return
    out = jax.block_until_ready(S.tri_attn_block(act, pmask, kp, orientation="ending", **flags, precision=precision))
    e = _rel_rms(out, ref)
    print(f"triattn f32 precision={precision}: rel_rms vs HIGHEST ref {e:.3e} (ceiling {ceiling:.0e}) f32_class={S.f32_class(precision)} jax {S.jax_line()}")
    assert not bool(jnp.any(jnp.isnan(out))) and e <= ceiling, (precision, e)


@gpu
def test_tri_attn_block_f32_fully_masked_input_is_nan_free_and_words_are_refused_by_name():
    n, c, h = 256, 64, 4
    act, pmask, ln, mw, kp, flags = _attn_case(n, c, h, "af2", dtype=jnp.float32)
    out = jax.block_until_ready(S.tri_attn_block(act, jnp.zeros_like(pmask), kp, orientation="starting", **flags, precision="tf32"))
    assert not bool(jnp.any(jnp.isnan(out)))
    bad = dict(kp); bad.pop("bo")
    with pytest.raises(S.Refusal) as e:
        S.tri_attn_block(act, pmask, bad, orientation="starting", **flags, precision="tf32")
    assert e.value.kind == S.BIAS_KEYS_INCOMPLETE
    with pytest.raises(S.Refusal) as e:
        S.tri_attn_block(act, pmask, kp, orientation="starting", **flags, precision="hi")
    assert e.value.kind == S.UNKNOWN_PRECISION


@gpu
@pytest.mark.parametrize("n,c,h", [(256, 128, 4), (448, 128, 4), (512, 128, 4), (1024, 128, 4), (1408, 128, 4), (256, 64, 4), (448, 64, 4), (1024, 64, 4)])
def test_tri_attn_block_f32_kernel_timing_print(n, c, h):
    """A kernel micro-timing inside the test box (not a timing run): the fused f32 block vs the pure-jnp f32 body jitted at DEFAULT
    precision (the XLA stock-like body), same device, interleaved, 20 repetitions after warm-up; prints ms per orientation."""
    import time
    act, pmask, ln, mw, kp, flags = _attn_case(n, c, h, "af2", dtype=jnp.float32)
    for orientation in ("starting", "ending"):
        fused = jax.jit(lambda a, m: S.tri_attn_block(a, m, kp, orientation=orientation, **flags, precision="tf32"))
        rc = next((r for r in (256, 128, 64) if n % r == 0), None) if n >= 1024 else None   # bound the [B,H,S,S] f32 logits (17 GB at N=1024) as a sub-batched model does
        body = jax.jit(lambda a, m: S.tri_attn_reference(a, m, **ln, **mw, orientation=orientation, **flags, precision=None, row_chunk=rc))
        for _ in range(3):
            jax.block_until_ready(fused(act, pmask)); jax.block_until_ready(body(act, pmask))
        tf, tb = [], []
        for _ in range(20):
            t0 = time.perf_counter(); jax.block_until_ready(fused(act, pmask)); tf.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); jax.block_until_ready(body(act, pmask)); tb.append(time.perf_counter() - t0)
        mf, mb = sorted(tf)[len(tf) // 2] * 1e3, sorted(tb)[len(tb) // 2] * 1e3
        print(f"TIMING triattn f32 N={n} C={c} H={h} {orientation}: fused {mf:.3f} ms | jnp body (XLA, DEFAULT precision{', row_chunk=%d' % rc if rc else ''}) {mb:.3f} ms | x{mb / mf:.2f}")


@gpu
def test_key_mask_by_line_is_the_transposed_mask_on_the_starting_node():
    n, c, h = 256, 128, 4
    d = S.head_dim(c, h)
    act, pmask, key = _inputs(n, c, seed=2)
    ks = jax.random.split(key, 8)
    pmask = pmask * (jax.random.uniform(ks[7], (n, n)) > 0.2).astype(pmask.dtype)          # asymmetric
    w = lambda k, shape: (jax.random.normal(k, shape, jnp.float32) / (c ** 0.5)).astype(jnp.bfloat16)  # noqa: E731
    kp = S.attn_params(ln_scale=jnp.ones((c,)), ln_offset=jnp.zeros((c,)), q_w=w(ks[0], (c, h, d)), k_w=w(ks[1], (c, h, d)), v_w=w(ks[2], (c, h, d)),
                       bias_w=w(ks[3], (c, h)), gate_w=w(ks[4], (c, h, d)), out_w=w(ks[5], (h, d, c)))
    a = S.tri_attn_block(act, pmask, kp, orientation="starting", key_mask="by_line")
    b = S.tri_attn_block(act, jnp.swapaxes(pmask, 0, 1), kp, orientation="starting", key_mask="by_column")
    assert bool(jnp.all(a == b))
    a = S.tri_attn_block(act, pmask, kp, orientation="ending", key_mask="by_line")
    b = S.tri_attn_block(act, pmask, kp, orientation="ending", key_mask="by_column")
    assert bool(jnp.all(a == b))


@gpu
def test_attention_core_matches_reference():
    B, S_, H, D = 256, 256, 4, 32
    ks = jax.random.split(jax.random.PRNGKey(3), 4)
    q, k, v = (jax.random.normal(ks[i], (B, S_, H, D), jnp.float32).astype(jnp.bfloat16) for i in range(3))
    bias = jax.random.normal(ks[3], (H, S_, S_), jnp.float32).astype(jnp.bfloat16)
    key_mask = jnp.broadcast_to(jnp.arange(S_) < S_ - 40, (B, S_))
    A = S.kernels()[1]
    ref = A.reference_attention_bshd(q, k, v, bias[None], key_mask[:, None, None, :], precision=jax.lax.Precision.HIGHEST)
    out = jax.block_until_ready(S.attention_core(q, k, v, bias, key_mask))
    e = _rel_rms(out, ref)
    print(f"attention core: rel_rms {e:.3e}")
    assert e <= 1.0e-2, e


@gpu
def test_dtypes_outside_bf16_and_f32_are_refused_by_name_not_cast():
    act, mask, key = _inputs(256, 128)
    p = _trimul_params(128, key, bias=False)
    with pytest.raises(S.Refusal) as e:
        S.trimul_block(act.astype(jnp.float16), mask, p, equation="outgoing")
    assert e.value.kind == S.DTYPE_NOT_SERVED


TRIMUL_F32_CASES = [(256, 128), (256, 64), (512, 128), (448, 128), (400, 128)]     # 400 pads to 448 inside


@gpu
@pytest.mark.parametrize("n,c", TRIMUL_F32_CASES)
@pytest.mark.parametrize("equation", ["outgoing", "incoming"])
@pytest.mark.parametrize("bias", [False, True])                    # True: the AlphaFold 2 parameterisation (projection/gate/output/gating biases)
def test_trimul_block_f32_matches_its_reference_in_class(n, c, equation, bias):
    """The f32 route (f32 activations AND parameters; tf32 MMAs in the two Pallas kernels, the cubic contraction one XLA GEMM at default
    precision, nothing rounded to bf16) against the pure-jnp f32 body at HIGHEST precision. Same-class rule: rel-rms <= 1.25 x that of the
    same body at XLA's DEFAULT precision (the stock-like comparator), bit-exact run-to-run, f32 out, NaN-free."""
    act, mask, p = _trimul_case_f32(n, c, bias)
    assert act.dtype == jnp.float32 and p["w_proj"].dtype == jnp.float32 and (("b_proj" in p) == bias)
    ref = S.trimul_reference(act, mask, p, equation=equation, precision=jax.lax.Precision.HIGHEST)
    stock_like = S.trimul_reference(act, mask, p, equation=equation, precision=None)
    out = jax.block_until_ready(S.trimul_block(act, mask, p, equation=equation, precision="tf32"))
    out2 = jax.block_until_ready(S.trimul_block(act, mask, p, equation=equation, precision="tf32"))
    assert out.shape == act.shape and out.dtype == jnp.float32
    assert not bool(jnp.any(jnp.isnan(out)))
    assert bool(jnp.all(out == out2)), "run-to-run bitwise"
    e_fused, e_stock = _rel_rms(out, ref), _rel_rms(stock_like, ref)
    amax = float(jnp.max(jnp.abs(out - ref)))
    print(f"trimul f32 {equation} bias={bias} N={n} (kernel_n {S.kernel_n('trimul', n, '9.0', 'float32')}) C={c}: rel_rms fused-tf32 vs HIGHEST ref {e_fused:.3e} | "
          f"stock-like (jnp body, DEFAULT precision) vs ref {e_stock:.3e} | fused max_abs {amax:.3e}")
    print(f"    f32_class={S.f32_class('tf32')} (jax line {S.jax_line()}); fused/body ratio {e_fused / max(e_stock, 1e-12):.2f}")
    _assert_f32_class("trimul", e_fused, e_stock)
    assert e_fused <= 2e-3, e_fused                                    # the tf32-word ceiling on any line
    assert e_fused <= 5e-3, e_fused


@gpu
@pytest.mark.parametrize("precision,ceiling", [("tf32", 5e-3), ("tf32x3", 5e-5), ("ieee", 2e-5)])
def test_trimul_block_f32_precision_words_name_their_class(precision, ceiling):
    n, c = 256, 128
    act, mask, p = _trimul_case_f32(n, c, bias=True)
    ref = S.trimul_reference(act, mask, p, equation="incoming", precision=jax.lax.Precision.HIGHEST)
    out = jax.block_until_ready(S.trimul_block(act, mask, p, equation="incoming", precision=precision))
    e = _rel_rms(out, ref)
    print(f"trimul f32 precision={precision}: rel_rms vs HIGHEST ref {e:.3e} (ceiling {ceiling:.0e})")
    assert not bool(jnp.any(jnp.isnan(out))) and e <= ceiling, (precision, e)
    with pytest.raises(S.Refusal) as r:
        S.trimul_block(act, mask, p, equation="incoming", precision="hi")
    assert r.value.kind == S.UNKNOWN_PRECISION


@gpu
@pytest.mark.parametrize("n,c", [(256, 128), (448, 128), (512, 128), (1024, 128), (1408, 128), (256, 64), (448, 64), (1024, 64)])
def test_trimul_block_f32_kernel_timing_print(n, c):
    """Kernel micro-timing inside the test box (not a timing run): the fused f32 block vs the pure-jnp f32 body jitted at DEFAULT
    precision (the XLA stock-like body), same device, interleaved, 20 repetitions after warm-up; ms per equation."""
    import time
    act, mask, p = _trimul_case_f32(n, c, bias=True)
    for equation in ("outgoing", "incoming"):
        fused = jax.jit(lambda a, m: S.trimul_block(a, m, p, equation=equation, precision="tf32"))
        body = jax.jit(lambda a, m: S.trimul_reference(a, m, p, equation=equation, precision=None))
        for _ in range(3):
            jax.block_until_ready(fused(act, mask)); jax.block_until_ready(body(act, mask))
        tf, tb = [], []
        for _ in range(20):
            t0 = time.perf_counter(); jax.block_until_ready(fused(act, mask)); tf.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); jax.block_until_ready(body(act, mask)); tb.append(time.perf_counter() - t0)
        mf, mb = sorted(tf)[len(tf) // 2] * 1e3, sorted(tb)[len(tb) // 2] * 1e3
        print(f"TIMING trimul f32 N={n} C={c} {equation}: fused {mf:.3f} ms | jnp body (XLA, DEFAULT precision) {mb:.3f} ms | x{mb / mf:.2f}")

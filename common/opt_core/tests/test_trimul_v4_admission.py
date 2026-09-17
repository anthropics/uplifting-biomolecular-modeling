"""fpf_trimul_v4 has no size ceiling and no memory admission: the served path is ONE launcher for every N >= N_MIN and every leading batch B
(structural refusals C/D/dtype/mask/N<N_MIN by name), its workspace is a documented formula (generic.workspace_bytes; HAZARDS), and an out-of-memory is
torch's own OutOfMemoryError propagating to the caller — nothing consults device memory, nothing catches or reroutes an OOM.  The launcher's arguments are
held to a recorded fixture recorded from this tree (tests/fixtures/fpf_trimul_v4_launch_args_0517.json; a [N,N,C] call launches the single-plane grids with a batch
axis of extent 1, a [B,N,N,C] call ONE K1 / bmm / K3 set with the batch on grid axis 2).  CPU-only: CUDA queries and kernels are recorded, never launched
(tests/_trimul_v4_launch_recorder.py; `write_launch_record` there regenerates the recorded fixture)."""
import importlib
import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(HERE)
LAUNCH_RECORD = os.path.join(HERE, "fixtures", "fpf_trimul_v4_launch_args_0517.json")
CELLS_FIXTURE = os.path.join(HERE, "fixtures", "fpf_trimul_v4_cells_fixture.json")

torch = pytest.importorskip("torch")
pytest.importorskip("triton")                     # kernels.py decorates its kernels at import (nothing is compiled or launched here)

sys.path.insert(0, HERE)
import _trimul_v4_launch_recorder as RC  # noqa: E402


def _pkg():
    pkg = importlib.import_module("opt_core.kernels.fpf_trimul_v4")
    for sub in ("kernels", "kdesc", "cells", "generic"):
        importlib.import_module("opt_core.kernels.fpf_trimul_v4." + sub)
    return pkg


def test_workspace_bytes_is_a_documented_formula_matching_hand_values():
    G = _pkg().generic
    assert G.workspace_bytes(705, 128) == 3 * 128 * 720 * 720 * 2 + 705 * 705 * 128 * 2 == 525369600          # Np = 720
    assert G.workspace_bytes(4096, 128) == 3 * 128 * 4096 * 4096 * 2 + 4096 * 4096 * 128 * 2 == 17179869184   # 16 GiB at N=4096, C=D=128 = 4 x z_bf16
    assert G.workspace_bytes(101, 256, 256) == 3 * 256 * 112 * 112 * 2 + 101 * 101 * 256 * 2 == 24490496       # Np = 112
    assert G.workspace_bytes(705, 128, 128, elem_out=4) == 3 * 128 * 720 * 720 * 2 + 705 * 705 * 128 * 4       # fp32 z: fp32 output, bf16 planes
    for N in (2048, 3072, 4096):                                                                                  # C = D, bf16, N % 16 == 0: exactly 4 x z_bf16 (the measured peak extra)
        assert G.workspace_bytes(N, 128, 128, 2, 16) == 4 * (N * N * 128 * 2)
    assert G.N_MIN == 101 and not hasattr(G, "N_MAX")


def test_workspace_bytes_is_what_the_launcher_allocates():
    """Single source: the formula equals the bytes kernels.trimul_v4_forward actually requests (a|b planes, x plane, output)."""
    pkg = _pkg()
    for N, C in ((705, 128), (1493, 256), (4096, 128)):
        rec = RC.record_call(pkg, CELLS_FIXTURE, N, C, C, True, False, False)
        allocs = [e for e in rec["log"] if "alloc" in e]
        planes = sum(2 * int(torch.Size(e["shape"]).numel()) for e in allocs if e["alloc"] == "empty")          # bf16 planes
        out = sum(2 * int(torch.Size(e["shape"]).numel()) for e in allocs if e["alloc"] == "empty_like")        # bf16 z -> bf16 out
        assert planes + out == pkg.generic.workspace_bytes(N, C, C, 2, 16), (N, C, planes, out)


def test_every_size_above_n_min_is_served_without_consulting_device_memory():
    pkg = _pkg()
    for N in (2048, 2049, 3072, 4096, 6144, 8192):
        rec = RC.record_call(pkg, CELLS_FIXTURE, N, 128, 128, True, True, True, free_bytes=1)                      # the driver "reports" 1 free byte: irrelevant, nothing asks
        assert "refused" not in rec and rec["Np"] == -(-N // 16) * 16 and [e["kernel"] for e in rec["log"] if "kernel" in e][1:] == ["bmm", "_k3c"], N
    assert RC.record_call(pkg, CELLS_FIXTURE, 100, 128, 128, True, False, False) == {"refused": "N<101"}       # the structural small-N route stays, by name


def test_an_out_of_memory_propagates_uncaught_and_uncounted():
    """No try-then-fallback on OOM: an OutOfMemoryError raised by any of the launcher's allocations reaches the caller of generic.trimul_packed
    as itself (not TrimulUnsupported, not a stock reroute), COUNTS records no refusal for it, and opt_core.trimul's Lever re-raises it instead of
    counting error:* and serving the engine forward."""
    pkg = _pkg(); G = pkg.generic
    torch_oom = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
    w = RC._weights(128, 128)
    for oom_at in (1, 2, 3):                                                                                       # a|b planes | x plane | the output (after K1 + bmm)
        before = json.dumps(G.COUNTS["unsupported"], sort_keys=True)
        z = RC._CudaLike(705, 128, torch.bfloat16)
        real_is_tensor = torch.is_tensor
        with RC.patched(pkg, CELLS_FIXTURE, oom_at=oom_at) as log:
            torch.is_tensor = lambda t: True if isinstance(t, RC._CudaLike) else real_is_tensor(t)
            try:
                with pytest.raises(torch_oom):
                    G.trimul_packed(z, None, outgoing=True, weights=w, residual=True)
            finally:
                torch.is_tensor = real_is_tensor
        assert json.dumps(G.COUNTS["unsupported"], sort_keys=True) == before and {"oom": oom_at} in log
    from opt_core import trimul as T                                                                               # the Lever: an OutOfMemoryError from a provider is re-raised
    class _OOM(Exception):
        pass
    _OOM.__name__ = "OutOfMemoryError"
    def boom(call):
        raise _OOM("CUDA out of memory (simulated)")
    prov = T.custom("boom", boom, eligible=lambda call: None)
    lever = T.Lever("t", "fast", provider=prov, expected=("mode_stock",), stream=open(os.devnull, "w"))
    zt = torch.zeros(128, 128, 8)
    call = T.Call(module=object(), z=zt, mask=None, direction="outgoing", residual=False, orig=lambda: "ENGINE-FORWARD")
    with pytest.raises(_OOM):
        lever.serve(call)
    assert not any(k.startswith("error:") for k in json.dumps(lever.census(), default=str).split('"')), lever.census()
    def other(call):
        raise RuntimeError("a kernel error that is not an OOM")
    lever2 = T.Lever("t2", "fast", provider=T.custom("other", other, eligible=lambda call: None), expected=("mode_stock",), stream=open(os.devnull, "w"))
    assert lever2.serve(call) == "ENGINE-FORWARD"                                                                 # every other provider error keeps the counted, gated engine-forward route


def test_no_size_ceiling_and_no_memory_admission_is_left_in_the_trimul_path():
    pkg_dir = os.path.join(CORE, "opt_core", "kernels", "fpf_trimul_v4")
    files = [os.path.join(pkg_dir, f) for f in os.listdir(pkg_dir) if f.endswith((".py", ".md", ".json"))]
    files += [os.path.join(pkg_dir, "tests", f) for f in os.listdir(os.path.join(pkg_dir, "tests")) if f.endswith(".py")]
    files += [os.path.join(CORE, "opt_core", "trimul.py"), os.path.join(CORE, "README.md")]
    ceiling = re.compile(r'"N>|\'N>|N>%d|N_MAX|NMAX|N <= 2048|<= N <= 2048|"n>|n>%d|\bn_max\b')          # fpf_trimul_v4 and the TM-K3 provider (opt_core.trimul.tmk3) alike
    admission = re.compile(r"mem_get_info|memory_reserved|workspace>|HEADROOM|except[^\n]*OutOfMemoryError|def admit\b|available_bytes")
    offenders = []
    for p in files:
        if os.path.basename(p) == "CHANGELOG.md":
            continue
        for n, line in enumerate(open(p, encoding="utf-8").read().split("\n"), 1):
            if ceiling.search(line) or admission.search(line):
                offenders.append("%s:%d:%s" % (os.path.relpath(p, CORE), n, line.strip()[:100]))
    assert offenders == [], offenders
    adm = open(os.path.join(pkg_dir, "generic.py")).read() + open(os.path.join(pkg_dir, "trimul.py")).read()
    assert "FPF_TRIMUL_V4_NMAX" not in adm and "2048" not in adm and not re.search(r"\bN\s*>\s*\d", adm)          # no size threshold of any value in the served path's gates
    src = open(os.path.join(pkg_dir, "generic.py")).read()
    body = src.split("def _trimul_packed" if "def _trimul_packed" in src else "def trimul_packed", 1)[1].split("\ndef ", 1)[0]   # the body moved behind the first-call trampoline (4.4.3)
    assert "try:" not in body[body.index("outs = []"):]                                                       # no try around the launcher call (the only try in that function guards the structural _check)


@pytest.mark.parametrize("stack", ["3.3", "3.7"])
def test_launch_arguments_for_admitted_sizes_equal_the_record(stack):
    """Every served [N,N,C] call for N in {101..2048} x C in {128,256} x direction x mask x residual selects the same cells row, resolves the same k1/k3,
    pads to the same Np, allocates the same planes, and launches K1 / bmm / K3 with the same grids, constexpr tiles and argument shapes as the
    recorded fixture recorded from this tree's launcher: an edit that changes any launch argument changes this record deliberately or fails here."""
    recorded = json.load(open(LAUNCH_RECORD))
    assert recorded["meta"]["sizes"] == [101, 256, 400, 705, 1000, 1400, 2048] and recorded["meta"]["channels"] == [128, 256]
    import hashlib
    assert hashlib.sha256(open(CELLS_FIXTURE, "rb").read()).hexdigest() == recorded["meta"]["cells_fixture_sha256"]
    cur = RC.record_all(_pkg(), CELLS_FIXTURE, tuple(recorded["meta"]["sizes"]), tuple(recorded["meta"]["channels"]), triton_mm=stack)
    ref = recorded["records"][stack]
    assert sorted(cur) == sorted(ref) and len(ref) == 112
    bad = [k for k in ref if cur[k] != ref[k]]
    assert bad == [], (bad[:5], cur[bad[0]] if bad else None, ref[bad[0]] if bad else None)
    assert not any("refused" in v for v in cur.values())


@pytest.mark.parametrize("stack", ["3.3", "3.7"])
def test_launch_arguments_above_2048_equal_their_record(stack):
    """N = 3072 and 4096 (the sizes exp2 qualifies above 2048) x C in {128,256} x direction x mask x residual: served, and their launch arguments
    (same recorder) equal the record in the record's `above_2048` section — the same launcher arithmetic continued past the former ceiling."""
    recorded = json.load(open(LAUNCH_RECORD))
    ref = recorded["above_2048"][stack]
    cur = RC.record_all(_pkg(), CELLS_FIXTURE, (3072, 4096), (128, 256), triton_mm=stack)
    assert sorted(cur) == sorted(ref) and len(ref) == 32 and not any("refused" in v for v in cur.values())
    bad = [k for k in ref if cur[k] != ref[k]]
    assert bad == [], bad[:5]
    r = cur["N4096_C128_out_mask_res"]
    assert r["Np"] == 4096 and r["log"][0]["shape"] == [2, 1, 128, 4096, 4096] and [e["grid"][1:] for e in r["log"] if e.get("kernel") in ("_k1c", "_k1t", "_k1d")] == [[4096, 1]]


@pytest.mark.parametrize("stack", ["3.3", "3.7"])
def test_a_batched_call_is_one_launch_set_and_equals_its_record(stack):
    """z [B,N,N,C] (B = 8): ONE K1 launch with the batch on grid axis 2, planes allocated [2, B, D, Np, Np] and [B, D, Np, Np], ONE bmm whose operands are the
    [B*D, Np, Np] plane views, ONE K3 launch (grid axis 2 = B), the [B,N,N,C] output; workspace_bytes(..., B=B) is exactly those allocations; and every
    launch argument equals the record's `batched` record."""
    recorded = json.load(open(LAUNCH_RECORD)); ref = recorded["batched"][stack]; pkg = _pkg(); B = RC.RECORD_BATCH["B"]
    cur = RC.record_all(pkg, CELLS_FIXTURE, RC.RECORD_BATCH["sizes"], (128, 256), B=B, triton_mm=stack)
    assert sorted(cur) == sorted(ref) and len(ref) == 48 and not any("refused" in v for v in cur.values())
    bad = [k for k in ref if cur[k] != ref[k]]
    assert bad == [], bad[:5]
    for N, C in ((101, 128), (705, 256)):
        r = cur["N%d_C%d_out_mask_res_B%d" % (N, C, B)]; Np = -(-N // 16) * 16
        kern = [e for e in r["log"] if "kernel" in e]
        assert [e["kernel"] for e in kern] in (["_k1c", "bmm", "_k3c"], ["_k1t", "bmm", "_k3c"], ["_k1d", "bmm", "_k3d"]), kern       # 705 x bf16 on the 3.7 row: the descriptor cells
        assert kern[0]["grid"][1:] == [Np, B] and kern[2]["grid"][1:] == [N, B]
        assert kern[1]["a"] == [B * C, Np, Np] and kern[1]["out"] == [B * C, Np, Np]
        allocs = [e for e in r["log"] if "alloc" in e and len(e["shape"]) >= 3]          # the workspaces (the descriptor K3's two [C, D] / [C, C] weight transposes are made once per PACK, not per call)
        assert [e["shape"] for e in allocs] == [[2, B, C, Np, Np], [B, C, Np, Np], [B, N, N, C]]
        assert sum(2 * int(torch.Size(e["shape"]).numel()) for e in allocs) == pkg.generic.workspace_bytes(N, C, C, 2, 16, B=B)


def _kernel_launches(log):
    return [e for e in log if e.get("kernel") in ("_k1c", "_k1t", "_k1d", "_k3c", "_k3d")]


def test_the_int32_row_edge_of_the_tma_launch_set_is_a_named_boundary():
    """The TMA K1 addresses the batch's B*N*N token rows with an int32 descriptor coordinate.  AT the boundary on the TMA stack (3.7, bf16, C 128, N 256 -> 65536 rows per
    plane): B = 32767 (2147418112 rows <= 2^31 - 1) is ONE launch set; B = 32768 (2^31 rows) is served by generic as 32768 single-plane launch sets, COUNTED BY NAME
    (COUNTS['batch_loop_fallback']['tma_rows>int32'] == 1) — and no recorded launch carries a row extent above 2^31 - 1; the kernel entry itself refuses that batch by name
    (kernels.BatchLimit 'tma_rows>int32') with NOTHING launched or allocated.  The pointer K1 (stack 3.3; int64 offsets) serves B = 32768 as one launch set."""
    pkg = _pkg(); K = pkg.kernels; N, C = 256, 128; INT32_MAX = 2 ** 31 - 1
    b_max = INT32_MAX // (N * N); assert b_max == 32767 and b_max * N * N <= INT32_MAX < (b_max + 1) * N * N
    ok = RC.record_generic_call(pkg, CELLS_FIXTURE, N, C, C, b_max, triton_mm="3.7")                       # largest servable batch: one K1 (grid axis 2 = B), one bmm, one K3
    kl = _kernel_launches(ok["log"])
    assert "refused" not in ok and "batch_limit" not in ok and [e["kernel"] for e in kl] == ["_k1t", "_k3c"] and [e["grid"][2] for e in kl] == [b_max, b_max], kl
    assert ok["counts"]["batch_loop_fallback"] == {} and ok["counts"]["batched_calls"] == 1
    assert max(a for e in kl for a in e["int_args"]) <= INT32_MAX and b_max * N * N in kl[0]["int_args"]          # the descriptor's row extent B*N*N is an int32 value
    over = RC.record_generic_call(pkg, CELLS_FIXTURE, N, C, C, b_max + 1, triton_mm="3.7")                 # first batch over the edge: the loop, by name, counted once
    kl = _kernel_launches(over["log"])
    assert "refused" not in over and "batch_limit" not in over
    assert over["counts"]["batch_loop_fallback"] == {"tma_rows>int32": 1} and over["counts"]["batched_calls"] == 0
    assert len(kl) == 2 * (b_max + 1) and {tuple(e["grid"][1:]) for e in kl if e["kernel"] == "_k1t"} == {(N, 1)}          # every plane its own single-plane launch (grid axis 2 = 1)
    assert max(a for e in kl for a in e["int_args"]) <= INT32_MAX and {e["int_args"][e["int_args"].index(N * N)] for e in kl if e["kernel"] == "_k1t"} == {N * N}   # no launch carries 2^31 rows
    direct = RC.record_generic_call(pkg, CELLS_FIXTURE, N, C, C, b_max + 1, triton_mm="3.7", kernel_entry=True)   # the kernel entry: refusal by name, nothing launched, nothing allocated
    assert direct.get("batch_limit") == "tma_rows>int32" and _kernel_launches(direct["log"]) == [] and [e for e in direct["log"] if "alloc" in e] == []
    ptr = RC.record_generic_call(pkg, CELLS_FIXTURE, N, C, C, b_max + 1, triton_mm="3.3")                  # the pointer K1 has no int32 row coordinate: one launch set
    kl = _kernel_launches(ptr["log"])
    assert [e["kernel"] for e in kl] == ["_k1c", "_k3c"] and kl[0]["grid"][2] == b_max + 1 and ptr["counts"]["batch_loop_fallback"] == {}
    assert K.batch_launch_limit(b_max, N, {"impl": "tma"}, False) is None and K.batch_launch_limit(b_max + 1, N, {"impl": "tma"}, False) == "tma_rows>int32"
    assert K.batch_launch_limit(b_max + 1, N, {"impl": "tma"}, True) is None and K.batch_launch_limit(b_max + 1, N, {"impl": "ptr"}, False) is None   # fp32 z runs the pointer K1


def test_the_cuda_grid_batch_edge_is_a_named_boundary():
    """Grid axis 2 carries the batch: B = 65535 is one launch set, B = 65536 is refused by name at the kernel entry ('grid_batch>65535', nothing launched) and is the named
    loop fallback in generic (kernels.batch_launch_limit) — checked at N = 101 on the pointer stack, where the int32 row edge does not bind first."""
    pkg = _pkg(); K = pkg.kernels; N, C = 101, 128
    assert K.batch_launch_limit(65535, N, {"impl": "ptr"}, False) is None and K.batch_launch_limit(65536, N, {"impl": "ptr"}, False) == "grid_batch>65535"
    one = RC.record_generic_call(pkg, CELLS_FIXTURE, N, C, C, 65535, triton_mm="3.3", kernel_entry=True)
    kl = _kernel_launches(one["log"]); assert "batch_limit" not in one and [e["grid"][2] for e in kl] == [65535, 65535]
    over = RC.record_generic_call(pkg, CELLS_FIXTURE, N, C, C, 65536, triton_mm="3.3", kernel_entry=True)
    assert over.get("batch_limit") == "grid_batch>65535" and _kernel_launches(over["log"]) == [] and [e for e in over["log"] if "alloc" in e] == []
    assert K.BATCH_LIMITS == ("tma_rows>int32", "grid_batch>65535")


def test_one_pair_representation_over_the_int32_row_edge_is_refused_by_name_without_a_launch():
    """N*N > 2^31 - 1 token rows in ONE pair representation (N = 46341; z >= 512 GiB) is beyond the TMA launch set and beyond the per-b loop alike: generic._check refuses it
    by the same name ('tma_rows>int32', a TrimulUnsupported counted in COUNTS['unsupported']) before any launch; N = 46340 (46340^2 = 2147395600 <= 2^31 - 1) is served.
    kernels.BatchLimit is a TrimulUnsupported: one `except` covers the kernel entry and the module-agnostic entry."""
    pkg = _pkg(); K, G = pkg.kernels, pkg.generic
    assert issubclass(K.BatchLimit, G.TrimulUnsupported) and G.TrimulUnsupported is K.TrimulUnsupported
    assert 46340 ** 2 <= 2 ** 31 - 1 < 46341 ** 2
    over = RC.record_generic_call(pkg, CELLS_FIXTURE, 46341, 128, 128, 1, triton_mm="3.7")
    assert over.get("refused") == "tma_rows>int32" and _kernel_launches(over["log"]) == [] and [e for e in over["log"] if "alloc" in e] == []
    assert over["counts"]["unsupported"].get("tma_rows>int32") == 1
    ok = RC.record_generic_call(pkg, CELLS_FIXTURE, 46340, 128, 128, 1, triton_mm="3.7")
    assert "refused" not in ok and [e["kernel"] for e in _kernel_launches(ok["log"])] == ["_k1t", "_k3c"]
    ptr = RC.record_generic_call(pkg, CELLS_FIXTURE, 46341, 128, 128, 1, triton_mm="3.3")          # the pointer K1: int64 offsets, no such edge
    assert "refused" not in ptr and [e["kernel"] for e in _kernel_launches(ptr["log"])] == ["_k1c", "_k3c"]


def test_the_descriptor_cells_serve_plane_extents_at_or_above_their_gate():
    """The host-descriptor cells (K1 'tma2' = kdesc._k1d, K3 'tma' = kdesc._k3d; the cells fixture's 9.0|3.7 row) serve a bf16 call whose plane extent Np is at least
    kdesc.N_MIN_DESC (512): below it (N 384 -> Np 384) and for fp32 input the replaced kernels of the same numbers run (K1 'tma' = _k1t or the pointer _k1c, K3 = _k3c); at and
    above it (N 512, 705) _k1d / _k3d run, and on the INCOMING direction K1 writes transposed planes, so the contraction is the outgoing GEMM `a @ b^T` (launch_bmm planes_t) instead
    of `a^T @ b` — visible in the recorded operand strides.  The tally kernels.LAUNCHES names which kernel ran."""
    pkg = _pkg(); K, KD = pkg.kernels, pkg.kdesc; C = 128
    assert KD.N_MIN_DESC == 512 and K.DESC_IMPLS == ("tma", "tma2")
    def kernels_of(N, outgoing, dtype="bfloat16", stack="3.7"):
        rec = RC.record_call(pkg, CELLS_FIXTURE, N, C, C, outgoing, True, True, dtype_name=dtype, triton_mm=stack)
        assert "refused" not in rec, rec
        return [e for e in rec["log"] if "kernel" in e], rec
    for N in (384, 400):                                                                      # Np 384 / 400 < 512: the replaced kernels
        kl, _ = kernels_of(N, True); assert [e["kernel"] for e in kl] == ["_k1t", "bmm", "_k3c"], kl
    for N in (512, 705):                                                                      # Np 512 / 720: the descriptor kernels, both directions
        for outgoing in (True, False):
            kl, rec = kernels_of(N, outgoing)
            assert [e["kernel"] for e in kl] == ["_k1d", "bmm", "_k3d"], (N, outgoing, kl)
            k1d, bmm, k3d = kl
            Np = rec["Np"]; assert Np == K.ceil_to(N, 16)
            assert k1d["grid"] == [-(-Np // 128), Np, 1] and k1d["constexpr"]["BM"] == 128 and k1d["constexpr"]["BN"] == 32 and k1d["constexpr"]["TRANS"] == int(not outgoing)
            assert k3d["grid"] == [-(-N // 64), N, 1] and k3d["constexpr"]["BM"] == 64 and k3d["constexpr"]["BN"] == 64
            zview = k1d["descriptors"][0]                                                    # the z box: outgoing reads [B*N*N, C] rows, incoming the transposed [B*N, N*C] view
            assert zview["box"] == [128, C] and zview["view"] == ([N * N, C] if outgoing else [N, N * C])
            assert [dsc["box"] for dsc in k3d["descriptors"]] == [[C, 64], [64, C], [64, 64], [1, 64, 64], [C, 64], [C, 64]]      # x [D, BM], z, z-residual, out [1, BM, BN], woT, wogT
            assert bmm["b_strides"][-2:] == [1, Np]                                           # a @ b^T for BOTH directions (transposed incoming planes)
    kl, _ = kernels_of(705, False, stack="3.3")                                               # the pointer stack: incoming is a^T @ b
    assert [e["kernel"] for e in kl] == ["_k1c", "bmm", "_k3c"] and kl[1]["a_strides"][-2:] == [1, 720]
    kl, _ = kernels_of(705, True, dtype="float32")                                            # fp32 input: the pointer K1 / K3 at every size
    assert [e["kernel"] for e in kl] == ["_k1c", "bmm", "_k3c"], kl
    assert set(K.LAUNCHES) == {"k1d", "k1t", "k1c", "k3d", "k3c"} and pkg.generic.COUNTS["kernels"] is K.LAUNCHES        # one tally, aliased into the exit COUNTS
    before = dict(K.LAUNCHES); kernels_of(705, False); kernels_of(384, True)
    assert {k: K.LAUNCHES[k] - before[k] for k in K.LAUNCHES} == {"k1d": 1, "k1t": 1, "k1c": 0, "k3d": 1, "k3c": 1}


def test_the_int32_row_edge_binds_the_descriptor_k3_as_well():
    """A row whose K3 alone is a descriptor cell is inside the 'tma_rows>int32' launch-set limit like a descriptor K1 (kernels.batch_launch_limit(B, N, k1, in_f32, k3)); on the
    fixture's 9.0|3.7 row at N = 512 (Np 512, both kernels descriptor cells) B = 8191 is one launch set of _k1d / _k3d, B = 8192 (2^31 rows) is generic's named per-b loop and
    the kernel entry's refusal by name with nothing launched or allocated; kdesc.serves itself declines a plane extent whose Np*Np exceeds int32 (N 46340 -> Np 46352)."""
    pkg = _pkg(); K, KD = pkg.kernels, pkg.kdesc; N, C = 512, 128; INT32_MAX = 2 ** 31 - 1
    assert K.INT32_MAX == INT32_MAX == pkg.table.INT32_MAX
    b_max = INT32_MAX // (N * N); assert b_max == 8191
    assert K.batch_launch_limit(b_max + 1, N, {"impl": "ptr"}, False, {"impl": "tma"}) == "tma_rows>int32"           # K3 descriptor cell alone binds
    assert K.batch_launch_limit(b_max + 1, N, {"impl": "ptr"}, False, {"impl": "ptr"}) is None and K.batch_launch_limit(b_max, N, {"impl": "tma2"}, False, {"impl": "tma"}) is None
    ok = RC.record_generic_call(pkg, CELLS_FIXTURE, N, C, C, b_max, triton_mm="3.7")
    kl = _kernel_launches(ok["log"]); assert [e["kernel"] for e in kl] == ["_k1d", "_k3d"] and [e["grid"][2] for e in kl] == [b_max, b_max], kl
    over = RC.record_generic_call(pkg, CELLS_FIXTURE, N, C, C, b_max + 1, triton_mm="3.7")
    assert over["counts"]["batch_loop_fallback"] == {"tma_rows>int32": 1} and {e["kernel"] for e in _kernel_launches(over["log"])} == {"_k1d", "_k3d"}
    direct = RC.record_generic_call(pkg, CELLS_FIXTURE, N, C, C, b_max + 1, triton_mm="3.7", kernel_entry=True)
    assert direct.get("batch_limit") == "tma_rows>int32" and _kernel_launches(direct["log"]) == [] and [e for e in direct["log"] if "alloc" in e] == []
    import torch
    z = RC._CudaLike(46340, C, torch.bfloat16)
    with RC.patched(pkg, CELLS_FIXTURE, triton_mm="3.7"):
        assert KD.serves(z, 46352, {"impl": "tma2", "BM": 128}, {"impl": "tma"}) == (False, False) and KD.serves(z, 46336, {"impl": "tma2", "BM": 128}, {"impl": "tma"}) == (True, True)

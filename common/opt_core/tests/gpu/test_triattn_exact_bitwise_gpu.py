"""GPU: the carried `triattn_exact` package against the live cuEquivariance op on this device.  For every `proven` cell of the cell table in
force (the carried CELLS.json, or the file `TRIATTN_EXACT_CELLS` names) that covers this (library version, ops build, device class): the face's
output equals the library op BIT FOR BIT on a grid of shapes inside the cell (contiguous and engine-style strided q / k / v, bias in fp32 and in
the input dtype, key mask with a fully-masked row, no mask), and the provider's row serves those calls (`served n/n`).  Calls outside every
proven cell are refused BY NAME and served by the stock callable through `exact_member.serve` (bitwise the library op by construction).  With an
cell table that covers nothing on the device the equality test is skipped by name and the refusal test still runs."""
import math
import os

import pytest

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def _lib():
    tri = pytest.importorskip("cuequivariance_torch.primitives.triangle")
    return tri.triangle_attention


def _face():
    from opt_core.kernels.triattn import exact_member as EM
    return EM, EM.load()


def _make(N, H, S, D=32, *, strided=False, bias_dtype=torch.float32, masked=True, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if strided:                                             # engine layout: [S, N, H*D] projections viewed [S, N, H, D] and permuted to [1, N, H, S, D]
        def one():
            return torch.randn(S, N, H * D, generator=g).to("cuda", dtype).view(S, N, H, D).permute(1, 2, 0, 3).unsqueeze(0)
        q, k, v = one(), one(), one()
    else:
        q, k, v = (torch.randn(1, N, H, S, D, generator=g).to("cuda", dtype) for _ in range(3))
    bias = torch.randn(1, 1, H, S, S, generator=g).to("cuda", bias_dtype)
    mask = None
    if masked:
        mask = (torch.rand(1, N, 1, 1, S, generator=g) > 0.15).to("cuda")
        mask[0, 0] = False                                  # a fully-masked row
    return q, k, v, bias, mask, 1.0 / math.sqrt(D)


def _covering_cells(EM, face):
    """The proven cells whose (lib_version, ops_build, device family or class, environment) match this process."""
    lib, build = face.library_version(), face.ops_build()
    dev = face.device_class(torch.device("cuda", torch.cuda.current_device()))
    fam = face.device_family(dev) if hasattr(face, "device_family") else dev
    env = os.environ.get("CUEQ_TRIATTN_FALLBACK_THRESHOLD")
    out = []
    for c in EM.proven_cells():
        w = c.get("when", {})
        as_list = lambda x: x if isinstance(x, list) else [x]                  # noqa: E731  scalar or list in the table
        if not ({None, "any", lib} & set(as_list(w.get("lib_version")))) or not ({None, "any", build} & set(as_list(w.get("ops_build")))):
            continue
        if "device_family" in w:
            fams = w["device_family"]; fams = fams if isinstance(fams, list) else [fams]
            if fam not in fams and "any" not in fams:
                continue
        else:
            dc = w.get("device_class"); dc = dc if isinstance(dc, list) else [dc]
            if dev not in dc and "any" not in dc and dc != [None]:
                continue
        fte = w.get("fallback_threshold_env", "unset")            # the cell's environment predicate: 'unset' | {'max': n} | '<value>'
        if fte == "unset" and env is not None:
            continue
        if isinstance(fte, dict) and (env is None or int(env) > int(fte.get("max", -1))):
            continue
        if isinstance(fte, str) and fte not in ("unset", "any") and env != fte:
            continue
        out.append(c)
    return out, (lib, build, fam, env)


@cuda
def test_face_equals_the_library_bitwise_inside_every_covering_proven_cell():
    lib = _lib()
    EM, face = _face()
    cells, key = _covering_cells(EM, face)
    if not cells:
        pytest.skip(f"no proven cell covers this process {key} (cells in force: {len(EM.cells())}, proven: {len(EM.proven_cells())})")
    EM.reset_counts()
    stock = lambda q, k, v, bias, mask=None, scale=None: lib(q, k, v, bias, mask=mask, scale=scale)   # noqa: E731 -- the kit's stock callable
    n = 0
    for c in cells:
        w = c["when"]
        D = int(w.get("head_dim", 32)); s_min = int(w.get("S_min", 101)); s_max = w.get("S_max")
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[w.get("dtype", "bfloat16")]
        sizes = [S for S in (max(s_min, 104), 128, 199, 256, 384, 512, 768) if S >= s_min and (s_max is None or S <= int(s_max))]
        for S in sizes:
            for H in (2, 4, 8):
                N = S + 8 if not w.get("N_equals_S", False) else S          # the non-square class unless the cell says square
                for strided in (False, True):
                    for bias_dtype in (torch.float32, dt):
                        for masked in (True, False):
                            q, k, v, bias, mask, sc = _make(N, H, S, D, strided=strided, bias_dtype=bias_dtype, masked=masked, seed=n, dtype=dt)
                            ref = lib(q, k, v, bias, mask=mask, scale=sc)
                            out = EM.serve(q, k, v, bias, mask, sc, stock)
                            assert out.dtype == ref.dtype and out.shape == ref.shape, (c["id"], S, H, strided, bias_dtype, masked)
                            assert torch.equal(out, ref), (c["id"], S, H, strided, str(bias_dtype), masked, (out.float() - ref.float()).abs().max().item())
                            n += 1
    counts = EM.counts()
    assert counts["served"] == n and not counts["refused"], EM.evidence_line()


@cuda
def test_calls_outside_every_proven_cell_are_refused_by_name_and_the_stock_callable_serves_them():
    lib = _lib()
    EM, face = _face()
    R = EM.refused_class()
    seen = []
    stock = lambda q, k, v, bias, mask=None, scale=None: (seen.append(1), lib(q, k, v, bias, mask=mask, scale=scale))[1]   # noqa: E731
    EM.reset_counts()
    # head_dim 64 is outside every proven cell of this package
    q, k, v, bias, mask, sc = _make(136, 4, 128, D=64)
    with pytest.raises(R):
        face.triangle_attention(q, k, v, bias, mask=mask, scale=sc)
    out = EM.serve(q, k, v, bias, mask, sc, stock)
    assert torch.equal(out, lib(q, k, v, bias, mask=mask, scale=sc)) and seen == [1]
    c = EM.counts()
    assert c["served"] == 0 and sum(c["refused"].values()) == 1, EM.evidence_line()
    # an fp32 call (no proven cell of this checkpoint) -- refused by name, the stock callable serves it
    q, k, v, bias, mask, sc = _make(136, 4, 128, dtype=torch.float32)
    out = EM.serve(q, k, v, bias, mask, sc, stock)
    assert torch.equal(out, lib(q, k, v, bias, mask=mask, scale=sc)) and len(seen) == 2
    assert "served 0/2" in EM.evidence_line(), EM.evidence_line()

"""Lever token_agg: the atom-attention encoder's atom -> token aggregation as one kernel (opt/forward/dtk/af3t_token_agg.py).
The kernel module's stated arithmetic (`reference`) equals xfold's two stock statements on the same operands (CPU, in its own interpreter: the
kit's model tree never enters this process); the lever is registered where the kit reads levers; GPU (AF3T_GPU_TESTS=1): the Triton kernel
equals that arithmetic to fp32 summation order and sample s of a batched call is bitwise the single-sample call."""
import os
import subprocess
import sys
import pytest
from af3_torch_opt import registry, stack

HERE = os.path.dirname(os.path.abspath(__file__))
DTK = os.path.join(stack.forward_dir(), "dtk")
KIT = os.path.join(stack.forward_dir(), "af3t", "af3_torch")

_COMMON = r"""
import sys, torch
sys.path.insert(0, DTK); sys.path.insert(0, KIT)
import af3t_token_agg as TA
from xfold.nn import atom_layout, utils

def problem(S, N, Q, C, A=24, seed=0, device="cpu", dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    M = Q * 32
    x = torch.randn((S, Q, 32, C), generator=g).to(device=device, dtype=dtype)
    idx = torch.randint(0, M, (N, A), generator=g).to(device)
    tmask = (torch.rand((N, A), generator=g) < 0.6).to(device)
    tmask[:, 0] = True; tmask[N - 1] = False                                   # a token with no atoms (the clamp(eps) path)
    gmask = tmask & (torch.rand((N, A), generator=g) < 0.95).to(device)       # the gather mask may clear extra slots
    return x, idx, gmask, tmask

def stock(x, idx, gmask, tmask):
    # xfold's statements (AtomCrossAttEncoder.forward): atom_layout.convert to the token-atoms layout, then utils.mask_mean of relu
    gi = atom_layout.GatherInfo(gather_idxs=idx, gather_mask=gmask, input_shape=torch.tensor([x.shape[-3], x.shape[-2]]))
    taa = atom_layout.convert(gi, x, layout_axes=(-3, -2))
    m = tmask.reshape((1,) * (taa.dim() - 3) + tuple(tmask.shape))[..., None]
    return utils.mask_mean(m.to(torch.float32), torch.relu(taa), dim=-2)
"""

_CPU_SCRIPT = _COMMON + r"""
x, idx, gmask, tmask = problem(3, 37, 5, 48)
ref = TA.reference(x.reshape(x.shape[0], -1, x.shape[-1]), idx, gmask, tmask)
st = stock(x, idx, gmask, tmask).to(torch.float32)
ops = TA.operands(idx, gmask, tmask)
print("RESULT", tuple(ref.shape) == tuple(st.shape) == (3, 37, 48), float((ref - st).abs().max()), tuple(ops["agg_w"].shape), tuple(ops["agg_inv"].shape),
      str(ops["agg_idx"].dtype) == str(idx.dtype), float(ops["agg_inv"][36]))
"""

_GPU_SCRIPT = _COMMON + r"""
out_lines = []
for dtype in (torch.bfloat16, torch.float32):
    x, idx, gmask, tmask = problem(5, 211, 29, 768, device="cuda", dtype=dtype, seed=1)
    xf = x.reshape(5, -1, 768)
    ops = TA.operands(idx, gmask, tmask)
    out = TA.token_mean_relu(xf, ops["agg_idx"], ops["agg_w"], ops["agg_inv"])
    st = stock(x, idx, gmask, tmask).to(torch.float32)
    one = TA.token_mean_relu(xf[2:3], ops["agg_idx"], ops["agg_w"], ops["agg_inv"])
    print("RESULT", str(dtype), tuple(out.shape) == tuple(st.shape), str(out.dtype), float((out - st).abs().max()), float(st.abs().max()), int(torch.equal(one[0], out[2])))
"""


def _run(script, timeout=600):
    src = "DTK = %r; KIT = %r\n" % (DTK, KIT) + script
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=timeout)
    lines = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    assert r.returncode == 0 and lines, (r.returncode, r.stdout[-2000:], r.stderr[-3000:])
    return [l.split() for l in lines]


def test_reference_is_the_stock_statements():
    """The kernel's stated arithmetic (af3t_token_agg.reference) IS the two stock statements, and the hoisted operands have the stated shapes."""
    (row,) = _run(_CPU_SCRIPT)
    _, shapes_ok, maxdiff, w_shape, inv_shape, idx_dtype_ok, inv_empty = row[0], row[1], float(row[2]), " ".join(row[3:5]), row[5], row[6], float(row[7])
    assert shapes_ok == "True" and idx_dtype_ok == "True", row
    assert maxdiff <= 1e-6, row
    assert "(37, 24)" in w_shape and inv_shape == "(37,)", row
    assert inv_empty == pytest.approx(1e10), row                               # no atoms: 1 / eps (times a zero sum)


def test_lever_is_registered_where_the_kit_reads_levers():
    assert "token_agg" in registry.LEVERS and "token_agg" in registry.NOT_BITWISE and registry.EVIDENCE["token_agg"] == "census"
    assert registry.IMPL["token_agg"] == ("af3t_token_agg.token_mean_relu", "kit") and registry.STRATEGY["token_agg"].startswith("LOCAL.")
    assert "token_agg" not in registry.EXACT
    api = open(os.path.join(KIT, "af3_torch_api.py"), encoding="utf-8").read()
    assert '"token_agg"' in api[api.index('"fastest": ('):api.index("\n", api.index('"fastest": ('))] and "def enable_token_agg(model)" in api and 'model._token_agg_state = enable_token_agg(model) if "token_agg" in levers else None' in api
    enc = open(os.path.join(KIT, "xfold", "nn", "atom_cross_attention.py"), encoding="utf-8").read()
    assert "TOKEN_AGG = None" in enc and "def _token_aggregate(self" in enc and 'cnt["refused"][why]' in enc          # bound by the api; refusals named per call
    assert os.path.isfile(os.path.join(DTK, "af3t_token_agg.py"))
    fwd = open(os.path.join(HERE, "..", "forward.py"), encoding="utf-8").read()
    assert '("token_agg", "composes:row_local")' in fwd and 'census["token_agg"]' in fwd and 'out["token_agg"] = ta' in fwd   # rowpair gate, census, build-time dead word


@pytest.mark.skipif(os.environ.get("AF3T_GPU_TESTS") != "1", reason="GPU numerics test: AF3T_GPU_TESTS=1 on a CUDA box with the kit's torch venv")
def test_kernel_equals_the_statements_and_is_sample_consistent_on_gpu():
    rows = _run(_GPU_SCRIPT, timeout=900)
    assert len(rows) == 2, rows
    for row in rows:
        _, dtype, shape_ok, out_dtype, maxdiff, scale, same = row
        assert shape_ok == "True" and out_dtype == "torch.float32", row
        assert float(maxdiff) <= 1e-5 * float(scale), row                        # fp32 summation order only
        assert same == "1", row                                                   # sample s of the batched call == the single-sample call, bit for bit

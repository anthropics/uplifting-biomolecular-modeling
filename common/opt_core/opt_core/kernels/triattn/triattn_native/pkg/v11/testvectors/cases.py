"""Test vectors of triattn_pkg: the case table, the deterministic input generator, the fp64 reference, digests.

Inputs are NOT stored: they are regenerated from `torch.Generator("cpu").manual_seed(seed)` (CPU RNG, fp32 draws, then cast) and every tensor's
bytes are checked against the SHA-256 recorded in manifest.json before use -- a mismatch is a named failure (`input_digest_mismatch`), never a silent
comparison against the wrong operands.  Stored per case: the package's output bytes on the device class the case belongs to (`<id>.expected.pt`;
for the square N = S cases whose output is large, rows 0-7 + the SHA-256 of the full output), the fp64 reference of pair row 0 as fp32
(`<id>.ref_rows.pt`), and in manifest.json the error of the full expected output against the full fp64 reference as measured at generation.

Every case names the compute capabilities it is a vector for (`cc`): the 17 cases of generation 10 are cc 9.0 vectors (H100; ids without a prefix,
inputs and expected bytes identical to generation 10's), the `a80_*` cases are cc 8.0 vectors (A100; the sm_80 member's routes).  test_pkg.py runs the
cases of the device's cc; the generator on one device class never touches the other's files.

Case classes mirror the bench harness: bias "bf16x" = fp32 tensor whose values are bf16-representable (what a trunk hands: the fp32 view of a bf16
GEMM output), "fp32" = generic fp32 values, "bias16" = the same bf16x values handed as a bf16 tensor; mask "prefix" = the last round(0.07 S) keys of
every row masked (one length for all rows), "prefixvar" = a different kept length per pair row drawn in [S - 3 round(0.07 S), S] with row 0 full;
layout "endstrided" = q/k/v stored [B, S, H, N, D] and handed as the dim-1 <-> dim-3 transposed views (an ending-node call's stride pattern; no
copies), "start" = contiguous [B, N, H, S, D].
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch

PAD_FRAC = 0.07
BIAS_SCALE = 0.5
REF_ROWS = 1                     # pair rows of the fp64 reference stored per case (fp32)
DIGEST_ROWS = 8                  # pair rows of the expected output stored for store="digest_rows8" cases
A80 = ("8.0",)                   # cc list of the A100 vectors


@dataclass(frozen=True)
class Case:
    id: str
    N: int
    S: int
    H: int = 4
    D: int = 32
    B: int = 1
    dtype: str = "bf16"          # bf16 | fp16
    mask: str = "none"           # none | prefix | prefixvar
    bias: str = "bf16x"          # bf16x | fp32 | bias16
    layout: str = "start"        # start | endstrided
    seed: int = 0
    store: str = "full"          # full | digest_rows8 (expected output too large to ship whole)
    note: str = ""
    cc: Tuple[str, ...] = ("9.0",)   # compute capabilities this case is a vector for (expected bytes are made on, and compared on, such a device)


CASES = [
    Case("s256_none",         N=8,   S=256,  seed=1,  note="k13 route (bf16 D32 S<512 contiguous)"),
    Case("s256_prefix",       N=8,   S=256,  seed=2,  mask="prefix"),
    Case("n256_s256_none",    N=256, S=256,  seed=3,  store="digest_rows8", note="square pair representation, N = S (every other case has N != S)"),
    Case("s384_endstrided",   N=8,   S=384,  seed=4,  layout="endstrided", note="cuda route (bf16 D32 S<512 strided)"),
    Case("s512_none",         N=8,   S=512,  seed=5,  note="cuda_b route from S=512"),
    Case("s512_prefixvar",    N=8,   S=512,  seed=6,  mask="prefixvar"),
    Case("s1024_none",        N=4,   S=1024, seed=7),
    Case("s1024_prefix",      N=4,   S=1024, seed=8,  mask="prefix"),
    Case("s1024_prefixvar",   N=4,   S=1024, seed=9,  mask="prefixvar"),
    Case("s1024_fp32bias",    N=4,   S=1024, seed=10, bias="fp32", note="generic fp32 pair bias (not bf16-representable)"),
    Case("s1024_bias16",      N=4,   S=1024, seed=11, bias="bias16", note="bias handed as a bf16 tensor"),
    Case("s1024_endstrided",  N=4,   S=1024, seed=12, layout="endstrided", mask="prefix", note="cuda_b on strided views"),
    Case("s2048_none",        N=4,   S=2048, seed=13),
    Case("s2048_prefixvar",   N=4,   S=2048, seed=14, mask="prefixvar"),
    Case("s3584_prefix",      N=2,   S=3584, seed=15, mask="prefix", note="cuda_c route (3072 < S <= 4096)"),
    Case("s256_fp16_prefix",  N=8,   S=256,  seed=16, dtype="fp16", mask="prefix", note="k13 fp16 route"),
    Case("s256_d64_none",     N=4,   S=256,  seed=17, D=64, note="tri route (k10, D=64)"),
    # cc 8.0 (A100): the sm_80 member `cuda_80` on every row below (bf16, D 16 / 32 / 64, S_q == S_kv, contiguous or transposed views)
    Case("a80_s256_none",          N=8,   S=256,  seed=101, cc=A80, note="cuda_80: bf16 D32, S < 512"),
    Case("a80_s256_prefix",        N=8,   S=256,  seed=102, cc=A80, mask="prefix"),
    Case("a80_s400_none",          N=8,   S=400,  seed=103, cc=A80, note="S not a multiple of the tile"),
    Case("a80_s400_prefixvar",     N=8,   S=400,  seed=104, cc=A80, mask="prefixvar"),
    Case("a80_n400_s400_prefix",   N=400, S=400,  seed=105, cc=A80, mask="prefix", store="digest_rows8", note="square pair representation, N = S"),
    Case("a80_s512_none",          N=8,   S=512,  seed=106, cc=A80),
    Case("a80_s512_prefixvar",     N=8,   S=512,  seed=107, cc=A80, mask="prefixvar"),
    Case("a80_s1024_none",         N=4,   S=1024, seed=108, cc=A80),
    Case("a80_s1024_prefix",       N=4,   S=1024, seed=109, cc=A80, mask="prefix"),
    Case("a80_s1024_prefixvar",    N=4,   S=1024, seed=110, cc=A80, mask="prefixvar"),
    Case("a80_s1024_fp32bias",     N=4,   S=1024, seed=111, cc=A80, bias="fp32", note="generic fp32 pair bias (staged exactly)"),
    Case("a80_s1024_bias16",       N=4,   S=1024, seed=112, cc=A80, bias="bias16", note="bias handed as a bf16 tensor"),
    Case("a80_s2048_none",         N=4,   S=2048, seed=113, cc=A80),
    Case("a80_s2048_prefixvar",    N=4,   S=2048, seed=114, cc=A80, mask="prefixvar"),
    Case("a80_s384_endstrided",    N=8,   S=384,  seed=115, cc=A80, layout="endstrided", note="transposed views (ending node), S < 512"),
    Case("a80_s1024_endstrided",   N=4,   S=1024, seed=116, cc=A80, layout="endstrided", mask="prefix"),
    Case("a80_d16_s512_prefix",    N=8,   S=512,  seed=117, cc=A80, D=16, mask="prefix", note="D = 16"),
    Case("a80_d64_s512_none",      N=4,   S=512,  seed=118, cc=A80, D=64, note="D = 64"),
    Case("a80_d64_s1024_endstrided", N=4, S=1024, seed=119, cc=A80, D=64, layout="endstrided", mask="prefix"),
    Case("a80_h8_s800_prefixvar",  N=4,   S=800,  seed=120, cc=A80, H=8, mask="prefixvar", note="H = 8"),
    Case("a80_n6_s1000_prefixvar", N=6,   S=1000, seed=121, cc=A80, mask="prefixvar", note="N = 6 pair rows of S = 1000"),
    Case("a80_b2_s400_prefix",     N=4,   S=400,  seed=122, cc=A80, B=2, mask="prefix", note="B = 2"),
]
BY_ID = {c.id: c for c in CASES}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
CCS = sorted({x for c in CASES for x in c.cc})


def device_cc(device=0) -> str:
    """'9.0' / '8.0' / ...: the compute capability of a CUDA device as the string the case table uses."""
    return "%d.%d" % tuple(torch.cuda.get_device_capability(device))


def cases_for_cc(cc: str):
    """The cases that are vectors for compute capability `cc` (e.g. '9.0'), in table order."""
    return [c for c in CASES if cc in c.cc]


def make_inputs(case: Case, device="cpu") -> Dict[str, Optional[torch.Tensor]]:
    """{q, k, v, bias, mask, scale}: generated on the CPU (deterministic), then moved to `device` keeping the layout (strided views stay views)."""
    g = torch.Generator("cpu").manual_seed(1000003 * case.seed + 131 * case.N + case.S)
    B, N, H, S, D = case.B, case.N, case.H, case.S, case.D
    dt = DTYPES[case.dtype]

    def rnd(*shape, s=1.0):
        return torch.randn(*shape, generator=g, dtype=torch.float32) * s

    if case.layout == "start":
        q, k, v = (rnd(B, N, H, S, D).to(dt) for _ in range(3))
    elif case.layout == "endstrided":
        q, k, v = (rnd(B, S, H, N, D).to(dt) for _ in range(3))          # storage [B, S, H, N, D]; handed below as the transposed views [B, N, H, S, D]
    else:
        raise ValueError(case.layout)
    bias = rnd(B, 1, H, S, S, s=BIAS_SCALE)
    if case.bias in ("bf16x", "bias16"):
        bias = bias.to(dt).float()                                           # 16-bit-representable values in an fp32 tensor
        if case.bias == "bias16":
            bias = bias.to(dt)
    elif case.bias != "fp32":
        raise ValueError(case.bias)
    mask = None
    npad = int(round(PAD_FRAC * S))
    if case.mask == "prefix":
        L = torch.full((B, N), max(S - npad, 1), dtype=torch.int64)
    elif case.mask == "prefixvar":
        L = torch.randint(max(1, S - 3 * npad), S + 1, (B, N), generator=g, dtype=torch.int64); L[:, 0] = S
    elif case.mask == "none":
        L = None
    else:
        raise ValueError(case.mask)
    if L is not None:
        mask = (torch.arange(S)[None, None, :] < L[:, :, None])[:, :, None, None, :].contiguous()      # [B, N, 1, 1, S] bool, True = attend
    t = {"q": q, "k": k, "v": v, "bias": bias, "mask": mask}
    t = {n: (x.to(device) if x is not None else None) for n, x in t.items()}
    if case.layout == "endstrided":
        for n in ("q", "k", "v"):
            t[n] = t[n].transpose(1, 3)                                      # [B, N, H, S, D] views with the strides of the [B, S, H, N, D] storage
    t["scale"] = float(D) ** -0.5
    return t


def tensor_bytes(x: torch.Tensor) -> bytes:
    """The tensor's values in logical (row-major) order as bytes — layout-independent, dtype-exact (bool as one byte per element)."""
    c = x.detach().contiguous().cpu()
    if c.dtype == torch.bool:
        c = c.to(torch.uint8)
    return c.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_digest(x: Optional[torch.Tensor]) -> Optional[str]:
    return None if x is None else hashlib.sha256(tensor_bytes(x)).hexdigest()


def input_digests(t) -> Dict[str, Optional[str]]:
    return {n: tensor_digest(t[n]) for n in ("q", "k", "v", "bias", "mask")}


def reference_fp64(t, rows=None) -> torch.Tensor:
    """softmax_k(scale q.k + bias (+ -inf where mask == 0)) @ v in fp64 for the pair rows `rows` (default all) -> [B, n_rows, H, S_q, D] fp64."""
    q, k, v, bias, mask = t["q"], t["k"], t["v"], t["bias"], t["mask"]
    idx = torch.arange(q.shape[1], device=q.device) if rows is None else torch.as_tensor(list(rows), device=q.device)
    qd, kd, vd = q[:, idx].double(), k[:, idx].double(), v[:, idx].double()
    lg = torch.einsum("bnhqd,bnhkd->bnhqk", qd * t["scale"], kd) + bias.double()
    if mask is not None:
        lg = lg.masked_fill(~mask[:, idx], float("-inf"))
    return torch.einsum("bnhqk,bnhkd->bnhqd", lg.softmax(-1), vd)


def error_stats(out: torch.Tensor, ref64: torch.Tensor) -> Dict[str, float]:
    """max |out - ref|, its ratio to max |ref|, RMS error, the same in units of the output dtype's spacing at max |ref|, count of non-finite values."""
    o = out.double(); d = (o - ref64).abs()
    mref = ref64.abs().max().item(); eps = torch.finfo(out.dtype).eps
    return {"max_abs": d.max().item(), "max_abs_over_max_ref": d.max().item() / max(mref, 1e-300), "rms": d.pow(2).mean().sqrt().item(),
            "max_ref": mref, "max_abs_in_ulp_at_max_ref": d.max().item() / max(eps * mref, 1e-300), "n_nonfinite": int((~torch.isfinite(o)).sum().item())}


def case_dict(case: Case) -> Dict:
    d = asdict(case)
    d["cc"] = list(case.cc)
    return d

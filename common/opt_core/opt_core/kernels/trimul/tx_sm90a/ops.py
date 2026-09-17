# SPDX-License-Identifier: Apache-2.0
"""Op assembly: K1 (LayerNorm_in + gated dual projection -> channel-major bf16 planes [512, Np, Np], Np = ceil16(N), zero pad) | cuBLAS bmm on the
planes (bf16 operands, fp32 accumulate, bf16 out) | K3 (LayerNorm_out + output projection x sigmoid(gate) [+ residual] -> [N, N, 256] bf16).
The CUDA extension is the prebuilt shared object under prebuilt/<abi tag>/ described by prebuilt/manifest.json (source sha256, binary sha256,
toolchain); it is loaded once per process and checked against the manifest and against a closed-form load-time numerical fingerprint."""
import hashlib, importlib.util, json, math, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
C = 256
_EXT = None
_STATE = {"loaded": None, "loadcheck": None}


class TrimulTxUnavailable(RuntimeError):
    """The lever cannot run in this process (no binary for this ABI, manifest mismatch, load-time fingerprint mismatch). Raised by name; the caller's
    mode refuses — nothing continues under this lever's name on another path."""
    def __init__(self, reason):
        super().__init__("protenix_fpf_trimul_tx cannot run: " + reason); self.reason = reason


def abi_tag():
    """The FULL ABI key of this process: torch version (+ CUDA), the interpreter's SOABI (CPython tag + platform), the GPU arch the binary targets --
    e.g. torch2.13.0+cu130-cpython-312-x86_64-linux-gnu-sm90.  One prebuilt directory per key under prebuilt/; a process whose key has no entry is
    refused by name (TrimulTxUnavailable), never handed another interpreter's binary."""
    import sysconfig
    return "torch%s-%s-sm90" % (torch.__version__, sysconfig.get_config_var("SOABI"))


def legacy_abi_tag():
    """The pre-full-key tag (torch major/minor + CUDA, no interpreter): looked up only as an alias when a manifest still carries it."""
    tv = torch.__version__.split("+")[0].split(".")
    return "torch%s%s_cu%s_sm90" % (tv[0], tv[1], (torch.version.cuda or "none").replace(".", ""))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load():
    """Load the prebuilt extension for this ABI; check binary and source digests against the manifest; run the load-time fingerprint once."""
    global _EXT
    if _EXT is not None:
        return _EXT
    man_path = os.path.join(HERE, "prebuilt", "manifest.json")
    if not os.path.exists(man_path):
        raise TrimulTxUnavailable("prebuilt/manifest.json missing")
    man = json.load(open(man_path))
    tag = abi_tag()
    ent = man.get("binaries", {}).get(tag)
    if ent is None and legacy_abi_tag() in man.get("binaries", {}):          # a manifest written before the full keys: same torch line, interpreter unchecked by the key
        tag = legacy_abi_tag(); ent = man["binaries"][tag]
    if ent is None:
        raise TrimulTxUnavailable("no prebuilt binary for ABI tag %s (have: %s)" % (tag, sorted(man.get("binaries", {}))))
    so = os.path.join(HERE, "prebuilt", ent["file"])
    if not os.path.exists(so):
        raise TrimulTxUnavailable("binary %s listed in the manifest is absent" % ent["file"])
    from opt_core.gates import binary_refusal  # noqa: PLC0415
    why = binary_refusal(so)
    if why:
        raise TrimulTxUnavailable("binary %s refused: %s" % (ent["file"], why))
    got = _sha256(so)
    if got != ent["sha256"]:
        raise TrimulTxUnavailable("binary sha256 %s.. != manifest %s.. (%s)" % (got[:12], ent["sha256"][:12], ent["file"]))
    for rel, want in man.get("sources", {}).items():                       # provenance: the shipped csrc is the source the binary was built from
        p = os.path.join(HERE, rel)
        if not os.path.exists(p) or _sha256(p) != want:
            raise TrimulTxUnavailable("source %s does not match the manifest digest the binary was built from" % rel)
    spec = importlib.util.spec_from_file_location(ent["module"], so)
    ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
    _EXT = ext
    _STATE["loaded"] = {"tag": tag, "file": ent["file"], "sha256": got[:16], "toolchain": ent.get("toolchain")}
    _loadcheck(man)
    return _EXT


def _closed_form_inputs(N, dev):
    """Deterministic inputs from closed-form expressions (no RNG): pair activations, mask, and a weight pack."""
    i = torch.arange(N, dtype=torch.float64, device=dev)
    c = torch.arange(C, dtype=torch.float64, device=dev)
    z = (torch.sin(0.37 * i[:, None, None] + 0.11 * c[None, None, :]) * 3.0 + torch.cos(0.23 * i[None, :, None] - 0.07 * c[None, None, :]) * 2.0
         + 0.5 * torch.sin(0.013 * (i[:, None, None] * i[None, :, None]) + 0.5 * c[None, None, :])).to(torch.bfloat16).contiguous()
    mask = ((i[:, None] * 7 + i[None, :] * 3) % 11 != 0).to(torch.float32).contiguous()
    o = torch.arange(2 * C, dtype=torch.float64, device=dev)
    def mat(rows, a, b):
        r = torch.arange(rows, dtype=torch.float64, device=dev)
        return (torch.sin(a * r[:, None] + b * c[None, :]) * 0.0625).float()
    class M: pass
    m = M()
    lin = lambda rows, a, b: type("L", (), {"weight": mat(rows, a, b)})()
    m.linear_a_g, m.linear_b_g = lin(C, 0.31, 0.17), lin(C, 0.29, 0.19); m.linear_a_p, m.linear_b_p = lin(C, 0.23, 0.13), lin(C, 0.21, 0.11)
    m.linear_z, m.linear_g = lin(C, 0.37, 0.07), lin(C, 0.41, 0.05)
    ln = lambda a: type("LN", (), {"weight": (1.0 + 0.25 * torch.sin(a * c)).float(), "bias": (0.1 * torch.cos(a * c)).float()})()
    m.layer_norm_in, m.layer_norm_out = ln(0.09), ln(0.07)
    return z, mask, pack_weights(m)


def _digest(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:32]


def _loadcheck(man):
    """Load-time numerical fingerprint on closed-form inputs (N = 160: two token tiles + a ragged edge): digests of the K1 planes and of the K3 output
    given fixed planes, for the fast-tier entry points (own LayerNorm order, padded planes) and for the exact-tier ones (stock LayerNorm order: K1 ->
    unpadded planes; K3 with either LN_out summation tree, N % 4 == 0 and N % 4 != 0), must equal the manifest's (these are this package's kernels: a
    mismatch means a broken binary -> refused by name); the cuBLAS contraction digest is recorded and compared for information only (cuBLAS may pick
    another kernel on another library version)."""
    dev = torch.device("cuda", torch.cuda.current_device())
    N = 160
    z, mask, w = _closed_form_inputs(N, dev)
    ab = k1(z, mask, w)
    x_fixed = (ab[:C].float() * 0.5 + ab[C:].float().flip(0) * 0.25).to(torch.bfloat16).contiguous()   # a plane tensor that does not depend on cuBLAS
    out = k3(x_fixed, z, w, residual=True)
    x = bmm(ab, True)
    abs_ = k1_stock_order(z, mask, w)                                    # exact tier: [512, 160, 160] unpadded planes, stock LayerNorm order
    xs_fixed = x_fixed[:, :N, :N].contiguous()
    outs2 = k3(xs_fixed, z, w, residual=True, lnmode=2)                  # stock order, the N % 4 == 0 tree of the transposing LayerNorm
    outs3 = k3(xs_fixed, z, w, residual=True, lnmode=3)                  # stock order, the N % 4 != 0 tree (numerically valid at any N; selected by N at run time)
    torch.cuda.synchronize(dev)
    got = {"k1": _digest(ab), "k3": _digest(out), "bmm": _digest(x), "k1s": _digest(abs_), "k3s2": _digest(outs2), "k3s3": _digest(outs3)}
    want = man.get("loadcheck", {})
    _STATE["loadcheck"] = {"got": got, "want": want}
    bad = [k for k in ("k1", "k3", "k1s", "k3s2", "k3s3") if want.get(k) is not None and want[k] != got[k]]
    if bad:
        raise TrimulTxUnavailable("load-time fingerprint mismatch for %s (binary does not reproduce the recorded kernels' output)" % ",".join(bad))
    if not want:
        sys.stderr.write("[protenix_fpf_trimul_tx] loadcheck: manifest carries no fingerprint; computed %s\n" % json.dumps(got))
    elif want.get("bmm") not in (None, got["bmm"]):
        sys.stderr.write("[protenix_fpf_trimul_tx] loadcheck: kernels OK; the cuBLAS contraction digest differs from the recorded one on this stack (informational)\n")


def describe():
    st = _STATE["loaded"]
    return "protenix_fpf_trimul_tx %s [%s %s]" % (sys.modules[__package__].__version__ if __package__ in sys.modules else "?", st["tag"] if st else "not loaded", st["sha256"] if st else "")


def pack_weights(m):
    """From a TriangleMultiplication module (c_z = c_hidden = 256): the bf16 block-interleaved gate|proj matrix K1 streams (16 blocks x [32 gate rows;
    32 proj rows], output channels a(256) then b(256)), the LayerNorm parameters in fp32, and the K3 matrix stream (Wg_b, Wz_b blocks)."""
    with torch.no_grad():
        wg = torch.cat([m.linear_a_g.weight, m.linear_b_g.weight], 0).float()
        wp = torch.cat([m.linear_a_p.weight, m.linear_b_p.weight], 0).float()
        blocks = []
        for b in range(16):
            blocks.append(wg[32 * b:32 * b + 32]); blocks.append(wp[32 * b:32 * b + 32])
        return dict(
            wgp=torch.cat(blocks, 0).to(torch.bfloat16).contiguous(),
            ln_in_w=m.layer_norm_in.weight.detach().float().contiguous(), ln_in_b=m.layer_norm_in.bias.detach().float().contiguous(),
            ln_out_w=m.layer_norm_out.weight.detach().float().contiguous(), ln_out_b=m.layer_norm_out.bias.detach().float().contiguous(),
            wgz=torch.cat([m.linear_g.weight, m.linear_z.weight], 0).detach().to(torch.bfloat16).contiguous())


def ceil16(n):
    return (n + 15) // 16 * 16


def k1(z, mask, w):
    N = z.shape[0]; Np = ceil16(N)
    ab = torch.empty((2 * C, Np, Np), dtype=torch.bfloat16, device=z.device)
    load().k1_forward(z, mask, w["ln_in_w"], w["ln_in_b"], w["wgp"], ab, 1e-5, 0, 0, 1, 1)
    return ab


def bmm(ab, outgoing):
    """x[d, i, j] = sum_k a[d, i, k] b[d, j, k] (outgoing) | sum_k a[d, k, i] b[d, k, j] (incoming): one strided-batched cuBLAS GEMM on the planes."""
    a, b = ab[:C], ab[C:]
    return torch.bmm(a, b.transpose(1, 2)) if outgoing else torch.bmm(a.transpose(1, 2), b)


def k3(x, z, w, residual=True, out=None, lnmode=1):
    out = torch.empty_like(z) if out is None else out
    load().k3_forward(x, z, w["ln_out_w"], w["ln_out_b"], w["ln_in_w"], w["ln_in_b"], w["wgz"], out, 1e-5, int(residual), 0, 0, lnmode)
    return out


def trimul(z, outgoing, mask, w, residual=True):
    """z [N, N, 256] bf16 contiguous; mask None or [N, N] float32 contiguous; returns [N, N, 256] bf16 = z + update (residual) or the update."""
    ab = k1(z, mask, w)
    x = bmm(ab, outgoing)
    return k3(x, z, w, residual=residual)


# ---------------------------------------------------------------- exact-tier assembly: bitwise class vs the stock cuEquivariance pipeline
# Both LayerNorms run INSIDE K1 / K3 in the summation order whose bf16 outputs equal the stock op's LayerNorm outputs bit for bit (csrc: ln_cueq; an
# order found by matching that op's observable outputs); the planes are unpadded [512, N, N] so that the contraction is the stock cuBLAS problem; the
# contraction is the stock expression.
def k1_stock_order(z, mask, w):
    """K1 with LayerNorm in the stock order -> UNPADDED planes [512, N, N] bf16.  N % 8 == 0: written directly (16-byte plane rows); otherwise written
    into 16-token-padded planes and packed by one strided copy (bitwise-neutral; faster than 2-byte plane stores)."""
    N = z.shape[0]
    if N % 8 == 0:
        ab = torch.empty((2 * C, N, N), dtype=torch.bfloat16, device=z.device)
        load().k1_forward(z, mask, w["ln_in_w"], w["ln_in_b"], w["wgp"], ab, 1e-5, 0, 0, 1, 3)
        return ab
    Np = ceil16(N)
    ab = torch.empty((2 * C, Np, Np), dtype=torch.bfloat16, device=z.device)
    load().k1_forward(z, mask, w["ln_in_w"], w["ln_in_b"], w["wgp"], ab, 1e-5, 0, 0, 1, 3)
    return ab[:, :N, :N].contiguous()


def contract_stock(ab, outgoing):
    """The stock contraction expression on [256, 1, N, N] chunk views of the unpadded planes (hence the stock cuBLAS problem, kernel and bits);
    returns x [256, N, N] contiguous."""
    N = ab.shape[-1]
    a, b = torch.chunk(ab.reshape(2 * C, 1, N, N), 2, dim=0)
    x = torch.einsum("dbik,dbjk->dbij", a, b) if outgoing else torch.einsum("dbki,dbkj->dbij", a, b)
    return x.reshape(C, N, N)


def trimul_exact(z, outgoing, mask, w, residual=True):
    """z [N, N, 256] bf16 contiguous; mask None | [N, N] fp32 holding 0/1; returns [N, N, 256] bf16 == the stock module output bit for bit
    (z + update with the module's separate residual add when residual, else the update)."""
    N = z.shape[0]
    x = contract_stock(k1_stock_order(z, mask, w), outgoing)
    if N % 8:                                    # K3 reads plane rows through 16-byte TMA boxes: ragged N takes one pad-copy of the contraction result
        Np = (N + 7) // 8 * 8
        xp = torch.empty((C, Np, Np), dtype=torch.bfloat16, device=z.device)
        xp[:, :N, :N].copy_(x)
        x = xp
    return k3(x, z, w, residual=residual, lnmode=2 if N % 4 == 0 else 3)   # the stock transposing LayerNorm sums in another order when N % 4 != 0

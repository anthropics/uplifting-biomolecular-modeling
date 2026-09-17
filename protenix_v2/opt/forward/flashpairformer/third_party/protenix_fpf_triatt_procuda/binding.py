"""binding.py — ctypes binding of the prebuilt sm_90a tri-attention prologue kernel (prebuilt/libtriatt_procuda_sm90a.so, source csrc/triatt_procuda_sm90.cu).

launch(z, cch, ending, out=None, head_major=False) computes, for the pair activation z [N0, N1, 256] bf16 (x = z, or x = z^T when ending=True; address
math only), LayerNorm(x) -> q|k|v|g = x_ln @ Wqkvg^T (bf16) and the pair bias fp32(bf16(x_ln @ Wb^T)) with the arithmetic of the kit's F1 prologue cell
(fpf_mkpf 'welford' LayerNorm emulation of the upstream fast_layernorm; fp32-accumulated bf16 tensor-core products, k ascending) — same output
layouts as fpf_mkpf.kernels.prologue_ln (row-major q/k/v [I,H,J,D]) or the head-major variant ([H,I,J,D] storage returned as [I,H,J,D]-shaped views),
unpadded or into the caller's padded [P,H,P,D] / [1,H,P,P] buffers (valid region only; pads are never written)."""
import os, json, ctypes, hashlib
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
PREBUILT_DIR = os.path.join(_HERE, "prebuilt")
SO_NAME = "libtriatt_procuda_sm90a.so"
C, H, D, HB = 256, 8, 32, 16
_LIB = None
_GRID = None


def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def manifest():
    with open(os.path.join(PREBUILT_DIR, "manifest.json")) as f:
        return json.load(f)


def source_sha256():
    return _sha256(os.path.join(_HERE, "csrc", "triatt_procuda_sm90.cu"))


def load():
    """Load the prebuilt library once; raises RuntimeError (named reason) when the binary is absent or does not match its manifest."""
    global _LIB, _GRID
    if _LIB is not None:
        return _LIB
    so = os.path.join(PREBUILT_DIR, SO_NAME)
    if not os.path.exists(so):
        raise RuntimeError(f"prebuilt binary missing: {so}")
    man = manifest()
    got = _sha256(so)
    if got != man.get("so_sha256"):
        raise RuntimeError(f"prebuilt binary sha256 {got[:16]} != manifest {str(man.get('so_sha256'))[:16]}")
    src = source_sha256()
    if src != man.get("src_sha256"):
        raise RuntimeError(f"shipped source sha256 {src[:16]} != manifest {str(man.get('src_sha256'))[:16]} (binary was built from a different source)")
    L = ctypes.CDLL(so)
    L.f1v2_launch.restype = ctypes.c_int
    L.f1v2_launch.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int, ctypes.c_int,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong,
                              ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong,
                              ctypes.c_void_p, ctypes.c_int]
    L.f1v2_smem_bytes.restype = ctypes.c_int
    _LIB = L
    _GRID = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return L


def fits(z, cch):
    """The kernel's compile-time envelope: c_z=256, 8 heads x 32, 16-row bias weight block, bf16 z with unit channel stride and 16-B aligned rows."""
    return (z.dim() == 3 and z.dtype == torch.bfloat16 and z.stride(-1) == 1 and int(z.shape[-1]) == C
            and (cch["C"], cch["H"], cch["D"], cch["HB"]) == (C, H, D, HB)
            and z.stride(0) % 8 == 0 and z.stride(1) % 8 == 0 and z.data_ptr() % 16 == 0
            and cch["wqkvg"].dtype == torch.bfloat16 and cch["wqkvg"].is_contiguous() and tuple(cch["wqkvg"].shape) == (4 * H * D, C)
            and cch["wb"].dtype == torch.bfloat16 and cch["wb"].is_contiguous() and tuple(cch["wb"].shape) == (HB, C)
            and cch["lnw"].dtype == torch.bfloat16 and cch["lnb"].dtype == torch.bfloat16)


def launch(z, cch, ending: bool, out: dict = None, head_major: bool = False):
    if ending: NI, NJ, s_zi, s_zj = int(z.shape[1]), int(z.shape[0]), z.stride(1), z.stride(0)
    else:      NI, NJ, s_zi, s_zj = int(z.shape[0]), int(z.shape[1]), z.stride(0), z.stride(1)
    HD = H * D; dev = z.device
    if out is None:
        if head_major:
            store = torch.empty((3, H, NI, NJ, D), dtype=torch.bfloat16, device=dev)
            qs, ks, vs = store[0], store[1], store[2]
            q, k, v = qs.permute(1, 0, 2, 3), ks.permute(1, 0, 2, 3), vs.permute(1, 0, 2, 3)
            q_si, q_sh = NJ * D, NI * NJ * D
        else:
            store = torch.empty((3, NI, H, NJ, D), dtype=torch.bfloat16, device=dev)
            q, k, v = store[0], store[1], store[2]; qs, ks, vs = q, k, v
            q_si, q_sh = H * NJ * D, NJ * D
        g = torch.empty((NI, NJ, HD), dtype=torch.bfloat16, device=dev); g_si = NJ * HD; gret = g
        bias = torch.empty((H, NI, NJ), dtype=torch.float32, device=dev); NIP, NJP = NI, NJ; bret = bias
    else:
        qb, kb, vb, bias = out["q"], out["k"], out["v"], out["bias"]
        P = int(qb.shape[0])
        assert tuple(qb.shape) == (P, H, P, D) and kb.shape == qb.shape and vb.shape == qb.shape and qb.is_contiguous() and kb.is_contiguous() and vb.is_contiguous()
        b4 = bias if bias.dim() == 4 else bias.unsqueeze(0)
        assert tuple(b4.shape) == (1, H, P, P) and b4.is_contiguous() and b4.dtype == torch.float32 and NI <= P and NJ <= P
        if head_major:
            qs, ks, vs = qb.view(H, P, P, D), kb.view(H, P, P, D), vb.view(H, P, P, D)
            q, k, v = qs.permute(1, 0, 2, 3), ks.permute(1, 0, 2, 3), vs.permute(1, 0, 2, 3)
            q_si, q_sh = P * D, P * P * D
        else:
            q, k, v = qb, kb, vb; qs, ks, vs = qb, kb, vb
            q_si, q_sh = H * P * D, P * D
        g = out.get("g")
        if g is None or tuple(g.shape) != (P, P, HD) or g.dtype != torch.bfloat16 or g.device != dev:
            g = torch.zeros((P, P, HD), dtype=torch.bfloat16, device=dev); out["g"] = g
        assert g.is_contiguous()
        g_si = P * HD; gret = g[:NI, :NJ]
        NIP, NJP = P, P; bret = bias; bias = b4
    L = load()
    rc = L.f1v2_launch(z.data_ptr(), s_zi, s_zj, NI, NJ, cch["lnw"].data_ptr(), cch["lnb"].data_ptr(), float(cch["eps"]),
                       cch["wqkvg"].data_ptr(), cch["wb"].data_ptr(),
                       qs.data_ptr(), ks.data_ptr(), vs.data_ptr(), q_si, q_sh, g.data_ptr(), g_si, bias.data_ptr(), NIP, NJP,
                       torch.cuda.current_stream(dev).cuda_stream, int(_GRID))
    if rc != 0:
        raise RuntimeError(f"kernel launch failed rc={rc} (NI={NI} NJ={NJ} strides={s_zi},{s_zj} padded={out is not None} head_major={head_major})")
    return q, k, v, gret, bret

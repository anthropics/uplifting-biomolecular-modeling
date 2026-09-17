"""TriangleMultiplication kernels of the ESM-family pair line re-tiled for other pair widths, with their measured launch cells.

One op boundary (the same as ``opt_core.kernels.trimul``): pre-LayerNorm pair ``z [N,N,c_z]`` (+ ``mask [N,N]``) in; LN_in, the four
projections and two gates, the contraction over ``c_hidden`` channels, LN_out and the gated output projection inside; ``update`` or
``z + update`` out, in z's dtype.  Kernels: ``kernels._k1s`` (LN_in + gated dual projection -> bf16 planes), cuBLAS bmm, ``kernels._k3s``
(LN_out + out-projection + gate (+ residual)).  What the variants add over the line they derive from (NOTICE):

    shapes      c_z / c_hidden walked as power-of-two channel chunks: (64,64), (64,128), (128,128), (256,256), (384,384) run natively
                (384 = 3 x 128, no padding of the reductions); B = 1 pairs.
    io          z bf16 (bf16 trunks) or fp32 (fp32 / TF32 trunks: fp32 activations in and out, bf16 tensor-core MMAs with fp32
                accumulation, fp32 LayerNorm statistics / gates / residual) -- tolerance class against those trunks' fp32 op.
    residual    fused into K3 or absent (the forward-only, no-residual entry).
    cards       cc 9.0 cells read z and the weights through tensor descriptors; cc 8.0 cells through pointer loads within sm_80 shared memory.

Face (pure Python here; torch / triton are imported by ``kernels`` when a call is served)::

    sel = select_cell(cc, io, c_z, c_hidden, n_tokens, direction)      # -> dict(key, cell, measured, size, note) or raises Unsupported(word)
    ok, why = supported(cc, io, c_z, c_hidden, n_tokens, batch=1)
    out = triangle_multiplication(z, mask, direction="outgoing", weights=w10, residual=False, levers=None, cache=my_dict)

``TILE_TABLES.json`` holds, per ``<cc>|<io>|C<c_z>|H<c_hidden>|N<=<tokens>|<dir>``, the launch cell chosen by measurement (K1 / K3 tiles,
warps, stages, descriptor use, chunk widths, contraction form) with the sweep's numbers, and per-cc ``defaults`` for widths inside the
envelope at unmeasured sizes (flagged ``measured: false``).  Nothing here reads the environment; a call outside the envelope raises
``Unsupported(word)`` before any launch -- the caller keeps its own path.
"""
import json
import os

__version__ = "1.2.1"
TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TILE_TABLES.json")
WEIGHT_KEYS = ("ln_in_w", "ln_in_b", "w_ag", "w_ap", "w_bg", "w_bp", "ln_out_w", "ln_out_b", "w_o", "w_og")
LEVERS = ("sigmoid", "stagger", "incnt", "form", "x32", "stock_round")
IO_WORDS = ("bf16", "fp32")
MAX_CHUNKS = 4
_T = {}


class Unsupported(Exception):
    """The call is outside these kernels' envelope; ``word`` is the reason for a LEVER line.  Nothing was launched."""

    def __init__(self, word):
        Exception.__init__(self, word)
        self.word = word


def table():
    if "t" not in _T:
        with open(TABLE_PATH, encoding="utf-8") as fh:
            _T["t"] = json.load(fh)
    return _T["t"]


def cc_word(cc):
    if isinstance(cc, (tuple, list)):
        return "%d.%d" % (int(cc[0]), int(cc[1]))
    s = str(cc)
    if s.isdigit() and len(s) >= 2:
        return "%s.%s" % (s[:-1], s[-1])
    return "%.1f" % float(s)


def io_word(dtype):
    s = str(dtype).replace("torch.", "").lower()
    s = {"bfloat16": "bf16", "float32": "fp32", "float": "fp32", "f32": "fp32", "tf32": "fp32", "f32z_bf16": "bf16"}.get(s, s)
    if s not in IO_WORDS:
        raise Unsupported("io:%s(bf16|fp32 pairs)" % s)
    return s


def direction_word(direction):
    d = str(direction).lower()
    if d in ("outgoing", "out"):
        return "out"
    if d in ("incoming", "in"):
        return "in"
    raise ValueError("direction must be outgoing | incoming (got %r)" % (direction,))


def chunk_plan(width, chunk=None):
    """(chunk, n_chunks) the kernels walk a channel width in: power-of-two chunks <= 256, at most MAX_CHUNKS."""
    width = int(width)
    if chunk is None:
        chunk = width if (width & (width - 1)) == 0 and width <= 256 else 128
    chunk = int(chunk)
    if chunk & (chunk - 1) or chunk > 256 or chunk < 16 or width % chunk or width // chunk > MAX_CHUNKS:
        raise Unsupported("width:%d(chunks of %d)" % (width, chunk))
    return chunk, width // chunk


def cell_key(cc, io, c_z, c_hidden, n_bucket, direction):
    return "%s|%s|C%d|H%d|N<=%d|%s" % (cc_word(cc), io_word(io), int(c_z), int(c_hidden), int(n_bucket), direction_word(direction))


def _family(cc, io, c_z, c_hidden, direction):
    pre = "%s|%s|C%d|H%d|" % (cc_word(cc), io_word(io), int(c_z), int(c_hidden))
    suf = "|%s" % direction_word(direction)
    out = {}
    for k in table().get("cells", {}):
        if k.startswith(pre) and k.endswith(suf):
            try:
                out[int(k[len(pre):-len(suf)].replace("N<=", ""))] = k
            except ValueError:
                continue
    return out


def supported(cc, io, c_z, c_hidden, n_tokens, *, batch=1, descriptor_api=True):
    """(True, cell key) or (False, word) -- never raises."""
    try:
        sel = select_cell(cc, io, c_z, c_hidden, n_tokens, "out", descriptor_api=descriptor_api, batch=batch)
        return True, sel["key"]
    except Unsupported as e:
        return False, e.word


def select_cell(cc, io, c_z, c_hidden, n_tokens, direction="outgoing", *, descriptor_api=True, batch=1):
    """The launch cell for a call: the smallest measured size >= n_tokens of the (cc, io, C, H, dir) family, else its largest measured size
    (``beyond_measured``), else the cc's default cell for a width inside the envelope (``measured: False``).  Raises Unsupported(word)."""
    ccw = cc_word(cc)
    iow = io_word(io)
    C, H, n = int(c_z), int(c_hidden), int(n_tokens)
    if int(batch) != 1:
        raise Unsupported("batch:%d(B=1 pairs)" % int(batch))
    t = table()
    env = t.get("envelope", {})
    if [C, H] not in env.get("shapes", []) and not env.get("any_chunked_width", False):
        raise Unsupported("shape:C%d/H%d(tabled: %s)" % (C, H, " ".join("%d/%d" % tuple(s) for s in env.get("shapes", []))))
    chunk_plan(C); chunk_plan(H)
    if n < int(env.get("n_min", 16)):
        raise Unsupported("n<%d" % int(env.get("n_min", 16)))
    if ccw not in t.get("defaults", {}) and not _family(ccw, iow, C, H, direction):
        raise Unsupported("cc:%s(no cells)" % ccw)
    fam = _family(ccw, iow, C, H, direction)
    key, size, measured, beyond = None, None, False, False
    if fam:
        sizes = sorted(fam)
        fit = [s for s in sizes if s >= n]
        size = fit[0] if fit else sizes[-1]
        beyond = not fit
        key = fam[size]
        cell = dict(t["cells"][key]["cell"])
        measured = True
    else:
        d = t["defaults"][ccw]
        cell = dict(d.get("C%d_H%d" % (C, H), d.get("any")))
        key = cell_key(ccw, iow, C, H, 0, direction) + "(default)"
    cell = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cell.items()}
    note = []
    alt_measured = cell.pop("k1_ptr", None)
    if cell.get("k1", {}).get("tma") and not descriptor_api:
        alt = alt_measured or t.get("defaults", {}).get(ccw, {}).get("no_descriptor_k1")
        if alt is None:
            raise Unsupported("descriptor_api(tl.make_tensor_descriptor absent; no pointer cell for cc %s)" % ccw)
        cell["k1"] = dict(alt)
        note.append("k1=%s pointer cell (descriptor api absent)" % ("measured" if alt_measured else "default"))
    return dict(key=key, cell=cell, measured=measured, size=size, beyond_measured=beyond, note="; ".join(note))


def describe_cell(sel):
    c = sel["cell"]
    k1, k3 = c.get("k1", {}), c.get("k3", {})
    return "%s%s k1[%s BM%s BN%s w%s s%s] bmm[%s] k3[BM%s BN%s w%s s%s] ck=%s chk=%s%s" % (
        sel["key"], "" if sel["measured"] else " UNMEASURED", "tma" if k1.get("tma") else "ptr", k1.get("BM"), k1.get("BN"), k1.get("num_warps"),
        k1.get("num_stages"), c.get("form") or "NT", k3.get("BM"), k3.get("BN"), k3.get("num_warps"), k3.get("num_stages"), c.get("ck"), c.get("chk"),
        (" (" + sel["note"] + ")") if sel.get("note") else "")



def pack_key(weights):
    """The pack cache's key: (data pointer, shape, stride, dtype, device) of the ten weight tensors -- never the tensors' version counter
    (inference tensors have none: reading ``_version`` raises 'Inference tensors do not track version counter' under torch.inference_mode).
    Weights are read as constants; a caller that updates them in place passes a fresh cache."""
    return ("pack",) + tuple((weights[k].data_ptr(), tuple(weights[k].shape), tuple(weights[k].stride()), str(weights[k].dtype), str(weights[k].device)) for k in WEIGHT_KEYS)

def _w10(weights):
    missing = [k for k in WEIGHT_KEYS if k not in weights]
    if missing:
        raise Unsupported("weights_missing:%s" % ",".join(missing))
    extra = sorted(set(weights) - set(WEIGHT_KEYS))
    if extra:
        raise Unsupported("weights_unknown:%s(no projection biases in these kernels)" % ",".join(extra))
    return weights


def triangle_multiplication(z, mask=None, *, direction="outgoing", weights=None, pack=None, residual=False, levers=None, cell=None,
                            cache=None, eps=1e-5, out=None, pad=16, pointer_k1=False):
    """Serve one call.  ``weights`` = the ten tensors (WEIGHT_KEYS) or ``pack`` = kernels.pack_weights(...) (``cache``: a dict the caller
    holds per weight set -- the pack is built once into it).  ``cell`` overrides the table's cell (a sweep passes its own).  ``pointer_k1``
    True serves the cell's measured pointer-load K1 tile even where the tensor-descriptor API exists (the same planes, byte for byte).  Raises
    Unsupported(word) outside the envelope; kernel errors propagate (nothing is substituted)."""
    import torch
    from . import kernels as K
    zz = z[0] if (z.dim() == 4 and int(z.shape[0]) == 1) else z
    if zz.dim() != 3:
        raise Unsupported("batch:%s(B=1 pairs)" % (tuple(z.shape)[:-3],))
    if not zz.is_cuda:
        raise Unsupported("device:%s" % zz.device)
    lv = dict(levers or {})
    bad = sorted(set(lv) - set(LEVERS))
    if bad:
        raise Unsupported("levers_unknown:%s" % ",".join(bad))
    if lv.get("x32") and not K.has_bmm_out_dtype():
        raise Unsupported("x32:torch.bmm(out_dtype) absent")
    if pack is None:
        if weights is None:
            raise Unsupported("weights_missing:all")
        w = _w10(weights)
        if cache is not None:
            key = pack_key(weights)
            hit = cache.get(key)
            pack = None if hit is None else hit[0]
            if pack is None:
                pack = K.pack_weights(**w)
                for k in [k for k in cache if isinstance(k, tuple) and k and k[0] == "pack"]:
                    del cache[k]
                cache[key] = (pack, tuple(weights[k] for k in WEIGHT_KEYS))     # the entry pins the source tensors: a data pointer cannot be reused while it lives
        else:
            pack = K.pack_weights(**w)
    C, H, N = int(zz.shape[-1]), int(pack["CH"]), int(zz.shape[0])
    if cell is None:
        cc = torch.cuda.get_device_capability(zz.device)
        sel = select_cell(cc, io_word(zz.dtype), C, H, N, direction, descriptor_api=(K.has_descriptor_api() and not pointer_k1))
        cell = sel["cell"]
    return K.forward(z, mask, outgoing=direction_word(direction) == "out", pack=pack, cell=cell, eps=eps, residual=residual, levers=lv, out=out, pad=pad)

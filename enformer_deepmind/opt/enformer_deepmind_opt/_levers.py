"""enformer_deepmind_opt._levers — the levers: rewrites of the Enformer prediction graph (``_graph.Graph``) that leave every output byte as
the stock graph computes it. ``LEVERS`` names them in the order they are applied (and the activation line prints them): a lever matches the
graph the levers before it left — ``poolgemm`` after ``pool``, ``bnconst`` after ``bngelu``; ``REWRITES[name](graph) -> stats``.

``pool`` — attention pooling (``SoftmaxPooling1D``, pool size 2: the stem's and the six conv-tower blocks'). Stock computes, on the
(B, L/2, 2, C) view ``x`` of a block's output and its per-channel logits ``l``: ``softmax(l, axis=-2)`` — which TensorFlow executes as a
transpose of the whole tensor, a softmax over rows of two, and a transpose back — then ``x * softmax`` and a sum over the pair axis (five
kernels, seven passes over the largest activations of the network). The op library's ``EdmSoftmaxPool2`` reads ``x`` and ``l`` once and
writes the pooled output once, performing per element exactly the stock kernels' float32 operations in their order (ops/README.md:
``m = max``, ``e_i = exp(l_i - m)``, ``s = e1 + e0``, ``w_i = e_i / s``, ``y = x0*w0 + x1*w1``).

``poscache`` — the relative-position keys of the eleven attention blocks. Stock recomputes, on every call, each block's positional basis
functions over the 3,071 relative distances (exponential, central-mask and gamma features: about a hundred small ops per block) and their
projection through the block's ``r_k_layer`` — none of which depends on the input. The lever evaluates exactly those stock ops once, on the
same device, when the graph is rewritten, and puts the resulting tensors into the graph as constants: the same bits the per-call computation
produces, about 1,100 fewer kernel launches per prediction.

``relsoftmax`` — the attention weights of the eleven attention blocks: the relative shift of the positional logits
(``attention_module.relative_shift``: pad, reshape, slice, reshape, slice — five kernels that materialize two extra copies of the
largest tensor of a block), their add to the content logits, and ``tf.nn.softmax`` over the rows of 1,536 logits, which TensorFlow
runs as three more kernels (row maximum, row sum of exponentials, normalization). One op (``EdmRelShiftSoftmax``) reads the content
and relative logits once, forms each row's logits with the shift expressed as an index and writes the normalized weights once, with
the Softmax kernels' float32 operations in their order — the two row reductions in the association TensorFlow's reduction library
uses for rows of this length (ops/README.md).

``biasres`` — a bias add followed by a residual add (the seven convolutional residual blocks, the eleven attention output projections, the
eleven MLP outputs): one op (``EdmBiasResidual``) doing both adds per element in the association the stock graph executes (ops/README.md).

``bngelu`` — every ``BatchNorm -> GELU`` in front of a convolution (the stem's pointwise block, the two of each conv-tower block, the final
pointwise block: 14 chains). Stock runs five elementwise kernels per chain — ``x * scale``, ``+ shift`` (``tf.nn.batch_normalization`` in
inference form; ``scale`` and ``shift`` are per-channel vectors derived from the moving statistics — ``bnconst`` below makes them constants),
``1.702 * h``, ``sigmoid``, ``* h`` — each a full pass over the activation. The op library's ``EdmScaleShiftGelu`` performs the same five
float32 operations per element in one pass, with TensorFlow's own sigmoid arithmetic (ops/README.md).

``layernorm`` — the 22 layer normalisations of the transformer (Sonnet ``LayerNorm`` over the channels, in front of each attention and
each MLP block). Stock computes ``tf.nn.moments`` (a row mean, the squared differences, their row mean: two segmented reductions and a
full pass) and ``tf.nn.batch_normalization`` with per-row statistics (add epsilon, rsqrt, then five full passes: ``rsqrt·gamma``, ``x·inv``,
``mean·inv``, ``beta − ·``, ``+``) — ten kernels per site. The op library's ``EdmLayerNorm`` does it in one kernel per site, one thread block
per row, reproducing the summation order of TensorFlow's row-reduction kernel and each elementwise operation as written (ops/README.md).

``poolgemm`` — the pooling logits ``l = x @ W`` (one C x C matrix per pooling module applied to every pair of positions). Stock runs them as a
strided-batched GEMM of B*L/2 two-row batches, which the GPU library serves with a small-batch float32 kernel at a fraction of the device's
rate. The op library's ``EdmPoolLogits`` computes the same product as one large GEMM with that kernel's exact per-element accumulation order
(ops/README.md), and the graph gives the stock op itself the batches the library computes in a separate small launch (the trailing ``nb mod 65535`` when
small, or a small call whole: ``LAUNCH_BATCHES`` / ``SMALL_BATCHES`` below) — so every row carries stock's bits at every batch size. Which
rows go where is integer arithmetic on the runtime shape that TensorFlow runs on the host: no kernel, copy or device round trip decides it.

``hostshape`` — Sonnet's ``BatchApply`` around each attention block's output projection reshapes (B, T, C) to (B·T, C) with B·T computed as
an int32 ``Prod`` over the tensor's runtime shape — an op TensorFlow places on the GPU, so the shape travels host → device → host and the
host waits for the device in every block. The lever gives ``Reshape`` the equivalent ``-1`` for that dimension: the same reshape, decided
on the host, eleven device round trips fewer per prediction.

``bnconst`` — the per-channel ``scale`` and ``shift`` vectors of the 14 ``BatchNorm -> GELU`` chains. Stock derives them on every call from the
moving statistics (``var + eps``, ``rsqrt``, ``* gamma``, ``mean * scale``, ``beta - ...``: five small kernels per chain, plus the variable
reads) although they depend on no input. The lever evaluates exactly those stock ops once on the device when the graph is rewritten and gives
``EdmScaleShiftGelu`` the resulting vectors as constants — the same bits, about a hundred fewer op dispatches per prediction.

``biasgelu`` — the bias add of the eight convolutions whose output feeds a GELU. The stem's and the six conv-tower blocks' first convolution:
its biased output ``h = conv + b`` (a ``BiasAdd`` kernel, one pass over the block's largest tensor) feeds the pointwise block's
``BatchNorm -> GELU`` and the block's residual add; ``EdmBiasScaleShiftGelu`` and ``EdmBias2Residual`` each take the raw convolution output
and add the bias in registers (TensorFlow's float32 add, the operation ``BiasAdd`` performs), so ``h`` is never written or read back. The
final pointwise convolution: its ``BiasAdd`` and the GELU after it (three kernels) become one ``EdmBiasGelu``.

``poolgelu`` — the six places where a pooling module's output feeds the next block's ``BatchNorm -> GELU`` and nothing else (the stem's and
conv-tower blocks 0-4's): ``EdmSoftmaxPool2Gelu`` performs the pooling and the chain in one pass, so the pooled tensor is never written or
read back. Per element the operations are ``EdmSoftmaxPool2``'s followed by ``EdmScaleShiftGelu``'s, unchanged.

``biasact`` — a bias add followed by an activation: the hidden layer of the eleven transformer MLPs (``Linear`` + bias -> ``relu``) and the
two heads (``Linear`` + bias -> ``softplus``). Stock runs the add and the activation as two kernels, two passes over the tensor; the op
library's ``EdmBiasAct`` does both per element in one pass with the same float32 operations (ops/README.md).

``qbias`` — each attention block's query after the head transpose: ``q * key_size**-0.5``, then ``q + r_w_bias`` for the content logits and
``q + r_r_bias`` for the relative-position logits — three kernels. The op library's ``EdmQScaleBias`` reads ``q`` once and writes both sums.

``hostcrop`` — the graph's first op crops each 393,216-position window to the central 196,608 positions the network reads, after the whole
window has crossed the bus. The lever removes that StridedSlice, gives the input placeholder the cropped length, and the call uploads only the
central positions of each window (``_graph.as_cropped_tensor``: a contiguous view per window, no host copy) — the same bytes on the device,
half the transfer.
"""
from __future__ import annotations

LEVERS: tuple = ("pool", "poolgemm", "bngelu", "bnconst", "poscache", "relsoftmax", "biasres", "biasgelu", "poolgelu", "hostshape", "layernorm", "biasact", "qbias", "hostcrop")
POOL_SCOPES = ("seqnn/trunk/stem/pooling/softmax_pooling",) + tuple(f"seqnn/trunk/downres/downres_block_{i}/pooling/softmax_pooling" for i in range(6))


def pool_region(g, scope: str) -> dict:
    """The nodes of one SoftmaxPooling1D in the stock graph, matched by op and wiring (KeyError / RuntimeError by name otherwise)."""
    P = scope
    r = {"x4": f"{P}/Reshape", "logits": f"{P}/linear/MatMul", "t_in": f"{P}/transpose", "softmax": f"{P}/Softmax", "t_out": f"{P}/transpose_1",
         "mul": f"{P}/mul", "sum": f"{P}/Sum"}
    want = {"x4": "Reshape", "logits": "BatchMatMulV2", "t_in": "Transpose", "softmax": "Softmax", "t_out": "Transpose", "mul": "Mul", "sum": "Sum"}
    for k, name in r.items():
        if not g.has(name, want[k]):
            raise RuntimeError(f"pool: {name} ({want[k]}) not found — not the released Enformer graph")
    pr = g.producer
    ok = (pr(g.node(r["logits"]).input[0]) == r["x4"] and pr(g.node(r["t_in"]).input[0]) == r["logits"] and pr(g.node(r["softmax"]).input[0]) == r["t_in"]
          and pr(g.node(r["t_out"]).input[0]) == r["softmax"] and {pr(i) for i in g.node(r["mul"]).input} == {r["x4"], r["t_out"]}
          and pr(g.node(r["sum"]).input[0]) == r["mul"])
    if not ok:
        raise RuntimeError(f"pool: the wiring under {P} is not the released Enformer's")
    for k in ("t_in", "softmax", "t_out", "mul"):                       # the replaced intermediates feed nothing else
        cons = [c.name for c in g.consumers(r[k])]
        expect = {"t_in": [r["softmax"]], "softmax": [r["t_out"]], "t_out": [r["mul"]], "mul": [r["sum"]]}[k]
        if sorted(cons) != sorted(expect):
            raise RuntimeError(f"pool: {r[k]} has consumers {cons} (expected {expect})")
    return r


def rewrite_pool(g) -> dict:
    n_rewired = 0
    for P in POOL_SCOPES:
        r = pool_region(g, P)
        y = g.add(f"{P}/kit/pool", "EdmSoftmaxPool2", [r["x4"], r["logits"]], arith_variant=0)     # (B, L/2, 2, C) x2 -> (B, L/2, C)
        n_rewired += g.rewire(r["sum"], y)
        g.remove([r["t_in"], r["softmax"], r["t_out"], r["mul"], r["sum"]])
    return {"regions": len(POOL_SCOPES), "consumers_rewired": n_rewired}


# ----------------------------------------------------------------------------------------------------------------- bngelu
GELU_COEFF = 1.702


def gelu_tails(g) -> list:
    """Every ``Mul(1.702, h) -> Sigmoid -> Mul(., h)`` in the graph — Enformer's GELU applied to a tensor ``h``, matched by op, wiring and the
    constant, the product and the sigmoid consumed only inside — as dicts {h, m1, s, m2} of node names."""
    import numpy as np
    from tensorflow.python.framework import tensor_util
    pr = g.producer
    out = []
    for sg in [n for n in g.gd.node if n.op == "Sigmoid" and n.name not in g.removed]:
        m1 = g.node(pr(sg.input[0]))
        if m1.op != "Mul" or len(m1.input) != 2:
            continue
        consts = [pr(i) for i in m1.input if g.node(pr(i)).op == "Const"]
        others = [pr(i) for i in m1.input if g.node(pr(i)).op != "Const"]
        if len(consts) != 1 or len(others) != 1:
            continue
        k = tensor_util.MakeNdarray(g.node(consts[0]).attr["value"].tensor)
        if k.size != 1 or np.float32(k.reshape(-1)[0]) != np.float32(GELU_COEFF):
            continue
        cs = g.consumers(sg.name)
        if len(cs) != 1 or cs[0].op != "Mul" or {pr(i) for i in cs[0].input} != {sg.name, others[0]} or [c.name for c in g.consumers(m1.name)] != [sg.name]:
            continue
        out.append({"h": others[0], "m1": m1.name, "s": sg.name, "m2": cs[0].name})
    return out


def bngelu_chains(g) -> list:
    """Every x*scale -> +shift -> GELU chain in the graph (a GELU tail whose ``h`` is the BatchNorm ``AddV2``; each intermediate consumed
    only inside the chain), as dicts {x, scale, shift, mb, a, m1, s, m2}."""
    pr = g.producer
    out = []
    for t in gelu_tails(g):
        a = g.node(t["h"])
        if a.op != "AddV2":
            continue                                                    # the GELU of a bias add (the final pointwise output): biasgelu's
        ins = [g.node(pr(i)) for i in a.input]
        mb = next((x for x in ins if x.op == "Mul"), None)
        if mb is None:
            continue
        shift = next(pr(i) for i in a.input if pr(i) != mb.name)
        scale = next((pr(i) for i in mb.input if pr(i).endswith("batchnorm/mul")), None)
        if scale is None or not shift.endswith("batchnorm/sub"):
            continue
        x = next(pr(i) for i in mb.input if pr(i) != scale)
        ok = sorted(c.name for c in g.consumers(mb.name)) == [a.name] and sorted(c.name for c in g.consumers(a.name)) == sorted([t["m1"], t["m2"]])
        if not ok:
            raise RuntimeError(f"bngelu: the chain at {t['s']} has consumers outside the chain — not the released Enformer graph")
        out.append({"x": x, "scale": scale, "shift": shift, "mb": mb.name, "a": a.name, "m1": t["m1"], "s": t["s"], "m2": t["m2"]})
    return out


def rewrite_bngelu(g) -> dict:
    chains = bngelu_chains(g)
    if len(chains) != 14:
        raise RuntimeError(f"bngelu: {len(chains)} BatchNorm -> GELU chains found (the released Enformer has 14)")
    n = 0
    for i, c in enumerate(chains):
        K = c["s"].rsplit("/", 1)[0] + "/kit/bngelu"
        flat = g.const_i32(f"{K}/flat", [-1])
        scale = g.add(f"{K}/scale", "Reshape", [c["scale"], flat], T=True, Tshape_i32=True)      # (1, 1, C) -> (C)
        shift = g.add(f"{K}/shift", "Reshape", [c["shift"], flat], T=True, Tshape_i32=True)
        y = g.add(f"{K}/y", "EdmScaleShiftGelu", [c["x"], scale, shift], arith_variant=0)
        n += g.rewire(c["m2"], y)
        g.remove([c["mb"], c["a"], c["m1"], c["s"], c["m2"]])
    return {"chains": len(chains), "consumers_rewired": n}


# ----------------------------------------------------------------------------------------------------------------- poolgemm
LAUNCH_BATCHES = 65535                 # the stock library computes a strided-batched GEMM of nb > 65535 batches as launches over multiples of 65535 batches plus one trailing launch of nb mod 65535 batches, whose kernel it chooses by that count
SMALL_BATCHES = 3072                   # below this batch count the library may choose a kernel with another accumulation order (observed: 3 on sm_80; 6, 9, 12, 1,536 on sm_90); such batches go through the stock op itself


def rewrite_poolgemm(g) -> dict:
    """The pooling logits ``l = x @ W`` of every SoftmaxPooling1D: stock = BatchMatMulV2 over (B*L/2, 2, C) x (C, C). The rows the stock library
    computes with its small-batch FP32 kernel — every full launch of LAUNCH_BATCHES batches, and a single or trailing launch of at least
    SMALL_BATCHES batches — are computed by the op library's EdmPoolLogits (that kernel's float32 accumulation order: ops/README.md); the rows
    of a trailing launch smaller than SMALL_BATCHES are computed by the stock BatchMatMulV2 itself on exactly those batches and placed behind
    them. Which rows go where is decided in the graph from the runtime batch count, so one graph serves every batch size; the decision is
    int32 arithmetic only (``Mul``, ``FloorMod``, ``Sub``, ``FloorDiv``, ``Add``), which TensorFlow computes on the host for a GPU graph —
    no boolean or select op, whose GPU kernels would take the shape scalars through device memory and make the host wait for them."""
    n = 0
    for P in POOL_SCOPES:
        mm = f"{P}/linear/MatMul"
        if not g.has(mm, "BatchMatMulV2"):
            raise RuntimeError(f"poolgemm: {mm} (BatchMatMulV2) not found")
        node = g.node(mm)
        if node.attr["adj_x"].b or node.attr["adj_y"].b:
            raise RuntimeError(f"poolgemm: {mm} has adj_x/adj_y set")
        x4, w = node.input[0], node.input[1]
        K = f"{P}/kit/gemm"
        # scalars from the runtime shape of x4 = (B, L/2, 2, C): nb = B * L/2 two-row batches, C channels
        shp = g.add(f"{K}/shape", "Shape", [x4], T=True, out_type_i32=True)
        def dim(i):
            return g.add(f"{K}/d{i}", "StridedSlice", [shp, g.const_i32(f"{K}/d{i}b", [i]), g.const_i32(f"{K}/d{i}e", [i + 1]), g.const_i32(f"{K}/d{i}s", [1])],
                         T_i32=True, Index_i32=True, shrink_axis_mask=1, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0)
        nb = g.add(f"{K}/nb", "Mul", [dim(0), dim(1)], T_i32=True)
        cdim = dim(3)
        launch = g.const_i32(f"{K}/launch", LAUNCH_BATCHES); zero = g.const_i32(f"{K}/zero", 0)
        r = g.add(f"{K}/r", "FloorMod", [nb, launch], T_i32=True)
        # rem = the trailing batches the stock op must compute itself: the trailing launch r = nb mod LAUNCH when r < SMALL (nb <= LAUNCH: r = nb, or 0
        # when nb = LAUNCH — the whole call when it is small). [r < SMALL] for 0 <= r < LAUNCH is (SMALL - 1 - r) // LAUNCH + 1 (floor division).
        rsmall = g.add(f"{K}/rsmall", "Add", [g.add(f"{K}/rsmall/q", "FloorDiv", [g.add(f"{K}/rsmall/d", "Sub", [g.const_i32(f"{K}/smallm1", SMALL_BATCHES - 1), r], T_i32=True),
                                                                                    launch], T_i32=True), g.const_i32(f"{K}/one", 1)], T_i32=True)
        rem = g.add(f"{K}/rem", "Mul", [r, rsmall], T_i32=True)
        main = g.add(f"{K}/main", "Sub", [nb, rem], T_i32=True)
        two = g.const_i32(f"{K}/two", 2); m1 = g.const_i32(f"{K}/m1", -1)
        x2 = g.add(f"{K}/x2", "Reshape", [x4, g.add(f"{K}/s2", "Pack", [m1, cdim], T_i32=True, N=2, axis=0)], T=True, Tshape_i32=True)          # (2 nb, C)
        x3 = g.add(f"{K}/x3", "Reshape", [x4, g.add(f"{K}/s3", "Pack", [m1, two, cdim], T_i32=True, N=3, axis=0)], T=True, Tshape_i32=True)     # (nb, 2, C)
        begin = g.add(f"{K}/begin", "Pack", [main, zero, zero], T_i32=True, N=3, axis=0)
        size = g.add(f"{K}/size", "Pack", [rem, two, cdim], T_i32=True, N=3, axis=0)
        xr = g.add(f"{K}/xrem", "Slice", [x3, begin, size], T=True, Index_i32=True)                                                              # the trailing small launch's batches
        yr = g.add(f"{K}/yrem", "BatchMatMulV2", [xr, w], T=True, adj_x=False, adj_y=False)                                                      # ... through the stock op
        yr2 = g.add(f"{K}/yrem2", "Reshape", [yr, f"{K}/s2"], T=True, Tshape_i32=True)
        y2 = g.add(f"{K}/logits2", "EdmPoolLogits", [x2, w, yr2])
        y4 = g.add(f"{K}/logits", "Reshape", [y2, shp], T=True, Tshape_i32=True)                                                                 # back to (B, L/2, 2, C)
        n += g.rewire(mm, y4)
        g.remove([mm])
    return {"regions": len(POOL_SCOPES), "consumers_rewired": n}


# ----------------------------------------------------------------------------------------------------------------- poscache
ATTENTION_BLOCKS = tuple(f"seqnn/trunk/transformer/transformer_block_{i}/mha/multihead_attention" for i in range(11))


def rewrite_poscache(g) -> dict:
    """Replace each block's ``r_k_layer/transpose`` (the projected relative-position keys, input-independent) by a constant holding the value
    the stock subgraph computes on this device. ``g.evaluate`` runs the stock nodes once through the same TensorFlow kernels."""
    names = [f"{A}/r_k_layer/transpose" for A in ATTENTION_BLOCKS]
    for nm in names:
        if not g.has(nm, "Transpose"):
            raise RuntimeError(f"poscache: {nm} (Transpose) not found — not the released Enformer graph")
    values = g.evaluate(names)                                           # stock ops, this device, once
    n = 0
    for A, nm, val in zip(ATTENTION_BLOCKS, names, values):
        if val.ndim != 4 or val.shape[0] != 1:
            raise RuntimeError(f"poscache: {nm} has shape {val.shape} (expected (1, heads, distances, key size): independent of the batch)")
        c = g.const_f32(f"{A}/kit/r_k", val)
        n += g.rewire(nm, c)
        g.remove([nm])
    return {"blocks": len(names), "consumers_rewired": n, "bytes": int(sum(v.nbytes for v in values))}


# --------------------------------------------------------------------------------------------------------------- relsoftmax
def rewrite_relsoftmax(g) -> dict:
    """Per attention block: ``Softmax(add_3)``, ``add_3 = MatMul (content logits) + Slice_1 (relative_shift(MatMul_1))`` ->
    EdmRelShiftSoftmax(MatMul, MatMul_1). The relative-shift chain feeds only the add, the add only the Softmax."""
    n = 0
    for A in ATTENTION_BLOCKS:
        content, rel, add = f"{A}/MatMul", f"{A}/MatMul_1", f"{A}/add_3"
        chain = [f"{A}/{x}" for x in ("strided_slice_5", "zeros_like", "concat_2", "Reshape_3", "Slice", "Reshape_4", "Slice_1")]
        for nm, op in [(content, "BatchMatMulV2"), (rel, "BatchMatMulV2"), (add, "AddV2"), (chain[0], "StridedSlice"), (chain[1], "ZerosLike"), (chain[2], "ConcatV2"),
                       (chain[3], "Reshape"), (chain[4], "Slice"), (chain[5], "Reshape"), (chain[6], "Slice")]:
            if not g.has(nm, op):
                raise RuntimeError(f"relsoftmax: {nm} ({op}) not found — not the released Enformer graph")
        pr = g.producer
        if {pr(i) for i in g.node(add).input} != {content, chain[6]} or [pr(i) for i in g.node(chain[6]).input][0] != chain[5] or \
           [pr(i) for i in g.node(chain[5]).input][0] != chain[4] or [pr(i) for i in g.node(chain[4]).input][0] != chain[3] or \
           [pr(i) for i in g.node(chain[3]).input][0] != chain[2] or sorted(pr(i) for i in g.node(chain[2]).input[:2]) != sorted([chain[1], rel]) or \
           pr(g.node(chain[1]).input[0]) != chain[0] or pr(g.node(chain[0]).input[0]) != rel:
            raise RuntimeError(f"relsoftmax: the relative-shift chain of {A} is wired differently — not the released Enformer graph")
        for nm in chain:                                                # the chain feeds only itself and add_3; MatMul_1 feeds only the chain
            outside = [c.name for c in g.consumers(nm) if c.name not in chain + [add]]
            if outside:
                raise RuntimeError(f"relsoftmax: {nm} feeds {outside} outside the chain")
        if sorted(c.name for c in g.consumers(rel)) != sorted([chain[0], chain[2]]):
            raise RuntimeError(f"relsoftmax: {rel} has consumers outside the chain")
        sm = f"{A}/Softmax"
        if not g.has(sm, "Softmax") or g.producer(g.node(sm).input[0]) != add or sorted(c.name for c in g.consumers(add)) != [sm]:
            raise RuntimeError(f"relsoftmax: {add} must feed exactly {sm} — not the released Enformer graph")
        y = g.add(f"{A}/kit/relsoftmax", "EdmRelShiftSoftmax", [content, rel])
        n += g.rewire(sm, y)
        g.remove(chain + [add, sm])
    return {"blocks": len(ATTENTION_BLOCKS), "consumers_rewired": n}


# ----------------------------------------------------------------------------------------------------------------- biasres
def biasres_sites(g) -> list:
    """Every residual add ``r + t`` where t is a bias add (``BiasAdd``, or ``Add``/``AddV2`` of a rank-1 variable read) of a tensor consumed
    nowhere else, possibly through one Reshape: [{res, y, bias, badd, reshape|None, add}]."""
    pr = g.producer
    sites = []
    for add in [n for n in g.gd.node if n.op == "AddV2" and n.name.endswith(("residual/add", "residual/add_1")) and n.name not in g.removed]:
        for k, inp in enumerate(add.input[:2]):
            t = g.node(pr(inp)); reshape = None
            if t.op == "Reshape" and len(g.consumers(t.name)) == 1:
                reshape = t; t = g.node(pr(t.input[0]))
            if t.op not in ("BiasAdd", "Add", "AddV2") or len(t.input) != 2:
                continue
            bias = pr(t.input[1]); ysrc = t.input[0]
            if g.node(bias).op != "ReadVariableOp":
                continue
            cons_t = g.consumers(t.name)
            shape_readers = [c.name for c in cons_t if c.op == "Shape"]          # readers of the biased tensor's SHAPE only (Sonnet's BatchApply plumbing)
            value_readers = sorted(c.name for c in cons_t if c.op != "Shape")
            if value_readers != [reshape.name if reshape is not None else add.name]:
                continue                                                # the biased tensor's values are used elsewhere (e.g. a block's first convolution): stays stock
            res = add.input[1 - k]
            sites.append({"add": add.name, "badd": t.name, "y": ysrc, "bias": bias, "reshape": reshape.name if reshape is not None else None, "res": res,
                          "shape_readers": shape_readers})
            break
    return sites


def rewrite_biasres(g) -> dict:
    sites = biasres_sites(g)
    if len(sites) != 29:                                                # 7 conv residual blocks + 11 attention output projections + 11 MLP outputs
        raise RuntimeError(f"biasres: {len(sites)} bias+residual sites found (the released Enformer has 29)")
    n = 0
    for st in sites:
        K = st["add"] + "/kit"
        y = st["y"]
        if st["reshape"] is not None:                                   # apply the stock reshape to the un-biased tensor (bias add and reshape commute: the bias is on the last axis, which the reshape keeps)
            rs = g.node(st["reshape"])
            y = g.add(f"{K}/reshape", "Reshape", [y, rs.input[1]], T=True)
            g.node(y).attr["Tshape"].CopyFrom(rs.attr["Tshape"])
        assoc = 1 if (st["reshape"] is None and g.node(st["badd"]).op in ("Add", "AddV2")) else 0   # an Add feeding the residual directly is executed regrouped: (res + y) + bias
        res = g.resolve(g.producer(st["res"])) + ("" if ":" not in st["res"] else ":" + st["res"].split(":")[1])
        out = g.add(f"{K}/biasres", "EdmBiasResidual", [y, st["bias"], res], assoc=assoc)
        for sr in st["shape_readers"]:                                  # the shape of y + bias is the shape of y
            node = g.node(sr)
            for q, inp in enumerate(node.input):
                if g.producer(inp) == st["badd"]:
                    node.input[q] = st["y"]
        n += g.rewire(st["add"], out)
        g.remove([st["badd"], st["add"]] + ([st["reshape"]] if st["reshape"] else []))
    return {"sites": len(sites), "consumers_rewired": n}


# ----------------------------------------------------------------------------------------------------------------- layernorm
def layernorm_sites(g) -> list:
    """Every LayerNorm over the last axis as Sonnet writes it (``tf.nn.moments`` then ``tf.nn.batch_normalization`` with per-row statistics):
    mean = Mean(x, -1) ; sd = SquaredDifference(x, StopGradient(mean)) ; var = Mean(sd, -1) ; a = var + eps ; rs = Rsqrt(a) ; inv = rs * gamma ;
    y = x * inv + (beta - mean * inv) — matched by op, wiring, operand order and constants, every intermediate consumed only inside the chain:
    [{x, gamma, beta, epsilon, y, chain (node names), consts (the chain's own constants)}]."""
    import numpy as np
    from tensorflow.python.framework import tensor_util
    pr = g.producer

    def full(t):                                                        # tensor name with its output index
        return t if ":" in t else t + ":0"

    def only(name, ops):                                                # the consumers of `name`, which must be exactly nodes of these ops (in any order)
        cs = g.consumers(name)
        return cs if sorted(c.op for c in cs) == sorted(ops) else None

    def last_axis_mean(n):
        if n.op != "Mean" or not n.attr["keep_dims"].b or len(n.input) != 2 or g.node(pr(n.input[1])).op != "Const":
            return False
        axes = tensor_util.MakeNdarray(g.node(pr(n.input[1])).attr["value"].tensor).reshape(-1)
        return axes.size == 1 and int(axes[0]) == -1

    sites = []
    for m1 in [n for n in g.gd.node if n.op == "Mean" and n.name not in g.removed]:
        if not last_axis_mean(m1):
            continue
        x = m1.input[0]
        c1 = only(m1.name, ["StopGradient", "Mul"])
        if c1 is None:
            continue
        sg = next(c for c in c1 if c.op == "StopGradient"); mul2 = next(c for c in c1 if c.op == "Mul")
        c2 = only(sg.name, ["SquaredDifference"])
        if c2 is None or sorted(full(i) for i in c2[0].input) != sorted([full(x), full(sg.name)]):
            continue
        sd = c2[0]
        c3 = only(sd.name, ["Mean"])
        if c3 is None or not last_axis_mean(c3[0]) or full(c3[0].input[0]) != full(sd.name):
            continue
        m2 = c3[0]
        c4 = only(m2.name, ["AddV2"])
        if c4 is None:
            continue
        a = c4[0]
        eps_name = next((pr(i) for i in a.input if pr(i) != m2.name), None)
        if eps_name is None or g.node(eps_name).op != "Const":
            continue
        eps_node = g.node(eps_name)
        eps = tensor_util.MakeNdarray(eps_node.attr["value"].tensor)
        if eps.size != 1 or eps.dtype != np.float32:
            continue
        c5 = only(a.name, ["Rsqrt"])
        if c5 is None:
            continue
        rs = c5[0]
        c6 = only(rs.name, ["Mul"])
        if c6 is None:
            continue
        mg = c6[0]
        gamma = next((i for i in mg.input if pr(i) != rs.name), None)
        c7 = only(mg.name, ["Mul", "Mul"])
        if gamma is None or c7 is None or mul2.name not in [c.name for c in c7]:
            continue
        mul1 = next((c for c in c7 if c.name != mul2.name), None)
        if mul1 is None:
            continue
        if sorted(full(i) for i in mul1.input) != sorted([full(x), full(mg.name)]) or sorted(full(i) for i in mul2.input) != sorted([full(m1.name), full(mg.name)]):
            continue
        c8 = only(mul2.name, ["Sub"])
        if c8 is None or full(c8[0].input[1]) != full(mul2.name):            # beta - mean*inv, in this operand order
            continue
        sb = c8[0]; beta = sb.input[0]
        c9 = only(mul1.name, ["AddV2"])
        if c9 is None or sorted(full(i) for i in c9[0].input) != sorted([full(mul1.name), full(sb.name)]) or [c.name for c in g.consumers(sb.name)] != [c9[0].name]:
            continue
        y = c9[0]
        chain = [m1.name, sg.name, sd.name, m2.name, a.name, rs.name, mg.name, mul1.name, mul2.name, sb.name, y.name]
        consts = [c for c in {pr(m1.input[1]), pr(m2.input[1]), eps_node.name} if all(k.name in chain for k in g.consumers(c))]
        sites.append({"x": x, "gamma": gamma, "beta": beta, "epsilon": float(eps.reshape(-1)[0]), "y": y.name, "chain": chain, "consts": consts})
    return sites


def rewrite_layernorm(g) -> dict:
    sites = layernorm_sites(g)
    if len(sites) != 22:                                                # 11 transformer blocks x (attention, MLP)
        raise RuntimeError(f"layernorm: {len(sites)} LayerNorm chains found (the released Enformer has 22)")
    n = 0
    for st in sites:
        K = st["y"].rsplit("/", 1)[0] + "/kit/layernorm"
        y = g.add(K, "EdmLayerNorm", [st["x"], st["gamma"], st["beta"]], epsilon=st["epsilon"])
        n += g.rewire(st["y"], y)
        g.remove(st["chain"] + st["consts"])
    return {"sites": len(sites), "consumers_rewired": n}
# ----------------------------------------------------------------------------------------------------------------- hostshape
def rewrite_hostshape(g) -> dict:
    """Per attention block: ``batch_apply/Prod`` (int32, keep_dims: the product B·T of the leading dimensions of the output projection's input,
    read by the ``ConcatV2`` that builds ``batch_apply/Reshape``'s target shape) -> the constant [-1], which ``Reshape`` resolves to the same B·T."""
    n = 0
    minus1 = g.const_i32("seqnn/trunk/transformer/kit/hostshape/leading", [-1])
    for A in ATTENTION_BLOCKS:
        prod, concat = f"{A}/batch_apply/Prod", f"{A}/batch_apply/concat"
        if not g.has(prod, "Prod") or not g.has(concat, "ConcatV2"):
            raise RuntimeError(f"hostshape: {prod} (Prod) / {concat} (ConcatV2) not found — not the released Enformer graph")
        if not g.node(prod).attr["keep_dims"].b or [c.name for c in g.consumers(prod)] != [concat] or [c.op for c in g.consumers(concat)] != ["Reshape"]:
            raise RuntimeError(f"hostshape: {prod} is not the keep_dims product feeding only {concat} -> Reshape — not the released Enformer graph")
        n += g.rewire(prod, minus1)
        g.remove([prod])
    return {"blocks": len(ATTENTION_BLOCKS), "consumers_rewired": n}


# ----------------------------------------------------------------------------------------------------------------- bnconst
def rewrite_bnconst(g) -> dict:
    """The ``scale`` / ``shift`` inputs of every EdmScaleShiftGelu (``rewrite_bngelu``'s reshapes of the stock ``batchnorm/mul`` and
    ``batchnorm/sub`` vectors: functions of the moving statistics only) evaluated once through the stock ops on this device and replaced by
    constants. The stock nodes that computed them feed nothing else and drop out of the executed graph."""
    sites = [n for n in g.gd.node if n.op == "EdmScaleShiftGelu" and n.name not in g.removed]
    if len(sites) != 14:
        raise RuntimeError(f"bnconst: {len(sites)} EdmScaleShiftGelu nodes found (bngelu makes 14; bnconst runs before biasgelu / poolgelu fuse them)")
    slots = []                                                           # (ssg node, input index, reshape node name)
    for n in sites:
        for k in (1, 2):
            node = g.node(g.producer(n.input[k]))
            if node.op != "Reshape" or not node.name.endswith(("/kit/bngelu/scale", "/kit/bngelu/shift")):
                raise RuntimeError(f"bnconst: {n.name} input {k} is {node.name} ({node.op}) — expected bngelu's scale / shift reshape")
            slots.append((n, k, node.name))
    values = g.evaluate([nm for _, _, nm in slots])                      # stock ops, this device, once
    for (n, k, nm), val in zip(slots, values):
        if val.ndim != 1:
            raise RuntimeError(f"bnconst: {nm} has shape {val.shape} (expected a per-channel vector)")
        n.input[k] = g.const_f32(nm + "/const", val)                     # bngelu made the reshape for this input alone (finish() names any other consumer)
        g.remove([nm])
    return {"chains": len(sites), "vectors": len(slots), "bytes": int(sum(v.nbytes for v in values))}


# ----------------------------------------------------------------------------------------------------------------- biasgelu
def nhwc(b) -> bool:
    """A BiasAdd whose bias runs along the last axis."""
    return (b.attr["data_format"].s if "data_format" in b.attr else b"NHWC") == b"NHWC" and len(b.input) == 2


def biasgelu_sites(g) -> tuple:
    """(conv sites, gelu sites). A conv site: a ``BiasAdd`` (channels last) consumed exactly by one EdmScaleShiftGelu (as ``x``) and one
    EdmBiasResidual with assoc 0 (as ``res``) — {badd, x, bias, ssg, br}. A gelu site: a ``BiasAdd`` consumed exactly by a GELU tail
    (``gelu_tails``) — {badd, x, bias, m1, s, m2}."""
    pr = g.producer
    conv, gelu = [], []
    for b in [n for n in g.gd.node if n.op == "BiasAdd" and n.name not in g.removed and nhwc(n)]:
        cons = g.consumers(b.name)
        ssg = [c for c in cons if c.op == "EdmScaleShiftGelu" and pr(c.input[0]) == b.name]
        br = [c for c in cons if c.op == "EdmBiasResidual" and pr(c.input[2]) == b.name and c.attr["assoc"].i == 0 and pr(c.input[0]) != b.name]
        if len(cons) == 2 and len(ssg) == 1 and len(br) == 1:
            conv.append({"badd": b.name, "x": b.input[0], "bias": b.input[1], "ssg": ssg[0].name, "br": br[0].name})
    for t in gelu_tails(g):
        b = g.node(t["h"])
        if b.op == "BiasAdd" and b.name not in g.removed and nhwc(b) and sorted(c.name for c in g.consumers(b.name)) == sorted([t["m1"], t["m2"]]):
            gelu.append({"badd": b.name, "x": b.input[0], "bias": b.input[1], "m1": t["m1"], "s": t["s"], "m2": t["m2"]})
    return conv, gelu


def rewrite_biasgelu(g) -> dict:
    conv, gelu = biasgelu_sites(g)
    if len(conv) != 7 or len(gelu) != 1:
        raise RuntimeError(f"biasgelu: {len(conv)} convolution bias sites and {len(gelu)} bias -> GELU sites found (the released Enformer has 7 and 1 once bngelu and biasres have run)")
    n = 0
    for st in conv:
        K = st["badd"] + "/kit"
        ssg, br = g.node(st["ssg"]), g.node(st["br"])
        y = g.add(f"{K}/bngelu", "EdmBiasScaleShiftGelu", [st["x"], st["bias"], ssg.input[1], ssg.input[2]])
        r = g.add(f"{K}/biasres", "EdmBias2Residual", [br.input[0], br.input[1], st["x"], st["bias"]])
        n += g.rewire(st["ssg"], y) + g.rewire(st["br"], r)
        g.remove([st["badd"], st["ssg"], st["br"]])
    for st in gelu:
        y = g.add(st["badd"] + "/kit/gelu", "EdmBiasGelu", [st["x"], st["bias"]])
        n += g.rewire(st["m2"], y)
        g.remove([st["badd"], st["m1"], st["s"], st["m2"]])
    return {"conv_sites": len(conv), "gelu_sites": len(gelu), "consumers_rewired": n}


# ----------------------------------------------------------------------------------------------------------------- poolgelu
def rewrite_poolgelu(g) -> dict:
    """Every EdmSoftmaxPool2 whose only consumer is an EdmScaleShiftGelu (as ``x``) -> EdmSoftmaxPool2Gelu(x, logits, scale, shift)."""
    pr = g.producer
    sites = []
    for p in [n for n in g.gd.node if n.op == "EdmSoftmaxPool2" and n.name not in g.removed]:
        cons = g.consumers(p.name)
        if len(cons) == 1 and cons[0].op == "EdmScaleShiftGelu" and pr(cons[0].input[0]) == p.name and p.attr["arith_variant"].i == 0 and cons[0].attr["arith_variant"].i == 0:
            sites.append((p, cons[0]))
    if len(sites) != 6:
        raise RuntimeError(f"poolgelu: {len(sites)} pooling -> BatchNorm-GELU sites found (the released Enformer has 6 once pool and bngelu have run)")
    n = 0
    for p, ssg in sites:
        y = g.add(p.name.rsplit("/kit/", 1)[0] + "/kit/poolgelu", "EdmSoftmaxPool2Gelu", [p.input[0], p.input[1], ssg.input[1], ssg.input[2]])
        n += g.rewire(ssg.name, y)
        g.remove([p.name, ssg.name])
    return {"sites": len(sites), "consumers_rewired": n}


# ----------------------------------------------------------------------------------------------------------------- hostcrop
INPUT_CROP = "seqnn/trunk/crop_input/strided_slice"


def rewrite_hostcrop(g) -> dict:
    """The graph's first op crops the (B, 393216, 4) input to its central positions (``crop_input``: a StridedSlice along the sequence axis)
    — after the whole window has been uploaded. The lever reads the crop bounds off that node, removes it, declares the input placeholder with
    the cropped length and records the bounds on the graph (``input_crop``); ``_graph.rewrite``'s ``predict`` then uploads only those positions
    (``as_cropped_tensor``) of each (B, L, 4) batch the runtime hands it. The device receives the bytes the StridedSlice produced: half of each window."""
    from tensorflow.python.framework import tensor_util
    pr = g.producer
    ph = pr(g.cf.inputs[0].name) if g.cf is not None else "args_0"
    if not g.has(ph, "Placeholder") or not g.has(INPUT_CROP, "StridedSlice") or pr(g.node(INPUT_CROP).input[0]) != ph:
        raise RuntimeError(f"hostcrop: {INPUT_CROP} (StridedSlice of the input placeholder {ph}) not found — not the released Enformer graph")
    if [c.name for c in g.consumers(ph)] != [INPUT_CROP]:
        raise RuntimeError(f"hostcrop: the input placeholder feeds {[c.name for c in g.consumers(ph)]} (expected only {INPUT_CROP})")
    node, dims = g.node(INPUT_CROP), g.node(ph).attr["shape"].shape.dim
    shape = [int(d.size) for d in dims]
    if len(shape) != 3 or shape[1] <= 0:
        raise RuntimeError(f"hostcrop: input placeholder shape {shape} (expected (B, L, 4) with L known)")
    begin, end, strides = (tensor_util.MakeNdarray(g.node(pr(i)).attr["value"].tensor).tolist() if g.node(pr(i)).op == "Const" else None for i in node.input[1:4])
    a = {k: node.attr[k].i for k in ("begin_mask", "end_mask", "ellipsis_mask", "new_axis_mask", "shrink_axis_mask")}
    def whole(i):                                                       # slice spec i (one per axis here: three specs, rank 3) takes axis i whole
        return bool(a["ellipsis_mask"] >> i & 1) or bool(a["begin_mask"] >> i & 1 and a["end_mask"] >> i & 1)
    if begin is None or end is None or strides is None or len(begin) != 3 or len(end) != 3 or strides != [1, 1, 1] or a["new_axis_mask"] or a["shrink_axis_mask"] \
            or a["ellipsis_mask"] >> 1 & 1 or not whole(0) or not whole(2):
        raise RuntimeError(f"hostcrop: {INPUT_CROP} is not a unit-stride crop of the sequence axis alone (begin={begin} end={end} strides={strides} {a})")
    L = shape[1]
    lo = 0 if a["begin_mask"] & 0b010 else (begin[1] + L if begin[1] < 0 else begin[1])
    hi = L if a["end_mask"] & 0b010 else (end[1] + L if end[1] < 0 else end[1])
    if not 0 <= lo < hi <= L:
        raise RuntimeError(f"hostcrop: crop [{lo}, {hi}) of {L} positions")
    n = g.rewire(INPUT_CROP, ph)
    g.remove([INPUT_CROP])
    dims[1].size = hi - lo
    g.input_crop = (lo, hi)
    return {"crop": [lo, hi], "of": L, "consumers_rewired": n}


# ----------------------------------------------------------------------------------------------------------------- biasact
def biasact_sites(g) -> list:
    """Every bias add (``Add``/``AddV2``, or a channels-last ``BiasAdd``, of a tensor and a variable read) consumed only by a ``Relu`` or a
    ``Softplus``: [{x, bias, badd, act, out}] with ``out`` the activation node the op replaces together with the add."""
    pr = g.producer
    sites = []
    for act_node in [n for n in g.gd.node if n.op in ("Relu", "Softplus") and n.name not in g.removed]:
        b = g.node(pr(act_node.input[0]))
        if b.op not in ("Add", "AddV2", "BiasAdd") or len(b.input) != 2 or (b.op == "BiasAdd" and not nhwc(b)):
            continue
        if act_node.input[0] not in (b.name, b.name + ":0") or [c.name for c in g.consumers(b.name)] != [act_node.name]:
            continue                                                    # output 0 of the add, read by the activation only
        var = [i for i in b.input if not i.startswith("^") and ":" not in i and g.node(pr(i)).op == "ReadVariableOp"]
        val = [i for i in b.input if i not in var]
        if len(var) != 1 or len(val) != 1 or val[0].startswith("^"):
            continue
        sites.append({"x": val[0], "bias": var[0], "badd": b.name, "act": act_node.op.lower(), "out": act_node.name})
    return sites


def rewrite_biasact(g) -> dict:
    sites = biasact_sites(g)
    found = {a: sum(1 for st in sites if st["act"] == a) for a in ("relu", "softplus")}
    if found != {"relu": 11, "softplus": 2}:                            # the eleven transformer MLPs' hidden layers; the two heads
        raise RuntimeError(f"biasact: bias+activation sites {found} found (the released Enformer has 11 relu, 2 softplus)")
    n = 0
    for st in sites:
        y = g.add(st["badd"] + "/kit/biasact", "EdmBiasAct", [st["x"], st["bias"]], act=st["act"])
        n += g.rewire(st["out"], y)
        g.remove([st["badd"], st["out"]])
    return {**found, "consumers_rewired": n}


# ----------------------------------------------------------------------------------------------------------------- qbias
def qbias_sites(g) -> list:
    """Every ``q * c`` (``c`` a scalar constant) consumed exactly by two ``AddV2`` of a variable read each: [{q, scale, mul, adds: [(add, bias)] sorted by name}]."""
    import numpy as np
    from tensorflow.python.framework import tensor_util
    pr = g.producer
    sites = []
    for m in [n for n in g.gd.node if n.op == "Mul" and n.name not in g.removed and len(n.input) == 2]:
        consts = [pr(i) for i in m.input if g.node(pr(i)).op == "Const"]
        if len(consts) != 1:
            continue
        k = tensor_util.MakeNdarray(g.node(consts[0]).attr["value"].tensor)
        if k.size != 1:
            continue
        cs = g.consumers(m.name)
        adds = []
        for a in cs:
            if a.op == "AddV2" and len(a.input) == 2 and pr(a.input[0]) == m.name and a.input[0] in (m.name, m.name + ":0") and g.node(pr(a.input[1])).op == "ReadVariableOp":
                adds.append((a.name, a.input[1]))
        if len(cs) != 2 or len(adds) != 2:
            continue
        q = next(i for i in m.input if pr(i) != consts[0])
        sites.append({"q": q, "scale": float(np.float32(k.reshape(-1)[0])), "mul": m.name, "adds": sorted(adds)})
    return sites


def rewrite_qbias(g) -> dict:
    sites = qbias_sites(g)
    if len(sites) != 11:                                                # one per attention block
        raise RuntimeError(f"qbias: {len(sites)} scaled-query bias pairs found (the released Enformer has 11)")
    n = 0
    for st in sites:
        (aw, bw), (ar, br) = st["adds"]                                 # add_1 (+ r_w_bias -> content logits), add_2 (+ r_r_bias -> relative logits)
        y = g.add(st["mul"] + "/kit/qbias", "EdmQScaleBias", [st["q"], bw, br], scale=st["scale"])
        n += g.rewire(aw, y + ":0") + g.rewire(ar, y + ":1")
        g.remove([st["mul"], aw, ar])
    return {"sites": len(sites), "consumers_rewired": n}


REWRITES = {"pool": rewrite_pool, "poolgemm": rewrite_poolgemm, "bngelu": rewrite_bngelu, "bnconst": rewrite_bnconst, "poscache": rewrite_poscache,
            "relsoftmax": rewrite_relsoftmax, "biasres": rewrite_biasres, "biasgelu": rewrite_biasgelu, "poolgelu": rewrite_poolgelu, "hostshape": rewrite_hostshape,
            "layernorm": rewrite_layernorm, "biasact": rewrite_biasact, "qbias": rewrite_qbias, "hostcrop": rewrite_hostcrop}


def build_label(sm) -> str:
    """The compiled levers' build serving a device of compute capability ``sm`` (RuntimeError when the op library cannot run there)."""
    from . import ops
    return ops.serves(sm)

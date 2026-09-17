# Derived from the exported TorchScript archives of Chai-1 (chai_lab 0.6.1, Copyright 2024 Chai Discovery, Inc., Apache-2.0).
"""Automatic step-invariant hoisting for a traced (straight-line SSA) TorchScript method run through ts2eager.

The diffusion module is called 2x199 times per sample with 14 inputs of which only `atom_noised_coords` and `noise_sigma` change
(the stack's wrapper re-hoists per item: when the trunk outputs, the atom features or the coordinate shape change). Every root-level
statement whose inputs do not (transitively) depend on
those two is executed ONCE (`precompute`) and its results that the variant part needs are cached; `step` runs only the variant
statements. Same ops, same operands, same order within every dependency chain -> bitwise identical to the un-hoisted call wherever
each op is run-to-run deterministic. `step_graphed` additionally captures the variant part in a CUDA graph (shapes are static per crop).

    hf = HoistedForward(flat_root, "forward_256", variant=("atom_noised_coords", "noise_sigma"))
    hf.precompute(**kw)          # once per (trunk outputs, features)  — kw = the full keyword set chai passes
    out = hf.step(**kw)           # per denoiser call (only the variant tensors are read from kw; pass the same dict)
"""
import ast, copy, __future__
import torch
from .ts2eager import _insert_frees

SURFACE = ("make", "HoistedForward", "HoistedForward2", "HoistedForward.precompute", "HoistedForward.step", "HoistedForward.step_graphed", "HoistedForward.compile", "HoistedForward.uncompile", "scalar_only_uses", "HOISTERS")   # the names the kit / this package's callers reach (chai1_opt tests/test_lever_surfaces.py checks they exist, statically)

INPLACE_EXTRA = {"masked_fill_", "index_put_", "copy_", "zero_", "fill_"}


def _names(node, ctx):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx)}


class HoistedForward:
    def __init__(self, root, method, variant=("atom_noised_coords", "noise_sigma"), free_dead=True):
        self.root, self.method, self.variant = root, method, tuple(variant)
        ts = object.__getattribute__(root, "_ts")
        rt = object.__getattribute__(root, "_rt")
        src = ts.methods_src[method]
        mod = ast.parse(src)
        fn = mod.body[0]
        assert isinstance(fn, ast.FunctionDef), type(fn)
        argnames = [a.arg for a in fn.args.args]            # includes self
        body = list(fn.body)
        assert isinstance(body[-1], ast.Return), "traced method must end in a single return"
        ret = body[-1]; stmts = body[:-1]
        tainted = set(self.variant)
        is_var = []
        for st in stmts:
            reads, writes = _names(st, ast.Load), _names(st, (ast.Store, ast.Del))
            v = bool(reads & tainted)
            if v:
                tainted |= writes
            is_var.append(v)
        pre_stmts = [st for st, v in zip(stmts, is_var) if not v]
        step_stmts = [st for st, v in zip(stmts, is_var) if v]
        pre_writes = set().union(*[_names(st, ast.Store) for st in pre_stmts]) if pre_stmts else set()
        step_reads = set().union(*[_names(st, ast.Load) for st in step_stmts], _names(ret, ast.Load))
        cache_names = sorted((step_reads & pre_writes) - set(argnames))
        # in-place ops in the step part whose first operand is a cached value or an input -> clone at step entry
        protect = set()
        for st in step_stmts:
            for node in ast.walk(st):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                        and node.func.value.id == "torch" and (node.func.attr.endswith("_") or node.func.attr in INPLACE_EXTRA) \
                        and not node.func.attr.startswith("__") and node.args and isinstance(node.args[0], ast.Name):
                    a0 = node.args[0].id
                    if a0 in cache_names or a0 in argnames:
                        protect.add(a0)
        self.stats = dict(hoister="base", n_stmts=len(stmts), n_pre=len(pre_stmts), n_step=len(step_stmts), n_cache=len(cache_names), protect=sorted(protect))
        # ---- build `_pre(self, <args>)` returning a dict of cached values
        pre_ret = ast.Return(value=ast.Dict(keys=[ast.Constant(n) for n in cache_names], values=[ast.Name(id=n, ctx=ast.Load()) for n in cache_names]))
        pre_fn = ast.FunctionDef(name="_pre", args=copy.deepcopy(fn.args), body=pre_stmts + [pre_ret], decorator_list=[], returns=None, type_comment=None)
        # ---- build `_step(self, <args>, __cache)`
        unpack = []
        for n in cache_names:
            val = ast.Subscript(value=ast.Name(id="__cache", ctx=ast.Load()), slice=ast.Constant(n), ctx=ast.Load())
            if n in protect:
                val = ast.Call(func=ast.Attribute(value=val, attr="clone", ctx=ast.Load()), args=[], keywords=[])
            unpack.append(ast.Assign(targets=[ast.Name(id=n, ctx=ast.Store())], value=val))
        for n in sorted(protect - set(cache_names)):        # protected *inputs*
            unpack.append(ast.Assign(targets=[ast.Name(id=n, ctx=ast.Store())],
                                     value=ast.Call(func=ast.Attribute(value=ast.Name(id=n, ctx=ast.Load()), attr="clone", ctx=ast.Load()), args=[], keywords=[])))
        step_args = copy.deepcopy(fn.args)
        step_args.args.append(ast.arg(arg="__cache", annotation=None))
        step_fn = ast.FunctionDef(name="_step", args=step_args, body=unpack + step_stmts + [ret], decorator_list=[], returns=None, type_comment=None)
        for f in (pre_fn, step_fn):
            for a in f.args.args:
                a.annotation = None          # drop TorchScript type annotations (names like __torch__ are not importable)
            f.returns = None
            if free_dead and len(f.body) > 8:
                f.body = _insert_frees(f)
        m = ast.Module(body=[pre_fn, step_fn], type_ignores=[])
        ast.fix_missing_locations(m)
        code = compile(m, f"<hoist:{ts.qualname}.{method}>", "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True)
        ns = {}
        exec(code, rt.globals, ns)
        self._pre, self._step = ns["_pre"], ns["_step"]
        self.argnames = argnames[1:]
        self.cache = None
        self._graph = None

    # ------------------------------------------------------------------------------------------------
    def _args(self, kw):
        return [kw[a] for a in self.argnames]

    @torch.no_grad()
    def precompute(self, **kw):
        self.cache = self._pre(self.root, *self._args(kw))
        self._graph = None
        return self

    @torch.no_grad()
    def compile(self, mode="default"):
        """`compiled` lever (fast tier): the per-step function through torch.compile / Inductor — the pointwise / layout glue between the step's
        GEMMs fused into Triton kernels. NOT bitwise (fused reductions reorder fp32 sums; numerics class: TF32 / Inductor lowering). The compiled
        function is what `step` / `step_graphed` then run (the CUDA graph captures the Inductor kernels). Compile cost lands on the first calls
        per crop (Dynamo + Inductor + Triton); a warm start reuses Inductor's on-disk caches (TORCHINDUCTOR_CACHE_DIR; the FX-graph and
        AOT-autograd caches are enabled here). Returns self."""
        import torch._dynamo, torch._inductor.config
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        torch._dynamo.config.capture_scalar_outputs = True          # the step reads one 0-d tensor's value (Tensor.item): captured, not a graph break (a break
                                                                    # would restart Dynamo's analysis of the whole traced step on the first call per crop)
        torch._inductor.config.fx_graph_cache = True
        try:
            import torch._functorch.config as _fc
            _fc.enable_autograd_cache = True
        except Exception:  # noqa: BLE001
            pass
        if not getattr(self, "compiled", None):
            self._step_eager = self._step
            self._step = torch.compile(self._step, mode=(None if mode == "default" else mode), fullgraph=False, dynamic=False)
            self.compiled = mode
            self.stats = dict(self.stats, compiled=mode)
        return self

    def uncompile(self):
        """Back to the eager per-step function (a compile that cannot engage on this machine steps aside by name in the wrapper)."""
        if getattr(self, "compiled", None):
            self._step = self._step_eager; self.compiled = None; self._graph = None
            self.stats = {k: v for k, v in self.stats.items() if k != "compiled"}
        return self

    def step(self, **kw):
        assert self.cache is not None, "call precompute() first"
        return self._step(self.root, *self._args(kw), self.cache)

    @torch.no_grad()
    def step_graphed(self, **kw):
        """CUDA-graph replay of the variant part. First call: 2 warm-up runs on a side stream + capture."""
        if self._graph is None:
            self._static_kw = dict(kw)
            for v in self.variant:
                self._static_kw[v] = kw[v].clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._step(self.root, *self._args(self._static_kw), self.cache)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._static_out = self._step(self.root, *self._args(self._static_kw), self.cache)
            self._graph = g
        for v in self.variant:
            self._static_kw[v].copy_(kw[v])
        self._graph.replay()
        o = self._static_out
        if torch.is_tensor(o):
            return o.clone()
        return type(o)(x.clone() if torch.is_tensor(x) else x for x in o)

    def cache_bytes(self):
        seen, tot = set(), 0
        for v in (self.cache or {}).values():
            if torch.is_tensor(v):
                s = v.untyped_storage()
                if s.data_ptr() not in seen:
                    seen.add(s.data_ptr()); tot += s.nbytes()
        return tot


# ====================================================================================================================================
# hoist2 — the VALUE-taint hoister: partitions the hoisted denoiser step by value dependence (a read through a metadata-only op does not
# taint), and snapshots each protected cached name right before its first per-step use (see _pre below).
#
# The base hoister above taints a statement as per-step when it reads ANY name derived from the two step inputs. In the traced denoiser
# many statements read only the METADATA of a variant tensor (`torch.size(x, d)`, `torch.new_empty(x, ...)`, `ops.prim.device(x)`): e.g. the
# length of the diffusion-sample axis is read off an `atom_noised_coords`-derived tensor and then used to `expand` step-invariant conditioning.
# That name-level taint drags into the step the whole atom-pair update block, both N=16 atom-pair LayerNorms, the 6 blocked pair-bias
# einsums + masked_fills, the AdaLN scale/shift and gate projections of the 6 atom-transformer blocks (their conditioning depends on neither
# coordinates nor sigma) and every CPU-side shape-arithmetic statement (NumToTensor / int / mul / floordiv).
#
# Here a read through a metadata-only op does not propagate taint. Shapes are static per (crop, n_samples) — the wrapper's key already
# re-hoists when the coordinate shape changes — so `precompute` runs the FULL traced forward once (on the first call's own inputs; every
# metadata read resolves to its real value) and caches what the per-step part needs. Same ops, same operands, same order inside every
# dependency chain: the per-step statements and their operands are the un-hoisted call's own => bitwise by construction where each op is
# run-to-run deterministic.
#
# Reordering safety (in-place ops):
#   * a per-step statement that mutates a name in place taints that name (and every later reader of it);
#   * a name mutated in place by a per-step statement and served from the cache is snapshotted (clone) inside the precompute run right
#     BEFORE the first per-step statement that touches it (so every precompute-side mutation the original order applies before that point is
#     in the snapshot), and cloned again at every step entry (the base's `protect` rule);
#   * a precompute-only statement with an in-place / out= target is kept precompute-only ONLY if no per-step statement located before it in
#     the original order value-reads that target or a name derived from it (else it is demoted to per-step: `stats["demoted"]`);
#   * traced bodies are single-assignment: a cached name stored twice is refused by name at build (AssertionError), never hoisted around;
#   * protected *inputs* are cloned at entry of both functions.
META_TORCH = {"size", "new_empty", "new_zeros", "new_ones", "empty_like", "zeros_like", "ones_like", "dim", "numel"}   # torch.<f>(x, ...): reads x's metadata only
META_PRIM = {"device", "dtype"}                                                                                      # ops.prim.<f>(x)
HOISTERS = ("base", "hoist2")                                                                                        # HoistedDiffusionWrapper(hoister=...)


SCALAR_CASTS = ("int", "float", "bool")


def scalar_only_uses(step_nodes, cache_names, protect=()):
    """{cached name: cast} for the cached names the per-step statements read ONLY as the sole argument of one and the same builtin cast
    (``int(x)`` / ``float(x)`` / ``bool(x)``) — never as a tensor operand, never mutated. `_pre` may store ``cast(x)`` for them: the step's
    ``cast(cast(x))`` is the same Python scalar, and a trace of the step (torch.compile / make_fx / export) meets no data-dependent tensor read."""
    want = set(cache_names) - set(protect)
    cast_of, spoiled = {}, set()
    for st in step_nodes:
        in_cast = set()
        for n in ast.walk(st):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in SCALAR_CASTS and len(n.args) == 1 and not n.keywords \
                    and isinstance(n.args[0], ast.Name) and n.args[0].id in want:
                nm = n.args[0].id
                in_cast.add(id(n.args[0]))
                if cast_of.setdefault(nm, n.func.id) != n.func.id:
                    spoiled.add(nm)                                  # read through two different casts: leave it a tensor
        for n in ast.walk(st):
            if isinstance(n, ast.Name) and n.id in want and isinstance(n.ctx, ast.Load) and id(n) not in in_cast:
                spoiled.add(n.id)                                    # read as a value somewhere else: leave it a tensor
            elif isinstance(n, ast.Name) and n.id in want and not isinstance(n.ctx, ast.Load):
                spoiled.add(n.id)                                    # stored / deleted in the step: leave it
    return {nm: c for nm, c in cast_of.items() if nm not in spoiled}


def value_reads(node):
    """(names read for their VALUE, names read only for metadata) in one statement."""
    meta_ids = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.args and isinstance(n.args[0], ast.Name):
            f = n.func
            if isinstance(f.value, ast.Name) and f.value.id == "torch" and f.attr in META_TORCH:
                meta_ids.add(id(n.args[0]))
            elif f.attr in META_PRIM and isinstance(f.value, ast.Attribute) and f.value.attr == "prim":
                meta_ids.add(id(n.args[0]))
    vals = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and id(n) not in meta_ids}
    metas = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and id(n) in meta_ids}
    return vals, metas - vals


def inplace_targets(st):
    """names mutated in place by a statement: first operand of torch.<op>_ / INPLACE_EXTRA calls, and `out=` keyword targets."""
    out = set()
    for n in ast.walk(st):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "torch" and not f.attr.startswith("__") \
                and (f.attr.endswith("_") or f.attr in INPLACE_EXTRA) and n.args and isinstance(n.args[0], ast.Name):
            out.add(n.args[0].id)
        for k in n.keywords:
            if k.arg == "out" and isinstance(k.value, ast.Name):
                out.add(k.value.id)
    return out


def partition_value(stmts, variant):
    """-> (is_var list, info dict). Value-taint pass + demotion of precompute-only in-place statements a preceding per-step statement could observe."""
    vr = [value_reads(st) for st in stmts]
    wr = [_names(st, (ast.Store, ast.Del)) for st in stmts]
    ip = [inplace_targets(st) for st in stmts]
    forced = set()
    while True:
        tainted, is_var = set(variant), []
        for k, st in enumerate(stmts):
            v = bool(vr[k][0] & tainted) or k in forced
            if v:
                tainted |= wr[k] | ip[k]
            is_var.append(v)
        demote = set()
        defined_at = {}
        for k in range(len(stmts)):
            for w in wr[k]:
                defined_at.setdefault(w, k)
        for p in range(len(stmts)):
            if is_var[p] or not ip[p]:
                continue
            for T in ip[p]:
                d = defined_at.get(T, -1)
                fam = {T}
                for k in range(max(d, 0), p):              # forward closure: names derived from T before p (conservative: any derivation counts as an alias)
                    if vr[k][0] & fam:
                        fam |= wr[k]
                if any(is_var[k] and (vr[k][0] & fam) for k in range(p)):
                    demote.add(p)
        if not (demote - forced):
            break
        forced |= demote
    meta_only = {}
    for k in range(len(stmts)):
        if not is_var[k]:
            for m in vr[k][1] & tainted:
                meta_only[m] = meta_only.get(m, 0) + 1
    return is_var, dict(demoted=sorted(forced), meta_only_reads=meta_only)


class HoistedForward2(HoistedForward):
    """The value-taint hoister; same surface as HoistedForward (precompute / step / step_graphed / cache_bytes; fields cache, _graph, _static_kw,
    _static_out), so the stack's wrapper and the kit's item-keyed adoption serve it unchanged. `stats` adds base_n_step / moved / demoted."""

    def __init__(self, root, method, variant=("atom_noised_coords", "noise_sigma"), free_dead=True):
        self.root, self.method, self.variant = root, method, tuple(variant)
        ts = object.__getattribute__(root, "_ts")
        rt = object.__getattribute__(root, "_rt")
        fn = ast.parse(ts.methods_src[method]).body[0]
        assert isinstance(fn, ast.FunctionDef), type(fn)
        argnames = [a.arg for a in fn.args.args]            # includes self
        body = list(fn.body)
        assert isinstance(body[-1], ast.Return), "traced method must end in a single return"
        ret, stmts = body[-1], body[:-1]
        is_var, info = partition_value(stmts, self.variant)
        t1, base_var = set(self.variant), []                # the base (name-taint) partition, for `stats` and the subset rule below
        for st in stmts:
            v = bool(_names(st, ast.Load) & t1)
            if v:
                t1 |= _names(st, (ast.Store, ast.Del))
            base_var.append(v)
        assert all(b or not v for b, v in zip(base_var, is_var)), "hoist2 may only move statements OUT of the base's per-step part"
        pre_idx = [k for k, v in enumerate(is_var) if not v]
        step_idx = [k for k, v in enumerate(is_var) if v]
        step_stmts = [stmts[k] for k in step_idx]
        pre_writes = set().union(*[_names(stmts[k], ast.Store) for k in pre_idx]) if pre_idx else set()
        step_reads = set().union(*[_names(st, ast.Load) for st in step_stmts], _names(ret, ast.Load))
        cache_names = sorted((step_reads & pre_writes) - set(argnames))
        n_stores = {}
        for st in stmts:
            for w in _names(st, ast.Store):
                n_stores[w] = n_stores.get(w, 0) + 1
        twice = sorted(n for n in cache_names if n_stores.get(n, 0) != 1)
        assert not twice, f"hoist2: cached name(s) stored more than once in the traced body (not single-assignment): {twice[:5]}"
        protect = set()
        for st in step_stmts:
            for a0 in inplace_targets(st):
                if a0 in cache_names or a0 in argnames:
                    protect.add(a0)
        first_touch = {}                                     # protected cached name -> index of the first PER-STEP statement that reads or mutates it
        for k in step_idx:
            for n in (_names(stmts[k], ast.Load) | inplace_targets(stmts[k])) & protect & set(cache_names):
                first_touch.setdefault(n, k)
        scalar_fold = scalar_only_uses(step_stmts + [ret], cache_names, protect)   # cached name -> "int" | "float" | "bool": read in the step ONLY as int(x) / float(x) / bool(x)
        self.stats = dict(hoister="hoist2", n_stmts=len(stmts), n_pre=len(pre_idx), n_step=len(step_stmts), n_cache=len(cache_names), protect=sorted(protect), n_scalar_folded=len(scalar_fold),
                          base_n_step=sum(base_var), moved=sum(base_var) - len(step_stmts), demoted=info["demoted"],
                          n_meta_only_names=len(info["meta_only_reads"]))

        def _clone(name_node):
            return ast.Call(func=ast.Attribute(value=name_node, attr="clone", ctx=ast.Load()), args=[], keywords=[])

        # ---- `_pre(self, <args>)`: the FULL traced body once (original order); a protected cached name is snapshotted right before its first per-step use
        prot_inputs = sorted(protect - set(cache_names))
        pre_body = [ast.Assign(targets=[ast.Name(id=n, ctx=ast.Store())], value=_clone(ast.Name(id=n, ctx=ast.Load()))) for n in prot_inputs]
        snap = {}
        snaps_at = {}
        for n, k in first_touch.items():
            snaps_at.setdefault(k, []).append(n)
        for k, st in enumerate(stmts):
            for n in sorted(snaps_at.get(k, ())):
                snap[n] = f"__snap_{n}"
                pre_body.append(ast.Assign(targets=[ast.Name(id=snap[n], ctx=ast.Store())], value=_clone(ast.Name(id=n, ctx=ast.Load()))))
            pre_body.append(copy.deepcopy(st))
        def _cached(n):                                      # the value _pre stores for a cached name: its snapshot if protected; its Python scalar if the step
            v = ast.Name(id=snap.get(n, n), ctx=ast.Load())      # reads it only through int()/float()/bool() (ts2eager keeps shape arithmetic on CPU 0-dim tensors:
            if n in scalar_fold:                                 # `int(t)` of a cached CPU scalar is the same int either way — folded here it is no tensor read in
                v = ast.Call(func=ast.Name(id=scalar_fold[n], ctx=ast.Load()), args=[v], keywords=[])   # the step, so a trace of the step sees no data-dependent scalar)
            return v
        pre_ret = ast.Return(value=ast.Dict(keys=[ast.Constant(n) for n in cache_names], values=[_cached(n) for n in cache_names]))
        pre_fn = ast.FunctionDef(name="_pre", args=copy.deepcopy(fn.args), body=pre_body + [pre_ret], decorator_list=[], returns=None, type_comment=None)
        # ---- `_step(self, <args>, __cache)`: as the base builds it
        unpack = []
        for n in cache_names:
            val = ast.Subscript(value=ast.Name(id="__cache", ctx=ast.Load()), slice=ast.Constant(n), ctx=ast.Load())
            if n in protect:
                val = _clone(val)
            unpack.append(ast.Assign(targets=[ast.Name(id=n, ctx=ast.Store())], value=val))
        for n in prot_inputs:
            unpack.append(ast.Assign(targets=[ast.Name(id=n, ctx=ast.Store())], value=_clone(ast.Name(id=n, ctx=ast.Load()))))
        step_args = copy.deepcopy(fn.args)
        step_args.args.append(ast.arg(arg="__cache", annotation=None))
        step_fn = ast.FunctionDef(name="_step", args=step_args, body=unpack + [copy.deepcopy(s) for s in step_stmts] + [copy.deepcopy(ret)],
                                  decorator_list=[], returns=None, type_comment=None)
        for f in (pre_fn, step_fn):
            for a in f.args.args:
                a.annotation = None          # drop TorchScript type annotations (names like __torch__ are not importable)
            f.returns = None
            if free_dead and len(f.body) > 8:
                f.body = _insert_frees(f)
        m = ast.Module(body=[pre_fn, step_fn], type_ignores=[])
        ast.fix_missing_locations(m)
        self.src = ast.unparse(m)                                                # the generated _pre/_step source, kept for diagnostics
        code = compile(m, f"<hoist2:{ts.qualname}.{method}>", "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True)
        ns = {}
        exec(code, rt.globals, ns)
        self._pre, self._step = ns["_pre"], ns["_step"]
        self.argnames = argnames[1:]
        self.cache = None
        self._graph = None


def make(hoister, root, method, **kw):
    """The hoister factory: 'base' -> HoistedForward (name taint), 'hoist2' -> HoistedForward2 (value taint)."""
    if hoister not in HOISTERS:
        raise ValueError(f"hoister {hoister!r} is not one of {HOISTERS}")
    return (HoistedForward2 if hoister == "hoist2" else HoistedForward)(root, method, **kw)

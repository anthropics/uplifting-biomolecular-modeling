# Derived from the exported TorchScript archives of Chai-1 (chai_lab 0.6.1, Copyright 2024 Chai Discovery, Inc., Apache-2.0).
"""ts2eager — turn the *printed TorchScript code tree* of an exported Chai-1 component (archive `code/__torch__/**.py`) into plain
eager-PyTorch callables with exactly the traced op order and dtype casts.

    tree = TSCodeTree("code_trees/trunk")                      # parsed class files
    root = build_module(tree, "__torch__.chai.model.af3.trunk.Trunk", state_dict, device)   # eager object tree (params = plain tensors)
    s, z = root.forward_256(**kwargs)                          # methods compiled lazily from the TorchScript source

No TorchScript executor is involved: every `torch.X(...)` line of the trace is executed by the Python interpreter against eager ATen ops
(`TorchShim` maps the schema-style names TorchScript prints — torch.to(x, 6), torch.slice, torch.linear, torch.size ... — onto their eager
equivalents; integer ScalarType codes are converted to torch.dtype; prim::NumToTensor shape arithmetic stays on CPU int64 scalars)."""
import os, re, glob, types, ast, zipfile, typing, __future__
import torch
import torch.nn.functional as F

DTYPES = {0: torch.uint8, 1: torch.int8, 2: torch.int16, 3: torch.int32, 4: torch.int64, 5: torch.float16, 6: torch.float32,
          7: torch.float64, 11: torch.bool, 15: torch.bfloat16}


def _dt(d):
    return DTYPES[d] if isinstance(d, int) and not isinstance(d, bool) else d


class TorchShim:
    """Object bound to the name `torch` inside transpiled method bodies."""
    Tensor = torch.Tensor

    def __getattr__(self, n):          # everything not shimmed below is the eager torch function of the same name
        return getattr(torch, n)

    @staticmethod
    def to(x, *a, **k):
        if len(a) >= 1 and isinstance(a[0], int):
            return x.to(_dt(a[0]), non_blocking=bool(a[1]) if len(a) > 1 else False, copy=bool(a[2]) if len(a) > 2 else False)
        if len(a) >= 1 and torch.is_tensor(a[0]):
            return x.to(a[0])
        if "dtype" in k:
            return x.to(_dt(k["dtype"]))
        raise NotImplementedError(f"torch.to shim: args {[type(v) for v in a]} {k}")

    @staticmethod
    def size(x, d=None):
        return list(x.size()) if d is None else x.size(d)

    @staticmethod
    def linear(x, w, b=None):
        return F.linear(x, w, b)

    @staticmethod
    def mul_(x, y):
        return x.mul_(y)

    @staticmethod
    def masked_fill_(x, m, v):
        return x.masked_fill_(m, v)

    @staticmethod
    def silu(x):
        return F.silu(x)

    @staticmethod
    def __and__(a, b):
        return a.__and__(b)

    @staticmethod
    def slice(x, dim=0, start=None, end=None, step=1):
        return torch.ops.aten.slice.Tensor(x, dim, start, end, step)

    @staticmethod
    def contiguous(x, memory_format=0):
        return x.contiguous()

    @staticmethod
    def new_empty(x, size, dtype=None, layout=None, device=None, pin_memory=False):
        return x.new_empty(size, dtype=_dt(dtype)) if dtype is not None else x.new_empty(size)

    @staticmethod
    def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
        return F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale=scale)

    @staticmethod
    def softmax(x, dim, dtype=None):
        return torch.softmax(x, dim, dtype=_dt(dtype))

    @staticmethod
    def sum(x, dim=None, keepdim=False, dtype=None):
        if dim is None:
            return torch.sum(x, dtype=_dt(dtype))
        return torch.sum(x, dim, keepdim, dtype=_dt(dtype))

    @staticmethod
    def cdist(x1, x2, p=2.0, compute_mode=None):
        return torch._VF.cdist(x1, x2, p, compute_mode)      # TorchScript passes the int enum; the python wrapper wants a string

    @staticmethod
    def index(x, indices):
        return torch.ops.aten.index.Tensor(x, indices)

    @staticmethod
    def expand(x, size, implicit=False):
        return x.expand(size)

    @staticmethod
    def new_zeros(x, size, dtype=None, layout=None, device=None, pin_memory=False):
        return x.new_zeros(size, dtype=_dt(dtype)) if dtype is not None else x.new_zeros(size)

    @staticmethod
    def new_ones(x, size, dtype=None, layout=None, device=None, pin_memory=False):
        return x.new_ones(size, dtype=_dt(dtype)) if dtype is not None else x.new_ones(size)

    @staticmethod
    def add_(x, y, alpha=1):
        return x.add_(y, alpha=alpha)

    @staticmethod
    def one_hot(x, num_classes=-1):
        return F.one_hot(x, num_classes)

    @staticmethod
    def arange(*a, dtype=None, layout=None, device=None, pin_memory=False, out=None):
        if out is not None:
            return torch.arange(*a, out=out)
        kw = {}
        if dtype is not None:
            kw["dtype"] = _dt(dtype)
        if device is not None:
            kw["device"] = device
        return torch.arange(*a, **kw)

    @staticmethod
    def zeros(size, dtype=None, layout=None, device=None, pin_memory=False):
        return torch.zeros(size, dtype=_dt(dtype), device=device)

    @staticmethod
    def ones(size, dtype=None, layout=None, device=None, pin_memory=False):
        return torch.ones(size, dtype=_dt(dtype), device=device)

    @staticmethod
    def full(size, fill_value, dtype=None, layout=None, device=None, pin_memory=False):
        return torch.full(size, fill_value, dtype=_dt(dtype), device=device)

    @staticmethod
    def cumsum(x, dim, dtype=None):
        return torch.cumsum(x, dim, dtype=_dt(dtype))

    @staticmethod
    def mean(x, dim=None, keepdim=False, dtype=None):
        if dim is None:
            return torch.mean(x, dtype=_dt(dtype))
        return torch.mean(x, dim, keepdim, dtype=_dt(dtype))

    @staticmethod
    def feature_dropout(x, p, train):
        if p == 0.0:
            return x            # aten::feature_dropout returns `input` itself for p == 0 (no RNG draw); keep it allocation-free
        return torch.feature_dropout(x, p, train)


class _Prim:
    @staticmethod
    def NumToTensor(n):
        return torch.tensor(n)    # CPU 0-dim (int64 for ints) — shape arithmetic only

    @staticmethod
    def device(x):
        return x.device

    @staticmethod
    def dtype(x):
        return x.dtype


class OpsShim:
    prim = _Prim()

    def __getattr__(self, n):
        return getattr(torch.ops, n)


class _Constants:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


# ------------------------------------------------------------------------------------------------------------------------------
class TSClass:
    __slots__ = ("qualname", "name", "params", "buffers", "attrs", "methods_src", "file")

    def __init__(self, qualname, name, file):
        self.qualname, self.name, self.file = qualname, name, file
        self.params, self.buffers, self.attrs, self.methods_src = [], [], {}, {}


class TSCodeTree:
    """Parses every class in a `code/__torch__` tree (as extracted from the .pt zip) into TSClass records."""
    _re_class = re.compile(r"^class (\w+)\((\w+)\):\s*$")
    _re_attr = re.compile(r"^  (\w+) : (.+)$")
    _re_ann = re.compile(r'^  __annotations__\["([^"]+)"\] = (.+)$')
    _re_list = re.compile(r"^  __(parameters|buffers)__ = \[(.*)\]\s*$")
    _re_def = re.compile(r"^  def (\w+)\(")

    def __init__(self, root, sources=None):
        """``sources``: {path relative to the code tree: text} -- the class files as text (``archive_sources``); then no file under ``root`` is read."""
        self.root = root
        self.classes = {}
        base = os.path.join(root, "__torch__")
        if sources is not None:
            for rel in sorted(sources):
                self._parse_file(rel, rel[:-3].replace("/", "."), text=sources[rel])   # the member's name is kept as the record's file; it is never opened
            return
        for f in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
            rel = os.path.relpath(f, root)[:-3].replace(os.sep, ".")      # __torch__.torch.nn.modules.linear.___torch_mangle_476
            self._parse_file(f, rel)

    def _parse_file(self, path, modprefix, text=None):
        lines = (open(path).read() if text is None else text).split("\n")
        i, n = 0, len(lines)
        cur = None
        while i < n:
            l = lines[i]
            m = self._re_class.match(l)
            if m:
                cur = TSClass(f"{modprefix}.{m.group(1)}", m.group(1), path)
                self.classes[cur.qualname] = cur
                i += 1
                continue
            if cur is None:
                i += 1
                continue
            m = self._re_list.match(l)
            if m:
                names = [x.strip().strip('"') for x in m.group(2).split(",") if x.strip()]
                (cur.params if m.group(1) == "parameters" else cur.buffers).extend(names)
                i += 1
                continue
            m = self._re_def.match(l)
            if m:
                j = i + 1
                while j < n and not self._re_def.match(lines[j]) and not self._re_class.match(lines[j]):
                    j += 1
                cur.methods_src[m.group(1)] = "\n".join(x[2:] if x.startswith("  ") else x for x in lines[i:j])
                i = j
                continue
            m = self._re_ann.match(l) or self._re_attr.match(l)
            if m:
                cur.attrs[m.group(1)] = m.group(2).strip()
            i += 1


class EagerTSModule:
    """One node of the eager object tree. Submodules / parameters are plain attributes; TorchScript methods compile lazily."""

    def __init__(self, tscls, path, runtime):
        object.__setattr__(self, "_ts", tscls)
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_rt", runtime)
        object.__setattr__(self, "training", False)

    def __getattr__(self, name):        # only called when normal lookup fails -> lazily compile a TorchScript method
        ts = object.__getattribute__(self, "_ts")
        if name in ts.methods_src:
            fn = object.__getattribute__(self, "_rt").compile(ts, name)
            bound = types.MethodType(fn, self)
            object.__setattr__(self, name, bound)
            return bound
        raise AttributeError(f"{ts.qualname} ({object.__getattribute__(self, '_path')}) has no attribute {name!r}")

    def named_parameters(self):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                yield (f"{self._path}.{k}" if self._path else k), v
            elif isinstance(v, EagerTSModule):
                yield from v.named_parameters()

    def modules(self):
        yield self
        for v in self.__dict__.values():
            if isinstance(v, EagerTSModule):
                yield from v.modules()

    def __repr__(self):
        return f"EagerTSModule({self._ts.name} @ '{self._path}')"


class Runtime:
    def __init__(self, tree, constants, verbose=False):
        self.tree, self.verbose = tree, verbose
        self.globals = {"torch": TorchShim(), "ops": OpsShim(), "CONSTANTS": _Constants(constants), "annotate": lambda t, v: v,
                        "unchecked_cast": lambda t, v: v, "uninitialized": lambda t: None, "Tensor": torch.Tensor,
                        "List": typing.List, "Dict": typing.Dict, "Optional": typing.Optional, "Tuple": typing.Tuple, "__torch__": None,
                        "__builtins__": __builtins__, "number": float, "Device": torch.device, "NoneType": type(None), "Final": typing.Final}
        self._cache = {}

    def compile(self, tscls, meth):
        key = (tscls.qualname, meth)
        if key not in self._cache:
            src = tscls.methods_src[meth]
            tree = ast.parse(src, f"<ts:{tscls.qualname}.{meth}>")
            fn = tree.body[0]
            if isinstance(fn, ast.FunctionDef) and len(fn.body) > 8:
                fn.body = _insert_frees(fn)          # free every SSA value right after its last use (the TS interpreter does the same)
            ast.fix_missing_locations(tree)
            code = compile(tree, f"<ts:{tscls.qualname}.{meth}>", "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True)
            ns = {}
            exec(code, self.globals, ns)
            self._cache[key] = ns[meth]
            if self.verbose:
                print(f"[ts2eager] compiled {tscls.qualname}.{meth} ({src.count(chr(10))} lines)")
        return self._cache[key]


def _insert_frees(fn):
    """Straight-line traced bodies keep every intermediate alive as a Python local until return; insert `del name` after the
    statement that last reads it. (Pure liveness bookkeeping — op order and values are untouched.)"""
    args = {a.arg for a in fn.args.args} - {"self"}
    last_use, stored = {}, set()
    for i, st in enumerate(fn.body):
        for node in ast.walk(st):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    last_use[node.id] = i
                elif isinstance(node.ctx, ast.Store):
                    stored.add(node.id)
                    last_use.setdefault(node.id, i)
                    if last_use[node.id] < i:
                        last_use[node.id] = i
    ret_names = set()
    if fn.body and isinstance(fn.body[-1], ast.Return):
        ret_names = {n.id for n in ast.walk(fn.body[-1]) if isinstance(n, ast.Name)}
    by_stmt = {}
    for name, i in last_use.items():
        if name in ret_names or name == "self" or (name not in stored and name not in args):
            continue
        by_stmt.setdefault(i, []).append(name)
    body = []
    for i, st in enumerate(fn.body):
        body.append(st)
        if i in by_stmt and not isinstance(st, ast.Return):
            body.append(ast.Delete(targets=[ast.Name(id=n, ctx=ast.Del()) for n in sorted(by_stmt[i])]))
    return body


def build_module(tree, root_qualname, state_dict, device=None, constants=None, verbose=False, param_dtype=None):
    """Instantiate the eager object tree for `root_qualname`, filling parameters/buffers from `state_dict` (keys = dotted instance paths)."""
    rt = Runtime(tree, constants or {}, verbose=verbose)
    missing, used = [], set()

    def make(qual, path):
        tscls = tree.classes[qual]
        node = EagerTSModule(tscls, path, rt)
        for an, at in tscls.attrs.items():
            if at.startswith("__torch__."):
                object.__setattr__(node, an, make(at, f"{path}.{an}" if path else an))
            elif an in tscls.params or an in tscls.buffers or at in ("Tensor", "Optional[Tensor]"):
                key = f"{path}.{an}" if path else an
                if key in state_dict:
                    t = state_dict[key]
                    if device is not None:
                        t = t.to(device)
                    if param_dtype is not None and t.is_floating_point():
                        t = t.to(param_dtype)
                    object.__setattr__(node, an, t)
                    used.add(key)
                elif at == "Optional[Tensor]":
                    object.__setattr__(node, an, None)
                else:
                    missing.append(key)
        return node

    root = make(root_qualname, "")
    unused = [k for k in state_dict if k not in used]
    if missing or unused:
        print(f"[ts2eager] WARNING missing params {missing[:5]} ({len(missing)}) unused state_dict keys {unused[:5]} ({len(unused)})")
    root._rt_info = dict(n_params=len(used), missing=missing, unused=unused)
    return root


def load_constants_from_pt(pt_path):
    """Read the tensor constants (archive/constants.pkl + constants/<key> blobs) of a TorchScript .pt -> {"c0": tensor, ...}.
    Uses a real unpickler with persistent_load (storage dtype per constant), so mixed int64/float64/... constants are decoded correctly."""
    import zipfile, pickle, io
    st_dtype = {"LongStorage": torch.int64, "FloatStorage": torch.float32, "DoubleStorage": torch.float64, "IntStorage": torch.int32,
                "HalfStorage": torch.float16, "BFloat16Storage": torch.bfloat16, "BoolStorage": torch.bool, "ByteStorage": torch.uint8,
                "CharStorage": torch.int8, "ShortStorage": torch.int16}
    out = {}
    with zipfile.ZipFile(pt_path) as z:
        names = z.namelist()
        pkl = [n for n in names if n.endswith("/constants.pkl") or n == "constants.pkl"]
        if not pkl:
            return out
        prefix = pkl[0][: -len("constants.pkl")]

        class _U(pickle.Unpickler):
            def find_class(self, mod, name):
                if name == "_rebuild_tensor_v2":
                    return torch._utils._rebuild_tensor_v2
                if mod.startswith("torch") and name in st_dtype:
                    return name
                return super().find_class(mod, name)

            def persistent_load(self, pid):
                kind, typename, key, location, numel = pid[0], pid[1], pid[2], pid[3], pid[4]
                typename = typename if isinstance(typename, str) else typename.__name__
                dtype = st_dtype[typename]
                raw = z.read(f"{prefix}constants/{key}")
                untyped = torch.UntypedStorage.from_buffer(bytearray(raw), dtype=torch.uint8)
                return torch.storage.TypedStorage(wrap_storage=untyped, dtype=dtype, _internal=True)

        consts = _U(io.BytesIO(z.read(pkl[0]))).load()
    for i, t in enumerate(consts):
        t = t.clone()
        out[f"c{i}"] = t[0] if (t.dim() == 1 and t.numel() == 1) else (t if t.dim() else t)
    return out

def extract_code_tree(pt_path, dest):
    """Extract archive `code/` of a TorchScript .pt into dest/ (dest/__torch__/...). Returns dest."""
    with zipfile.ZipFile(pt_path) as z:
        for n in z.namelist():
            if "/code/" in n:
                rel = n.split("/code/", 1)[1]
                p = os.path.join(dest, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f:
                    f.write(z.read(n))
    return dest


def archive_sources(pt_path):
    """{path relative to the archive's `code/`: text} of every class file in a TorchScript .pt -- the text extract_code_tree would write,
    read the way TSCodeTree reads a file (universal newlines)."""
    out = {}
    with zipfile.ZipFile(pt_path) as z:
        for n in z.namelist():
            if "/code/" in n and n.endswith(".py"):
                out[n.split("/code/", 1)[1]] = z.read(n).decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return out


def load_eager_component(pt_path, root_qualname=None, device="cuda", code_dir=None, state_dict=None, verbose=False):
    """One-stop: code tree + constants + weights of an exported Chai-1 component -> eager root object.
    If state_dict is None the weights are read with torch.jit.load(pt_path, map_location='cpu').state_dict()."""
    code_dir = code_dir or (pt_path + ".code")
    # the class files are parsed from the archive's own members: a tree already at code_dir (a shared or temp path) is neither read nor
    # written here, so no file found on disk is ever compiled; extract_code_tree stays for a caller who wants the files
    tree = TSCodeTree(code_dir, sources=archive_sources(pt_path))
    if root_qualname is None:
        # the root class is the only one never referenced as an attribute type
        referenced = {t for c in tree.classes.values() for t in c.attrs.values()}
        roots = [q for q in tree.classes if q not in referenced]
        assert len(roots) == 1, roots
        root_qualname = roots[0]
    if state_dict is None:
        state_dict = torch.jit.load(pt_path, map_location="cpu").state_dict()
    consts = load_constants_from_pt(pt_path)
    return build_module(tree, root_qualname, state_dict, device=device, constants=consts, verbose=verbose)

"""The `compiled` lever's ahead-of-time route: the hoisted per-step function of one crop exported once (torch.export) and compiled by AOTInductor into a
package on disk, so that a later process LOADS the step's Inductor kernels instead of compiling them through Dynamo at its first fold of
that crop. Same lowering as the Dynamo route (Inductor, the same Triton kernels, TF32 as the row sets it): the fast tier's tolerance
class, not bitwise to the eager statements; per-step outputs sit where the Dynamo-compiled step's sit.

Packages live under ``$MODEL_OPT_JIT_ROOT/<key>/aoti/`` (``chai1_opt.jit``'s key: torch / CUDA / card word) and are named by crop, number of
diffusion samples, the step's lever word and a digest of the generated step source + versions: a package that does not match the running code is
not found (the Dynamo route serves, by name). A package holds the kernels and the C++ launcher only: the denoiser's weights are
bound at load from the live module (``load_constants``, user managed — no copy, no second set on the device); the traced program's small literal
constants ride in a side file. Building needs the C++ toolchain + CUDA development headers of the kit's image (``CUDA_HOME``) and the target GPU
present (Inductor tunes and compiles the kernels on it): ``python -m chai1_opt.warm_aoti`` builds a card's set.

Surface: ``package_dir`` ``step_digest`` ``package_name`` ``find`` ``listing`` ``build`` ``Package`` ``attach`` ``synthetic_inputs`` ``report``
(+ ``cpu_isa``: a package whose launcher was built for CPU instructions this host lacks is refused by name at load, ``cpu_isa_mismatch``)."""
from __future__ import annotations

import glob, hashlib, json, os, re, sys, time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from . import cpu_isa                                                              # the package launcher's build ISA vs this host's (stdlib only)

NAME, SUBDIR, SCHEMA = "aoti", "aoti", 1
STATS: Dict[str, Any] = {"attached": [], "asked": [], "loaded": [], "absent": [], "mismatch": [], "built": [], "load_s": [], "errors": [], "aside": [], "aside_reason": None, "realigned": [], "refused": [], "isa": []}
SURFACE = ("package_dir", "step_digest", "package_name", "find", "listing", "build", "Package", "attach", "card_aside_reason", "step_aside", "synthetic_inputs", "report")


class Refused(RuntimeError):
    """A package refused BY NAME (``word``): ``no_layout_meta`` — built before input layouts were recorded (its generated code assumes build-time
    strides this process cannot check); ``layout_unproducible:<input>`` — an input whose recorded layout this process cannot hand over;
    ``cpu_isa_mismatch`` — the package's launcher was compiled (-march=native) for CPU instructions this host lacks (cpu_isa.py: the archive's
    recorded AOTI_CPU_ISA vs the host's word; loading it would end the process with SIGILL). The step is then served by the EAGER hoisted step
    on every card (no per-process build in any mode; EXIT compiled=stepped_aside:<word>)."""
    def __init__(self, package: str, word: str):
        super().__init__(f"{package}: {word}"); self.package, self.word = package, word


class Unusable(RuntimeError):
    """A package that cannot serve this item (inputs' structure or the traced constants differ, a constant without a source): the caller steps aside by name."""


# ---------------------------------------------------------------------------------------------------------------- naming
def package_dir(environ=None, create=False) -> Optional[str]:
    """``$MODEL_OPT_JIT_ROOT/<key>/aoti`` (the kit's one JIT root, chai1_opt.jit.cache_key's key) or None when the root is unset."""
    env = os.environ if environ is None else environ
    root = env.get("MODEL_OPT_JIT_ROOT")
    if not root: return None
    try:
        from chai1_opt import jit as J
        key = env.get(J.ENV_KEY) or J.cache_key(torch, env)
    except Exception:  # noqa: BLE001 — the add-on used without the kit: the same words spelled here
        cc = "gpu"
        key = f"torch{str(torch.__version__).split('+')[0]}-cu{(torch.version.cuda or 'none').replace('.', '')}-{env.get('MODEL_OPT_TARGET_GPU', cc)}"
    d = os.path.join(root, key, SUBDIR)
    if create: os.makedirs(d, exist_ok=True)
    return d


ASIDE_ISA = cpu_isa.WORD                      # "cpu_isa_mismatch": a package whose launcher this host's CPU cannot run (cpu_isa.py) — refused by name at load, the eager hoisted step serves
ASIDE_CC80 = "no_aoti_packages_cc80"          # the compiled step's step-aside word on cc 8.0 (A100): no package for the crop → the EAGER hoisted step, never a
                                              # per-process Dynamo compile (its cost per crop per process exceeds what it returns there)
_CC80_WORDS = ("a100", "sm80", "cc80", "cc8.0", "8.0")


def card_aside_reason(environ=None, cc=None) -> Optional[str]:
    """The reason word under which the compiled step steps aside on this card when no ahead-of-time package serves a crop — ``ASIDE_CC80`` on
    cc 8.0, None elsewhere (cc 9.0 keeps the Dynamo route as its fallback). The card is the device's compute
    capability when CUDA is ALREADY initialised (never initialised here: the allocator lever owns the first CUDA touch), else the configuration's
    ``MODEL_OPT_TARGET_GPU`` word (configs/<card>.env) — the word the ACTIVE line's token reads before any model exists."""
    if cc is None:
        try:
            if torch.cuda.is_initialized(): cc = tuple(torch.cuda.get_device_capability())
        except Exception:  # noqa: BLE001
            cc = None
    if cc is not None:
        return ASIDE_CC80 if tuple(cc)[:2] == (8, 0) else None
    env = os.environ if environ is None else environ
    w = str(env.get("MODEL_OPT_TARGET_GPU") or "").strip().lower()
    return ASIDE_CC80 if w and any(w.startswith(x) for x in _CC80_WORDS) else None


def step_aside(hf, crop: int, reason: str, *, wrapper=None, events=None, n_samples=None, where=None) -> None:
    """Record and say (once per crop) that the compiled step steps aside on this card for `crop`: the eager hoisted step serves it."""
    key = (crop, n_samples)
    if key in STATS["aside"]: return
    STATS["aside"].append(key); STATS["aside_reason"] = reason
    if wrapper is not None and not getattr(wrapper, "compile_aside", None): wrapper.compile_aside = reason
    if events is not None: events.append((crop, f"compile_aside:{reason}", 0.0))
    hf.compiled = None; hf.stats = dict(getattr(hf, "stats", {}) or {}, compiled=f"aside:{reason}")
    print(f"[chai1-fastln] compile steps aside for crop {crop} on this card ({reason}): no ahead-of-time package"
          + (f" under {where}" if where else " directory") + " and a per-process compile costs more than it returns here — the eager hoisted step serves it "
          "(python -m chai1_opt.warm_aoti builds the package that switches it back on)", file=sys.stderr, flush=True)


def _versions() -> Dict[str, str]:
    """What shapes a package's lowering besides the step source: torch (the launcher's ABI, Inductor's kernels) and this module's package format.
    The kit's own versions are NOT in it: a release that leaves the generated step unchanged keeps its packages (a changed step source, input plan
    or op signature is caught by the digest, the per-item constants check or the loader, each by name)."""
    return {"torch": str(torch.__version__), "schema": str(SCHEMA)}


def step_digest(hf) -> str:
    """12 hex digits over the generated step source of this hoisted forward + _versions(): a change to the transpiled module or the hoister's
    output renames the package (an old one is then not found; nothing stale is ever loaded)."""
    h = hashlib.sha256()
    h.update((getattr(hf, "src", None) or repr(sorted(getattr(hf, "cache_names", [])))).encode())
    h.update(json.dumps(_versions(), sort_keys=True).encode()); h.update(getattr(hf, "method", "").encode())
    return h.hexdigest()[:12]


def crop_of(hf) -> int:
    m = re.search(r"(\d+)$", getattr(hf, "method", "")); return int(m.group(1)) if m else -1


def package_name(crop: int, n_samples: int, levers_word: str, digest: str) -> str:
    return f"dstep_c{crop}_s{n_samples}_{levers_word}_{digest}.pt2"


def find(dirpath: Optional[str], crop: int, n_samples: Optional[int], levers_word: str, digest: str) -> List[str]:
    """The package files for (crop, levers, digest) — one per n_samples present when n_samples is None."""
    if not dirpath or not os.path.isdir(dirpath): return []
    pat = package_name(crop, n_samples if n_samples is not None else "*", levers_word, digest)   # type: ignore[arg-type]
    return sorted(p for p in glob.glob(os.path.join(dirpath, pat)) if os.path.exists(p[:-4] + ".meta.json"))


def listing(dirpath: Optional[str]) -> List[str]:
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(dirpath, "dstep_*.pt2"))) if dirpath and os.path.isdir(dirpath) else []


def levers_word_of(wrapper=None, *, hoister="hoist2", dit_attn=False) -> str:
    """``<hoister>[+dit_attn]`` — the levers that change the step's graph as it runs in THIS process (a dit_attn that stepped aside is absent)."""
    if wrapper is not None:
        hoister = getattr(wrapper, "hoister_now", lambda: getattr(wrapper, "hoister", hoister))()
        r = getattr(wrapper, "dit_attn", None)
        dit_attn = bool(r is not None and not getattr(r, "aside", None))
    return hoister + ("+dit_attn" if dit_attn else "")


# ---------------------------------------------------------------------------------------------------------------- flattening
def _is_input(x) -> bool:
    """Device tensors and CPU tensors with elements ride as INPUTS of the package; a cached 0-d CPU tensor (the traced program's shape arithmetic)
    is a per-crop constant like the ints and lists beside it."""
    return torch.is_tensor(x) and not (x.device.type == "cpu" and x.dim() == 0)


def _plan(args: Sequence[Any], cache: Dict[str, Any]):
    """(input getters, constant leaves rendered for the digest, treespec) of one (args, cache) pair — tensors nested in cached lists included."""
    import torch.utils._pytree as P
    leaves, spec = P.tree_flatten((list(args), dict(cache)))
    kinds = ["i" if _is_input(v) else "c" for v in leaves]
    return leaves, kinds, spec


def _leaf_names(args: Sequence[Any], cache: Dict[str, Any], argnames: Sequence[str] = ()) -> List[str]:
    """One name per flattened leaf of (args, cache), in _plan's order: ``arg:<name>`` / ``cache:<key>`` (+ ``[i]`` / ``.k`` for nested leaves)."""
    import torch.utils._pytree as P
    out = []
    paths, _ = P.tree_flatten_with_path((list(args), dict(cache)))
    for path, _leaf in paths:
        top, rest = path[0], path[1:]
        idx = getattr(top, "idx", getattr(top, "index", 0))
        if idx == 0:                                                           # args[i]
            i = getattr(rest[0], "idx", getattr(rest[0], "index", 0)) if rest else 0
            s = f"arg:{argnames[i] if i < len(argnames) else i}"; rest = rest[1:]
        else:
            k = getattr(rest[0], "key", None) if rest else None
            s = f"cache:{k}"; rest = rest[1:]
        for e in rest:
            s += f"[{getattr(e, 'idx', getattr(e, 'index', getattr(e, 'key', '?')))}]"
        out.append(s)
    return out


ALIGN = 16                                                                     # bytes: Inductor compiles an input it saw aligned with aligned loads; AOTInductor's launcher copies an
                                                                                # input that arrives unaligned at EVERY call and says so (torch's W… "Input N was compiled as 16-bytes aligned")


def _aligned(t) -> bool:
    return (not torch.is_tensor(t)) or t.numel() == 0 or t.data_ptr() % ALIGN == 0


def _unaligned_example(t):
    """A same-valued view of `t` whose data pointer is NOT 16-byte aligned: an input built from it is compiled without the alignment assumption
    (the launcher then takes the live tensor as it comes — no per-call copy, no warning)."""
    flat = torch.empty(t.numel() + ALIGN, dtype=t.dtype, device=t.device)
    off = 1 if t.element_size() < ALIGN else 0
    v = flat[off:off + t.numel()].view(t.shape) if t.is_contiguous() else torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
    v.copy_(t); return v


def _const_render(v):
    if isinstance(v, (bool, int, float, str)) or v is None: return v
    if torch.is_tensor(v): return {"t0": v.tolist(), "dtype": str(v.dtype)}
    p = getattr(v, "_path", None)
    return {"obj": type(v).__name__, "path": p} if p is not None or not isinstance(v, (list, tuple, dict)) else repr(v)[:80]


def _consts_digest(leaves, kinds, spec) -> str:
    body = {"spec": repr(spec), "consts": [_const_render(v) for v, k in zip(leaves, kinds) if k == "c"],
            "shapes": [[list(v.shape), str(v.dtype), v.device.type] for v, k in zip(leaves, kinds) if k == "i"]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------- build
def build(hf, kw: Dict[str, Any], dirpath: str, *, n_samples: Optional[int] = None, levers_word: str = "hoist2", weights: str = "user", log=print,
          unaligned: Sequence[str] = ()) -> Dict[str, Any]:
    """Export + compile the step of `hf` (precomputed on `kw` here) into ``dirpath``: <name>.pt2 (kernels + launcher), <name>.meta.json (inputs' structure,
    constants' sources, versions, timings), <name>.consts.pt (small literal constants). Returns the meta dict. Raises on failure (the caller names it)."""
    import torch._dynamo.config as _dc, torch._inductor
    from torch.fx.experimental.proxy_tensor import make_fx
    _dc.capture_scalar_outputs = True
    t0 = time.perf_counter()
    with torch.no_grad(): hf.precompute(**kw)
    args = hf._args(kw); crop = crop_of(hf); ns = int(kw["atom_noised_coords"].shape[1]) if n_samples is None else n_samples
    leaves, kinds, spec = _plan(args, hf.cache); names = _leaf_names(args, hf.cache, list(getattr(hf, "argnames", []) or []))
    in_names = [n for n, k in zip(names, kinds) if k == "i"]
    want_unaligned = {n for n in in_names if any(n == u or n.split(":", 1)[-1] == u for u in (unaligned or ()))}   # inputs compiled WITHOUT the 16-byte assumption (by name)
    inputs = tuple((_unaligned_example(v) if n in want_unaligned else v) for v, n in zip((v for v, k in zip(leaves, kinds) if k == "i"), in_names))
    consts = [v for v, k in zip(leaves, kinds) if k == "c"]; cdig = _consts_digest(leaves, kinds, spec); digest = step_digest(hf)
    name = package_name(crop, ns, levers_word, digest)[:-4]
    step_fn, root = getattr(hf, "_step_eager", None) or hf._step, hf.root
    import torch.utils._pytree as P
    def tensors_only(*flat_in):
        it = iter(flat_in); ci = iter(consts)
        full = P.tree_unflatten([next(it) if k == "i" else next(ci) for k in kinds], spec)
        return step_fn(root, *full[0], full[1])
    class _Step(torch.nn.Module):
        def forward(self, *flat_in): return tensors_only(*flat_in)
    route, ep = "export", None
    try:
        with torch.no_grad(): ep = torch.export.export(_Step().eval(), inputs, strict=False)
    except Exception as e:  # noqa: BLE001 — a size read from a cached 0-d tensor (crop 256's reshape): pre-trace once for real, sizes become literals
        route = f"make_fx+export ({type(e).__name__})"
        with torch.no_grad(): gm = make_fx(tensors_only, tracing_mode="real", _error_on_data_dependent_ops=False)(*inputs)
        gm.graph.eliminate_dead_code(); gm.recompile()
        with torch.no_grad(): ep = torch.export.export(gm, inputs, strict=False)
    t1 = time.perf_counter()
    params = dict(root.named_parameters()) if hasattr(root, "named_parameters") else {}
    named = {t.data_ptr(): n for n, t in params.items()}
    os.makedirs(dirpath, exist_ok=True); base = os.path.join(dirpath, name); tmp = base + ".partial"
    cfg = {"aot_inductor.package_constants_in_so": False} if weights == "user" else {}
    captured: Dict[str, torch.Tensor] = {}                                   # the lowered graph's constants under the names the compiled launcher asks for
    from torch._inductor.codecache import AotCodeCompiler                     # (Inductor names them itself; the sources are matched by storage, not by name)
    raw = AotCodeCompiler.__dict__["compile"]; fn = getattr(raw, "__func__", raw)
    def _grab(graph):
        try: captured.update({n: v for n, v in dict(graph.constants).items() if torch.is_tensor(v)})
        except Exception as e: STATS["errors"].append(f"constants capture: {type(e).__name__}: {e}")  # noqa: BLE001
    if isinstance(raw, staticmethod): AotCodeCompiler.compile = staticmethod(lambda graph, *a, **k: (_grab(graph), fn(graph, *a, **k))[1])
    else: AotCodeCompiler.compile = classmethod(lambda cls, graph, *a, **k: (_grab(graph), fn(cls, graph, *a, **k))[1])
    try:
        path = torch._inductor.aoti_compile_and_package(ep, package_path=tmp + ".pt2", inductor_configs=cfg)
    finally:
        AotCodeCompiler.compile = raw
    t2 = time.perf_counter()
    sources, small = {}, {}
    for fqn, t in captured.items():
        n_ = named.get(t.data_ptr())
        if weights == "user" and n_ is not None and t.shape == params[n_].shape and t.stride() == params[n_].stride(): sources[fqn] = ["w", n_, list(t.shape), str(t.dtype)]
        else: sources[fqn] = ["c", list(t.shape), str(t.dtype)]; small[fqn] = t.detach().clone().cpu()
    meta = {"schema": SCHEMA, "name": name, "crop": crop, "n_samples": ns, "levers": levers_word, "digest": digest, "consts_digest": cdig, "weights": weights,
            "kinds": "".join(kinds), "n_inputs": len(inputs), "input_shapes": [list(t.shape) for t in inputs], "constants": sources, "versions": _versions(),
            "inputs": in_names, "input_strides": [list(t.stride()) for t in inputs], "input_dtypes": [str(t.dtype) for t in inputs], "unaligned_inputs": sorted(want_unaligned),
            "export_route": route, "export_s": round(t1 - t0, 1), "compile_s": round(t2 - t1, 1), "package_bytes": os.path.getsize(path),
            "small_constants_bytes": int(sum(t.numel() * t.element_size() for t in small.values())), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    torch.save(small, tmp + ".consts.pt")
    with open(tmp + ".meta.json", "w") as fh: json.dump(meta, fh, indent=1)
    os.replace(tmp + ".consts.pt", base + ".consts.pt"); os.replace(tmp + ".pt2", base + ".pt2"); os.replace(tmp + ".meta.json", base + ".meta.json")   # meta last: find() lists complete packages only
    STATS["built"].append(name); log(f"[chai1-fastln] AOTI built {name}: export {meta['export_s']} s ({route}), compile {meta['compile_s']} s, {meta['package_bytes']/1e6:.1f} MB, {sum(1 for s in sources.values() if s[0]=='w')} weights bound by name + {len(small)} literal constants")
    return meta


# ---------------------------------------------------------------------------------------------------------------- load / serve
class Package:
    """One loaded package bound to one hoisted forward: ``step(root, *args, cache)`` has the eager step's signature and returns what it returns."""

    def __init__(self, hf, path: str):
        t0 = time.perf_counter()
        self.hf = hf                                                                 # the hoisted forward this package serves (its argnames name the input leaves of a package whose meta lists none)
        import torch._inductor                                                       # seconds at a process's first package when nothing compiled yet imported it
        t1 = time.perf_counter()
        with open(path[:-4] + ".meta.json") as fh: self.meta = json.load(fh)
        self.path, self.name, self.kinds = path, self.meta["name"], self.meta["kinds"]
        if not self.meta.get("input_strides") or not self.meta.get("inputs"):       # built before input layouts were recorded (names + strides + dtypes per input):
            raise Refused(self.name, "no_layout_meta")                             # its generated code assumes build strides nothing here can check — refused by name
        isa = cpu_isa.verdict(path)                                                  # the launcher's build ISA (the archive's AOTI_CPU_ISA) vs this host's word — BEFORE the dlopen:
        STATS["isa"].append(f"{self.name}:{isa['package']}|{isa['host']}|{isa['source']}|{'ok' if isa['ok'] else ('unrecorded' if isa['ok'] is None else 'lacking')}")
        if isa["ok"] is False:                                                       # an AVX-512 launcher on an AVX2 host dies of SIGILL inside aoti_load_package / load_constants
            print(f"[chai1-fastln] AOTI cpu isa: {self.name} — launcher built for {isa['package']} (the archive's AOTI_CPU_ISA, -march=native at the build), "
                  f"this host {isa['host']} ({isa['source']}): not loadable here (an instruction this CPU lacks would end the process) — {cpu_isa.WORD}", file=sys.stderr, flush=True)
            raise Refused(self.name, cpu_isa.WORD)                                   # by name: the eager hoisted step serves the crop (Dispatch._refused), exit 0
        self.runner = torch._inductor.aoti_load_package(path, run_single_threaded=True)      # single-threaded runner: capturable in the step's CUDA graph
        t2 = time.perf_counter()
        want = list(self.runner.get_constant_fqns())
        named = dict(hf.root.named_parameters()) if hasattr(hf.root, "named_parameters") else {}
        small = torch.load(path[:-4] + ".consts.pt", map_location="cpu") if os.path.exists(path[:-4] + ".consts.pt") else {}
        dev = next((t.device for t in named.values() if torch.is_tensor(t) and t.device.type == "cuda"), torch.device("cuda"))
        cmap, missing, self._keep = {}, [], []
        for fqn in want:
            src = self.meta["constants"].get(fqn)
            if src and src[0] == "w" and src[1] in named and list(named[src[1]].shape) == list(src[2]): cmap[fqn] = named[src[1]]
            elif fqn in small: t = small[fqn].to(dev); self._keep.append(t); cmap[fqn] = t
            else: missing.append(fqn)
        if missing: raise Unusable(f"{self.name}: {len(missing)} constant(s) without a source in this process ({missing[0]} …)")
        if want: self.runner.load_constants(cmap, check_full_update=True, user_managed=True)
        self._cache_id, self._checked = None, None
        self._fix_args, self._said = (), set()                                # arg-leaf input positions that arrive unaligned for this item (re-checked per call: new tensors each step)
        self.in_names = list(self.meta.get("inputs") or []); self.in_strides = [tuple(s) for s in (self.meta.get("input_strides") or [])]
        self.free = set(self.meta.get("unaligned_inputs") or [])              # inputs this package was compiled to take unaligned
        t3 = time.perf_counter()
        self.load_s = round(t3 - t0, 2); self.load_split = {"import_s": round(t1 - t0, 2), "dlopen_s": round(t2 - t1, 2), "bind_s": round(t3 - t2, 2), "n_constants": len(want)}
        print(f"[chai1-fastln] AOTI loaded {self.name}: {self.load_s} s (launcher {self.load_split['dlopen_s']} s, {len(want)} constants bound from the live module) — the compiled step of crop {self.meta.get('crop')} without a compile", file=sys.stderr, flush=True)
        STATS["loaded"].append(self.name); STATS["load_s"].append(self.load_s)

    def _name(self, i):
        return self.in_names[i] if i < len(self.in_names) else f"input{i}"

    def _realign(self, i, t, how):
        """An aligned (and, when the package recorded strides, identically strided) copy of input `i`; named once per package on stderr."""
        if self._layout_ok(i, t):
            fixed = t.clone()                                                  # same layout, fresh (aligned) storage
        else:                                                                  # the recorded layout, materialised (an expanded axis gets its own elements)
            rec = tuple(self.in_strides[i])
            if any(n > 1 and st == 0 for n, st in zip(t.shape, rec)):        # a recorded broadcast axis cannot hold this input's per-element values
                raise Refused(self.name, f"layout_unproducible:{self._name(i)}")
            try:
                fixed = torch.empty_strided(tuple(t.shape), rec, dtype=t.dtype, device=t.device); fixed.copy_(t)
            except RuntimeError:
                raise Refused(self.name, f"layout_unproducible:{self._name(i)}") from None
        key = (i, how)
        if key not in self._said:
            self._said.add(key); STATS["realigned"].append(f"{self.name}:{self._name(i)}:{how}")
            lay = f", strides {tuple(t.stride())} → {tuple(fixed.stride())}" if tuple(fixed.stride()) != tuple(t.stride()) else ""
            print(f"[chai1-fastln] AOTI input realigned: {self._name(i)} (input {i}, {list(t.shape)} {t.dtype}, data offset {t.data_ptr() % ALIGN} B{lay}) — "
                  f"copied {how} into the layout the launcher was compiled for (16-byte aligned, the package's recorded strides; a compiled launcher does not re-stride its inputs)",
                  file=sys.stderr, flush=True)
        return fixed

    def _inputs(self, args, cache):
        leaves, kinds, spec = _plan(args, cache)
        pos = [j for j, k in enumerate(kinds) if k == "i"]
        if id(cache) != self._cache_id:                                   # a new item (precompute made a new cache): its structure and constants must be the package's
            k = "".join(kinds)
            if k != self.kinds: raise Unusable(f"{self.name}: the item's step inputs differ in structure ({len(k)} leaves vs {len(self.kinds)})")
            if _consts_digest(leaves, kinds, spec) != self.meta["consts_digest"]: raise Unusable(f"{self.name}: the item's traced constants / input shapes differ from the package's")
            n_args = len(list(args)); fix_args = []                                # an input whose strides differ from the package's (an expanded / transposed view a mode hands over)
            names = _leaf_names(args, cache, list(getattr(self.hf, "argnames", []) or [])) if not self.in_names else None
            if names is not None: self.in_names = [n for n, kk in zip(names, kinds) if kk == "i"]
            for i, j in enumerate(pos):                                    # is COPIED into the recorded layout (a compiled launcher does not re-stride: it would read a stride-0
                v = leaves[j]                                              # sample axis as materialised) and an unaligned input the package expects aligned is copied to an
                lay_ok = self._layout_ok(i, v)                             # aligned buffer — cache leaves once per item (replaced IN the cache: later steps and the eager /
                if lay_ok and (self._name(i) in self.free or _aligned(v)): # Dynamo routes read the copy), argument leaves per call (small tensors)
                    continue
                nm = self._name(i)
                if nm.startswith("cache:"):
                    if not lay_ok:                                             # a precompute leaf is produced ONCE per item in the layout the build traced: a different
                        raise Refused(self.name, f"layout_unproducible:{nm}") # layout here is a different producer — refused by name, never re-laid
                    fixed = self._realign(i, v, "once for this item"); self._store(cache, nm, v, fixed); leaves[j] = fixed   # unaligned only: an aligned copy, same layout
                else:
                    fix_args.append(i)
            self._fix_args = tuple(fix_args); self._cache_id = id(cache)
        out = [leaves[j] for j in pos]
        for i in self._fix_args:                                           # per-step arguments (noised coordinates, noise levels, …): small; copied per call while unaligned / foreign-strided
            if not self._layout_ok(i, out[i]) or (not _aligned(out[i]) and self._name(i) not in self.free):
                out[i] = self._realign(i, out[i], "at each step")
        return out

    def _layout_ok(self, i, v) -> bool:
        """True when input `i`'s strides are the package's recorded ones on every axis longer than 1 (the layouts ``warm_aoti`` recorded at the build:
        every ARG contiguous, the CACHE leaves as hoist2's precompute produces them — some legitimately non-contiguous views)."""
        if not torch.is_tensor(v) or v.dim() == 0: return True
        if not self.in_strides or i >= len(self.in_strides): return True                        # (a package without recorded layouts is refused at load: no_layout_meta)
        rec = self.in_strides[i]
        return len(rec) == v.dim() and all(n <= 1 or a == b for n, a, b in zip(v.shape, v.stride(), rec))

    @staticmethod
    def _store(cache, name, old, new):
        """Put `new` where `old` sits under cache[<key>] (the value itself, or a leaf of a list / tuple / dict below it)."""
        key = name.split(":", 1)[1].split("[", 1)[0]
        if key not in cache: return
        if cache[key] is old: cache[key] = new; return
        def swap(node):
            if isinstance(node, list):
                for n, x in enumerate(node):
                    if x is old: node[n] = new; return True
                    if swap(x): return True
            elif isinstance(node, dict):
                for n, x in node.items():
                    if x is old: node[n] = new; return True
                    if swap(x): return True
            elif isinstance(node, tuple):
                if any(x is old for x in node):
                    return None                                            # immutable: left as is (the per-call path realigns it)
            return False
        swap(cache[key])

    def step(self, root, *rest):
        *args, cache = rest
        return self.runner(*self._inputs(args, cache))


class Dispatch:
    """Installed as ``hf._step`` by ``attach``: serves the package for (crop, n_samples of the call, levers, digest) when the directory holds it, else —
    or when a package proves unusable for an item — the fallback the caller gave (the Dynamo-compiled step, built lazily at the first step), by name."""

    def __init__(self, hf, dirpath, levers_word, fallback_factory, events=None, aside_reason=None, wrapper=None, eager=None):
        self.hf, self.dir, self.levers, self.factory, self.events = hf, dirpath, levers_word, fallback_factory, events
        self.aside_reason, self.wrapper, self.eager = aside_reason, wrapper, eager      # a card where the compile steps aside without a package: the fallback is the EAGER step, by name
        self.digest, self.crop = step_digest(hf), crop_of(hf)
        self.pkg, self.fallback, self.tried = None, None, set()

    def _event(self, what, secs=0.0):
        if self.events is not None: self.events.append((self.crop, what, secs))

    def _refused(self, e: "Refused") -> None:
        """A package refused by name: counted (STATS refused), said once, and the card's no-package route serves the step — the named eager aside
        where the compile steps aside without a package (cc 8.0), else the per-process compile."""
        STATS["refused"].append(f"{e.package}:{e.word}"); self._event(f"aoti_refused:{e.word}")
        if self.eager is not None:                                               # every card: a refused package is served by the EAGER hoisted step — never a per-process
            print(f"[chai1-fastln] AOTI refused {e.package}: {e.word} → eager ({e.word.split(':')[0]}): the eager hoisted step serves crop {self.crop}", file=sys.stderr, flush=True)
            step_aside(self.hf, self.crop, e.word.split(":")[0], wrapper=self.wrapper, events=self.events, where=self.dir); self.fallback = self.eager   # build (EXIT compiled=stepped_aside:<word>)
        else:                                                                    # (no eager step handed to this dispatcher: the caller's own route, named)
            print(f"[chai1-fastln] AOTI refused {e.package}: {e.word} → the caller's compiled route", file=sys.stderr, flush=True)
            self.hf.compiled = getattr(self.hf, "_compile_mode", None) or "default"

    def __call__(self, root, *rest):
        *args, cache = rest
        if self.pkg is None:
            names = list(getattr(self.hf, "argnames", []) or [])
            i = names.index("noise_sigma") if "noise_sigma" in names else -1
            ns = int(args[i].shape[-1]) if 0 <= i < len(args) and torch.is_tensor(args[i]) else None    # (batch, n_samples)
            key = (self.crop, ns)
            if key not in self.tried:
                self.tried.add(key); STATS["asked"].append(key)
                hits = find(self.dir, self.crop, ns, self.levers, self.digest)
                if hits:
                    try:
                        self.pkg = Package(self.hf, hits[0]); self.hf.compiled = "aoti"; self.hf.stats = dict(getattr(self.hf, "stats", {}) or {}, compiled="aoti", aoti=self.pkg.name)
                        STATS["attached"].append(self.pkg.name); self._event(f"aoti_loaded:{self.pkg.name}", self.pkg.load_s)
                    except Refused as e:
                        self._refused(e)
                    except Exception as e:  # noqa: BLE001
                        STATS["errors"].append(f"{type(e).__name__}: {str(e)[:160]}"); self._event(f"aoti_unusable:{type(e).__name__}:{str(e)[:120]}")
                else:
                    STATS["absent"].append(package_name(self.crop, ns or 0, self.levers, self.digest)); self._event(f"aoti_absent:c{self.crop}_s{ns}_{self.levers}_{self.digest}")
                    if self.aside_reason and self.eager is not None:            # cc 8.0: no package for (crop, n_samples) → the eager hoisted step, not a per-process compile
                        step_aside(self.hf, self.crop, self.aside_reason, wrapper=self.wrapper, events=self.events, n_samples=ns, where=self.dir)
                        self.fallback = self.eager
                    else:
                        print(f"[chai1-fastln] AOTI absent: no package {package_name(self.crop, ns or 0, self.levers, self.digest)} under {self.dir} — this crop compiles through Dynamo at this first step (python -m chai1_opt.warm_aoti builds it)", file=sys.stderr, flush=True)
        if self.pkg is not None:
            try:
                return self.pkg.step(root, *args, cache)
            except Refused as e:
                self.pkg = None; self._refused(e)
            except Unusable as e:
                STATS["mismatch"].append(str(e)[:200]); self._event(f"aoti_mismatch:{str(e)[:120]}"); self.pkg = None
                if self.aside_reason and self.eager is not None:
                    print(f"[chai1-fastln] AOTI package set aside for this item ({e}) — the eager hoisted step serves it on this card ({self.aside_reason})", file=sys.stderr, flush=True)
                    step_aside(self.hf, self.crop, self.aside_reason, wrapper=self.wrapper, events=self.events, where=self.dir); self.fallback = self.eager
                else:
                    print(f"[chai1-fastln] AOTI package set aside for this item ({e}) — the Dynamo route serves it", file=sys.stderr, flush=True)
                    self.hf.compiled = getattr(self.hf, "_compile_mode", None) or "default"
        if self.fallback is None:
            self.fallback = self.factory(); self._event("aoti_fallback:dynamo")
            self.hf.stats = dict(getattr(self.hf, "stats", {}) or {}, compiled=self.hf.compiled)
        return self.fallback(root, *args, cache)


def attach(hf, *, wrapper=None, levers_word: Optional[str] = None, mode: str = "default", dirpath: Optional[str] = None, events=None, aside_reason: Optional[str] = None) -> bool:
    """Route `hf`'s compiled step through a package when ``$MODEL_OPT_JIT_ROOT/<key>/aoti`` holds one for its crop and digest (any n_samples), the
    Dynamo compile of the same statements otherwise (lazily, at the first step — where its cost has always landed) — or, on a card where the
    compile steps aside without a package (``aside_reason``, card_aside_reason(): cc 8.0), the EAGER hoisted step by name. False (nothing
    installed) when there is no directory or no candidate package: the caller compiles through Dynamo (or steps aside on such a card: step_aside)."""
    try:
        d = dirpath if dirpath is not None else package_dir()
        lw = levers_word or levers_word_of(wrapper)
        if not find(d, crop_of(hf), None, lw, step_digest(hf)):
            if d: STATS["absent"].append(f"c{crop_of(hf)}_{lw}_{step_digest(hf)}")
            return False
    except Exception as e:  # noqa: BLE001 — nothing installed: the caller compiles through Dynamo
        STATS["errors"].append(f"attach: {type(e).__name__}: {str(e)[:160]}"); return False
    import torch._dynamo, torch._inductor.config
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit); torch._dynamo.config.capture_scalar_outputs = True
    torch._inductor.config.fx_graph_cache = True
    eager = getattr(hf, "_step_eager", None) or hf._step
    hf._step_eager, hf._compile_mode = eager, mode
    hf._step = Dispatch(hf, d, lw, lambda: torch.compile(eager, mode=(None if mode == "default" else mode), fullgraph=False, dynamic=False), events=events,
                        aside_reason=aside_reason, wrapper=wrapper, eager=eager)
    hf.compiled = mode; hf._graph = None
    hf.stats = dict(getattr(hf, "stats", {}) or {}, compiled=mode, aoti_dir=d)
    return True


# ---------------------------------------------------------------------------------------------------------------- warm inputs
def synthetic_inputs(crop: int, n_samples: int = 5, device="cuda", seed: int = 0, batch: int = 1) -> Dict[str, torch.Tensor]:
    """Structurally valid denoiser inputs of one crop (token / atom / atom-pair features, masks, sequence-local atom attention blocks of 32 queries x 128
    keys, noised coordinates and sigmas for n_samples): what the hoisted forward's precompute and step take. Values are random; shapes, dtypes and the
    index layout are the model's at that crop — the package built on them serves real items of the crop (their traced constants are checked at load)."""
    g = torch.Generator().manual_seed(seed)
    n_tok, n_atoms, stride, kvb = crop, 23 * crop, 32, 128
    nb = n_atoms // stride
    q_idx = torch.arange(n_atoms).reshape(nb, stride)
    kv = q_idx[:, :1] + (stride - kvb) // 2 + torch.arange(kvb)
    kv_mask = (kv < n_atoms) & (kv >= 0); kv = kv % n_atoms
    n_valid_tok = int(crop * 0.9); n_valid_atoms = int(n_atoms * 0.9 * 0.35)
    atom_tok = torch.zeros(n_atoms, dtype=torch.int64); atom_tok[:n_valid_atoms] = torch.arange(n_valid_atoms) * n_valid_tok // n_valid_atoms
    atom_mask = torch.zeros(n_atoms, dtype=torch.bool); atom_mask[:n_valid_atoms] = True
    tok_mask = torch.zeros(n_tok, dtype=torch.bool); tok_mask[:n_valid_tok] = True
    pair_mask = atom_mask[q_idx][:, :, None] & atom_mask[kv][:, None, :] & kv_mask[:, None, :]
    rn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)  # noqa: E731
    b = batch
    kw = dict(token_single_initial_repr=rn(b, n_tok, 384), token_pair_initial_repr=rn(b, n_tok, n_tok, 256),
              token_single_trunk_repr=rn(b, n_tok, 384), token_pair_trunk_repr=rn(b, n_tok, n_tok, 256),
              atom_single_input_feats=rn(b, n_atoms, 128), atom_block_pair_input_feats=rn(b, nb, stride, kvb, 16),
              atom_single_mask=atom_mask[None].expand(b, -1).contiguous(), atom_block_pair_mask=pair_mask[None].expand(b, -1, -1, -1).contiguous(),
              token_single_mask=tok_mask[None].expand(b, -1).contiguous(), block_indices_h=q_idx, block_indices_w=kv,
              atom_noised_coords=rn(b, n_samples, n_atoms, 3) * 16.0, noise_sigma=torch.full((b, n_samples), 16.0),
              atom_token_indices=atom_tok[None].expand(b, -1).contiguous())
    return {k: v.to(device) for k, v in kw.items()}


def report() -> Dict[str, Any]:
    """The lever's counters for the kit's EXIT line (chai1_opt.report: ``aoti=<crops served by a package>/<compiled crops entered>``):
    every list of STATS plus ``dir`` (the package directory in force, None when no root is set) and the two counts."""
    out = dict(unit=NAME, dir=package_dir(), **{k: (list(v) if isinstance(v, list) else v) for k, v in STATS.items()})
    out["n_asked"], out["n_loaded"], out["n_aside"], out["n_realigned"] = len(STATS["asked"]), len(STATS["attached"]), len(STATS["aside"]), len(STATS["realigned"])
    out["refused"] = list(STATS["refused"]); out["n_refused"] = len(STATS["refused"])
    out["isa"] = list(STATS["isa"])                                            # per loaded/refused package: <name>:<package word>|<host word>|<source>|ok|lacking|unrecorded
    return out

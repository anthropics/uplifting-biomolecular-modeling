"""Drop-in stack for chai_lab 0.6.1 `run_inference`.

    import chai1_eager.stack as S
    h = S.install(levers="tier1")      # patches chai_lab.chai1.load_exported in place
    chai_lab.chai1.run_inference(...)  # unchanged call
    h.restore()

levers='tier1' : structured eager trunk + copy-free layouts (bitwise vs scripted), hoisted denoiser step replayed from a CUDA graph
                 (graphed=False under the DET recipe -> byte-identical end-to-end outputs vs stock), flat eager embedders / confidence head.
The stack keeps the six scripted modules that chai_lab loads as the weight source (memoised; nothing is downloaded here).
"""
import os, time
import torch

SURFACE = ("install", "build_parts", "LEVERS", "HoistedDiffusionWrapper", "HoistedDiffusionWrapper.forward", "HoistedDiffusionWrapper.hoister_now")   # the names the kit / this package's callers reach (chai1_opt tests/test_lever_surfaces.py checks they exist, statically)

LEVERS = ("stock", "tier1")


class Timed:
    """wraps a ModuleWrapper-like object; records CUDA-synchronised wall per forward call into sink[name]."""
    def __init__(self, name, inner, sink):
        self.name, self.inner, self.sink = name, inner, sink

    def __getattr__(self, n):
        return getattr(self.inner, n)

    def forward(self, *a, **kw):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = self.inner.forward(*a, **kw)
        torch.cuda.synchronize(); self.sink.setdefault(self.name, []).append(time.perf_counter() - t0)
        return r


def _move(kw, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in kw.items()} if dev is not None else kw


def _cpu(out):
    if isinstance(out, dict):
        return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in out.items()}
    if isinstance(out, (tuple, list)):
        return type(out)(o.cpu() if torch.is_tensor(o) else o for o in out)
    return out.cpu() if torch.is_tensor(out) else out


class FlatWrapper:
    """ModuleWrapper replacement backed by a ts2eager flat object (any component)."""
    def __init__(self, flat):
        self.flat = flat; self.jit_module = flat

    def forward(self, crop_size, *, return_on_cpu=False, move_to_device=None, **kw):
        kw = _move(kw, move_to_device)
        with torch.no_grad():
            out = getattr(self.flat, f"forward_{crop_size}")(**kw)
        return _cpu(out) if return_on_cpu else out


class EagerTrunkWrapper:
    """trunk.pt replacement: structured eager trunk (cfg_fn fixes chai1_eager.trunk.CFG before each call)."""
    def __init__(self, trunk, cfg_fn=None):
        self.trunk, self.cfg_fn = trunk, cfg_fn
        self.jit_module = trunk

    def forward(self, crop_size, *, return_on_cpu=False, move_to_device=None, **kw):
        kw = _move(kw, move_to_device)
        with torch.no_grad():
            if self.cfg_fn:
                self.cfg_fn()
            out = self.trunk(**kw, crop_size=crop_size)
        return tuple(o.cpu() for o in out) if return_on_cpu else out


class HoistedDiffusionWrapper:
    """diffusion_module.pt replacement: hoists the step-invariant part once per sample, runs the variant part per call (optionally from a CUDA
    graph).  Re-hoists automatically when the static inputs change (new sample / new run).  hoister: 'base' (name-taint, hoist.HoistedForward)
    or 'hoist2' (value-taint, hoist.HoistedForward2: metadata-only reads of the step inputs do not drag work into the step; bitwise to 'base')."""
    def __init__(self, flat, graphed=False, sink=None, graph_fallback=True, hoister="base"):
        from . import hoist as H
        if hoister not in H.HOISTERS:
            raise ValueError(f"hoister {hoister!r} is not one of {H.HOISTERS}")
        self.flat, self.graphed, self.sink = flat, graphed, sink
        self.graph_fallback = graph_fallback
        self.hoister = hoister
        self.stand_down = None    # per-item hook for a memory line (the kit's big.py sets it at the item's trunk call, like `graphed`): "base" = THIS item's
                                  # denoiser runs the base hoister — hoist2 stood down by name (its larger hoist cache does not fit beside the top crops on the
                                  # smaller memory class); None = self.hoister. base and hoist2 are bitwise identical: standing down moves memory and speed only.
        self.hoister_used = None  # the hoister the current item's denoiser actually runs (read by the kit's census / MEMORY line)
        self.compile = None       # `compiled` lever: torch.compile mode for the per-step function ("default"), None = eager statements; a compile that fails on
                                  # this machine steps aside by name (event compile_failed:…, self.compile_failed set, the item re-run on the eager step)
        self.compile_failed = None
        self.compile_aside = None                                                 # the compile steps aside BY CARD (chai1_fastln.aoti.card_aside_reason: cc 8.0 without an ahead-of-time package): the reason word
        self.jit_module = flat
        self.hf = {}; self.key = None
        self.events = []          # (crop, what, seconds) — hoist build / capture fallback records

    def hoister_now(self):
        """The hoister this item runs: the stand-down hook's when set, else the wrapper's own."""
        return self.stand_down if self.stand_down in ("base",) and self.hoister != "base" else self.hoister

    def _hf(self, crop_size):
        h = self.hoister_now()
        k = (crop_size, h)
        if k not in self.hf:
            from . import hoist as H
            t0 = time.perf_counter()
            self.hf[k] = H.make(h, self.flat, f"forward_{crop_size}")
            self.events.append((crop_size, "hoist_build_s" if h == "base" else f"{h}_build_s", time.perf_counter() - t0))
            if self.compile and not self.compile_failed:
                a = getattr(self, "aoti", None)                 # the add-on's ahead-of-time route (chai1_fastln.aoti, set by its lever builder): a package on
                why = a.card_aside_reason() if a is not None and hasattr(a, "card_aside_reason") else None   # disk for this crop and code serves the compiled
                if not (a and a.attach(self.hf[k], wrapper=self, mode=self.compile, events=self.events, **({"aside_reason": why} if why else {}))):
                    if why:                                      # step without a Dynamo pass; none there: the Dynamo route — except on a card where a
                        a.step_aside(self.hf[k], crop_size, why, wrapper=self, events=self.events)   # per-process compile costs more than it returns (cc 8.0):
                    else:                                        # the compile steps aside by name for this crop, the eager hoisted step serves it
                        self.hf[k].compile(self.compile)
        return self.hf[k]

    def forward(self, crop_size, *, return_on_cpu=False, move_to_device=None, **kw):
        kw = _move(kw, move_to_device)
        key = (crop_size, kw["token_pair_trunk_repr"].data_ptr(), kw["atom_single_input_feats"].data_ptr(), tuple(kw["atom_noised_coords"].shape))
        if self.hoister_now() != self.hoister_used:
            self.key = None                                           # the stand-down hook changed the hoister: this item re-hoists on the one it names
        hf = self._hf(crop_size)
        if key != self.key:
            if self.hoister_now() != self.hoister:
                self.events.append((crop_size, f"{self.hoister}_stood_down", 0.0))
            for k_, other in self.hf.items():                     # one live hoist cache at a time: the other hoister's cache of a previous item is released here
                if other is not hf and getattr(other, "cache", None) is not None:
                    other.cache = None; other._graph = None
            self.hoister_used = self.hoister_now()
            torch.cuda.synchronize(); t0 = time.perf_counter(); hf.precompute(**kw); torch.cuda.synchronize(); dt = time.perf_counter() - t0
            if self.sink is not None:
                self.sink.setdefault("diffusion_precompute", []).append(dt)
            self.key = key
        with torch.no_grad():
            if getattr(hf, "compiled", None):
                try:
                    return self._run(hf, crop_size, return_on_cpu, kw)
                except Exception as e:               # the compiled step cannot engage on this machine (Dynamo / Inductor / Triton): step aside BY NAME, once,
                    from opt_core.oom import is_oom  # and run the item on the eager statements (same lever census: compiled shows off, the event names why)
                    if is_oom(e):
                        raise
                    import sys, traceback                # the traceback goes to stderr ONCE (the EXIT word carries a named reason, never a bare class)
                    sys.stderr.write(f"[chai1-eager] the compiled denoiser step raised at launch (crop {crop_size}); it steps aside by name and the eager hoisted step serves:\n")
                    traceback.print_exc(file=sys.stderr); sys.stderr.flush()
                    self.compile_failed = f"launch_error:{type(e).__name__}: {str(e)[:160]}"
                    self.events.append((crop_size, "compile_failed:launch_error:" + repr(e)[:160], 0.0))
                    for h_ in self.hf.values():
                        if getattr(h_, "compiled", None):
                            h_.uncompile()
                    torch.cuda.synchronize()
            out = self._run(hf, crop_size, return_on_cpu=False, kw=kw)
        return out.cpu() if return_on_cpu else out

    def _run(self, hf, crop_size, return_on_cpu, kw):
        if True:
            if self.graphed:
                try:
                    out = hf.step_graphed(**kw)
                except Exception as e:               # e.g. capture refused under torch.use_deterministic_algorithms
                    from opt_core.oom import is_oom  # an out-of-memory error propagates — it is never rerouted to the eager step
                    if is_oom(e):
                        raise
                    if not self.graph_fallback:
                        raise
                    self.events.append((crop_size, "graph_capture_failed:" + repr(e)[:160], 0.0))
                    self.graphed = False
                    hf._graph = None
                    torch.cuda.synchronize()
                    out = hf.step(**kw)
            else:
                out = hf.step(**kw)
        return out.cpu() if return_on_cpu else out


class StackHandle:
    def __init__(self, C1, orig, loader, parts):
        self.C1, self.orig, self.loader, self.parts = C1, orig, loader, parts

    def restore(self):
        self.C1.load_exported = self.orig

    def stats(self):
        out = {}
        d = self.parts.get("diffusion")
        if d is not None:
            out["diffusion_events"] = list(d.events)
            out["diffusion_hoist_stats"] = {(f"{c[0]}:{c[1]}" if isinstance(c, tuple) else c): getattr(h, "stats", None) for c, h in d.hf.items()}
            out["diffusion_hoister"] = {"row": getattr(d, "hoister", None), "used": getattr(d, "hoister_used", None), "stand_down": getattr(d, "stand_down", None)}
            out["diffusion_compile"] = {"mode": getattr(d, "compile", None), "failed": getattr(d, "compile_failed", None), "aside": getattr(d, "compile_aside", None)}
        return out


class Components:
    """Resident components shared by several stacks in one process (a caller may build stock / tier1 side by side)."""
    def __init__(self, downloads_dir=None, device="cuda:0", code_root=None):
        import chai_lab.chai1 as C1
        self.C1 = C1
        self.orig_load = C1.load_exported
        self.dev = torch.device(device)
        self.downloads = downloads_dir or os.environ["CHAI_DOWNLOADS_DIR"]
        # names the per-component code directories handed on; nothing is read from or written to them (ts2eager parses the archive), and the
        # default is per user, never a path shared between users
        self.code_root = code_root or os.path.join(os.environ.get("TMPDIR") or "/tmp", "chai1_eager_code-uid%d" % os.getuid())
        self.scripted = {}       # comp_key -> stock ModuleWrapper (weights source + stock arm)
        self.flats = {}
        self._trunk = None

    def stock(self, comp_key, device=None):
        if comp_key not in self.scripted:
            self.scripted[comp_key] = self.orig_load(comp_key, device or self.dev)
        return self.scripted[comp_key]

    def flat(self, comp_key):
        if comp_key not in self.flats:
            from . import ts2eager as T
            sdk = {k: v.detach() for k, v in self.stock(comp_key).jit_module.state_dict().items()}
            self.flats[comp_key] = T.load_eager_component(f"{self.downloads}/models_v2/{comp_key}", device=None,
                                                          code_dir=f"{self.code_root}/{comp_key}.code", state_dict=sdk)
        return self.flats[comp_key]

    def trunk(self):
        if self._trunk is None:
            from . import trunk as TR
            sd = {k: v.detach().clone() for k, v in self.stock("trunk.pt").jit_module.state_dict().items()}
            self._trunk = TR.load_trunk(sd, device=self.dev)
        return self._trunk

    @staticmethod
    def cfg_lc():
        from . import trunk as TR, kernels as K
        TR.CFG.update(precast_bf16=True, trimul_impl=K.trimul_bmm, triattn_impl=None, record_ranges=False)   # triangle attention: the module's own statement (a kit lever may set the plug: chai1_opt's triattn binds the shared provider by tier word)


def build_parts(comps, levers, *, graphed=True, sink=None, hoister="base"):
    """returns {'trunk': wrapper, 'diffusion': wrapper, 'flat_rest': bool} for levers == 'tier1'; {} for 'stock'.  hoister: HoistedDiffusionWrapper's."""
    assert levers in LEVERS, levers
    if levers == "stock":
        return {}
    return dict(trunk=EagerTrunkWrapper(comps.trunk(), Components.cfg_lc),
                diffusion=HoistedDiffusionWrapper(comps.flat("diffusion_module.pt"), graphed=graphed, sink=sink, hoister=hoister),
                flat_rest=True)


def make_loader(comps, parts, sink=None):
    """a chai_lab.chai1.load_exported replacement serving `parts` (see build_parts); every component is wrapped in Timed if sink is given."""
    def load(comp_key, device):
        base = comps.stock(comp_key, device)
        if comp_key == "trunk.pt" and parts.get("trunk") is not None:
            base = parts["trunk"]
        elif comp_key == "diffusion_module.pt" and parts.get("diffusion") is not None:
            base = parts["diffusion"]
        elif parts.get("flat_rest") and comp_key in ("confidence_head.pt", "token_embedder.pt", "feature_embedding.pt", "bond_loss_input_proj.pt"):
            base = FlatWrapper(comps.flat(comp_key))
        return Timed(comp_key.replace(".pt", ""), base, sink) if sink is not None else base
    return load


def install(levers="tier1", *, graphed=None, sink=None, comps=None, device="cuda:0", hoister="base"):
    """Patch chai_lab.chai1.load_exported for `levers`.  graphed defaults to True unless torch.use_deterministic_algorithms is on (DET recipe).
    hoister: the denoiser's hoisting rule ('base' | 'hoist2', HoistedDiffusionWrapper)."""
    comps = comps or Components(device=device)
    if graphed is None:
        graphed = not torch.are_deterministic_algorithms_enabled()
    parts = build_parts(comps, levers, graphed=graphed, sink=sink, hoister=hoister)
    loader = make_loader(comps, parts, sink)
    comps.C1.load_exported = loader
    return StackHandle(comps.C1, comps.orig_load, loader, parts)

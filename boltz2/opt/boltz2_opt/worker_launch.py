"""``python -m boltz2_opt.worker_launch [--route NAME[,NAME]] [--attach NAME[,NAME]] -- <script.py> <args...>``: the kit's worker script run
under the core's kernel route and/or with an attach-time lever module applied. First of all the process enters the INVOKING directory
(``enter_invoking_dir``: the kit directory stays importable; relative paths inside the input YAMLs resolve as under ``boltz predict``) and the
parsing zygote is forked (``prep.start``: every input
is parsed in a fresh child of a process that has initialised no CUDA, installed no hook and parsed nothing — prep.py). Then, before anything else is
imported: ``opt_core.kernels.route(name)``
(one meta-path finder serving exactly that name from the core's carried copy), its ``exports`` into the environment, ``route_check(name)`` (the
resolved file is the core copy, its bytes the sums file's) — a refusal exits 3 with a `[boltz2-opt route] REFUSED` line, never a silent fallback.
``--attach NAME`` first holds the attachment to this kit package (``attach_origin``: ``boltz2_opt.trimul`` — the fused TriMul —, ``boltz2_opt.big`` — the engine adapter of the memory
lines —, or any adapter of ``ATTACH``: triattn_exact / sampler / ditexact / pairfuse / exactln / waste / msa / msa2 / conf / graph / atom — must resolve to a file under HOME, the package directory this launcher runs from; a module resolved from outside that package — the
working directory, another copy on sys.path — is refused by name, exit 3. The version of the package itself is not this check's: it is the
git commit this tree is checked out at, and a shadow of the whole package would shadow this launcher with it), then installs one import hook per
attachment: right after the attachment's trigger module has executed (``ATTACH``: ``boltz.model.layers.triangular_mult`` for ``trimul``,
``boltz.model.models.boltz2`` — the worker's own model import, bz_worker_lev.py:59 — for ``xl``), the adapter's ``apply()`` runs from the checked
origin (an adapter with no levers installed refuses ``… has no levers installed yet``; one whose switch is absent from the row applies nothing and
refuses), and at interpreter exit the adapter's ``report()`` is written under its ``report_key`` (``trimul_report`` / ``xl_report``) into the
worker's own log (the ``--batch`` json names ``kit_dir`` and ``tag``; the worker writes ``<kit_dir>/<tag>_worker_log.json`` before this hook
runs: atexit is last-in first-out).
An attach that cannot import or apply exits 3 with a `[boltz2-opt attach] REFUSED` line (the kit's `pred` then exits 1: EXIT_FAILED, the worker
process failed — report.py run_exit). Every launch also places the per-item PHASE timing line (``boltz2_opt.phase``: trunk / conditioning / sampler / confidence walls of each
``Boltz2.predict_step``, synchronize + perf_counter at each boundary, numerics unchanged) right after ``boltz.model.models.boltz2`` executes — the
same instrument the stock process holds (``stock_pred``). Then the script runs as ``__main__`` with
``sys.argv = [script, args...]`` (runpy), exactly as ``python <script.py> <args...>`` would run it.
"""
import functools
import importlib.abc
import importlib.util
import json
import os
import runpy
import sys
from typing import Optional

EXIT_REFUSED = 3
CALLER_CWD_ENV = "BOLTZ_OPT_CALLER_CWD"     # the INVOKING directory (stack.run_worker exports the caller's cwd): the worker runs there, as the stock CLI does —
                                           # relative paths inside an input YAML (msa: / templates / constraints files) resolve where `boltz predict` resolves them
WORKDIR_ENV = "BOLTZ_OPT_WORKDIR"           # the kit directory the worker was launched in (stack.run_worker: cwd=<out_dir>/_kit): its staged modules import from
                                           # it whatever the process's cwd (the worker scripts and make_worker_variant.py put it on sys.path by this name)


def enter_invoking_dir(launch_dir: str) -> str:
    """Run the worker in the invoking directory. ``launch_dir`` (the kit directory this process was started in) goes on ``sys.path``
    and into ``WORKDIR_ENV`` so the staged lever modules keep importing by name; then, when ``CALLER_CWD_ENV`` names a directory, the process
    ``chdir``s there — BEFORE the parsing zygote forks (main: every input is parsed in a fresh child of it, which inherits this cwd), before
    any hook, before the model script. No YAML is rewritten: boltz's parser sees the caller's relative paths from the caller's directory,
    exactly as under ``boltz predict``. Without the variable (a launch outside stack.run_worker) the process stays where it was started.
    Returns the cwd after the call."""
    launch_dir = os.path.abspath(launch_dir)
    os.environ[WORKDIR_ENV] = launch_dir
    if launch_dir not in sys.path:
        sys.path.insert(0, launch_dir)
    dest = (os.environ.get(CALLER_CWD_ENV) or "").strip()
    if dest and os.path.realpath(dest) != os.path.realpath(launch_dir):
        if os.path.isdir(dest):
            os.chdir(dest)
            sys.stderr.write(f"[boltz2-opt] worker cwd={dest} (the invoking directory: relative paths in the input YAMLs resolve there, as for the stock CLI); kit directory {launch_dir}\n")
        else:
            sys.stderr.write(f"[boltz2-opt] {CALLER_CWD_ENV}={dest} is not a directory: the worker stays in {launch_dir}\n")
    return os.getcwd()


def script_path(script: str, launch_dir: str) -> str:
    """The worker line names its script relative to the kit directory (stack.worker_command: ``bz_worker.py``); absolute once the process has left it."""
    return script if os.path.isabs(script) else os.path.join(launch_dir, script)
ATTACH = {   # one attach name per engine adapter: the module (held to this kit package), the trigger module it is applied after, the key its report() takes in the worker log.
             # Hooks sharing a trigger run in the order the mode row lists them (modes._EXACT_ATTACH / _FAST_ATTACH: same-trigger hooks chain in attach order).
    "trimul": {"module": "boltz2_opt.trimul", "trigger": "boltz.model.layers.triangular_mult", "report_key": "trimul_report"},   # the fused TriMul (registry: fpf_trimul, fpf_trimul_exact)
    "transition": {"module": "boltz2_opt.transition", "trigger": "boltz.model.layers.transition", "report_key": "transition_report"},   # the fused transition (registry: fused_transition)
    "pairblock": {"module": "boltz2_opt.pairblock", "trigger": "boltz.model.layers.triangular_attention.attention", "report_key": "pairblock_report"},   # the fused triangle-attention block (registry: pairblock; fast: flash_triattn as its core)
    "triattn_exact": {"module": "boltz2_opt.triattn_exact", "trigger": "boltz.model.layers.triangular_attention.attention", "report_key": "triattn_exact_report"},   # the block's cueq core through the core provider's exact word (registry: triattn_exact): pairblock's trigger, attached right after it (hooks sharing a trigger run in row order)
    "pairfuse": {"module": "boltz2_opt.pairfuse", "trigger": "boltz.model.modules.trunkv2", "report_key": "pairfuse_report"},          # the PAIRFUSE layer driver (registry: pairfuse): PairformerModule / PairformerNoSeqModule / MSALayer forwards — every C=128 pair stack on one resident bf16 z per call over the same cores; the three block adapters serve what it hands back [PAIRFUSE]
    "sampler": {"module": "boltz2_opt.sampler", "trigger": "boltz.model.modules.diffusionv2", "report_key": "sampler_report"},       # the sampler roll-out (registry: rollout, kabsch_device, dit_fused): AtomDiffusion.sample class-level, right after diffusionv2's own import so the worker's hoist wraps it outermost [SAMPLER]
    "ditexact": {"module": "boltz2_opt.ditexact", "trigger": "boltz.model.modules.diffusionv2", "report_key": "ditexact_report"},   # the DITEXACT token-transformer schedule (registry: dit_par, dit_mask, dit_sba): DiffusionModule.__init__ wrapped, the schedule the instance forward of every token transformer built; after `sampler` (both on diffusionv2); reads the DiT hoist's level-2 cache [DITEXACT]
    "xl": {"module": "boltz2_opt.big", "trigger": "boltz.model.models.boltz2", "report_key": "xl_report"},                      # the memory lines (registry: xl_trans, xl_cond, xl_free, expandable_segments)
    "templ": {"module": "boltz2_opt.templates", "trigger": "boltz.model.models.boltz2", "report_key": "templ_report", "kind": "guard"},  # the template guard (registry.GUARDS: templates) — applies no lever: counts, names, refuses
    "templskip": {"module": "boltz2_opt.templskip", "trigger": "boltz.model.models.boltz2", "report_key": "templskip_report"},   # the dummy-template elision (registry: templ_skip): TemplateV2Module.forward class-level, attached AFTER `graph` (whose templ unit restates that forward and pins its stock source) — an all-dummy pass' update (the zero tensor) served without the module
    "tp": {"module": "boltz2_opt.rowpair", "trigger": "boltz.model.models.boltz2", "report_key": "tp_report"},                      # the row-sharded trunk at n_gpu > 1 (registry: rowpair_tp; big only): Boltz2.forward + the pair stacks
    "exactln": {"module": "boltz2_opt.exactln", "trigger": "boltz.model.models.boltz2", "report_key": "exactln_report"},       # the bitwise ATen layer_norm replica (registry: exactln): torch.nn.functional.layer_norm replaced process-wide after the worker's model import, the pair-track adapters' LayerNorm->bf16 cast fused into it (exactln.ln_to_bf16) [EXACTLN]
    "waste": {"module": "boltz2_opt.waste", "trigger": "boltz.model.models.boltz2", "report_key": "waste_report"},             # the WASTE exact hoists of the MSA module's stock chunked paths (registry: waste_chunkcast, waste_opmmask): Transition (chunked branch) / PairWeightedAveraging / OuterProductMean forwards; before `msa` and `msa2` in the rows (both wrap the forward they find and hand calls back to it by name) [WASTE]
    "writer": {"module": "boltz2_opt.writer", "trigger": "boltz.model.models.boltz2", "report_key": "writer_report"},         # the output writer off the critical path (registry: writer_overlap): spawns its writer process through the zygote at attach, displaces BoltzWriter.write_on_batch_end (pinned D2H staging + background write / move); attach it beside the featurizer
    "prefetch": {"module": "boltz2_opt.prefetch", "trigger": "boltz.model.models.boltz2", "report_key": "prefetch_report"},   # the persistent featurizer (registry: prefetch): spawns its helper through the zygote at attach, patches Boltz2InferenceDataModule.predict_dataloader; attach it LAST
    "msa": {"module": "boltz2_opt.msa_kernels", "trigger": "boltz.model.models.boltz2", "report_key": "msa_report"},            # the fused MSA-module kernels (registry: fpf_opm, fpf_pwa): OuterProductMean.forward / PairWeightedAveraging.forward, patched after the worker's model import (both layer modules imported by then)
    "msa2": {"module": "boltz2_opt.msa2", "trigger": "boltz.model.models.boltz2", "report_key": "msa2_report"},                 # the fused dim-64 MSA-module transition (registry: msa_trans2, msa_trans2_exact): Transition.forward for the dim-64 / hidden-256 instances, every other call handed to the forward installed before it (transition / waste) [MSA]
    "conf": {"module": "boltz2_opt.conf", "trigger": "boltz.model.models.boltz2", "report_key": "conf_report"},                 # the CONF levers (registry: condproj — the fast row's word; tfeat / tdummy parked, in no row; opt/forward/conf): DiffusionConditioning.forward (TemplateV2Module.forward / Boltz2.forward for the parked ones), patched after the worker's model import [CONF]
    "graph": {"module": "boltz2_opt.graph", "trigger": "boltz.model.models.boltz2", "report_key": "graph_report"},              # the trunk CUDA-graph lever (registry: graph_trunk): PairformerModule / PairformerNoSeqModule / MSAModule forwards captured per item shape and replayed, the template boundaries hoisted; after the layer adapters so a captured stack re-issues their kernels [GRAPH]
    "atom": {"module": "boltz2_opt.atom", "trigger": "boltz.model.models.boltz2", "report_key": "atom_report"},                 # the atom-attention levers (registry: atom_keys_gather, atom_glue_hoist, atom_fused, atom_gemm): AtomDiffusion.__init__ hooked so the structure module is installed at instance level when the checkpoint loader builds it; last in the rows [ATOM]
    "precision": {"module": "boltz2_opt.precision", "trigger": "boltz.model.models.boltz2", "report_key": "precision_report"},  # the fast tier's precision units (registry: dit_tf32 — the fast row's word; dit_bf16 / dit_attn_bf16 / seq_bf16 available, in no row): class-level __call__ installs on DiffusionModule / AttentionPairBias / FourierEmbedding / SingleConditioning / DiffusionTransformerLayer from opt/forward/precision/boltz2_precision.py; one row word BOLTZ_PRECISION=<unit,...>, `off:<unit>` = the row's ablation entry (R1) [PRECISION]
}
HOME = os.path.dirname(os.path.abspath(__file__))   # the kit package directory: every attachment is a module of this package and must resolve under it
KERNELS_CENSUS = "{tag}_kernels_census.json"        # the KERNELS reader's record of a worker pass, beside <tag>_worker_log.json in the launch's kit_dir (kernels.emit; worker.run reads it back for the exit rule and removes it)


def _kernels_json(script_args) -> Optional[str]:
    """``<kit_dir>/<tag>_kernels_census.json`` from the worker's ``--batch`` json (rank r > 0 of an n_gpu run has its own kit_dir, stack.rank_batch); None without one."""
    if "--batch" not in script_args:
        return None
    try:
        b = json.load(open(script_args[script_args.index("--batch") + 1]))
        return os.path.join(b["kit_dir"], KERNELS_CENSUS.format(tag=b["tag"]))
    except (OSError, ValueError, KeyError, IndexError):
        return None


class _AfterImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Runs `hook(module)` once, right after `trigger`'s own body has executed (the module's real spec/loader do the work)."""

    def __init__(self, trigger, hook):
        self.trigger, self.hook, self._busy = trigger, hook, False

    def find_spec(self, name, path=None, target=None):
        if name != self.trigger or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self._busy = False
        if spec is None:
            return None
        self._inner = spec.loader
        return _wrap(spec, self)

    def create_module(self, spec):
        return self._inner.create_module(spec) if hasattr(self._inner, "create_module") else None

    def exec_module(self, module):
        self._inner.exec_module(module)
        sys.meta_path.remove(self)
        self.hook(module)


def _wrap(spec, finder):
    spec.loader = finder
    return spec


def attach_origin(name):
    """The attachment's module resolved by name and held to this kit package: the module is ``boltz2_opt.<adapter>`` and its file must lie under
    HOME (the directory this launcher runs from); a module resolved from outside that package (the working directory, another copy on sys.path)
    is refused by name. The package's own version is the git commit this tree is checked out at, not this check's.
    Returns (origin, sha256) or raises with the reason."""
    import hashlib
    module = ATTACH[name]["module"]
    pkg = importlib.import_module(module.split(".")[0])
    pkg_dir = os.path.realpath(os.path.dirname(os.path.abspath(pkg.__file__)))
    if pkg_dir != os.path.realpath(HOME):
        raise RuntimeError(f"{module.split('.')[0]} imports from {pkg_dir}, not this kit package {HOME}")
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        raise ModuleNotFoundError(f"No module named {module!r} in the kit package {HOME}")
    origin = os.path.realpath(spec.origin)
    if not origin.startswith(os.path.realpath(HOME) + os.sep):
        raise RuntimeError(f"{module} resolves to {origin}, outside the kit package {HOME}")
    with open(origin, "rb") as fh:
        return origin, hashlib.sha256(fh.read()).hexdigest()


def _attach(name, script_args):
    a = ATTACH[name]
    batch = None
    if "--batch" in script_args:
        batch = script_args[script_args.index("--batch") + 1]
    try:
        origin, sha = attach_origin(name)
    except Exception as e:                            # a refusal by name, never a module from elsewhere
        sys.stderr.write(f"[boltz2-opt attach] REFUSED {name}: {type(e).__name__}: {e}\n"); return False
    sys.stderr.write(f"[boltz2-opt attach] {name}: {a['module']} = {origin} (sha256 {sha[:16]}) under the kit package\n")

    def hook(module):
        try:
            xl = importlib.import_module(a["module"])
            if os.path.realpath(getattr(xl, "__file__", "")) != origin:
                raise RuntimeError(f"{a['module']} imported from {getattr(xl, '__file__', None)}, not the checked origin {origin}")
            applied = xl.apply()
        except Exception as e:                        # a refusal by name, never a silent stock run
            sys.stderr.write(f"[boltz2-opt attach] REFUSED {name}: {type(e).__name__}: {e}\n"); os._exit(EXIT_REFUSED)
        if not (getattr(xl, "GUARDS", ()) if a.get("kind") == "guard" else getattr(xl, "LEVERS", ())):
            sys.stderr.write(f"[boltz2-opt attach] REFUSED {name}: {a['module']} has no {'guards' if a.get('kind') == 'guard' else 'levers'} installed yet\n"); os._exit(EXIT_REFUSED)
        disposed = getattr(xl, "dispositions", dict)() if not applied else {}   # levers the adapter names as disposed of rather than installed (msa_kernels at n_gpu > 1: replaced_by_rowpair) — accepted BY NAME, its report carries the word
        if not applied and not disposed:
            sys.stderr.write(f"[boltz2-opt attach] REFUSED {name}: {a['module']}.apply() applied nothing (its switch is not in the environment row?)\n"); os._exit(EXIT_REFUSED)
        if applied:
            sys.stderr.write(f"[boltz2-opt attach] {name}: {a['module']} applied {applied} after {a['trigger']} in the worker process\n")
        else:
            sys.stderr.write(f"[boltz2-opt attach] {name}: {a['module']} installed nothing after {a['trigger']} — " + ", ".join(f"{k}={v}" for k, v in disposed.items()) + " (named disposition; the adapter's report carries it)\n")
        if batch:
            import atexit

            def dump():
                try:
                    b = json.load(open(batch)); p = os.path.join(b["kit_dir"], f"{b['tag']}_worker_log.json")   # stack.worker_log_path's rule
                    log = json.load(open(p)) if os.path.exists(p) else {}
                    log[a["report_key"]] = xl.report(); json.dump(log, open(p, "w"), indent=1)
                except Exception as e:
                    sys.stderr.write(f"[boltz2-opt attach] {a['report_key']} not written: {type(e).__name__}: {e}\n")
            atexit.register(dump)
    sys.meta_path.insert(0, _AfterImport(a["trigger"], hook))
    return True


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        sys.stderr.write("usage: python -m boltz2_opt.worker_launch [--route NAME[,NAME]] [--attach NAME[,NAME]] [--kernels-route R --kernels-expect SPEC ...] -- <script.py> <args...>\n"); return 2
    launch_dir = os.getcwd()                              # the kit directory (stack.run_worker launches here); the process moves to the INVOKING directory now, before the
    enter_invoking_dir(launch_dir)                        # zygote forks, so every parse child resolves an input YAML's relative paths where the stock CLI would
    from . import prep                                    # the parsing zygote, forked FIRST: before CUDA, before any hook / route / lever, before anything is parsed —
    prep.start(forked_at="launch")                        # every input is parsed in a fresh child of it (stock's per-process parsing context; prep.py says why)
    from . import kernels as kcensus                       # the KERNELS census + REQUIRE guard (boltz2_opt/kernels.py): its options ride this launcher's argv (stack.kernels_opts); bound under its
    try:                                                  # own name — `kernels` below is the core's kernel router (opt_core.kernels)
        kopts, head = kcensus.parse_cli_opts(argv[:argv.index("--")])
    except ValueError as e:
        sys.stderr.write(f"[boltz2-opt attach] REFUSED kernels: {e}\n"); return EXIT_REFUSED
    argv = head + argv[argv.index("--"):]
    if "--route" not in argv and "--attach" not in argv and not kopts:
        sys.stderr.write("usage: python -m boltz2_opt.worker_launch [--route NAME[,NAME]] [--attach NAME[,NAME]] [--kernels-route R --kernels-expect SPEC ...] -- <script.py> <args...>\n"); return 2
    names = [n for n in argv[argv.index("--route") + 1].split(",") if n] if "--route" in argv else []
    attaches = [n for n in argv[argv.index("--attach") + 1].split(",") if n] if "--attach" in argv else []
    i = argv.index("--"); script, args = script_path(argv[i + 1], launch_dir), argv[i + 2:]
    from . import phase                                   # the per-item PHASE timing line (lm/trunk/sampler/conf), placed right after the worker's model import on every launch; numerics unchanged
    sys.meta_path.insert(0, _AfterImport(phase.MODEL_MODULE, phase.install_after))
    if kopts:                                             # the KERNELS census: installed right after the model import, AFTER the PHASE hook (its predict_step wrapper is the outer one: the first item's
        kopts.setdefault("json_path", _kernels_json(args))   # census + probe + REQUIRE run before the timed step); one pass line at exit, the record beside the worker log
        rk = os.environ.get("ROWPAIR_RANK", "").strip()   # opt_core.mem.rowpair.launch ENV_RANK: set in every rank process of an n_gpu > 1 run, absent at n_gpu = 1
        if rk.isdigit():
            kopts.setdefault("rank", int(rk))
        sys.meta_path.insert(0, _AfterImport(kcensus.MODEL_MODULE, functools.partial(kcensus.install_after, **kopts)))
    for name in attaches:
        if name not in ATTACH:
            sys.stderr.write(f"[boltz2-opt attach] REFUSED {name}: not an attachment of this tree ({sorted(ATTACH)})\n"); return EXIT_REFUSED
        if not _attach(name, args):
            return EXIT_REFUSED
    if names:
        from opt_core import kernels
        from . import modes, stack
    for name in names:
        try:
            kernels.route(name)
            os.environ.update(kernels.exports(name, **{k: stack.kit_path(v) for k, v in modes.kernel_exports(name).items()}))
            g = kernels.route_check(name)
            reason = None if g.ok else g.reason
        except Exception as e:                        # an unknown name, a missing sums file: a refusal by name, never a fallback to sys.path
            reason = f"{type(e).__name__}: {e}"
        if reason:
            sys.stderr.write(f"[boltz2-opt route] REFUSED {name}: {reason}\n"); return EXIT_REFUSED
        sys.stderr.write(f"[boltz2-opt route] {name} served from {os.path.abspath(g.details['resolved'])} (core copy) in the worker process\n")
    warm_imports()                                        # the stack's heavy model libraries imported once, here, before the model script runs (opt_core.warm_imports)
    sys.argv = [script] + args
    runpy.run_path(script, run_name="__main__")
    return 0


def warm_imports() -> None:
    """``opt_core.warm_imports()`` once at worker start-up (every mode's worker and every rank of the xP line pass here, after the parsing zygote's
    fork, before the model script's first import): cuEquivariance's import-time table build runs now, through the core's large-frame trampoline,
    instead of lazily inside the first item from whatever call depth first reaches it (the core measured 56-75 s of host time there on an unlucky
    depth; ~1.4 s otherwise). The core's provider faces warm themselves at their first serving call anyway; this line moves that ahead of the first
    forward and also covers the paths that reach cuEquivariance without a core face (the exact tier's stock kernels). torch is imported first, then
    the core's library list (cuEquivariance ops, bindings): the order every stock import path has. Imports only — no numerics,
    no bytes; free when the libraries are already imported. One stderr line: ``[boltz2-opt] WARM_IMPORTS <library>=<seconds|present|absent|…> … total_s=<t>``;
    an error is that line's ``error=<Type>`` word, never a refusal (the libraries then import lazily, as before)."""
    import time
    t0 = time.perf_counter()
    try:
        import opt_core
        from opt_core import warm as core_warm
        libs = ("torch",) + tuple(n for n in core_warm.DEFAULT_LIBRARIES if n != "torch")   # torch FIRST: cuEquivariance's ops library resolves the CUDA
        rep = dict(opt_core.warm_imports(libraries=libs))                                      # runtime (libnvrtc) through torch's bundled libraries, so it must
                                                                                                # not be the process's first CUDA import (CUDA-13 wheels fail to load it)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[boltz2-opt] WARM_IMPORTS error={type(e).__name__} total_s={time.perf_counter() - t0:.2f}\n"); return
    words = " ".join(f"{k}={v}" for k, v in rep.items()) or "-"
    sys.stderr.write(f"[boltz2-opt] WARM_IMPORTS {words} total_s={time.perf_counter() - t0:.2f}\n")


if __name__ == "__main__":
    sys.exit(main())

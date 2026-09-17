"""atlasfold-opt — the run interface.

    atlasfold-opt pred --mode <off|exact|fast|big> [--det 0|1] [--n_gpu P] [--allow-partial] [--weights DIR] -- monomer|multimer <stock atlasfold flags verbatim>
    atlasfold-opt check [--mode M]               # DRY-RUN: gates, the lever plan and the kernel routes; loads no model
    atlasfold-opt warm  [--mode M] [--multimer]  # one small fold (AFO_WARM_TOKENS residues, default 640; 1 recycle, 1 sample, 2 steps): compiles every first-use kernel of the mode
    atlasfold-opt install [--weights DIR]        # compares the weights root with stock/PINS.json (sha256) and prints WEIGHTS present n/n or the missing/differing files; information only, exit 0 (2 without a weights root); the environment install itself is run.sh install

Everything after ``--`` is the stock ``atlasfold`` command line (subcommand first). The kit adds ``--model-path``/``--lm-path`` from the weights
root when the stock line does not carry them (printed on the EXECUTION line); run settings (steps, recycles, samples) are never changed.
Exit: 0 complete; 1 the engine failed or wrote fewer outputs than targets; 2 usage; 3 NOT ACTIVE / partial refused / a lever gate refused."""
import argparse
import glob
import os
import sys
import time

from . import TAG, ENV, __version__
from .modes import MODE_NAMES

P = f"[{TAG}]"


def _err(s):
    sys.stderr.write(s + "\n"); sys.stderr.flush()


def _split(argv):
    if "--" in argv:
        i = argv.index("--"); return argv[:i], argv[i + 1:]
    return argv, []


def _flag_value(args, name):
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _inject_weights(stock, weights_root):
    """Add --model-path / --lm-path from the weights root unless present. Returns (argv, notes)."""
    from . import weights as W
    notes = []
    if not stock or stock[0] not in ("monomer", "multimer"):
        return stock, notes
    root = W.weights_dir(weights_root)
    if not root:
        return stock, notes
    p = W.paths(root)
    out = list(stock)
    if _flag_value(out, "--model-path") is None:
        out += ["--model-path", p["monomer" if stock[0] == "monomer" else "multimer"]]; notes.append("model-path<-weights_dir")
    if _flag_value(out, "--lm-path") is None:
        out += ["--lm-path", p["lm"]]; notes.append("lm-path<-weights_dir")
    return out, notes


def _expected_outputs(stock):
    """(n_targets, out_dir) from the stock line: one <name>/done.txt per FASTA record."""
    fasta, out_dir = _flag_value(stock, "--input-fasta"), _flag_value(stock, "--out-dir")
    n = 0
    if fasta and os.path.isfile(fasta):
        with open(fasta) as f:
            n = sum(1 for line in f if line.startswith(">"))
    return n, out_dir


def _count_done(out_dir):
    return len(glob.glob(os.path.join(out_dir, "*", "done.txt"))) if out_dir and os.path.isdir(out_dir) else 0


JIT_ROOT_ENV = "MODEL_OPT_JIT_ROOT"        # set by configs/*.env; opt_core.jit_cache and the core's NVRTC kernels (exactln) read the same variable
JIT_KEY_ENV = "MODEL_OPT_STACK_KEY"


def jit_env(environ=None):
    """Key the first-use compile caches by the running stack under ``$MODEL_OPT_JIT_ROOT`` (opt_core.jit_cache, the family contract:
    ``<root>/torch<version>-cu<cuda>-sm<cc>/{triton,inductor,torch_extensions}``; the key is resolved WITHOUT importing torch): every cache
    variable the caller did not point at an existing directory is set to its keyed directory (created), a pre-set one is kept, and
    ``MODEL_OPT_STACK_KEY`` names the key for the core's NVRTC cubin caches (``<root>/<key>/<family>``). No root configured -> nothing is touched
    (torch's and Triton's own defaults: ``~/.triton/cache`` …). Returns the NOTE line's fields, or None. The stock subprocess of ``--mode off``
    inherits the same variables (stock_pred keeps them), so stock's own cuEquivariance Triton kernels land in the same keyed cache."""
    env = os.environ if environ is None else environ
    root = (env.get(JIT_ROOT_ENV) or "").strip()
    if not root:
        return None
    try:
        from opt_core import jit_cache as J
        key = (env.get(JIT_KEY_ENV) or "").strip() or J.key_facts()["key"]
        fields = [f"jit_root={root}", f"cache_key={key}"]
        for var, (val, how) in sorted(J.cache_env(root, key, environ=env).items()):
            if how == "keyed":
                try:
                    os.makedirs(val, exist_ok=True)
                except OSError as e:
                    fields.append(f"{var}=unwritable:{type(e).__name__}"); continue
            env[var] = val; fields.append(f"{var}={how}")
        env.setdefault(JIT_KEY_ENV, key)
        return " ".join(fields)
    except Exception as e:  # noqa: BLE001  (a cache location is never a refusal: named, and the defaults apply)
        return f"jit_root={root} cache_key=unresolved:{type(e).__name__}"


def cmd_pred(a, stock):
    if not stock:
        _err(f"{P} usage: atlasfold-opt pred --mode M -- monomer|multimer --input-fasta X --out-dir Y [stock flags]"); return 2
    note = jit_env()                                                   # before anything compiles: Triton reads TRITON_CACHE_DIR at its first compile
    if note:
        _err(f"{P} NOTE {note}")
    stock, notes = _inject_weights(stock, a.weights)
    n_expected, out_dir = _expected_outputs(stock)
    t0 = time.time()
    if a.mode == "off":
        from . import stock_pred, det as D
        _err(f"{P} EXECUTION mode=off route=subprocess(clean env) det={a.det} argv={' '.join(stock)} injected={','.join(notes) or 'none'}")
        rc = stock_pred.run(stock, extra_env=D.env_for_subprocess(a.det))
    else:
        from . import enable, inputs_hint
        enable(a.mode, det=a.det, n_gpu=a.n_gpu, allow_partial=a.allow_partial, trigger="cli", inputs=inputs_hint.from_argv(stock))
        _err(f"{P} EXECUTION mode={a.mode} route=in-process det={a.det} argv={' '.join(stock)} injected={','.join(notes) or 'none'}")
        from . import phase_timing, workers
        phase_timing.install()                                        # the per-item PHASE / PEAK lines: CUDA events around trunk / sampler / confidence, outside every lever's wrapper (timing only)
        w = workers.install(a.mode, a.det, a.allow_partial)           # stock --gpu-ids workers are fresh interpreters: each activates this mode itself (never a silent stock worker under a kit word)
        ids = workers.gpu_ids_of(stock)
        if ids:
            _err(f"{P} WORKERS gpu_ids={','.join(ids)} activation={'per-worker' if w['installed'] else 'unavailable:' + str(w['reason'])} mode={a.mode} det={a.det}")
        from atlasfold.cli import main as stock_main
        from .hooks import LeverAborted
        old = sys.argv
        try:
            sys.argv = ["atlasfold", *stock]
            rc = stock_main(list(stock))          # stock cli/__init__.py L41: main(argv: Sequence[str] | None)
            rc = int(rc or 0)
        except SystemExit as x:
            rc = x.code if isinstance(x.code, int) else 1
        except LeverAborted as x:                                     # a lever ended the item by name (denoiser_graph: a failed CUDA-graph capture): the kit's refusal, exit 3 through the gate fold below
            rc = 1
            _err(f"{P} REFUSED: {x.lever} {x.reason} ({x.hint})")
        except Exception as x:  # noqa: BLE001 — the run raised (a lever that ends an item by name, or the stock code): named here, folded into the exit code below (a refused gate reads exit 3), never a bare traceback
            rc = 1
            _err(f"{P} RUN raised {type(x).__name__}: {str(x)[:400]}")
        finally:
            sys.argv = old
    n_done = _count_done(out_dir)
    complete = (n_expected == 0) or (n_done >= n_expected)
    _err(f"{P} OUTPUTS {'complete' if complete else 'incomplete'} files={n_done} expected={n_expected} out_dir={out_dir}")
    gates_ok, refused = True, []
    if a.mode != "off":
        from .stack import gates_verdict
        v = gates_verdict(); gates_ok, refused = v["ok"], v["refused"]
    code = 0
    if rc != 0 or not complete:
        code = 1
    if not gates_ok:
        code = 3
    _err(f"{P} FINAL mode={a.mode} rc={rc} exit={code} outputs={n_done}/{n_expected} gates={'ok' if gates_ok else 'refused:' + ';'.join(f'{l}:{r}' for l, r in refused)} wall_s={time.time() - t0:.1f}")
    return code


def _accepts_argv(fn):
    try:
        import inspect
        ps = list(inspect.signature(fn).parameters.values())
        return bool(ps) and ps[0].default is not inspect.Parameter.empty or (bool(ps) and ps[0].kind == inspect.Parameter.VAR_POSITIONAL) or (bool(ps) and ps[0].default is None)
    except Exception:  # noqa: BLE001
        return False


def kernel_probe(names=("fpf_trimul", "flash_triattn", "fpf_triatt_pro", "fpf_triatt_epi", "fpf_triatt_k2b", "fpf_transition")):   # the opt_core Triton kernels the levers route by name (the TriMul levers bind opt_core.kernels.trimul by tier word: no fpf_trimul_v4 route, no kit cell table)
    """The check verb's kernel rows: route each carried kernel name to the core copy exactly as activation does (the levers' install), THEN ask
    opt_core.kernels.route_check — which answers 'not importable' for a name that was never routed."""
    rows = []
    try:
        from opt_core import kernels as Kr
    except Exception as e:  # noqa: BLE001
        return [{"kernel": "*", "ok": None, "reason": f"opt_core.kernels unavailable: {type(e).__name__}: {e}"}]
    for name in names:
        try:
            if name not in Kr.names():
                rows.append({"kernel": name, "ok": None, "reason": "not carried by this core"}); continue
            Kr.route(name)
            g = Kr.route_check(name)
            get = (lambda k: g.get(k) if isinstance(g, dict) else getattr(g, k, None))
            det = get("details") or {}
            rows.append({"kernel": name, "ok": bool(get("ok")), "routed": det.get("routed"), "resolved": det.get("resolved"), "reason": get("reason")})
        except Exception as e:  # noqa: BLE001
            rows.append({"kernel": name, "ok": False, "reason": f"{type(e).__name__}: {e}"})
    return rows


def cmd_check(a, stock):
    from . import _core
    core = _core.gate()
    from .stack import _gpu_probe, _stock_facts
    from .modes import MODES, PLANNED
    from .registry import LEVERS
    from . import ablation as A
    gpu, st = _gpu_probe(), _stock_facts()
    ablated = A.requested()
    try:                                                            # MODEL_OPT_LEVERS_OFF: the same by-name validation pred applies (an unknown / foreign name exits 3 here too)
        why = A.reasons(a.mode, ablated)                            # requested -> levers_off; the levers that require them -> requires:<lever>(levers_off)
    except A.AblationError as e:
        _err(f"{P} NOT ACTIVE: reason=levers_off_refused: {e}"); return 3
    ablated = list(why)
    abl = f" {A.TOKEN}={A.token(ablated)}" if ablated else ""
    cfg = os.path.basename(os.environ.get("ATLASFOLD_KIT_CONFIG", "")).removesuffix(".env") or "none"
    _err(f"{P} DRY-RUN mode={a.mode}{abl} config={cfg} kit={__version__} core={core['installed']['version']} (pinned>={core['pinned']['version']} at {core['pinned']['path']}) "
         f"stock_importable={st.get('importable')} atlasfold_dist={st.get('dist_version')} cueq={st.get('cuequivariance_torch')} gpu={gpu.get('name')} cc={gpu.get('cc')} torch={gpu.get('torch')}")
    for lever in MODES.get(a.mode, []):
        d = LEVERS.get(lever, {})
        if lever in why:
            _err(f"{P} PLAN lever={lever} state=off reason={why[lever]} class={d.get('cls')} strategy={d.get('strategy')}"); continue
        _err(f"{P} PLAN lever={lever} class={d.get('cls')} strategy={d.get('strategy')} site={d.get('site', '')[:110]}")
    for lever in PLANNED.get(a.mode, []):
        _err(f"{P} PLAN lever={lever} state=planned(not in {__version__})")
    from .hooks import atom_sdpa as _asd, denoiser_graph as _dg
    rows = kernel_probe() + [_asd.probe(), _dg.probe()]                        # the carried Triton cells, the fused-SDPA route atom_sdpa takes, one tiny CUDA-graph capture + replay through denoiser_graph's backend
    for row in rows:
        _err(f"{P} KERNEL {row['kernel']} ok={row['ok']} routed={row.get('routed')} resolved={row.get('resolved')} reason={row.get('reason')}")
    if a.weights or os.environ.get("ATLASFOLD_WEIGHTS_DIR"):
        from . import weights as W
        root = W.weights_dir(a.weights); v = W.compare(root, quick=True)
        _err(f"{P} WEIGHTS {'present' if v['ok'] else 'MISSING'}: {v['checked']}/3 under {root} (bytes compared; sha256 on install --weights)")
    graph_row = next((r for r in rows if r["kernel"] == "cuda_graph"), None)
    if "denoiser_graph" in MODES.get(a.mode, []) and "denoiser_graph" not in ablated and graph_row is not None and graph_row["ok"] is False:   # a GPU box that cannot capture: refused here, before any item (ok None = no CUDA device: the GPU box's own check decides)
        _err(f"{P} CHECK refused: cuda_graph {graph_row['reason']} — mode {a.mode} carries denoiser_graph (exit 3; AFO_DENOISER_GRAPH_MAX_TOKENS=0 keeps the denoiser eager)")
        return 3
    return 0


def cmd_install(a, stock):
    from . import weights as W
    root = W.weights_dir(a.weights)
    if not root:
        _err(f"{P} install: pass --weights DIR or set ATLASFOLD_WEIGHTS_DIR (layout: DIR/atlaslm-3b-base/weights/atlaslm_3b_base.pth, DIR/atlasfold-260703/..., DIR/atlasfold-m-260725/...)"); return 2
    v = W.compare(root, quick=False)                  # information only: weights are never pinned or gated (exit 0 either way)
    if v["ok"]:
        _err(f"{P} WEIGHTS present: {v['checked']}/3, sha256 == stock/PINS.json under {root}"); return 0
    _err(f"{P} WEIGHTS NOTE: missing={v['missing']} differing={v['mismatched']} under {root} (information; not a gate)"); return 0


WARM_TOKENS_ENV = "AFO_WARM_TOKENS"
WARM_TOKENS_DEFAULT = 640     # the smallest stock bucket (atlasfold.common.featurize DEFAULT_BUCKETS) at or above the size floor of every lever whose kernel
                              # compiles at first use, so one warm fold compiles them all (lm_sdpa's floor lies higher, but that lever is torch SDPA: nothing
                              # to compile). One size serves every bucket: the Triton / NVRTC kernels are not specialised per N.
_WARM_UNIT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"   # 64 residues


def warm_fasta_text(tokens=None) -> str:
    """The warm fold's one record: ``AFO_WARM_TOKENS`` residues (default WARM_TOKENS_DEFAULT, at least 16) — large enough to engage every
    size-gated kernel of the mode, so the first real item finds them compiled."""
    raw = os.environ.get(WARM_TOKENS_ENV, "")
    try:
        n = int(tokens if tokens is not None else (raw.strip() or WARM_TOKENS_DEFAULT))
    except ValueError:
        n = WARM_TOKENS_DEFAULT
    n = max(16, n)
    return ">warm\n" + (_WARM_UNIT * (n // len(_WARM_UNIT) + 1))[:n] + "\n"


def cmd_warm(a, stock):
    import tempfile
    d = tempfile.mkdtemp(prefix="afo_warm_")
    fa = os.path.join(d, "warm.fasta")
    with open(fa, "w") as f:
        f.write(warm_fasta_text())
    sub = "multimer" if a.multimer else "monomer"
    extra = ["--num-recycles", "1", "--num-samples", "1", "--num-steps", "2", "--seed", "1"]
    return cmd_pred(a, [sub, "--input-fasta", fa, "--out-dir", os.path.join(d, "out"), *extra])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ours, stock = _split(argv)
    ap = argparse.ArgumentParser(prog="atlasfold-opt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("pred", "check", "warm", "install"))
    ap.add_argument("--mode", default=os.environ.get(ENV) or "off", choices=MODE_NAMES)
    ap.add_argument("--det", type=int, default=int(os.environ.get("AFO_DET", "0") or 0), choices=(0, 1))
    ap.add_argument("--n_gpu", type=int, default=1)
    ap.add_argument("--allow-partial", action="store_true", default=os.environ.get("AFO_ALLOW_PARTIAL") == "1")
    ap.add_argument("--weights", default=None, help="weights root (else ATLASFOLD_WEIGHTS_DIR)")
    ap.add_argument("--multimer", action="store_true", help="warm: use the multimer model")
    a = ap.parse_args(ours)
    os.environ[ENV] = a.mode                       # children (stock subprocess excluded: stock_pred scrubs it) see the same word
    return {"pred": cmd_pred, "check": cmd_check, "warm": cmd_warm, "install": cmd_install}[a.verb](a, stock)


if __name__ == "__main__":
    raise SystemExit(main())

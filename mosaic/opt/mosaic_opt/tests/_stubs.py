"""Test fixtures: a stub kit directory (the real kit's files with a stand-in driver), gate stubs, and a fresh package state.

The stand-in driver (`tools/public_design_run.py`) keeps the real driver's argparse block (settings.driver_defaults reads it by ast) and
writes the same `results.json` shape the real one does — `manifest.load_path`, `manifest.features.source`, `manifest.identity_key_pre`
from the environment it sees, a `run` record with the run fields — without jax; it prints the `[run] <tag>: ...` line. Nothing here
is a lever: the tests check what the package composes and records, never a design.
"""
import importlib.util
import json
import os
import shutil
import sys
import textwrap
import types

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # opt/mosaic_opt
OPT_DIR = os.path.dirname(PKG_DIR)                                                # opt/
TREE = os.path.dirname(OPT_DIR)                                                   # mosaic/
KIT = PKG_DIR                                                                     # the kit files (tools/, mosaic_fast/) live in the package directory

STANDINS = ("jax", "jax.numpy", "einops", "equinox")

NO_JAX = importlib.util.find_spec("jax") is None


def _standins():
    """sys.modules entries for the array stack a jax-less box lacks (attribute access returns a placeholder; nothing is computed at import).
    Shared by every test file that loads a kit lever module standalone (via spec_from_file_location) on a box without the real stack."""
    class _Inert(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _Inert(f"{self.__name__}.{name}")
    out = {}
    for name in STANDINS:
        if importlib.util.find_spec(name.split(".")[0]) is None:
            out[name] = _Inert(name)
    if "jax" in out:
        out["jax"].numpy = out["jax.numpy"]
    return out


def tree_present() -> bool:
    return os.path.isfile(os.path.join(TREE, "stock", "PINS.json")) and os.path.isfile(os.path.join(KIT, "tools", "public_design_run.py"))


STUB_DRIVER = textwrap.dedent('''\
    #!/usr/bin/env python
    # stand-in for tools/public_design_run.py (tests only): same arguments, same results.json shape, no jax.
    import argparse, hashlib, json, os, sys, time
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--out", default="out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--binder-length", type=int, default=80)
    p.add_argument("--target-copies", type=int, default=1)
    p.add_argument("--weights", default="torch", choices=["torch", "fastinit"])
    p.add_argument("--features-in", default=None)
    p.add_argument("--features-sha", default=None)
    p.add_argument("--features-out", default=None)
    p.add_argument("--steps1", type=int, default=75)
    p.add_argument("--steps2", type=int, default=50)
    p.add_argument("--target-fasta", default=None)
    p.add_argument("--first-record", action="store_true")
    p.add_argument("--epitope", default=None)
    p.add_argument("--msa", default=None)
    args = p.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import recipe        # the kit's recipe module beside this driver, as the real driver imports it
    TARGET = recipe.target(fasta=args.target_fasta, copies=args.target_copies, first_record=args.first_record)   # (the stand-in leaves --msa unread: no alignment is parsed here)
    EPITOPE = recipe.epitope(args.epitope, TARGET) if args.epitope is not None else None
    for _line in recipe.input_lines(TARGET, EPITOPE):                                     # the real driver's INPUT lines, from the same function
        print(_line, flush=True)
    TLEN = TARGET["length"]
    OUT = Path(args.out) / args.tag; OUT.mkdir(parents=True, exist_ok=True)
    def identity_key():
        fl = os.environ.get("XLA_FLAGS", ""); info = {"xla_flags": fl, "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR", "")}
        cd = info["jax_compilation_cache_dir"]
        if cd and os.path.isdir(cd):
            info["jax_cache_n_entries"] = len(os.listdir(cd))
        return info
    if os.environ.get("STUB_DRIVER_FAIL"):
        print("[run] " + args.tag + " done in 0s status=error", flush=True); sys.exit(1)
    if os.environ.get("STUB_DRIVER_SLEEP"):
        time.sleep(float(os.environ["STUB_DRIVER_SLEEP"]))
    DROP = set(filter(None, os.environ.get("STUB_DRIVER_DROP", "").split(",")))    # levers the stand-in silently does not apply (its manifest shows the stock path)
    if "P2" in DROP:
        args.weights = "torch"
    if "P3" in DROP:
        args.features_in = None
    if "P1" in DROP:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = ""
    if os.environ.get("STUB_DRIVER_NO_RESULTS"):                                     # exit 0 without the results file (an incomplete design)
        sys.exit(0)
    manifest = {"tag": args.tag, "args": vars(args), "identity_key_pre": identity_key(), "pid": os.getpid(), "n_tokens": TLEN * args.target_copies + args.binder_length, "epitope": EPITOPE,
                "target": {k: TARGET[k] for k in ("name", "id", "length", "copies", "source", "fasta", "fasta_records", "fasta_record_used", "first_record")},
                "versions": {"jax": "stub"}, "host_class": {"gpu_nvidia_smi": os.environ.get("STUB_GPU", "stub GPU")}, "boltz2_ckpt_sha256": "stub", "param_fingerprint": "stub",
                "load_path": "P2 load_stock_fast_init (torch ckpt -> joltz, random init skipped)" if args.weights == "fastinit" else "stock Boltz2() (torch ckpt -> joltz.from_torch)",
                "t_load_boltz2_s": 0.0, "t_features_s": 0.0, "env_seen": {k: v for k, v in os.environ.items() if k.startswith(("JAX_", "XLA_", "MOSAIC", "PYTHONUNBUFFERED"))}}
    if args.features_in:
        got = hashlib.sha256(open(args.features_in, "rb").read()).hexdigest()
        assert args.features_sha is None or got == args.features_sha, "frozen feature sha256 mismatch"
        manifest["features"] = {"source": "frozen npz " + args.features_in, "sha256": got}
    else:
        manifest["features"] = {"source": "in-process boltz featurization of " + ("target " + args.target_fasta if args.target_fasta else "the public target") + (" (msa: staged a3m; no server)" if args.msa else " (msa: empty -> single-sequence; no server)")}
        if args.features_out:
            open(args.features_out, "wb").write(b"stub features " + str(args.seed).encode())
            manifest["features"]["frozen_out"] = args.features_out; manifest["features"]["sha256"] = hashlib.sha256(open(args.features_out, "rb").read()).hexdigest()
    lt = [float(i) for i in range(args.steps1 + args.steps2)]
    rec = {"loss_traj": lt, "loss_traj_sha256": hashlib.sha256(str(lt).encode()).hexdigest(), "x0_sha16": "x0" * 8, "x1_sha16": "x1" * 8, "best1_sha16": "b1" * 8,
           "x2_sha16": "x2" * 8, "best2_sha16": "b2" * 8, "seq_stage2_x": "A" * args.binder_length, "seq_stage2_best": "A" * args.binder_length,
           "refold_coords_sha16": "rc" * 8, "refold_pae_sha16": "rp" * 8, "refold_plddt_sha16": "rl" * 8, "refold_iptm": 0.5, "numeric_state_unchanged": True,
           "t_first_iter_s": 0.0, "t_iter_stage1_steady_s": 0.0, "t_design_opt_s": 0.0, "t_refold_s": 0.0}
    (OUT / f"pssm_seed{args.seed}.npz").write_bytes(b"stub"); (OUT / f"refold_seed{args.seed}.npz").write_bytes(b"stub")
    results = {"manifest": manifest, "run": rec, "status": "ok", "t_process_total_s": 0.0}
    fl = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_dump_autotune_results_to=" in fl:                       # what XLA does at the end of a process under the dump flag
        af = fl.split("--xla_gpu_dump_autotune_results_to=", 1)[1].split()[0]
        os.makedirs(os.path.dirname(af), exist_ok=True); open(af, "wb").write(b"stub autotune")
        cd = os.environ.get("JAX_COMPILATION_CACHE_DIR")
        if cd:
            os.makedirs(cd, exist_ok=True); open(os.path.join(cd, "jit_stub_executable"), "wb").write(b"stub cache entry")
    results["manifest"]["identity_key_post"] = identity_key()
    json.dump(results, open(OUT / "results.json", "w"), indent=1)
    print(f"[run] {args.tag}: tokens {manifest['n_tokens']} loss[0]={lt[0]!r} best2={rec['best2_sha16']} seq={rec['seq_stage2_best']}", flush=True)
    print(f"PEAK item={args.tag} peak_bytes_in_use_gib=none bytes_limit_gib=none source=stub", flush=True)
    print(f"[run] {args.tag} done in 0s status=ok", flush=True)
    sys.exit(0)
''')

STUB_FETCH_TAIL = textwrap.dedent('''\
    if __name__ == "__main__":
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument("--weights", action="store_true"); ap.add_argument("--out", default="inputs_public")
        a = ap.parse_args()
        os.makedirs(a.out, exist_ok=True)
        print("provenance: stub (inline sequence)"); print('{"sha256_ok": true}')
''')


def make_stub_kit(root: str) -> str:
    """A kit directory under `root`: the real kit's mosaic_fast/ and the constants of tools/fetch_public_inputs.py, with the
    stand-in driver."""
    kit = os.path.join(root, "kit")
    os.makedirs(os.path.join(kit, "tools"), exist_ok=True)
    for sub in ("mosaic_fast",):
        if os.path.isdir(os.path.join(KIT, sub)):
            shutil.copytree(os.path.join(KIT, sub), os.path.join(kit, sub))
    src = open(os.path.join(KIT, "tools", "fetch_public_inputs.py"), encoding="utf-8").read()
    head = []
    for line in src.splitlines():
        if line.startswith("def ") or line.startswith("if __name__"):
            break
        head.append(line)
    open(os.path.join(kit, "tools", "fetch_public_inputs.py"), "w", encoding="utf-8").write("\n".join(head) + "\n" + STUB_FETCH_TAIL)
    open(os.path.join(kit, "tools", "public_design_run.py"), "w", encoding="utf-8").write(STUB_DRIVER)
    shutil.copyfile(os.path.join(KIT, "tools", "recipe.py"), os.path.join(kit, "tools", "recipe.py"))          # the recipe module is the kit's own (the package accessor reads it from the kit home)
    return kit


def _is_upstream_module(name: str) -> bool:
    """`mosaic` / `mosaic.*` (upstream, real or a case's fake) — never `mosaic_opt*`."""
    return (name == "mosaic" or name.startswith("mosaic.")) and not name.startswith("mosaic_opt")


class StubState:
    """Fresh package state per test: the environment, sys.meta_path, the fake modules, the activation report, the instance counter,
    the kit P1 module, the exit tally."""

    def __init__(self):
        self.env = dict(os.environ)
        self.meta_path = list(sys.meta_path)
        for m in [m for m in sys.modules if _is_upstream_module(m)]:              # a real upstream `mosaic` an earlier case or import left in the interpreter (a stack venv has one): removed, so this
            del sys.modules[m]                                                       # case's fake package on sys.path[0] is what `import mosaic…` finds and the autoload trigger fires; re-imported on demand later
        self.modules = set(sys.modules)
        self.path = list(sys.path)

    def restore(self):
        os.environ.clear(); os.environ.update(self.env)
        sys.meta_path[:] = self.meta_path
        sys.path[:] = self.path
        for m in list(sys.modules):
            if m not in self.modules and _is_upstream_module(m):
                del sys.modules[m]
        from mosaic_opt import report, stack
        stack._REPORT = None
        stack._ACTIVATING = False
        stack._INSTANCES.update({"state": None, "finder": None})
        stack.pins.cache_clear()
        report._TALLY.update({"fn": None})


def gpu_h100():
    return {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0", "sm": "sm90", "memory_mib": 81559, "probe": "stub"}


def stub_gates(monkeypatch, gpu=None, pins_ok=True, versions_ok=True):
    """Make the box look pinned, GPU-equipped and before any jax computation, without jax or the upstream packages installed."""
    from mosaic_opt import stack
    monkeypatch.setattr(stack, "pins_gate", lambda p, force=False: (pins_ok, {"stub": True}, None if pins_ok else "upstream not at the pinned commit: stub"))
    monkeypatch.setattr(stack, "version_gate", lambda p, force=False: (True, {"stub": True}, None if versions_ok else "stack differs from the kit's pins: stub"))   # a note, never a refusal
    monkeypatch.setattr(stack, "gpu_identity", lambda: gpu if gpu is not None else gpu_h100())
    monkeypatch.setattr(stack, "backend_initialised", lambda: (False, "jax imported, no backend yet (stub)"))   # a box look before any jax computation, whatever another test of this interpreter computed (the gate's own test sets it True after this call)


class FakeReproCache:
    """Stand-in for the P1 lever module (`stack.p1_lever`, opt_core.jax_design.pcc): records the call, mimics dump-vs-load through XLA_FLAGS;
    the phase, file and never-re-apply readers are the real primitive's (pure functions)."""

    def __init__(self):
        from opt_core.jax_design import pcc
        self.calls = []
        self.autotune_file_of, self.autotune_mode, self.already_enabled, self.backend_initialized = (
            pcc.autotune_file_of, pcc.autotune_mode, pcc.already_enabled, pcc.backend_initialized)

    def enable(self, cache_dir, autotune="auto", autotune_file=None):
        os.makedirs(cache_dir, exist_ok=True)
        af = autotune_file or os.path.join(cache_dir, "xla_autotune_results.pb")
        self.calls.append(cache_dir)
        flag = " --xla_gpu_load_autotune_results_from=" if os.path.exists(af) else " --xla_gpu_dump_autotune_results_to="
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + flag + af).strip()
        return {"cache_dir": cache_dir, "autotune_file": af, "xla_flags": os.environ["XLA_FLAGS"]}

    def identity_key(self, cache_dir, autotune_file=None):
        return {"autotune_sha256": None, "cache_listing_sha256": None, "n_cache_entries": len(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else 0}


def fake_mosaic_package(tmp: str, with_model: bool = True) -> str:
    """A fake upstream `mosaic` package on a temp path: an empty __init__ (as upstream's) and mosaic.models.boltz2.Boltz2."""
    pkg = os.path.join(tmp, "site", "mosaic")
    os.makedirs(os.path.join(pkg, "models"), exist_ok=True)
    open(os.path.join(pkg, "__init__.py"), "w").write("")
    open(os.path.join(pkg, "models", "__init__.py"), "w").write("")
    if with_model:
        open(os.path.join(pkg, "models", "boltz2.py"), "w").write("class Boltz2:\n    def __init__(self):\n        self.model = None\n")
    return os.path.join(tmp, "site")

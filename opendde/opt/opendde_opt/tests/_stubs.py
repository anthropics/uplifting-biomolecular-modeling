"""Stubs of the upstream package for the CPU tests: a site directory with a fake `opendde` distribution (the pinned version by default:
stock/PINS.json), a fake `runner` package (`cli.opendde_cli` = the console entry, `batch_inference.get_default_runner`,
`inference.InferenceRunner`) and `opendde.model.opendde`, so the package's gates, the autoload finder and the kits' own shims can be
exercised without torch or the model. The kit bytes are never stubbed. `ADMIT_ALL` is the one-line prefix a fresh interpreter runs to test
every carried lever on the pinned version (registry.TESTED_ON) — the activation-mechanics tests use it; the pending-rebase tests do not."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

TREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
with open(os.path.join(TREE, "stock", "PINS.json")) as _fh:
    PINNED = json.load(_fh)["upstream"]["version"]                  # the tree's stock pin: the one place the tests read the upstream version
ADMIT_ALL = (f"import opendde_opt.registry as _R, opendde_opt.stack as _S; _R.TESTED_ON[{PINNED!r}] = frozenset(_R.LEVERS); _R.LINES_ON_HOLD.clear(); "
           "_S.stack_mismatch = lambda tree=None: None\n")             # test-only, this interpreter: every lever tested on the pin and the box's stack taken as the pin's


def make_site(root: str, version: str = PINNED, with_runner: bool = True) -> str:
    """Create `<root>/site` holding the stub packages and dist-info; returns the site path (put it FIRST on sys.path / PYTHONPATH)."""
    site = os.path.join(root, "site")
    os.makedirs(os.path.join(site, "opendde", "model"), exist_ok=True)
    open(os.path.join(site, "opendde", "__init__.py"), "w").write("")
    open(os.path.join(site, "opendde", "version.py"), "w").write(f'__version__ = "{version}"\n')
    open(os.path.join(site, "opendde", "model", "__init__.py"), "w").write("")
    open(os.path.join(site, "opendde", "model", "opendde.py"), "w").write(                   # the surface the memory mode's census unit and the offload / XL
        "class OpenDDE:\n"                                                                    # units patch: the pin's inference stage methods (opendde.py @1.1.1), as no-ops
        "    def __init__(self, *a, **kw):\n        pass\n"
        + "".join(f"    def {m}(self, *a, **kw):\n        return None\n" for m in (
            "run_sample_diffusion_stage", "prepare_diffusion_cache_for_sampling", "expand_to_structural_tokens", "get_pairformer_output",
            "select_pair_output_branch", "run_confidence_head_stage", "run_confidence_head", "compute_distogram_contact_probs",
            "_bound_pairformer_chunk_size", "_main_inference_loop", "forward")))
    os.makedirs(os.path.join(site, "opendde", "model", "triangular"), exist_ok=True)         # the classes the kit's FPF adapter re-binds
    open(os.path.join(site, "opendde", "model", "triangular", "__init__.py"), "w").write("")
    open(os.path.join(site, "opendde", "model", "triangular", "triangular.py"), "w").write(textwrap.dedent("""
        class TriangleMultiplicationOutgoing:
            def forward(self, z, mask=None, **kw):
                return ("stock_out", z)
        class TriangleMultiplicationIncoming:
            def forward(self, z, mask=None, **kw):
                return ("stock_in", z)
        """))
    di = os.path.join(site, f"opendde-{version}.dist-info")
    os.makedirs(di, exist_ok=True)
    open(os.path.join(di, "METADATA"), "w").write(f"Metadata-Version: 2.1\nName: opendde\nVersion: {version}\n")
    open(os.path.join(di, "RECORD"), "w").write("")
    if with_runner:
        os.makedirs(os.path.join(site, "runner"), exist_ok=True)
        open(os.path.join(site, "runner", "__init__.py"), "w").write(textwrap.dedent("""
            import sys
            SHIM_LOADED_DURING_BODY = "odde_served_levers" in sys.modules      # the finder must fire after this body, never before
            """))
        open(os.path.join(site, "runner", "inference.py"), "w").write(textwrap.dedent("""
            class InferenceRunner:
                def __init__(self, configs=None):
                    self.configs = configs
                    self.model = object()
            """))
        open(os.path.join(site, "runner", "cli.py"), "w").write(textwrap.dedent("""
            def opendde_cli():                                       # the console entry (runner.cli:opendde_cli); the tests that call it replace this file
                pass
            """))
        open(os.path.join(site, "runner", "batch_inference.py"), "w").write(textwrap.dedent("""
            from runner.cli import opendde_cli                       # upstream re-exports the same group object here
            from runner.inference import InferenceRunner
            def get_default_runner(**kw):
                return InferenceRunner(kw)
            """))
    return site


def run_py(code: str, env: dict | None = None, cwd: str | None = None, pythonpath: list[str] | None = None) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter of this venv (the .pth autoload installed), with PYTHONPATH prepended."""
    e = dict(os.environ)
    for k in list(e):
        if k.startswith(("ODDE_", "OPENDDE_OPT", "DIT_", "CUEQ_", "CUBLAS_", "PYTORCH_CUDA")):
            e.pop(k)
    e.update(env or {})
    e["MODEL_OPT"] = TREE
    e["PYTHONDONTWRITEBYTECODE"] = "1"                        # the kit tree stays as carried (no __pycache__ under opt/forward)
    e.pop("PYTHONPATH", None)                                 # nothing of the tree on the path unless the test says so
    if pythonpath:
        e["PYTHONPATH"] = os.pathsep.join(pythonpath + ([e["PYTHONPATH"]] if e.get("PYTHONPATH") else []))
    return subprocess.run([sys.executable, "-c", code], env=e, cwd=cwd, capture_output=True, text=True, timeout=120)


def fake_stock_cli(args) -> int:
    """A stand-in for cli.run_stock_cli in the CPU tests: writes the structure files the arguments owe (`-i` items x `--seeds` x
    `--sample`, the CLI route's layout) under `-o` and returns 0 — a run that produced its outputs."""
    import json as _json
    def last(*flags):
        v = None
        for i, a in enumerate(args[:-1]):
            if a in flags:
                v = args[i + 1]
        return v
    out, q = last("-o", "--out_dir"), last("-i", "--input")
    seeds = [s for s in str(last("--seeds") or "101").split(",") if s]
    n = int(last("--sample", "-e") or 5)
    for j in _json.load(open(q)):
        for s in seeds:
            d = os.path.join(out, j["name"], f"seed_{s}", "predictions"); os.makedirs(d, exist_ok=True)
            for k in range(n):
                open(os.path.join(d, f"{j['name']}_seed_{s}_sample_{k}.cif"), "w").write("data_stub\n")
    return 0


def weights_root(dirpath: str) -> str:
    """A frozen-weights root for `pred` tests: every file the boot gate requires (frozen.REQUIRED_FILES) present under ``dirpath`` (stub
    bytes). A test that reaches `pred`'s completeness accounting owns its root: it never depends on the operator's OPENDDE_ROOT_DIR."""
    from opendde_opt import frozen
    sizes = frozen.managed_sizes(TREE)                               # upstream size-checks the CCD files: the stubs are sparse files of exactly those sizes
    for rel in frozen.REQUIRED_FILES:
        p = os.path.join(dirpath, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p):
            with open(p, "wb") as fh:
                if rel in sizes:
                    fh.truncate(sizes[rel])                          # sparse: no blocks written
                else:
                    fh.write(b"x")
    return dirpath


TEARDOWN_SIGSEGV = -11          # a CPU torch+triton interpreter can SIGSEGV at exit AFTER the kit printed its verdict (image-of-record boxes without a GPU)


def verdict_ok(r, line: str, rc: int = 0) -> bool:
    """The subprocess printed ``line`` on stdout AND exited ``rc`` — or died with the interpreter-teardown SIGSEGV after printing it (torch +
    triton CPU teardown, rc -11: the verdict line is the evidence; the crash after it is the interpreter's, named here, not the kit's)."""
    if line not in r.stdout:
        return False
    return r.returncode == rc or r.returncode == TEARDOWN_SIGSEGV

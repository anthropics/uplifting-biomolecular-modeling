"""A stand-in stack for the package's CPU tests: a temporary site directory holding the stand-in `colabdesign` + `jax` of tests/stubsite/
(the tests' recording design surface `colabdesign_opt.tests.recipe` installed on a stand-in `_af_design`, so BindCraft's own design step runs
on it end to end: deterministic logs, a call transcript, `save_pdb` writing real geometry sliced from BindCraft's example PDB so its biopython
checks and DSSP run) and the dist-info the pins read; helpers for the tree's paths, a synthetic target PDB, the AF2 params layout, a child
process environment. Nothing here is imported by the package itself.
"""
import importlib
import importlib.util
import json
import os
import shutil
import sys
from typing import Optional

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # opt/colabdesign_opt
OPT_DIR = os.path.dirname(PKG_DIR)                                                # opt/
TREE = os.path.dirname(OPT_DIR)                                                   # colabdesign/
PALLAS = os.path.join(OPT_DIR, "forward", "af2_pallas_flash")
PINS = os.path.join(TREE, "stock", "PINS.json")
GPU = {"name": "NVIDIA H100 80GB HBM3", "memory_mib": 81559, "cc": "9.0"}


BINDCRAFT = os.path.join(TREE, "stock", "src", "bindcraft")
EXAMPLE_PDB = os.path.join(BINDCRAFT, "example", "PDL1.pdb")                      # BindCraft's own example target: the tests' case and the stand-in's structure source
DSSP = os.path.join(BINDCRAFT, "functions", "dssp")                              # BindCraft's DSSP executable: not carried in the tree, fetched by `run.sh install` (stock/PINS.json upstream.bindcraft.fetched)
SKIP_DSSP = ("BindCraft's design step runs its DSSP executable (settings optimise_beta), which this tree does not carry: "
             "`bash run.sh install` (or `python -I stock/check_pins.py --fetch`) fetches it into stock/src/bindcraft/functions/ first")


def dssp_present() -> bool:
    """The install-fetched DSSP executable is in place and executable (an installed tree) — every test whose arm runs BindCraft's design step
    needs it: the step refuses to start without it and runs it after stage 1 (settings optimise_beta)."""
    return os.path.isfile(DSSP) and os.access(DSSP, os.X_OK)


BINDCRAFT_IMPORTS = ("numpy", "pandas", "scipy", "matplotlib", "Bio")                    # what BindCraft's design step imports at module level (generic_utils / biopython_utils / colabdesign_utils)


def bindcraft_stack_missing() -> list:
    """The packages of BINDCRAFT_IMPORTS this interpreter lacks ([] = the driver tests can run BindCraft's code): `pip install -e opt[test]`."""
    return [m for m in BINDCRAFT_IMPORTS if importlib.util.find_spec(m) is None]


SKIP_BINDCRAFT = f"BindCraft's design step imports {', '.join(BINDCRAFT_IMPORTS)}; missing here: pip install -e opt[test]"


def tree_present() -> bool:
    return os.path.isfile(PINS) and os.path.isfile(os.path.join(BINDCRAFT, "functions", "colabdesign_utils.py"))


def nvidia_smi_stub(tmp: str, gpu: dict = None) -> str:
    """A stand-in `nvidia-smi` under `<tmp>/bin` answering `--query-gpu=<fields>` in the requested field order (name, memory.total,
    compute_cap) for the stub GPU; returns the directory to prepend to PATH."""
    gpu = GPU if gpu is None else gpu
    d = os.path.join(tmp, "bin"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "nvidia-smi")
    fields = {"name": gpu.get("name", ""), "memory.total": gpu.get("memory_mib", ""), "compute_cap": gpu.get("cc", "")}
    with open(p, "w") as fh:
        fh.write(f"#!{sys.executable}\nimport sys\nF = {fields!r}\n"
                 "q = [a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--query-gpu=')]\n"
                 "names = q[0].split(',') if q else ['name', 'memory.total', 'compute_cap']\n"
                 "print(', '.join(str(F[n.strip()]) for n in names))\n")
    os.chmod(p, 0o755)
    return d


def pins() -> dict:
    with open(PINS, "r", encoding="utf-8") as fh:
        return json.load(fh)


STUBSITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stubsite")   # the stand-in colabdesign + jax as files (copied into each test's site dir)


def _dist_info(root: str, name: str, version: str, direct_url: dict = None, record: list = None):
    d = os.path.join(root, f"{name.replace('-', '_')}-{version}.dist-info")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "METADATA"), "w") as fh:
        fh.write(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    if direct_url is not None:
        with open(os.path.join(d, "direct_url.json"), "w") as fh:
            json.dump(direct_url, fh)
    if record is not None:
        with open(os.path.join(d, "RECORD"), "w") as fh:
            fh.write("".join(f"{r},,\n" for r in record))
    return d


def make_stub_stack(tmp: str, commit: str = None, versions: dict = None, with_record: bool = False) -> str:
    """A site directory with the stand-in `colabdesign` + `jax` (tests/stubsite/) and the dist-info the pins read; returns its path (put it first
    on sys.path / PYTHONPATH). `commit` defaults to the pinned commit; `versions` overrides the stack versions."""
    p = pins()
    commit = commit or p["upstream"]["colabdesign"]["commit"]
    v = {"colabdesign": p["upstream"]["colabdesign"]["version"], "jax": p["pins"]["jax"], "jaxlib": p["pins"]["jaxlib"], "dm-haiku": p["pins"]["dm-haiku"],
         "numpy": p["pins"]["numpy"], "jax-cuda12-plugin": p["pins"]["jax-cuda12-plugin"]}
    v.update(versions or {})
    site = os.path.join(tmp, "site")
    if os.path.isdir(site):
        shutil.rmtree(site)
    shutil.copytree(STUBSITE, site, ignore=shutil.ignore_patterns("__pycache__", "README.md"))
    files = sorted(os.path.relpath(os.path.join(d, f), site) for d, _, fs in os.walk(site) for f in fs)
    _dist_info(site, "colabdesign", v["colabdesign"], {"url": "https://github.com/sokrypton/ColabDesign.git",
                                                       "vcs_info": {"commit_id": commit, "requested_revision": commit, "vcs": "git"}},
               record=files if with_record else None)
    for name in ("jax", "jaxlib", "dm-haiku", "numpy", "jax-cuda12-plugin"):
        _dist_info(site, name, v[name])
    importlib.invalidate_caches()
    return site


def make_params(root: str, models=(1, 2, 3, 4, 5)) -> str:
    """<root>/params/params_model_<k>_multimer_v3.npz as empty files (the driver refuses by name when a design model's params file is absent;
    the stand-in never reads them); returns <root>."""
    d = os.path.join(root, "params"); os.makedirs(d, exist_ok=True)
    for k in models:
        open(os.path.join(d, f"params_model_{k}_multimer_v3.npz"), "a").close()
    return root


def write_target(path: str, chains=("A",), n_res: int = 108) -> str:
    """A PDB with `n_res` CA atoms per chain (the token estimate = n_res * len(chains) + binder length)."""
    lines = []
    i = 1
    for c in chains:
        for r in range(1, n_res + 1):
            lines.append(f"ATOM  {i:5d}  N   ALA {c}{r:4d}    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           N")
            lines.append(f"ATOM  {i+1:5d}  CA  ALA {c}{r:4d}    {1.5:8.3f}{0.0:8.3f}{float(r):8.3f}  1.00 20.00           C")
            i += 2
        lines.append("TER")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\nEND\n")
    return path


class StubState:
    """Save/restore the package's process state and the environment around a test."""

    def __init__(self):
        from colabdesign_opt import stack
        self.stack = stack
        self.env = dict(os.environ)
        self.path = list(sys.path)
        self.mods = set(sys.modules)

    def restore(self):
        os.environ.clear(); os.environ.update(self.env)
        sys.path[:] = self.path
        for m in list(sys.modules):
            if m not in self.mods and (m.startswith("colabdesign") and not m.startswith("colabdesign_opt") or m == "jax" or m.startswith("jax.")):
                del sys.modules[m]
        self.stack.reset_for_tests()
        importlib.invalidate_caches()


def host_nvidia_smi() -> Optional[str]:
    """The host's own nvidia-smi on PATH, if any: the no-GPU refusal tests need a host without one (the stand-in is removed from PATH, a real one cannot be)."""
    return shutil.which("nvidia-smi")


def child_env(site: str, **extra) -> dict:
    """An environment for a CLI/launcher test: the stub site and the package on PYTHONPATH, no package or kit variable and no params root
    inherited from the host (COLABDESIGN_PARAMS_DIR: the tests pass --params-dir or test the default root)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COLABDESIGN_OPT", "COLABDESIGN_PARAMS_DIR", "AF2M_", "AF_PALLAS_", "MODEL_OPT", "KEEP_XLA", "JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_"))}
    env["PYTHONPATH"] = os.pathsep.join([site, OPT_DIR] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("STUB_EXAMPLE_PDB", EXAMPLE_PDB)
    env["XDG_CACHE_HOME"] = os.path.join(os.path.dirname(os.path.abspath(site)), "xdg")   # lever compilecache's keyed default directory (and stack.py's digest memo): under the test's tmp, never ~/.cache
    env.update(extra)
    return env


# A child interpreter that runs one arm exactly as `python -s -m colabdesign_opt.<arm>_launch <argv>` does (launch.main) and then, test-side,
# writes what the tests inspect and the package itself never writes: the observer's rows (units.rows()), what the one optional step
# callback received (set before the arm starts: [step, state] per call), the XLA_FLAGS in force at exit, and the run records — the return
# value of every bindcraft.run_design call the design script made (captured by wrapping it here; keyed by seed, in run order). The first
# run's end row (units.end_row over its terminate verdict) is appended to the rows. Importing colabdesign_opt.bindcraft here loads names /
# settings / units only: it adds no module the arm's clean-process proof flags (no lever, core or upstream module).
ARM_PROBE = ("import json, os, sys\n"
             "from colabdesign_opt import bindcraft, launch, units\n"
             "arm, probe, argv = sys.argv[1], sys.argv[2], sys.argv[3:]\n"
             "seen, runs = [], {}\n"
             "units.set_step_callback(lambda step, state: seen.append([step, state]))\n"
             "_run_design = bindcraft.run_design\n"
             "def run_design(prep, **kw):\n"
             "    run = _run_design(prep, **kw)\n"
             "    runs[str(kw['seed'])] = run\n"
             "    return run\n"
             "bindcraft.run_design = run_design\n"
             "rc = launch.main(arm, argv)\n"
             "rows = units.rows()\n"
             "if runs:\n"
             "    rows = rows + [units.end_row(next(iter(runs.values()))['terminate'], rows)]\n"
             "json.dump({'rows': rows, 'callback': seen, 'xla_flags': os.environ.get('XLA_FLAGS'), 'runs': runs}, open(probe, 'w'), default=str)\n"
             "sys.exit(rc)\n")


def arm_cmd(arm: str, probe: str, launcher_argv):
    """argv for an arm run through ARM_PROBE: [python, -s, -c, ARM_PROBE, arm, probe, *launcher_argv]."""
    import sys as _sys
    return [_sys.executable, "-s", "-c", ARM_PROBE, arm, probe, *launcher_argv]


def read_probe(probe: str) -> dict:
    import json as _json
    with open(probe) as fh:
        return _json.load(fh)


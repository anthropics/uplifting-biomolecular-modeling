"""Test scaffolding: a pinned `openfold3` distribution stub (metadata only) and a fresh package state, so the CPU tests exercise the
package's resolution, gates and hook route without openfold3 or torch installed."""
import importlib
import os
import sys
import tempfile
import types

PIN = "0.4.1"
HOOK_TARGETS = ("openfold3.core.model.primitives.linear", "openfold3.core.model.structure.diffusion_module", "openfold3.core.model.latent.pairformer",
                "openfold3.projects.of3_all_atom.model")


def tree_home():
    import openfold3_opt.env as env
    return env.tree_home()


def core_dir():
    """Wherever `opt_core` is importable from in THIS process — however it got there (`pip install -e`, or a raw PYTHONPATH entry,
    as in a CPU test run). A subprocess a test spawns inherits neither this process's sys.path nor its import machinery, only the
    environment it is given, so a subprocess PYTHONPATH that omits this drops the core silently whenever it reached this process
    via PYTHONPATH rather than an install."""
    import opt_core
    return os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))


def subprocess_pythonpath(home, *extra):
    """PYTHONPATH for a test that spawns a subprocess: `<home>/opt` (this package) plus `core_dir()` plus any extra dirs given, in
    that order. The common-case builder for such tests — do not construct one ad hoc alongside this; when a dir must come BEFORE
    `<home>/opt` (e.g. a stub meant to shadow a real install), compose with `core_dir()` directly instead."""
    return os.pathsep.join([os.path.join(home, "opt"), core_dir(), *extra])


def stub_dist(version=PIN):
    """A site directory carrying `openfold3-<version>.dist-info/METADATA` so importlib.metadata reports the pin; returns the dir."""
    d = tempfile.mkdtemp(prefix="of3stub_")
    di = os.path.join(d, f"openfold3-{version}.dist-info")
    os.makedirs(di)
    with open(os.path.join(di, "METADATA"), "w") as fh:
        fh.write(f"Metadata-Version: 2.1\nName: openfold3\nVersion: {version}\n")
    with open(os.path.join(di, "RECORD"), "w") as fh:
        fh.write("")
    sys.path.insert(0, d)
    importlib.invalidate_caches()
    return d


def unstub_dist(d):
    if d in sys.path:
        sys.path.remove(d)
    importlib.invalidate_caches()


def reset_package():
    """Fresh activation state: drop the package's modules, the kit hook finders and the kit hook dirs from sys.path/sys.modules."""
    import openfold3_opt.env as env
    for f in list(sys.meta_path):
        if type(f).__name__ in ("_LeverFinder", "_Finder", "_CounterFinder", "Finder", "_PostImportFinder", "_TemplGuardFinder"):
            sys.meta_path.remove(f)
    for d in list(sys.path):
        if any(os.path.abspath(d).startswith(os.path.abspath(k)) for k in env.kit_dirs()):
            sys.path.remove(d)
    for n in list(sys.modules):
        if n == "openfold3" or n.startswith("openfold3.") or n.startswith("openfold3_opt") or n in ("of3_fastinit", "of3_graphs", "of3t_levers"):
            del sys.modules[n]
    for k in list(os.environ):
        if k.startswith(("OF3_", "OF3T_", "OF3FPF_", "OF3FLASHPF_", "OF3TP_", "BFTP_", "OPENFOLD3_OPT")) or k in ("CUBLAS_WORKSPACE_CONFIG",):
            del os.environ[k]
    import openfold3_opt  # noqa: F401
    return importlib.import_module("openfold3_opt")


def stub_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def stub_model_module():
    class OpenFold3:
        def __init__(self, *a, **k):
            self.built = True

        def forward(self, batch):
            return batch
    return stub_module("openfold3.projects.of3_all_atom.model", OpenFold3=OpenFold3)


def pin_for_pickling(globals_dict):
    """Register the calling test module in sys.modules under its own name before multiprocessing pickles its top-level functions by reference
    (fresh() drops every openfold3_opt.* module, this package's test modules included; pickle then imports a second copy and refuses the first's
    functions as 'not the same object')."""
    import types
    name = globals_dict["__name__"]
    mod = sys.modules.get(name)
    if mod is None or mod.__dict__ is not globals_dict:
        m = types.ModuleType(name); m.__dict__.update(globals_dict); sys.modules[name] = m


def caller_yaml(dst_dir, seeds=(42, 66, 101, 2024, 8888), num_recycles=10, name="own_predict.yml"):
    """A caller's own runner yaml for `--runner-yaml`: the kernels-off configuration (modes.KERNELS_OFF_YAML) plus a seed LIST
    (experiment_settings.seeds) and a recycle count (model_update.custom.architecture.shared.num_recycles). Returns its path."""
    import yaml
    from openfold3_opt import modes
    doc = yaml.safe_load(open(os.path.join(tree_home(), modes.KERNELS_OFF_YAML), encoding="utf-8"))
    doc.setdefault("model_update", {}).setdefault("custom", {}).setdefault("architecture", {}).setdefault("shared", {})["num_recycles"] = int(num_recycles)
    doc["experiment_settings"] = {"seeds": list(seeds)}
    os.makedirs(str(dst_dir), exist_ok=True)
    path = os.path.join(str(dst_dir), name)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, default_flow_style=False, sort_keys=False)
    return path

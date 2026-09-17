"""The self-containment property of the release tree, proven statically: every kit stands on its own stock dependencies + opt_core,
opt_core stands on the standard library, opt_core's generic modules parse and run under the Python floor, nothing in opt_core is named
after an engine, and no kit installs a top-level name outside its own namespace.

Pure Python, ast-based; imports nothing of the scanned code (no kit, no opt_core, no framework); runs on Python >= 3.8 itself (no
ast.unparse; tomllib / tomli when importable, else the line reader). Two forms:

    python -m pytest common/opt_core/tests/test_selfcontained.py          # the assertions (part of the core's CPU suite)
    python common/opt_core/tests/test_selfcontained.py [<tree>] [--census] [--json <out.json>]
                                                                          # exit 1 on any violation; --census adds the name censuses; the JSON is the full record

<tree> is the release tree (the directory holding common/opt_core): this file's own tree by default, or MODEL_OPT, or
SELFCONTAINED_TREE for the pytest form. The kit set, the kit package names and the engine names are DERIVED from the tree — every
<engine>/opt/ beside common/; a kit package is a directory directly under <engine>/opt/ holding __init__.py plus every package or module
its opt/pyproject.toml installs — nothing engine-named is listed in this file except today's KNOWN offenders (lists that only shrink:
an entry whose offence is gone fails the test until the entry is deleted).

Assertions:
  (i)    no kit imports another kit's package: `import <pkg>`, `from <pkg> import ...`, importlib.import_module / __import__ / find_spec
         with a literal naming <pkg>, and no kit's opt/pyproject.toml declares another kit's project or package as a dependency
         ([project] dependencies / optional-dependencies); a `sys.path.insert|append(...)` expression naming another engine's
         directory is a violation (`syspath_cross`);
  (ii)   opt_core imports no kit package (same forms);
  (iii)  opt_core modules import nothing from HEAVY (torch, triton, jax, numpy ...) at module top level unless the file is a carried
         kernel file (the live file set of a kernels/META/*.json name, see carried_files() below) or a FRAMEWORK_EXTRA row (files a
         kit imports only when it calls them); module-level if/try/with/for
         bodies count as top level, `if TYPE_CHECKING:` does not, function and class bodies do not (the lazy pattern);
  (iv)   every kit that imports opt_core declares a pin ([tool.opt_core] path / version in <engine>/opt/pyproject.toml — version the
         MINIMUM core version the kit needs; equality is the git commit, not a restated hash, so no digest is compared here);
  (v)    the Python floor: every opt_core/**/*.py parses with ast feature_version=PY_FLOOR (3.8); a FRAMEWORK_EXTRA row may
         declare a higher floor (its modules refuse by name behind their lazy import on an older interpreter); a module that evaluates a
         `X | Y` annotation or a subscripted builtin (`list[int]`) at run time — i.e. without `from __future__ import annotations` — is a
         violation at its floor (TypeError below 3.10 / 3.9); and the 3.9+/3.10+ standard-library calls that parse under 3.8 but fail there
         (str.removeprefix/removesuffix, functools.cache, ast.unparse, math.lcm, zoneinfo, graphlib) are violations in a 3.8-floor file.
         A dict union `a | b` is not statically typed here: reviewers hold it;
  (vi)   no file or directory under common/opt_core/ is named after an engine (word match against the derived engine names) — except the
         KNOWN_ENGINE_NAMED entries, and no research script (sweep_* / diag_* / exp_* / debug_* / bench_* / export_*) ships in the package —
         except the KNOWN_RESEARCH_SCRIPTS entries;
  (vii)  every top-level name a kit INSTALLS (opt/pyproject.toml: [tool.setuptools] packages / py-modules, [tool.setuptools.packages.find]
         include or auto-discovery, *.pth files under opt/) is prefixed by the kit's namespace (the engine directory name or the root of its
         [project] name), and no two kits install the same top-level name; AND every name the kit makes importable at RUN time — the
         directories and modules directly under opt/ (the editable install's path entry) and the entries of every implied sys.path root
         inside opt/ (the parent of a package-hierarchy root), confirmed by the kit's own top-level imports — is namespaced the same way or
         private (`_...`); a generic name (`engines`, `compare`, `kits`, ...) collides with another kit in one process and is a violation
         (`generic_toplevel`, reported per kit with its evidence; today's names are listed in KNOWN_GENERIC_TOPLEVEL, which only shrinks). A plain directory without __init__.py that no kit file imports by name
         (the opt/forward/ layout directory) is layout, not an import name: `inert`, reported, never failed; a third-party distribution
         declared in the VENDOR.json of the directory that holds it (contents: <name>-<ver>.dist-info, source sha256, licence) keeps its
         upstream import name: `vendored`, reported, never failed. The carried directories a kit puts on sys.path are declared in
         KNOWN_RUNTIME_INJECTIONS (census).
Censuses (reported, never failed; --census / --json): other-engine NAMES inside a kit's files, classified by where they occur
(`mentions`); each kit's importable top-level names (`toplevel`, the record behind (vii)) with the generic ones two kits share (`shared_generic`).

"""
from __future__ import annotations

import ast
import fnmatch
import glob
import json
import os
import re
import sys

HEAVY = ("torch", "triton", "jax", "jaxlib", "tensorflow", "transformer_engine", "flash_attn", "xformers", "numpy", "cupy", "cuequivariance_torch",
         "deepspeed", "einops", "scipy", "pandas")
# FRAMEWORK files MAY import HEAVY at top level because a kit imports them only when it calls them: the CARRIED kernel files — the
# live file set of every opt_core/kernels/META/<name>.json name (see carried_files() below), never listed here — plus the rows below (globs relative to
# opt_core/, with the syntax floor the file is held to; None = PY_FLOOR). Every other module under opt_core/ is standard-library-at-import
# and 3.8. Owners add rows through CORE.
FRAMEWORK_EXTRA = (
    ("ops/msa_opm/**", (3, 8)),      # the fused MSA outer-product-mean cell: torch + triton at module top; a kit imports it only when its lever engages
    ("ops/msa_pwa/**", (3, 8)),      # the fused MSA pair-weighted-averaging cell (fast class): same
    ("ops/msa_pwa2/**", (3, 8)),     # the exact-replica MSA pair-weighted-averaging cell: same
    ("ops/msa_fused/msa_triton.py", None),     # Triton kernels of the fused MSA-module ops (ln_linear, opm_out, pwa_ln_vg, pwa_out2), moved verbatim from the producing kit
                                              #   tree (one shared copy; kits import the module by path when the lever engages): torch / triton at module top, as @triton.jit requires
)
PY_FLOOR = (3, 8)                    # the syntax + stdlib floor of every other opt_core/**/*.py (an engine's tested interpreter is 3.8)
BUILTIN_GENERICS = ("list", "dict", "tuple", "set", "frozenset", "type")
LIB_ABOVE_FLOOR = {                  # name -> first Python that has it: calls that parse under 3.8 and fail there
    "attr:removeprefix": "3.9", "attr:removesuffix": "3.9", "functools.cache": "3.9", "ast.unparse": "3.9", "math.lcm": "3.9",
    "module:zoneinfo": "3.9", "module:graphlib": "3.9", "functools.cached_property": None,   # cached_property is 3.8: listed to document it is allowed
}
CORE_REL = os.path.join("common", "opt_core")
CORE_PKG = "opt_core"
NOT_KITS = ("common", "tools")
RESEARCH_GLOBS = ("sweep_*.py", "diag_*.py", "exp_*.py", "debug_*.py", "bench_*.py", "export_*.py")

# ---- KNOWN offenders: today's debt, named. Each list ONLY SHRINKS: a listed entry whose offence is gone fails until it is deleted here.
KNOWN_ENGINE_NAMED = (               # (vi) files under common/opt_core/ named after an engine — none today
)
KNOWN_RESEARCH_SCRIPTS = (           # (vi) research scripts (sweep_* / diag_* / exp_* / debug_* / bench_* / export_*) inside the shipped package — none today
)
KNOWN_SHARED_TOPLEVEL = (            # (vii) top-level names two kits install today: "<name>: <engine>,<engine>"
)
KNOWN_UNNAMESPACED = (               # (vii) installed top-level names outside the kit's namespace today: "<engine>: <name>"
)
KNOWN_RUNTIME_INJECTIONS = {         # (vii, census only) carried directories a kit puts on sys.path at run time (never installed): engine -> [(what, cite)]
    "af3_jax": [("model process only (the fork interpreter): af3_pallas_levers (sys.path += opt/forward/pallas_addon/patches via levers_launch.py); "
                 "af3_flashpairformer (sys.path += opt/forward/flashpairformer, own launcher); the fork repo dir as cwd/__main__ (run_alphafold(_fast).py); "
                 "af3_jax_opt.inprocess.{dattn,templates} loaded by file path; fpf_launch imported by levers_launch (script dir on sys.path)",
                 "af3_jax_opt levers_launch.py / fpf_launch.py")],
    "chai1": [("opt/forward/fast_inference/kit bare driver modules (chai_proto, chai_worker, chai_multiseed, chai_graphs, compare_probes, compare_runs, compare_runs_lib, "
               "diag_denoiser, diag_graphs, diag_trunk_rng, ipsae_lite, selftest_report, stock_fold; errata_02/chai_worker)", "chai1_opt driver.py:82, cli.py:119"),
              ("opt/forward/eager_trunk package chai1_eager (+ jobs/, scripts/ namespace dirs); opt/forward/dstep_megakernel package chai1_fastln (+ jobs/, analysis/)", "chai1_opt driver"),
              ("common/opt_core appended to sys.path by _core.ensure_importable when the core is not installed", "chai1_opt _core.py")],
    "colabfold": [("opt/forward/af2_pallas_flash/af2_pallas_flash on sys.path at activation in fast: bare modules af2_pallas_attn, af2_flash_pallas (the carried kit's own names)",
                   "colabfold/opt/colabfold_opt/stack.py activate")],
    "opendde": [("sitecustomize x4 (levers/{ACCEL,ARMT,XL,OFFLOAD}), src/{fpf_trimul,fpf_engines}, triatt_impls, odde_* + odde_arm_t package, KIT tools (analyze_gate, d59_compare, host_cpu, "
                 "jobs_tg_tune_cache, lazy_state_check, rows_to_harness, run_verdicts, score_interface, tuned_bitwise, verify_patched_tree), DITFAST tools (capture_recovery, deadskip_odde)",
                 "opendde_opt stack / the levers' sitecustomize dirs")],
    "openfold3": [("sitecustomize hook dirs; of3_fastinit, of3_graphs, of3_offload, of3o_confidence, ptx_tp (pkg), "
                   "of3t_census, of3t_flash_triattn (shadowed by the core route under fast), of3t_levers, of3t_paircache, of3t_realcert, of3flashpf (pkg): carried add-on "
                   "modules on sys.path at activation, never installed", "openfold3_opt modes.KITS / the hook dirs' sitecustomize")],
    "protenix_v1": [("v05_addon/{lib,lib/kit112_src,ptxfpf} on sys.path when activated: levers_ptx1 levers_ptx05 bench bwcheck opnum optimer tune, detpatch fastln_prebuilt "
                     "ptx_flash_triattn_patch ptx_trunk_graphs kit112_src, biascache_static dit_hoist dit_levers infopt_graphs (carried add-on modules import each other by bare name)",
                     "protenix_v1_opt kit.py / stack.py")],
    "protenix_v2": [("the activation-time env.sh sys.path list (~45 bare names incl. sitecustomize, fpf, fpf_*, flash_triattn, ptx_*, infopt_graphs, dit_hoist)",
                     "protenix_opt stack / opt/forward env.sh")],
    "rosettafold3": [("opt/forward/rf3_fpf_trimul_addon/rf3fpf on sys.path under any FPF-arm row: bare module fpf_rf3_adapter (+ add-on tooling modules imported by nothing); "
                      "renaming = editing carried bytes", "rosettafold3_opt stack.fpf_apply / modes.FPF_SYS_PATH")],
}


# ----------------------------------------------------------------------------------------------------------------- the tree
def find_tree(start=None):
    env = os.environ.get("MODEL_OPT")
    if env and os.path.isdir(os.path.join(env, CORE_REL)):
        return os.path.abspath(env)
    d = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    while True:
        if os.path.isdir(os.path.join(d, CORE_REL)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise SystemExit(f"no release tree (a directory holding {CORE_REL}) at or above {start or __file__}; pass it or set MODEL_OPT")
        d = parent


def _norm(name):
    """PEP 503 name normalisation (case, runs of -_. to one -)."""
    return re.sub(r"[-_.]+", "-", str(name).strip().lower())


def kits(tree):
    """{engine: {"opt", "packages": [dirs under opt/ with __init__.py], "pyproject": {...}, "installs": [top-level names], "project": name}}."""
    out = {}
    for name in sorted(os.listdir(tree)):
        opt = os.path.join(tree, name, "opt")
        if name in NOT_KITS or name.startswith(".") or not os.path.isdir(opt):
            continue
        pkgs = sorted(p for p in os.listdir(opt) if os.path.isfile(os.path.join(opt, p, "__init__.py")))
        pp = read_pyproject(os.path.join(opt, "pyproject.toml"))
        out[name] = {"opt": opt, "packages": pkgs, "pyproject": pp, "project": (pp.get("project") or {}).get("name"),
                     "installs": installed_toplevel(opt, pp)}
    return out


SKIP_DIRS = ("__pycache__", ".git", "build", "dist", ".pytest_cache")


def py_files(root):
    for d, dns, fns in os.walk(root):
        dns[:] = sorted(x for x in dns if x not in SKIP_DIRS and not x.endswith(".egg-info"))
        for f in sorted(fns):
            if f.endswith(".py"):
                yield os.path.join(d, f)


def all_files(root):
    for d, dns, fns in os.walk(root):
        dns[:] = sorted(x for x in dns if x not in SKIP_DIRS and not x.endswith(".egg-info"))
        for x in list(dns) + sorted(fns):
            yield os.path.join(d, x)


# ----------------------------------------------------------------------------------------------------------------- pyproject
def read_pyproject(path):
    """The kit's pyproject.toml as a dict (tomllib / tomli when importable, else the line reader for the tables this test needs)."""
    if not os.path.isfile(path):
        return {}
    for mod in ("tomllib", "tomli"):
        try:
            lib = __import__(mod)
            with open(path, "rb") as fh:
                return lib.load(fh)
        except ModuleNotFoundError:
            continue
        except Exception:  # noqa: BLE001  an unparseable toml: the line reader names what it can
            break
    return _toml_lines(path)


_ARR = re.compile(r'"([^"]*)"|\'([^\']*)\'')


def _toml_lines(path):
    """A reader for the flat tables and string / string-array values this test needs (no inline tables, no nested arrays)."""
    doc, table, key, buf = {}, {}, None, None
    doc["__root__"] = table
    for raw in open(path, encoding="utf-8"):
        ln = raw.split("#", 1)[0].rstrip() if raw.count('"') % 2 == 0 else raw.rstrip()
        s = ln.strip()
        if buf is not None:
            buf += " " + s
            if "]" in s:
                table[key] = [a or b for a, b in _ARR.findall(buf)]; buf = None
            continue
        if not s:
            continue
        if s.startswith("[") and s.endswith("]") and not s.startswith("[["):
            table = doc
            for part in s.strip("[]").split("."):
                table = table.setdefault(part.strip(), {})
            continue
        if "=" in s:
            key, val = [x.strip() for x in s.split("=", 1)]
            if val.startswith("["):
                if "]" in val:
                    table[key] = [a or b for a, b in _ARR.findall(val)]
                else:
                    buf = val
            elif val.startswith(('"', "'")):
                table[key] = val.strip("\"'")
            else:
                table[key] = val
    return doc


def installed_toplevel(opt, pp):
    """The top-level importable names `pip install <opt>` puts on site-packages: setuptools packages / py-modules / packages.find, or
    auto-discovery (every package root directly under opt/ but tests), plus every *.pth under opt/ (two levels)."""
    st = (pp.get("tool") or {}).get("setuptools") or {}
    names = set()
    pk = st.get("packages")
    if isinstance(pk, list):
        names |= {p.split(".")[0] for p in pk}
    elif isinstance(pk, dict) and isinstance(pk.get("find"), dict):
        names |= _find_names(opt, pk["find"])
    if isinstance(st.get("py-modules"), list):
        names |= set(st["py-modules"])
    find = (st.get("packages") or {}).get("find") if isinstance(st.get("packages"), dict) else None
    if find is None and isinstance((st.get("packages.find") if "packages.find" in st else None), dict):
        find = st["packages.find"]
    if pk is None and "py-modules" not in st:
        fnd = ((pp.get("tool") or {}).get("setuptools") or {}).get("packages", {})
        fnd = fnd.get("find") if isinstance(fnd, dict) else None
        names |= _find_names(opt, fnd if isinstance(fnd, dict) else {})
    for d, dns, fns in os.walk(opt):
        if os.path.relpath(d, opt).count(os.sep) >= 1:
            dns[:] = []
        dns[:] = [x for x in dns if x not in SKIP_DIRS and not x.endswith(".egg-info")]
        names |= {f[:-4] for f in fns if f.endswith(".pth")}
    return sorted(n for n in names if n)


def _find_names(opt, find):
    where = find.get("where") or ["."]
    where = where if isinstance(where, list) else [where]
    inc = find.get("include") or ["*"]
    exc = (find.get("exclude") or []) + ["tests", "tests.*", "test", "docs", "build", "dist"]
    names = set()
    for w in where:
        root = os.path.normpath(os.path.join(opt, w))
        if not os.path.isdir(root):
            continue
        for x in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, x, "__init__.py")) and any(fnmatch.fnmatchcase(x, p.split(".")[0]) for p in inc) \
                    and not any(fnmatch.fnmatchcase(x, p) for p in exc):
                names.add(x)
    return names


def dependencies(pp):
    prj = pp.get("project") or {}
    deps = list(prj.get("dependencies") or [])
    for v in (prj.get("optional-dependencies") or {}).values():
        deps += list(v or [])
    out = []
    for d in deps:
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", str(d))
        if m:
            out.append(m.group(1))
    return out


# ----------------------------------------------------------------------------------------------------------------- one file
def _str_arg(call):
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None


def _parse(path, feature_version=None):
    src = open(path, "rb").read().decode("utf-8", "replace")
    if feature_version is None:
        return ast.parse(src, filename=path), src
    return ast.parse(src, filename=path, feature_version=feature_version), src


def _expr_text(lines, node):
    """The source text of an expression node from the file's lines (positions are the parser's; no re-split of the source per call)."""
    try:
        if node.end_lineno == node.lineno:
            return lines[node.lineno - 1].encode("utf-8")[node.col_offset:node.end_col_offset].decode("utf-8", "replace")
        first = lines[node.lineno - 1].encode("utf-8")[node.col_offset:].decode("utf-8", "replace")
        last = lines[node.end_lineno - 1].encode("utf-8")[:node.end_col_offset].decode("utf-8", "replace")
        return " ".join([first] + [ln.strip() for ln in lines[node.lineno:node.end_lineno - 1]] + [last.strip()])
    except Exception:  # noqa: BLE001
        return "?"


def imports_of(path, deep=False, need=None):
    """{"top": [(module, line)], "local": [(module, line)], "dynamic": [(module, line)], "syspath": [(expr, line)],
    "future_annotations": bool, "runtime_unions": [line], "runtime_generics": [line], "lib_above_floor": [(name, line)], "error": str|None}."""
    out = {"top": [], "local": [], "dynamic": [], "syspath": [], "future_annotations": False, "runtime_unions": [], "runtime_generics": [],
           "lib_above_floor": [], "error": None, "skipped": False}
    if need is not None:                                        # the prefilter: a file naming none of the tokens cannot import, load or path-insert them
        try:
            text = open(path, "rb").read().decode("utf-8", "replace")
        except OSError as e:
            out["error"] = f"OSError: {e}"
            return out
        if not any(t in text for t in need):
            out["skipped"] = True
            return out
    try:
        tree, src = _parse(path)
    except (SyntaxError, ValueError) as e:                      # a python-2 file or a template: recorded by the caller, not fatal here
        out["error"] = f"{type(e).__name__}: {getattr(e, 'msg', e)} (line {getattr(e, 'lineno', '?')})"
        return out
    lines = src.split("\n")

    def names(node):
        if isinstance(node, ast.Import):
            return [a.name for a in node.names]
        if isinstance(node, ast.ImportFrom):
            return [] if node.level else [node.module or ""]
        return []

    def is_type_checking(test):
        return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")

    try_types = (ast.Try,) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())

    def walk_top(body):
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                out["top"] += [(n, node.lineno) for n in names(node)]
                if isinstance(node, ast.ImportFrom) and node.module == "__future__" and any(a.name == "annotations" for a in node.names):
                    out["future_annotations"] = True
            elif isinstance(node, ast.If):
                if not is_type_checking(node.test):
                    walk_top(node.body)
                walk_top(node.orelse)
            elif isinstance(node, try_types):
                walk_top(node.body); walk_top(node.orelse); walk_top(node.finalbody)
                for h in node.handlers:
                    walk_top(h.body)
            elif isinstance(node, (ast.With, ast.For, ast.While)):
                walk_top(node.body); walk_top(getattr(node, "orelse", []) or [])

    walk_top(tree.body)
    top_lines = {ln for _, ln in out["top"]}

    def union_in(ann):
        return ann is not None and any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr) for n in ast.walk(ann))

    def builtin_generic_in(ann):
        return ann is not None and any(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in BUILTIN_GENERICS
                                       for n in ast.walk(ann))

    for node in ast.walk(tree):                                 # ONE walk: imports (local, dynamic, sys.path) and, when deep, the floor checks
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if node.lineno not in top_lines:
                out["local"] += [(n, node.lineno) for n in names(node)]
            if deep:
                for n in names(node):
                    if LIB_ABOVE_FLOOR.get("module:" + n.split(".")[0]):
                        out["lib_above_floor"].append((n.split(".")[0], node.lineno))
                if isinstance(node, ast.ImportFrom) and node.module:
                    for a in node.names:
                        if LIB_ABOVE_FLOOR.get(f"{node.module}.{a.name}"):
                            out["lib_above_floor"].append((f"{node.module}.{a.name}", node.lineno))
        elif isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if fname in ("import_module", "__import__", "find_spec"):
                s_ = _str_arg(node)
                if s_:
                    out["dynamic"].append((s_, node.lineno))
            elif fname in ("insert", "append") and isinstance(f, ast.Attribute):
                tgt = f.value
                if isinstance(tgt, ast.Attribute) and tgt.attr == "path" and isinstance(tgt.value, ast.Name) and tgt.value.id == "sys":
                    out["syspath"].append((_expr_text(lines, node.args[-1]) if node.args else "?", node.lineno))
            if deep and isinstance(f, ast.Attribute) and LIB_ABOVE_FLOOR.get("attr:" + f.attr):
                out["lib_above_floor"].append((f.attr, node.lineno))
        elif not deep:
            continue
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            full = f"{node.value.id}.{node.attr}"
            if LIB_ABOVE_FLOOR.get(full):
                out["lib_above_floor"].append((full, node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            anns = [a.annotation for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs] + [node.returns]
            anns += [node.args.vararg.annotation if node.args.vararg else None, node.args.kwarg.annotation if node.args.kwarg else None]
            if any(union_in(a) for a in anns):
                out["runtime_unions"].append(node.lineno)
            if any(builtin_generic_in(a) for a in anns):
                out["runtime_generics"].append(node.lineno)
        elif isinstance(node, ast.AnnAssign) and not isinstance(getattr(node, "target", None), ast.Attribute):
            if union_in(node.annotation):
                out["runtime_unions"].append(node.lineno)
            if builtin_generic_in(node.annotation):
                out["runtime_generics"].append(node.lineno)
    return out


def top_package(mod):
    return (mod or "").split(".")[0]


# ----------------------------------------------------------------------------------------------------------------- pins
def read_pin(pyproject):
    """[tool.opt_core] of a kit's pyproject.toml — the two string keys (path, the minimum version)."""
    pp = read_pyproject(pyproject)
    t = (pp.get("tool") or {}).get("opt_core") or {}
    return {k: str(v) for k, v in t.items() if k in ("path", "version")}


# ----------------------------------------------------------------------------------------------------------------- censuses
def package_roots(opt_dir):
    """The roots of every package hierarchy under opt/: a directory holding __init__.py whose parent does not. {name: [relative parent dirs]}."""
    out = {}
    for d, dns, fns in os.walk(opt_dir):
        dns[:] = sorted(x for x in dns if x not in SKIP_DIRS and not x.endswith(".egg-info"))
        if "__init__.py" in fns and not os.path.isfile(os.path.join(os.path.dirname(d), "__init__.py")):
            out.setdefault(os.path.basename(d), []).append(os.path.relpath(os.path.dirname(d), opt_dir).replace(os.sep, "/"))
    return out


# ----------------------------------------------------------------------------------------------------------------- importable top-level names (vii, run time)
KNOWN_ENGINE_MENTIONS = {              # (ix-b) core files that name an engine today: the row-sharded pair stacks' recipe / adapter DOCS and worked examples
    # (a recipe may be model-keyed — alphafold, alphafold3 are upstream libraries — but an ENGINE (kit) name belongs in the kit); each row is
    # reworded by its family owner in the next core version; the list only shrinks. Read by tests/test_instances_backend_hygiene.py.
    "opt_core/mem/README.md": ("openfold3",),
    "opt_core/mem/rowpair/ADAPTER_GUIDE.md": ("openfold3", "openfold3_opt"),  # F8: the worked example of an adapter
    "opt_core/mem/rowpair/confidence.py": ("openfold3",),                   # F8: docstring naming the summary's engine
    "opt_core/mem/rowpair_jax/API.md": ("af3_jax", "colabfold"),                           # F9: recipe / adapter-pinned docs
    "opt_core/mem/rowpair_jax/alphafold.py": ("colabfold", "colabfold_opt"),                     # F9: recipe / adapter-pinned docs
    "tests/test_rowpair_jax_logic.py": ("colabfold",),                           # F9: recipe / adapter-pinned docs
}

KNOWN_GENERIC_TOPLEVEL = {            # (vii, H1) today's generic importable top-level names per kit — a list that only shrinks (an entry whose name is gone fails until deleted)
    "af3_jax": ('af3_flashpairformer', 'run_alphafold_flashpairformer'),
    "af3_torch": ('alphafold3', 'constants', 'fastnn', 'feat_batch', 'features', 'geometry', 'nn', 'of3', 'params', 'protein_data_processing', 'scoring', 'version'),
    "boltz2": ('conf', 'fpf_msa', 'msa2',),
    "borzoi": ('kitlib', 'pipeline_tf'),
    "chrombpnet": ('pred_bw_fast', 'test_apply_identity_cpu', 'test_batch_bound_cpu', 'test_cache_pin_cpu', 'test_cache_witness_cpu', 'test_class_by_cc_cpu', 'test_counts_form_cpu', 'test_device_input_cpu', 'test_exit_fast_by_path_cpu', 'test_exit_fast_cpu', 'test_fail_fast_cpu', 'test_finish_stage_cpu', 'test_image_key_cpu', 'test_k1_import_fallback_cpu', 'test_per_mode_route_cpu', 'test_predict_no_io', 'test_preimport_cpu', 'test_preimport_hook_cpu', 'test_route_agreement_cpu', 'test_spawned_helpers_cpu', 'test_te_guard_cpu', 'test_tile_regimes_cpu', 'test_v01264_cpu', 'test_v01266_cpu', 'test_v01269_cpu', 'test_v01270_cpu'),
    "e1": ('engines',),
    "enformer": ('engines',),
    "flashzoi": ('engines',),
    "opendde": ('fpf_engines', 'fpf_transition_odde', 'odde_arm_t', 'odde_transition_bind', 'odde_triattn_bind', 'odde_trimul_bind', 'sitecustomize',),
    "progen2": ('compare', 'engines'),
    "protenix_v1": ('dit_hoist', 'fastln_prebuilt', 'infopt_graphs', 'protenix_fpf_ditfast', 'protenix_fpf_msa', 'ptx1_keep_pool', 'ptx1_lazy_init', 'ptx1_summary_host',),
    "protenix_v2": ('blockfuse', 'blockfuse_xl', 'deadskip', 'dit_hoist', 'fastln_prebuilt', 'fpf', 'fpf_clisampler', 'fpf_cueq_pad8exact', 'fpf_pad8exact', 'fpf_smalln', 'fpf_stackgraph', 'fpf_trunkgraph', 'infopt_graphs', 'ptx_c64_routes', 'ptx_native_core', 'ptx_drop_bond_mask', 'ptx_fpf_v02', 'ptx_lazy_init', 'ptx_msa_adapt', 'ptx_tp', 'ptx_transition_core', 'ptx_trimul_routes', 'ptx_trunk2_levers', 'sitecustomize',),
    "pxdesign": ('pxd_xattempt',),
    "rfdiffusion1": ('rfd_se3fast',),
}

NOT_IMPORT_ROOTS = ("tests", "test", "fixtures", "_fixtures")      # directories never placed on sys.path as roots (their packages are test data)


def _first_py(d):
    for root, dns, fns in os.walk(d):
        dns[:] = [x for x in dns if x not in SKIP_DIRS]
        if any(f.endswith(".py") for f in fns):
            yield root
            return


def _importable_entries(d):
    """The importable names directly under directory ``d`` when ``d`` is on sys.path: sub-directories holding python code (a package, or a
    namespace directory with a .py below) and ``*.py`` modules. {name: kind}."""
    out = {}
    try:
        entries = sorted(os.listdir(d))
    except OSError:
        return out
    for n in entries:
        p = os.path.join(d, n)
        if n in SKIP_DIRS or n.endswith(".egg-info"):
            continue
        if os.path.isdir(p):
            if n.isidentifier() and (os.path.isfile(os.path.join(p, "__init__.py")) or any(True for _ in _first_py(p))):
                out[n] = "package" if os.path.isfile(os.path.join(p, "__init__.py")) else "namespace-dir"
        elif n.endswith(".py") and n[:-3].isidentifier():
            out[n[:-3]] = "module"
    return out


def implied_roots(opt_dir):
    """The directories inside opt/ that some sys.path entry must name for the kit's packages to import: opt/ itself (the editable install's
    path entry) and the parent of every package-hierarchy root, excluding test-data parents. Relative to opt/; '.' = opt/."""
    roots = {"."}
    for d, dns, fns in os.walk(opt_dir):
        dns[:] = sorted(x for x in dns if x not in SKIP_DIRS and not x.endswith(".egg-info"))
        parent = os.path.dirname(d)
        if "__init__.py" in fns and d != opt_dir and not os.path.isfile(os.path.join(parent, "__init__.py")):
            relp = os.path.relpath(parent, opt_dir).replace(os.sep, "/")
            if not any(part in NOT_IMPORT_ROOTS for part in relp.split("/")):
                roots.add(relp)
    return sorted(roots)


def vendored_names(root_dir):
    """The third-party distributions a VENDOR.json directly inside ``root_dir`` declares: the import names of its ``contents`` entries
    (``<name>-<version>.dist-info`` -> the dist's top_level.txt names when that file is beside, else <name> normalised) and of its top-level
    keys that name a ``file``. A declared vendored distribution keeps its upstream import name (renaming it would fork it); the declaration
    (source, sha256, licence) is what makes it reviewable. set() when there is no VENDOR.json."""
    vj = os.path.join(root_dir, "VENDOR.json")
    if not os.path.isfile(vj):
        return set()
    try:
        doc = json.load(open(vj, encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    out = set()
    for entry in (doc.get("contents") or []) if isinstance(doc, dict) else []:
        entry = str(entry)
        if entry.endswith(".dist-info"):
            tl = os.path.join(root_dir, entry, "top_level.txt")
            if os.path.isfile(tl):
                out |= {ln.strip() for ln in open(tl, encoding="utf-8") if ln.strip()}
            out.add(re.split(r"-\d", entry, maxsplit=1)[0].replace("-", "_").lower())
        else:
            out.add(entry.replace("-", "_"))
    if isinstance(doc, dict):
        for k, v in doc.items():
            if isinstance(v, dict) and v.get("file"):
                out.add(str(v["file"])[:-3] if str(v["file"]).endswith(".py") else str(v["file"]))
    return {n for n in out if n}


def toplevel_names(engine, project, opt_dir, packages, imported_tops):
    """{name: {"ok", "inert", "vendored", "evidence"}} for every name the kit makes importable at top level at RUN time: the importable
    entries of opt/ and of every implied root. ``imported_tops`` ({top-level module name: "file:line"} from the kit's own import statements)
    adds evidence to a name found at a root and never introduces a name by itself. ok = namespaced (engine / project root / declared
    package) or private (`_...`) or inert or vendored; inert = a plain directory without __init__.py that no kit file imports by name (layout
    such as opt/forward/, not an import name); vendored = a third-party distribution declared in the VENDOR.json of the root that holds it
    (upstream import name kept, declaration reviewed instead)."""
    names, vend = {}, {}
    for root in implied_roots(opt_dir):
        d = os.path.join(opt_dir, root) if root != "." else opt_dir
        declared = vendored_names(d)
        for n, kind in _importable_entries(d).items():
            names.setdefault(n, []).append("%s under opt/%s" % (kind, "" if root == "." else root + "/"))
            if n.lower() in declared or n.lower().replace("_", "") in {x.replace("_", "") for x in declared}:
                vend[n] = "declared in opt/%sVENDOR.json" % ("" if root == "." else root + "/")
    for n, where in imported_tops.items():
        if n in names:
            names[n].append("imported top-level at " + where)
    out = {}
    for n, ev in sorted(names.items()):
        ns = n.startswith("_") or n in packages or namespaced(n, engine, project)
        inert = all(e.startswith("namespace-dir") for e in ev)
        if n in vend:
            ev = ev + ["vendored: " + vend[n]]
        out[n] = {"ok": ns or inert or n in vend, "inert": inert and not ns, "vendored": n in vend, "evidence": ev}
    return out


def carried_files(core_pkg_dir):
    """The carried kernel files as paths relative to opt_core/ — one name per kernels/META/<name>.json found, its file set read
    live off disk (a package <name>/ walked whole, or a module's kernels/<name>.* siblings): META carries no file list, equality
    is the git commit, not a restated inventory."""
    out = set()
    kernels_dir = os.path.join(core_pkg_dir, "kernels")
    meta_dir = os.path.join(kernels_dir, "META")
    if not os.path.isdir(meta_dir):
        return out
    for j in sorted(os.listdir(meta_dir)):
        if not j.endswith(".json"):
            continue
        try:
            kind = (json.load(open(os.path.join(meta_dir, j), encoding="utf-8")) or {}).get("kind")
        except (OSError, ValueError):
            continue
        name = j[:-5]
        if kind == "package":
            pkg_dir = os.path.join(kernels_dir, name)
            for r, dirs, files in os.walk(pkg_dir):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for f in files:
                    out.add(f"kernels/{name}/{os.path.relpath(os.path.join(r, f), pkg_dir)}")
        else:
            for p in glob.glob(os.path.join(kernels_dir, name + ".*")):
                out.add(f"kernels/{os.path.basename(p)}")
    return out


def framework_row(sub_parts, carried=frozenset()):
    """("carried", PY_FLOOR) for a carried kernel file, the FRAMEWORK_EXTRA row (glob, floor) matching the path (parts relative to opt_core/), or None."""
    rel = "/".join(sub_parts)
    if rel in carried:
        return "carried", PY_FLOOR
    for glob_, floor in FRAMEWORK_EXTRA:
        if fnmatch.fnmatchcase(rel, glob_) or (glob_.endswith("/**") and (rel + "/").startswith(glob_[:-2])):
            return glob_, floor
    return None


def namespaced(name, engine, project):
    roots = {engine.lower().replace("-", "_")}
    if project:
        p = str(project).lower().replace("-", "_")
        roots |= {p, re.sub(r"_opt$", "", p)}
    n = name.lower().replace("-", "_")
    # `<root>_...` is namespaced for a root of any length (the engine `e1`); a bare prefix only for roots of 3+ characters
    return any(n == r or n.startswith(r + "_") or (len(r) >= 3 and n.startswith(r)) for r in roots)


# ----------------------------------------------------------------------------------------------------------------- the scan
def scan(tree, census=False):
    K = kits(tree)
    pkg_owner = {}                                                          # importable kit package / project name -> engine
    for e, k in K.items():
        for p in list(k["packages"]) + list(k["installs"]):
            pkg_owner.setdefault(p, e)
    proj_owner = {_norm(k["project"]): e for e, k in K.items() if k["project"]}
    for p, e in list(pkg_owner.items()):
        proj_owner.setdefault(_norm(p), e)
    engines = set(K)
    rel = lambda p: os.path.relpath(p, tree).replace(os.sep, "/")  # noqa: E731
    R = {"tree": tree, "kits": {e: {"packages": k["packages"], "installs": k["installs"], "project": k["project"]} for e, k in K.items()},
         "violations": [], "syspath_cross": [], "opt_core_kit_imports": [], "opt_core_heavy_top": [], "opt_core_py_floor": [],
         "opt_core_engine_named": [], "opt_core_research_scripts": [], "known_stale": [], "namespacing": [], "shared_toplevel": [], "generic_toplevel": [],
         "pins": {}, "parse_errors": [], "mentions": [], "toplevel": {}, "shared_generic": [], "runtime_injections": KNOWN_RUNTIME_INJECTIONS, "counts": {}}
    word = {e: re.compile(r"(?<![A-Za-z0-9])" + re.escape(e) + r"(?![A-Za-z0-9])") for e in engines if len(e) >= 4}

    # (i) kits: imports, pyproject dependencies, sys.path expressions
    n_files, n_parsed, uses_core = 0, 0, {}
    need_all = sorted(set(pkg_owner) | {CORE_PKG, "sys.path", "import_module", "__import__", "find_spec"})
    for e, k in K.items():
        own = set(k["packages"]) | set(k["installs"])
        need = [t for t in need_all if t not in own]
        imported_tops = {}
        for path in py_files(k["opt"]):
            n_files += 1
            imp = imports_of(path, need=need)
            n_parsed += 0 if imp["skipped"] else 1
            if imp["error"]:
                R["parse_errors"].append(f"{rel(path)}: {imp['error']}")
            for kind, items in (("import", imp["top"]), ("local-import", imp["local"]), ("dynamic", imp["dynamic"])):
                for mod, ln in items:
                    tp = top_package(mod)
                    if tp and tp not in imported_tops:
                        imported_tops[tp] = f"{rel(path)}:{ln}"
                    if tp == CORE_PKG:
                        uses_core.setdefault(e, []).append(f"{rel(path)}:{ln}")
                    owner = pkg_owner.get(tp)
                    if owner and owner != e:
                        R["violations"].append(f"{e}: {rel(path)}:{ln}: {mod} ({kind} of {owner}'s kit package)")
            for expr, ln in imp["syspath"]:
                hit = [o for o in engines if o != e and o in word and word[o].search(expr)]
                if hit:
                    R["syspath_cross"].append(f"{e}: {rel(path)}:{ln}: sys.path <- {expr} (names {','.join(hit)}: another engine's directory)")
            if census:
                try:
                    text = open(path, encoding="utf-8", errors="replace").read()
                except OSError:
                    text = ""
                others = sorted(o for o in engines if o != e and o in word and word[o].search(text))
                if others:
                    parts = rel(path).split("/")
                    where = "tests" if "tests" in parts else ("compare" if any(p.startswith(("compare", "bench", "stock")) for p in parts) else "code")
                    R["mentions"].append({"kit": e, "file": rel(path), "names": others, "where": where})
        for dep in dependencies(k["pyproject"]):
            owner = proj_owner.get(_norm(dep))
            if owner and owner != e:
                R["violations"].append(f"{e}: {e}/opt/pyproject.toml: declares {dep!r} ({owner}'s kit) as a dependency")
        # (vii) importable names at run time: generic (not namespaced, not private, not inert layout) = violation
        R["toplevel"][e] = toplevel_names(e, k["project"], k["opt"], k["packages"], imported_tops)
        known_generic = set(KNOWN_GENERIC_TOPLEVEL.get(e, ()))
        for n, v in R["toplevel"][e].items():
            if not v["ok"] and n not in known_generic:
                R["generic_toplevel"].append(f"{e}: top-level name {n!r} is importable outside the kit namespace ({'; '.join(v['evidence'])})")
        for n in sorted(known_generic):
            if n not in R["toplevel"][e] or R["toplevel"][e][n]["ok"]:
                R["known_stale"].append(f"{e}: KNOWN_GENERIC_TOPLEVEL lists {n!r}, which is not a generic importable name in this tree — delete the entry")
        # (vii) installed names
        for n in k["installs"]:
            if not namespaced(n, e, k["project"]):
                entry = f"{e}: {n}"
                if entry not in KNOWN_UNNAMESPACED:
                    R["namespacing"].append(f"{e}: {e}/opt installs top-level {n!r}, outside the kit namespace ({e} / {k['project']})")
    installed_by = {}
    for e, k in K.items():
        for n in k["installs"]:
            installed_by.setdefault(n, []).append(e)
    for n, es in sorted(installed_by.items()):
        if len(es) >= 2 and f"{n}: {','.join(sorted(es))}" not in KNOWN_SHARED_TOPLEVEL:
            R["shared_toplevel"].append(f"{n}: installed by {','.join(sorted(es))} (two kits in one environment collide)")
    for entry in KNOWN_UNNAMESPACED:
        e, n = [x.strip() for x in entry.split(":", 1)]
        if e not in K or n not in K[e]["installs"] or namespaced(n, e, K[e]["project"]):
            R["known_stale"].append(f"KNOWN_UNNAMESPACED {entry!r}: the offence is gone — delete the entry")
    for entry in KNOWN_SHARED_TOPLEVEL:
        n, es = [x.strip() for x in entry.split(":", 1)]
        if sorted(installed_by.get(n, [])) != sorted(es.split(",")):
            R["known_stale"].append(f"KNOWN_SHARED_TOPLEVEL {entry!r}: the installers differ now ({','.join(sorted(installed_by.get(n, [])))}) — update or delete the entry")
    seen = {}
    for e, tl in R["toplevel"].items():
        for n, v in tl.items():
            if not v["ok"]:
                seen.setdefault(n, []).append(e)
    R["shared_generic"] = sorted(f"{n}: {','.join(es)}" for n, es in seen.items() if len(es) >= 2)
    R["counts"]["kit_py_files"] = n_files
    R["counts"]["kit_py_parsed"] = n_parsed

    # (ii) (iii) (v) (vi) opt_core
    core_dir = os.path.join(tree, CORE_REL)
    core_pkg_dir = os.path.join(core_dir, CORE_PKG)
    carried = frozenset(carried_files(core_pkg_dir))
    R["counts"]["carried_files"] = len(carried)
    n_core = 0
    for path in py_files(core_dir):
        n_core += 1
        imp = imports_of(path, deep=True)
        r = rel(path)
        if imp["error"]:
            R["parse_errors"].append(f"{r}: {imp['error']}")
        sub = os.path.relpath(path, core_pkg_dir).split(os.sep)
        in_pkg = not sub[0].startswith("..")
        row = framework_row(sub, carried) if in_pkg else None
        generic = in_pkg and row is None
        floor = (row[1] if row and row[1] else PY_FLOOR) if in_pkg else None
        for kind, items in (("import", imp["top"]), ("local-import", imp["local"]), ("dynamic", imp["dynamic"])):
            for mod, ln in items:
                if top_package(mod) in pkg_owner and top_package(mod) != CORE_PKG:
                    R["opt_core_kit_imports"].append(f"opt_core: {r}:{ln}: {mod} ({kind} of {pkg_owner[top_package(mod)]}'s kit package)")
        if generic:
            for mod, ln in imp["top"]:
                if top_package(mod) in HEAVY:
                    R["opt_core_heavy_top"].append(f"opt_core: {r}:{ln}: {mod} (a framework import at module top level in a module that is not a carried kernel file or a FRAMEWORK_EXTRA row)")
        if in_pkg:
            try:
                _parse(path, feature_version=floor)
            except SyntaxError as e:
                R["opt_core_py_floor"].append(f"opt_core: {r}:{e.lineno}: syntax above Python {floor[0]}.{floor[1]}: {e.msg}")
            if imp["runtime_unions"] and not imp["future_annotations"]:
                R["opt_core_py_floor"].append(f"opt_core: {r}:{imp['runtime_unions'][0]}: `X | Y` annotation evaluated at run time without `from __future__ import annotations` (TypeError below 3.10; floor {floor[0]}.{floor[1]})")
            if floor < (3, 9) and imp["runtime_generics"] and not imp["future_annotations"]:
                R["opt_core_py_floor"].append(f"opt_core: {r}:{imp['runtime_generics'][0]}: subscripted builtin annotation evaluated at run time without `from __future__ import annotations` (TypeError on 3.8)")
            if floor < (3, 9):
                for name, ln in imp["lib_above_floor"]:
                    key = name if name in LIB_ABOVE_FLOOR else ("attr:" + name if "attr:" + name in LIB_ABOVE_FLOOR else "module:" + name)
                    R["opt_core_py_floor"].append(f"opt_core: {r}:{ln}: {name} needs Python {LIB_ABOVE_FLOOR.get(key)} (floor {floor[0]}.{floor[1]})")
    R["counts"]["opt_core_py_files"] = n_core
    for path in all_files(core_dir):
        r = os.path.relpath(path, core_dir).replace(os.sep, "/")
        base = os.path.basename(path)
        stem = base[:-3] if base.endswith(".py") else base
        named = [e for e in engines if e in word and word[e].search(stem.replace("_", " ").replace("-", " ")) or stem.lower() == e.lower()]
        if named and os.path.isfile(path) or (named and os.path.isdir(path)):
            if r not in KNOWN_ENGINE_NAMED:
                R["opt_core_engine_named"].append(f"opt_core: {r}: named after engine {','.join(sorted(named))}")
        if os.path.isfile(path) and r.startswith(CORE_PKG + "/") and any(fnmatch.fnmatchcase(base, g) for g in RESEARCH_GLOBS):
            if r not in KNOWN_RESEARCH_SCRIPTS:
                R["opt_core_research_scripts"].append(f"opt_core: {r}: a research script inside the shipped package")
    for r in KNOWN_ENGINE_NAMED:
        if not os.path.exists(os.path.join(core_dir, r)):
            R["known_stale"].append(f"KNOWN_ENGINE_NAMED {r!r}: gone from the tree — delete the entry")
    for r in KNOWN_RESEARCH_SCRIPTS:
        if not os.path.isfile(os.path.join(core_dir, r)):
            R["known_stale"].append(f"KNOWN_RESEARCH_SCRIPTS {r!r}: gone from the tree — delete the entry")

    # (iv) pins
    R["pins"] = {"kits": {}, "problems": []}
    for e, k in K.items():
        pin = read_pin(os.path.join(k["opt"], "pyproject.toml"))
        R["pins"]["kits"][e] = {"pinned": bool(pin), "version": pin.get("version"), "imports_opt_core": len(uses_core.get(e, []))}
        if uses_core.get(e) and not pin:
            R["pins"]["problems"].append(f"{e}: imports opt_core ({len(uses_core[e])} sites, first {uses_core[e][0]}) but opt/pyproject.toml has no [tool.opt_core] pin")
        if pin:
            missing = [x for x in ("path", "version") if not pin.get(x)]
            if missing:
                R["pins"]["problems"].append(f"{e}: [tool.opt_core] lacks {','.join(missing)}")
    c = R["counts"]
    c.update(kits=len(K), violations=len(R["violations"]), syspath_cross=len(R["syspath_cross"]), opt_core_kit_imports=len(R["opt_core_kit_imports"]),
             opt_core_heavy_top=len(R["opt_core_heavy_top"]), opt_core_py_floor=len(R["opt_core_py_floor"]), opt_core_engine_named=len(R["opt_core_engine_named"]),
             opt_core_research_scripts=len(R["opt_core_research_scripts"]), known_stale=len(R["known_stale"]), namespacing=len(R["namespacing"]),
             shared_toplevel=len(R["shared_toplevel"]), generic_toplevel=len(R["generic_toplevel"]),
             kits_with_generic_toplevel=len({v.split(":")[0] for v in R["generic_toplevel"]}), pin_problems=len(R["pins"]["problems"]), parse_errors=len(R["parse_errors"]),
             kits_pinning_opt_core=sum(1 for v in R["pins"]["kits"].values() if v["pinned"]), kits_importing_opt_core=len(uses_core),
             mentions=len(R["mentions"]), shared_generic=len(R["shared_generic"]))
    return R


def failures(R):
    """Every violation line of the assertions, in order; [] = the tree is self-contained."""
    return (R["violations"] + R["syspath_cross"] + R["opt_core_kit_imports"] + R["opt_core_heavy_top"] + R["opt_core_py_floor"]
            + R["opt_core_engine_named"] + R["opt_core_research_scripts"] + R["known_stale"] + R["namespacing"] + R["shared_toplevel"]
            + R["generic_toplevel"] + ["pin: " + p for p in R["pins"]["problems"]])


# ----------------------------------------------------------------------------------------------------------------- pytest form
_CACHE = {}


def _scan():
    if "R" not in _CACHE:
        _CACHE["R"] = scan(find_tree(os.environ.get("SELFCONTAINED_TREE")))
    return _CACHE["R"]


def test_no_kit_imports_or_declares_another_kit():
    R = _scan()
    assert not (R["violations"] or R["syspath_cross"]), "\n".join(R["violations"] + R["syspath_cross"])


def test_opt_core_imports_no_kit():
    assert not _scan()["opt_core_kit_imports"], "\n".join(_scan()["opt_core_kit_imports"])


def test_opt_core_generic_modules_import_no_framework_at_top_level():
    assert not _scan()["opt_core_heavy_top"], "\n".join(_scan()["opt_core_heavy_top"])


def test_opt_core_modules_hold_the_python_floor():
    assert not _scan()["opt_core_py_floor"], "\n".join(_scan()["opt_core_py_floor"])


def test_nothing_in_opt_core_is_named_after_an_engine_and_no_research_script_ships():
    R = _scan()
    bad = R["opt_core_engine_named"] + R["opt_core_research_scripts"]
    assert not bad, "\n".join(bad)


def test_known_offender_lists_only_shrink():
    assert not _scan()["known_stale"], "\n".join(_scan()["known_stale"])


def test_every_installed_top_level_name_is_kit_namespaced_and_unique():
    R = _scan()
    assert not (R["namespacing"] or R["shared_toplevel"]), "\n".join(R["namespacing"] + R["shared_toplevel"])


def test_no_kit_makes_a_generic_top_level_name_importable_at_run_time():
    assert not _scan()["generic_toplevel"], "\n".join(_scan()["generic_toplevel"])


def test_every_kit_that_imports_the_core_pins_this_core():
    assert not _scan()["pins"]["problems"], "\n".join(_scan()["pins"]["problems"])


# ----------------------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    out, census = None, False
    if "--json" in argv:
        i = argv.index("--json"); out = argv[i + 1]; del argv[i:i + 2]; census = True
    if "--census" in argv:
        argv.remove("--census"); census = True
    tree = find_tree(argv[0]) if argv else find_tree()
    R = scan(tree, census=census)
    if out:
        with open(out, "w") as fh:
            json.dump(R, fh, indent=1, sort_keys=True, default=str)
    c = R["counts"]
    print("tree=%s kits=%d kit_py=%d core_py=%d violations=%d syspath_cross=%d core_kit_imports=%d core_heavy_top=%d core_py_floor=%d "
          "core_engine_named=%d core_research=%d known_stale=%d namespacing=%d shared_toplevel=%d generic_toplevel=%d (kits %d) pin_problems=%d parse_errors=%d" % (
              tree, c["kits"], c["kit_py_files"], c["opt_core_py_files"], c["violations"], c["syspath_cross"], c["opt_core_kit_imports"],
              c["opt_core_heavy_top"], c["opt_core_py_floor"], c["opt_core_engine_named"], c["opt_core_research_scripts"], c["known_stale"],
              c["namespacing"], c["shared_toplevel"], c["generic_toplevel"], c["kits_with_generic_toplevel"], c["pin_problems"], c["parse_errors"]))
    bad = failures(R)
    for v in bad:
        print("  " + v)
    for v in R["shared_generic"]:
        print("  shared generic top-level name: " + v)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())



# ----------------------------------------------------------------------------------------------------------------- the core in a clean venv (viii)
MODEL_LIBRARIES = ("torch", "jax", "jaxlib", "haiku", "flax", "alphafold3", "alphafold", "colabfold", "colabdesign", "openfold", "openfold3",
                   "boltz", "chai_lab", "protenix", "rf3", "esm", "deepspeed", "triton", "cuequivariance_torch")


def _core_import_targets():
    """``opt_core``, ``opt_core.mem`` and every model-keyed recipe package under opt_core/mem (``rowpair*``: the row-sharded pair stacks) —
    discovered from the tree so a recipe package that lands is held the moment it exists."""
    import opt_core
    memdir = os.path.join(os.path.dirname(opt_core.__file__), "mem")
    recipes = sorted("opt_core.mem." + d for d in os.listdir(memdir)
                     if d.startswith("rowpair") and os.path.isfile(os.path.join(memdir, d, "__init__.py")))
    modules = sorted("opt_core.mem." + f[:-3] for f in os.listdir(memdir) if f.startswith("rowpair") and f.endswith(".py"))   # a single-module recipe
    subs = []
    for r in recipes:
        rd = os.path.join(memdir, r.rsplit(".", 1)[1])
        for dp, _dn, fn in os.walk(rd):                                  # every module of the package, sub-packages included
            if "__pycache__" in dp:
                continue
            pkg = r + "".join("." + x for x in os.path.relpath(dp, rd).split(os.sep) if x not in (".", ""))
            subs += sorted(pkg + "." + f[:-3] for f in fn if f.endswith(".py") and f != "__init__.py")
            subs += sorted(pkg + "." + d for d in _dn if os.path.isfile(os.path.join(dp, d, "__init__.py")))
    return ["opt_core", "opt_core.mem"] + recipes + modules + sorted(set(subs))


_CLEAN_VENV_PROBE = r"""
import importlib, importlib.abc, importlib.machinery, sys, json
BLOCKED = tuple(json.loads(sys.argv[1]))
class _NoModelLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("clean-venv probe: %s is not installed here" % name)
        return None
for b in BLOCKED:
    for k in [m for m in sys.modules if m.split(".")[0] == b]:
        del sys.modules[k]
sys.meta_path.insert(0, _NoModelLibraries())
out = {}
for target in json.loads(sys.argv[2]):
    try:
        importlib.import_module(target)
        out[target] = "ok"
    except Exception as e:  # noqa: BLE001
        out[target] = "%s: %s" % (type(e).__name__, e)
pulled = sorted({m.split(".")[0] for m in sys.modules} & set(BLOCKED))
print(json.dumps({"imports": out, "pulled": pulled}))
"""


def test_core_and_recipe_packages_import_in_a_clean_venv():
    """(viii) `import opt_core`, `import opt_core.mem` and every model-keyed recipe package / module (opt_core.mem.rowpair*) import with NO
    model library importable (torch, jax, haiku, alphafold3, colabfold, …): a recipe imports its upstream library lazily at call time,
    pin-gated, refusing by name — so another model's recipe files are inert for a single-engine install and the core never pulls a framework
    at import. Held in a fresh interpreter whose import system refuses every name in MODEL_LIBRARIES."""
    import json
    import subprocess
    CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # common/opt_core (the directory holding the opt_core package)
    targets = _core_import_targets()
    r = subprocess.run([sys.executable, "-c", _CLEAN_VENV_PROBE, json.dumps(list(MODEL_LIBRARIES)), json.dumps(targets)],
                       cwd=CORE, env=dict(os.environ, PYTHONPATH=CORE), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    bad = {t: v for t, v in res["imports"].items() if v != "ok"}
    assert not bad, "imports that need a model library at import time (must be lazy, at call time):\n" + "\n".join(f"{t}: {v}" for t, v in bad.items())
    assert res["pulled"] == [], f"importing the core pulled model libraries into sys.modules: {res['pulled']}"



_REFUSAL_PROBE = r"""
import importlib, importlib.abc, sys, json
BLOCKED = tuple(json.loads(sys.argv[1]))
class _NoModelLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("clean-venv probe: %s is not installed here" % name)
        return None
for b in BLOCKED:
    for k in [m for m in sys.modules if m.split(".")[0] == b]:
        del sys.modules[k]
sys.meta_path.insert(0, _NoModelLibraries())
out = {}
for pkg_name in json.loads(sys.argv[2]):
    pkg = importlib.import_module(pkg_name)
    probes = []
    try:
        lt = importlib.import_module(pkg_name + "._torch")        # torch recipes: lazy proxies; first attribute access imports torch
        probes.append(("torch", lambda lt=lt: getattr(lt.torch, "float32")))
    except ImportError:
        pass
    try:
        lz = importlib.import_module(pkg_name + "._lazy")         # jax recipes: lazy import functions
        for fname in ("jax", "haiku"):
            if hasattr(lz, fname):
                probes.append((fname, getattr(lz, fname)))
    except ImportError:
        pass
    res = {}
    for lib, call in probes:
        try:
            call()
            res[lib] = "NO REFUSAL (the library imported?)"
        except Exception as e:  # noqa: BLE001
            res[lib] = [type(e).__name__, str(e)[:200]]
    out[pkg_name] = res
print(json.dumps(out))
"""


def test_recipe_packages_refuse_by_name_without_their_library():
    """(ix) A model-keyed recipe package whose upstream library is absent refuses BY NAME at call time (a `...Refused` exception naming the
    library) — never an ImportError at import, never a silent pass. Probed per recipe package found in the tree through its lazy-import seam
    (`<pkg>._torch` proxies for torch recipes, `<pkg>._lazy.jax/haiku` for JAX recipes) in a fresh interpreter refusing MODEL_LIBRARIES."""
    import json
    import subprocess
    CORE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkgs = [t for t in _core_import_targets() if t.count(".") == 2 and t.startswith("opt_core.mem.rowpair")]
    import pytest
    if not pkgs:
        pytest.skip("no model-keyed recipe package (opt_core/mem/rowpair*) in this tree")
    r = subprocess.run([sys.executable, "-c", _REFUSAL_PROBE, json.dumps(list(MODEL_LIBRARIES)), json.dumps(pkgs)],
                       cwd=CORE, env=dict(os.environ, PYTHONPATH=CORE), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    problems = []
    for pkg, libs in res.items():
        if not libs:
            problems.append(f"{pkg}: no lazy-import seam found (_torch / _lazy) — the refusal cannot be probed")
        for lib, v in libs.items():
            if not (isinstance(v, list) and v[0].endswith("Refused") and lib in v[1]):
                problems.append(f"{pkg}: {lib}: {v}")
    assert problems == [], "\n".join(problems)

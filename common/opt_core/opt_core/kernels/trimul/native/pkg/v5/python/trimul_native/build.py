#!/usr/bin/env python3
"""trimul_native.build -- compile the units of csrc/ to arch-keyed cubins with nvcc and record them in build/manifest.json.

    python -m trimul_native.build [--archs sm_90a,sm_80] [--units probe,...] [--src <tree>/csrc] [--out <tree>/build]
                                  [--nvcc /usr/local/cuda/bin/nvcc] [--fast-math] [--lineinfo] [--define K=V ...]

One unit = one cubin per architecture.  By default every ``.cu`` file under csrc/ (found recursively) is a unit named by its file stem; a
source restricts its architectures with a first-lines comment ``// build: archs=sm_90a sm_80``, adds flags with ``// build: flags=<nvcc args>``,
and declares VARIANT units (one cubin per preprocessor configuration of the same source) with
``// build: variant=<unit name> defines=K=V,K=V``, one line per variant.  A source whose width instantiations are guarded by
``TMN_ONLY_CZ == <c_z> && TMN_ONLY_CH == <c_h>`` and declares no variants gets one unit per guarded width pair, named
by a recipe in ``RECIPES`` (none active: the sm_90a member ships one ``tmn90_z<c_z>_h<c_h>.cu`` unit file per width pair).
Units marked ``// build: dev`` (and everything under csrc/dev*/) are development units: built on request, never sealed.
NAMING AUTHORITY: unit and kernel names belong to the op assembly (kernel.py: unit_name / k1_name / k3_name / TILE_TABLE; sm80_ops.py:
TILES).  A width unit file (one defining TMN_CZ / TMN_CH) must be named kernel.unit_name(arch, c_z, c_hidden); after compiling, the build
asks the authority for the kernel names the op launches per (c_z, c_hidden, form) (``kernel.serve_names``; ``exact=True`` adds the
reference-order epilogue instantiations) and records which are present per unit as ``serves`` in the manifest -- the face serves exactly
that, and resolves the recorded kernel names at load.  One command rebuilds everything at a new source state:
``python -m trimul_native.build --all && python -m trimul_native.vectors make``.
``--jobs`` compiles units in parallel (nvcc processes); the manifest is written once at the end.
Output: ``build/<arch>/<unit>.cubin``, ``build/<arch>/<unit>.ptxas.txt`` (the ``-Xptxas -v`` resource report) and the manifest entry
(cubin sha256, sha256 of every source file the unit depends on (``nvcc -M``), the nvcc release line, the exact flags, UTC build time, the
ELF header facts, registers / shared / spill per entry from ptxas).  Flags are fixed and deterministic: ``-O3 -std=c++17 -DNDEBUG
--expt-relaxed-constexpr -Xptxas -v``; ``--use_fast_math`` and ``-lineinfo`` are OFF unless asked (numerics and byte-stable release builds).
The compile runs from a scratch copy of csrc/ so no absolute path of the build host enters the object.  Requires nvcc of a CUDA toolkit that
knows every requested architecture (12.0+ for sm_90a); a cubin loads on any driver at least as new as the toolkit's major line requires.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from trimul_native import manifest as M                      # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_ARCHS = ["sm_90a", "sm_80"]
BASE_FLAGS = ["-O3", "-std=c++17", "-DNDEBUG", "--expt-relaxed-constexpr", "-Xptxas", "-v"]
ARCH_RE = re.compile(r"^\s*//\s*build:\s*archs\s*=\s*([\w ,]+)", re.M)
SMEM_RE = re.compile(r"^\s*//\s*build:\s*max_dynamic_smem\s*=\s*(\{.*\})", re.M)
FLAGS_RE = re.compile(r"^\s*//\s*build:\s*flags\s*=\s*(.+?)\s*$", re.M)
VARIANT_RE = re.compile(r"^\s*//\s*build:\s*variant\s*=\s*(\w+)\s+defines\s*=\s*([\w=,.]+)", re.M)
DEV_RE = re.compile(r"^\s*//\s*build:\s*dev\b", re.M)
WIDTH_GUARD_RE = re.compile(r"TMN_ONLY_CZ\s*==\s*(\d+)\s*&&\s*TMN_ONLY_CH\s*==\s*(\d+)")
# Built-in recipes for sources that carry no ``// build:`` lines of their own (the source's owner may replace them with header lines at any time):
# per-width variant naming + the flags their reference build script uses.  Keyed by path relative to csrc/.
RECIPES = {
    # "some_widths.cu": {"archs": ["sm_90a"], "width_variants": "tmn_{archtag}_z{cz}_h{ch}", "flags": ["-Xptxas", "--warn-on-spills"]},
}
ARCH_TAG = {"sm_90a": "sm90a", "sm_80": "sm80"}
DEFINE_RE = re.compile(r"^\s*#define\s+TMN_(CZ|CH)\s+(\d+)", re.M)


def _naming():
    """The op assembly's naming functions (kernel.py for sm_90a, sm80_ops.py for sm_80) -- the single authority for unit and kernel names.
    None when they cannot be imported here (they import the framework); the build then keeps file stems and records no served table."""
    try:
        from trimul_native import kernel as K
    except Exception as e:                                                  # noqa: BLE001
        return None, None, "kernel.py not importable (%s)" % str(e).splitlines()[0][:80]
    try:
        from trimul_native import sm80_ops as S
    except Exception as e:                                                  # noqa: BLE001
        return K, None, "sm80_ops.py not importable (%s)" % str(e).splitlines()[0][:80]
    return K, S, None


def compute_serves(unit, arch, kernel_names, naming):
    """What a built unit serves, decided by the naming authority: for an sm_90a width unit, per form ('b' bf16 z | 'f' fp32-resident z) whether
    the tile table's default K1 (mask / no mask) and K3 instantiations are in the cubin (fast) and whether the reference-order LayerNorm
    instantiations are (exact); for an sm_80 unit, which (c_z, c_hidden, form) rows of the member's tile table it holds.  The kernel names
    checked are recorded so the face can resolve them at load."""
    K, S, why = naming
    have = set(kernel_names)
    out = []
    if arch == "sm_90a":
        if K is None:
            return {"error": why}
        if not hasattr(K, "serve_names"):
            return {"error": "kernel.serve_names absent (the naming authority must list the kernels the op launches per width / form)"}
        for (a, cz, ch, form), tt in sorted(K.TILE_TABLE.items()):
            if a != arch or K.unit_name(arch, cz, ch) != unit:
                continue
            row = {"c_z": cz, "c_hidden": ch, "form": form, "fast": False, "exact": False, "kernels": [], "missing": []}
            if form == "f":
                row["k3_mode"] = tt.get("k3_mode", "f")
            try:
                need = list(K.serve_names(arch, cz, ch, form))              # exactly what the op launches with table defaults + default numerics
                need_x = list(K.serve_names(arch, cz, ch, form, exact=True))[len(need):] if form == "b" else []   # reference-order epilogue instantiations
            except Exception as e:                                          # noqa: BLE001 -- the authority could not name this width/form: not served, said why
                row["missing"].append("serve_names raised %s: %s" % (type(e).__name__, str(e).splitlines()[0][:80]))
                out.append(row)
                continue
            if not need:
                row["missing"].append("serve_names: none for this width/form")
                out.append(row)
                continue
            miss = [n for n in need if n not in have]
            row["fast"] = not miss
            row["kernels"] += [n for n in need if n in have]
            row["missing"] += miss
            if form == "b":
                miss_x = [n for n in need_x if n not in have]
                row["exact"] = row["fast"] and bool(need_x) and not miss_x
                row["kernels"] += [n for n in need_x if n in have]
                row["missing"] += miss_x
            out.append(row)
        return out
    if arch == "sm_80":
        if S is None:
            return {"error": why}
        part = {"trimul_k1_sm80": 0, "trimul_k3_sm80": 1}.get(unit)
        if part is None:
            return out
        for (C, Dh, zt), names in sorted(S.TILES.items()):
            name = names[part]
            out.append({"c_z": C, "c_hidden": Dh, "form": "b" if zt == "bf16" else "f", "part": "k1" if part == 0 else "k3", "kernel": name, "present": name in have})
        for (C, Dh, zt), names in sorted((getattr(S, "TILES_EXACT", None) or {}).items()):      # the bit-exact rows: K1 (stock order) in the K1 unit, both K3 variants in the K3 unit
            mine = [n for n in names if (n.startswith("k1") if part == 0 else n.startswith("k3"))]
            if mine:
                out.append({"c_z": C, "c_hidden": Dh, "form": "b" if zt == "bf16" else "f", "part": "k1x" if part == 0 else "k3x", "kernels": mine,
                            "present": all(n in have for n in mine), "missing": [n for n in mine if n not in have]})
        return out
    return out


def find_nvcc(explicit=None):
    cands = [explicit] if explicit else []
    for env in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(env):
            cands.append(os.path.join(os.environ[env], "bin", "nvcc"))
    cands += ["/usr/local/cuda/bin/nvcc", shutil.which("nvcc") or ""]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    sys.exit("[build] nvcc not found (tried %s); run inside an image with the CUDA toolkit or pass --nvcc" % [c for c in cands if c])


def nvcc_version(nvcc):
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
    rel = [l for l in out.splitlines() if "release" in l]
    build = [l for l in out.splitlines() if l.startswith("Build")]
    return {"release": (rel[0].strip() if rel else out.strip().splitlines()[-1]), "build": (build[0].strip() if build else ""), "path": nvcc}


def _head(cu_path, n=8192):
    with open(cu_path, encoding="utf-8") as f:
        return f.read(n)


def discover_units(src, archs_requested=None):
    """{unit name: {"source": path, "archs": [...] or None (= every requested arch), "defines": [...], "flags": [...], "dev": bool}}."""
    units = {}

    def add(name, spec):
        if name in units:
            sys.exit("[build] two units named %s (%s, %s)" % (name, units[name]["source"], spec["source"]))
        units[name] = spec
    for dp, _, fns in os.walk(src):
        reldir = os.path.relpath(dp, src)
        in_dev_dir = reldir.split(os.sep)[0].startswith("dev") if reldir != "." else False
        for fn in sorted(fns):
            if not fn.endswith(".cu"):
                continue
            path = os.path.join(dp, fn)
            rel = os.path.normpath(os.path.join(reldir, fn)) if reldir != "." else fn
            head = _head(path)
            whole = open(path, encoding="utf-8").read()
            m = ARCH_RE.search(head)
            archs = [a.strip() for a in re.split(r"[ ,]+", m.group(1)) if a.strip()] if m else None
            flags = []
            for fm in FLAGS_RE.finditer(head):
                flags += fm.group(1).split()
            dev = bool(DEV_RE.search(head)) or in_dev_dir
            recipe = RECIPES.get(rel.replace(os.sep, "/"), {})
            if archs is None and recipe.get("archs"):
                archs = list(recipe["archs"])
            flags = flags or list(recipe.get("flags", []))
            defs = dict(DEFINE_RE.findall(head))
            if "CZ" in defs and "CH" in defs and archs and len(archs) == 1:            # a width unit of the op assembly: its name is kernel.unit_name's
                K = _naming()[0]
                if K is not None:
                    want_name = K.unit_name(archs[0], int(defs["CZ"]), int(defs["CH"]))
                    if want_name != fn[:-3]:
                        sys.exit("[build] %s defines TMN_CZ=%s TMN_CH=%s but kernel.unit_name(%s, ...) is %r: rename the file or fix the table" % (rel, defs["CZ"], defs["CH"], archs[0], want_name))
            variants = [(vm.group(1), [d for d in vm.group(2).split(",") if d]) for vm in VARIANT_RE.finditer(head)]
            widths = sorted(set((int(a), int(b)) for a, b in WIDTH_GUARD_RE.findall(whole)))
            if variants:
                for name, defs in variants:
                    add(name, {"source": path, "rel": rel, "archs": archs, "defines": defs, "flags": flags, "dev": dev})
            elif widths and recipe.get("width_variants"):
                for arch in (archs or archs_requested or DEFAULT_ARCHS):
                    for cz, ch in widths:
                        name = recipe["width_variants"].format(archtag=ARCH_TAG.get(arch, arch.replace("_", "")), cz=cz, ch=ch)
                        add(name, {"source": path, "rel": rel, "archs": [arch], "defines": ["TMN_ONLY_CZ=%d" % cz, "TMN_ONLY_CH=%d" % ch], "flags": flags, "dev": dev})
            else:
                add(fn[:-3], {"source": path, "rel": rel, "archs": archs, "defines": [], "flags": flags, "dev": dev})
    return units


def unit_archs(spec, requested):
    declared = spec.get("archs")
    return list(requested) if not declared else [a for a in requested if a in declared]


def unit_dynamic_smem(cu_path):
    with open(cu_path, encoding="utf-8") as f:
        head = f.read(4096)
    m = SMEM_RE.search(head)
    return json.loads(m.group(1)) if m else {}


PTXAS_FUNC_RE = re.compile(r"Compiling entry function '([^']+)' for '([^']+)'")
PTXAS_USED_RE = re.compile(r"Used (\d+) registers(.*)")
PTXAS_SPILL_RE = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")


def parse_ptxas(log):
    """{entry: {regs, smem, cmem0, stack, spill_stores, spill_loads, barriers}} from a -Xptxas -v log."""
    out, cur, pending_spill = {}, None, None
    for line in log.splitlines():
        m = PTXAS_FUNC_RE.search(line)
        if m:
            cur = m.group(1)
            out[cur] = {"arch": m.group(2)}
            if pending_spill:
                out[cur].update(pending_spill)
                pending_spill = None
            continue
        m = PTXAS_SPILL_RE.search(line)
        if m:
            d = {"stack": int(m.group(1)), "spill_stores": int(m.group(2)), "spill_loads": int(m.group(3))}
            if cur is not None and "regs" not in out[cur]:
                out[cur].update(d)
            else:
                pending_spill = d
            continue
        m = PTXAS_USED_RE.search(line)
        if m and cur is not None:
            out[cur]["regs"] = int(m.group(1))
            rest = m.group(2)
            for key, pat in (("smem", r"(\d+) bytes smem"), ("cmem0", r"(\d+) bytes cmem\[0\]"), ("barriers", r"used (\d+) barriers")):
                mm = re.search(pat, rest)
                if mm:
                    out[cur][key] = int(mm.group(1))
            cur = None
    return out


def deps(nvcc, cu_rel, scratch, arch, flags):
    """Source files under the scratch csrc/ the unit depends on (nvcc -M), as paths relative to the scratch root."""
    r = subprocess.run([nvcc, "-M", "-arch=" + arch] + [f for f in flags if f not in ("-Xptxas", "-v")] + [cu_rel], cwd=scratch, capture_output=True, text=True)
    if r.returncode != 0:
        return [cu_rel]
    found = []
    for tok in r.stdout.replace("\\\n", " ").split():
        tok = tok.strip()
        if tok.endswith(":") or not tok:
            continue
        p = os.path.normpath(os.path.join(scratch, tok)) if not os.path.isabs(tok) else os.path.normpath(tok)
        if p.startswith(os.path.join(scratch, "csrc") + os.sep) and os.path.isfile(p):
            rel = os.path.relpath(p, scratch)
            if rel not in found:
                found.append(rel)
    return found or [cu_rel]


def build_unit(nvcc, nv, unit, spec, arch, src, out, base_flags):
    """Compile one unit for one arch; returns (ok, "<unit>/<arch>", manifest entry or None)."""
    cu_path = spec["source"]
    cu_dir_rel = os.path.relpath(os.path.dirname(cu_path), os.path.dirname(src))     # csrc or csrc/<member>
    flags = (list(base_flags) + ["-I", "csrc"] + (["-I", cu_dir_rel] if cu_dir_rel != "csrc" else []) + list(spec.get("flags", []))
             + ["-D" + d for d in spec.get("defines", [])])                            # includes resolve against csrc/ and the unit's own directory
    scratch = tempfile.mkdtemp(prefix="tn_build_")
    try:
        shutil.copytree(src, os.path.join(scratch, "csrc"))
        cu_rel = os.path.relpath(cu_path, os.path.dirname(src))                       # csrc/<...>/<file>.cu
        arch_dir = os.path.join(out, arch)
        os.makedirs(arch_dir, exist_ok=True)
        cubin_rel = os.path.join(arch, unit + ".cubin")
        log_rel = os.path.join(arch, unit + ".ptxas.txt")
        cmd = [nvcc, "-cubin", "-arch=" + arch] + flags + ["-o", os.path.join(scratch, unit + ".cubin"), cu_rel]
        r = subprocess.run(cmd, cwd=scratch, capture_output=True, text=True)
        log = (r.stderr or "") + (r.stdout or "")
        with open(os.path.join(out, log_rel), "w", encoding="utf-8") as f:
            f.write("$ " + " ".join(["nvcc"] + cmd[1:]) + "\n" + log)
        if r.returncode != 0:
            print("[build] FAILED %s/%s (rc %d):\n%s" % (unit, arch, r.returncode, log[-4000:]), file=sys.stderr)
            return False, "%s/%s" % (unit, arch), None
        shutil.copy2(os.path.join(scratch, unit + ".cubin"), os.path.join(out, cubin_rel))
        with open(os.path.join(out, cubin_rel), "rb") as f:
            cub = f.read()
        srcs = {}
        for rel in deps(nvcc, cu_rel, scratch, arch, flags):
            srcs[rel] = M.sha256_file(os.path.join(scratch, rel))
        kern = parse_ptxas(log)
        dyn = unit_dynamic_smem(cu_path)
        for k, v in dyn.items():
            if k in kern:
                kern[k]["max_dynamic_smem"] = int(v)
        try:
            serves = [] if spec.get("dev") else compute_serves(unit, arch, list(kern.keys()), _naming())
        except Exception as e:                                              # noqa: BLE001 -- a naming-authority failure must not lose the build record
            serves = {"error": "compute_serves raised %s: %s" % (type(e).__name__, str(e).splitlines()[0][:120])}
        ent = {"unit": unit, "arch": arch, "cubin": cubin_rel, "cubin_sha256": M.sha256_bytes(cub), "cubin_bytes": len(cub), "sources": srcs,
               "source": spec.get("rel", os.path.basename(cu_path)), "defines": list(spec.get("defines", [])), "dev": bool(spec.get("dev")), "serves": serves,
               "nvcc": nv["release"], "nvcc_build": nv["build"], "flags": ["-cubin", "-arch=" + arch] + flags,
               "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "elf": M.elf_facts(cub), "kernels": kern, "ptxas_log": log_rel}
        spills = [k for k, v in kern.items() if v.get("spill_stores")]
        nk = len([k for k in kern if k != "tmn_info"])
        regs = ", ".join("%s: %s regs %s B smem%s" % (k, v.get("regs"), v.get("smem", 0), (" SPILL %d/%d" % (v["spill_stores"], v["spill_loads"])) if v.get("spill_stores") else "")
                         for k, v in sorted(kern.items())) if nk <= 4 else "%d kernels, regs %d..%d%s" % (
                             nk, min(v.get("regs", 0) for v in kern.values()), max(v.get("regs", 0) for v in kern.values()), (", %d WITH SPILLS: %s" % (len(spills), ",".join(spills[:3])) if spills else ", no spills"))
        print("[build] %s/%s -> %s (%d B, sha256 %s..) [%s]" % (unit, arch, cubin_rel, len(cub), ent["cubin_sha256"][:12], regs))
        if isinstance(serves, dict) and serves.get("error"):
            print("[build]   serves: NOT RECORDED (%s) -- the face will not serve this unit" % serves["error"])
        elif arch == "sm_90a" and serves:
            print("[build]   serves: %s" % ", ".join("z%d_h%d_%s:%s%s" % (r["c_z"], r["c_hidden"], "f32z" if r["form"] == "f" else "bf16",
                  "fast+exact" if r["exact"] else ("fast" if r["fast"] else "NONE"), (" missing " + ",".join(r["missing"])) if r["missing"] else "") for r in serves))
        elif arch == "sm_80" and serves:
            miss = [r["kernel"] for r in serves if not r["present"]]
            print("[build]   serves: %d/%d tile-table kernels present%s" % (len(serves) - len(miss), len(serves), (" missing " + ",".join(miss)) if miss else ""))
        return True, "%s/%s" % (unit, arch), ent
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archs", default=",".join(DEFAULT_ARCHS))
    ap.add_argument("--units", default="", help="comma list of unit names or glob patterns (default: every non-development unit)")
    ap.add_argument("--all", action="store_true", help="every non-development unit for every architecture (the default selection; explicit for scripts)")
    ap.add_argument("--dev", action="store_true", help="also build development units (csrc/dev*/, '// build: dev')")
    ap.add_argument("--list", action="store_true", help="print the unit table and exit")
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)), help="parallel nvcc processes")
    ap.add_argument("--src", default=os.path.join(ROOT, "csrc"))
    ap.add_argument("--out", default=os.path.join(ROOT, "build"))
    ap.add_argument("--nvcc", default=None)
    ap.add_argument("--fast-math", action="store_true", help="add --use_fast_math (OFF by default: it changes numerics)")
    ap.add_argument("--lineinfo", action="store_true", help="add -lineinfo (OFF for release builds)")
    ap.add_argument("--define", action="append", default=[], help="K=V preprocessor definition added to every unit (recorded in the manifest flags)")
    a = ap.parse_args(argv)
    import fnmatch
    from concurrent.futures import ThreadPoolExecutor
    archs = [x.strip() for x in a.archs.split(",") if x.strip()]
    src, out = os.path.abspath(a.src), os.path.abspath(a.out)
    units = discover_units(src, archs)
    if a.list:
        for name in sorted(units):
            u = units[name]
            print("%-28s %-28s archs=%-14s defines=%-32s flags=%s%s" % (name, u["rel"], ",".join(u["archs"] or ["*"]), ",".join(u["defines"]) or "-", " ".join(u["flags"]) or "-", "  [dev]" if u["dev"] else ""))
        return 0
    pats = [] if a.all else [u.strip() for u in a.units.split(",") if u.strip()]
    if pats:
        want = [n for n in sorted(units) if any(fnmatch.fnmatch(n, p) for p in pats)]
        unknown = [p for p in pats if not any(fnmatch.fnmatch(n, p) for n in units)]
        if unknown:
            sys.exit("[build] no unit matches %s under %s (have %s)" % (unknown, src, sorted(units)))
    else:
        want = [n for n in sorted(units) if a.dev or not units[n]["dev"]]
    nvcc = find_nvcc(a.nvcc)
    nv = nvcc_version(nvcc)
    flags = list(BASE_FLAGS) + (["--use_fast_math"] if a.fast_math else []) + (["-lineinfo"] if a.lineinfo else []) + ["-D" + d for d in a.define]
    os.makedirs(out, exist_ok=True)
    try:
        man = M.load(out)
    except M.ManifestError:
        man = {"units": {}}
    man["nvcc"] = nv
    man["schema"] = M.SCHEMA
    man["package"] = "trimul_native"
    jobs = [(u, arch) for u in want for arch in unit_archs(units[u], archs)]
    print("[build] nvcc: %s | %s ; %d unit x arch jobs, %d in parallel" % (nv["release"], nv["build"], len(jobs), a.jobs))
    ok = True
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for good, key, ent in ex.map(lambda ja: build_unit(nvcc, nv, ja[0], units[ja[0]], ja[1], src, out, flags), jobs):
            ok &= good
            if ent is not None:
                man.setdefault("units", {})[key] = ent
    stale = [k for k, e in man["units"].items()
             if not os.path.isfile(os.path.join(src, os.path.relpath(e.get("source", ""), ".") if not e.get("source", "").startswith("csrc") else e["source"][5:]))
             and not os.path.isfile(os.path.join(os.path.dirname(src), "csrc", e.get("source", "")))
             or not os.path.isfile(os.path.join(out, e.get("cubin", "")))]
    for k in stale:                                                         # units whose source left the tree or whose cubin is gone
        ent = man["units"].pop(k)
        for rel in (ent.get("cubin"), ent.get("ptxas_log")):
            if rel and os.path.isfile(os.path.join(out, rel)):
                os.remove(os.path.join(out, rel))
        print("[build] pruned stale unit %s (source %s)" % (k, ent.get("source")))
    M.save(out, man)
    print("[build] manifest -> %s (%d entries)%s" % (M.path(out), len(man["units"]), "" if ok else "  -- WITH FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

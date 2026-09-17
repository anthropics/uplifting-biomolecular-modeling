#!/usr/bin/env python3
"""build_prebuilt.py — the BUILD route of the kit's shipped CUDA C++ / CuTe kernels (driver/prebuilt/<arch>/): compile every kernel a driver module
declares (ef2_nvjit.kernel_specs() over ef2_nvjit.PREBUILD_MODULES — empty in this tree: no cubin ships, the manifest's "cubins" is {})
with NVRTC from the header wheels (nvidia-cutlass 4.2.x: cute/ cutlass/; nvidia-cuda-cccl; the CUDA 13 runtime headers), write <stem>.cubin +
manifest.json + PROVENANCE.md, and — on a compute-capability-9.0 GPU — load each cubin through the run-time path (ef2_nvjit: shipped cubin ->
cuModuleLoadData) and run the modules' load checks, recording registers / spill / shared memory and the load-check output digests.

  python build_prebuilt.py build   [--arch sm_90a]      compile (no GPU needed) + manifest; device fields filled when a 9.0 GPU is visible
  python build_prebuilt.py check   [--write]            GPU, no headers needed: load the shipped cubins through the run-time path, run the load
                                                        checks, print the device record as JSON (--write: merge it into manifest.json / PROVENANCE.md)
  python build_prebuilt.py annotate --from check.json   merge a `check` record taken on another box into manifest.json / PROVENANCE.md

The run time never compiles when the shipped cubin's source_key (sha256 of source text + compile options) equals the tree's: editing
a kernel source or its compile options without rebuilding here makes `run.sh install` fail by name (ef2_nvjit's `cubins` check).
"""
import argparse, datetime, hashlib, json, os, platform, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.dirname(HERE)
sys.path.insert(0, DRIVER)
import ef2_nvjit as J  # noqa: E402


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _versions():
    from importlib import metadata
    out = {}
    for dist in ("nvidia-cuda-nvrtc", "nvidia-cutlass", "nvidia-cuda-cccl", "nvidia-cuda-runtime", "cuda-bindings", "torch"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def manifest_path(arch):
    return os.path.join(J.shipped_dir(arch), J.MANIFEST)


def load_manifest(arch):
    m = J.shipped_manifest(arch)
    return m if m else {"arch": arch, "kit": "esmfold2", "helper": "ef2_nvjit", "cubins": {}}


def write_manifest(arch, man):
    os.makedirs(J.shipped_dir(arch), exist_ok=True)
    with open(manifest_path(arch), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, sort_keys=True); f.write("\n")
    write_provenance(arch, man)


def write_provenance(arch, man):
    lines = [f"# esmfold2 kit — shipped {arch} cubins (driver/prebuilt/{arch}/)", "",
             "Built by `driver/prebuilt/build_prebuilt.py build` with NVRTC from the header wheels named below (each cubin's sha256 is also its `SHA256SUMS` line here,",
             "re-hashed before loading); loaded at run time by `driver/ef2_nvjit.py` through the CUDA",
             "driver API (cuda-bindings) when the manifest's `source_key` (sha256 of the kernel source text + compile options) equals the tree's. No header wheel and no",
             "compiler is needed at run time; a tree whose kernel source moved refuses `run.sh install` by name until this directory is rebuilt.", ""]
    b = man.get("build") or {}
    if b:
        lines.append(f"- build: {b.get('built_utc')} — nvrtc {b.get('nvrtc')} (nvidia-cuda-nvrtc {b.get('versions', {}).get('nvidia-cuda-nvrtc')}), nvidia-cutlass "
                     f"{b.get('versions', {}).get('nvidia-cutlass')}, nvidia-cuda-cccl {b.get('versions', {}).get('nvidia-cuda-cccl')}, nvidia-cuda-runtime "
                     f"{b.get('versions', {}).get('nvidia-cuda-runtime')}; python {b.get('python')}; host {b.get('host')}")
    c = man.get("check") or {}
    if c:
        lines.append(f"- load check: {c.get('utc')} on {c.get('gpu')} (driver {c.get('driver')}, torch {c.get('torch')}, cuda-bindings {c.get('cuda_bindings')}): every cubin loaded through "
                     "ef2_nvjit's shipped route and launched once against its module's reference (digests below; an install re-runs the same check and compares)")
    lines.append("")
    for stem, e in sorted((man.get("cubins") or {}).items()):
        lines += [f"## {e.get('file')}",
                  f"- lever {e.get('lever')} — module `{e.get('module')}` ({e.get('tag')}), kernel entry `{e.get('entry')}`, source `{e.get('source_file')}` sha256 {str(e.get('source_sha256'))[:16]}…",
                  f"- compile options: `{' '.join(e.get('options') or [])}`",
                  f"- source_key {e.get('source_key')}",
                  f"- cubin sha256 {e.get('cubin_sha256')} ({e.get('bytes')} B), compiled in {e.get('compile_secs')} s",
                  f"- ptxas: {e.get('regs_ptxas')} registers, stack frame {e.get('stack_bytes')} B, spill stores {e.get('spill_stores')} B, spill loads {e.get('spill_loads')} B "
                  "(a build with any of these non-zero, or with a C75xx advisory other than C7517, is refused: spill-free is an enforced build property of the shipped object, "
                  "so that the audited binary is the binary that runs); "
                  f"scheduling fingerprint: C7517 x{e.get('c7517_count')} {[d['text'][:80] for d in (e.get('diagnostics') or [])]}; SASS scan verdict: {e.get('sass_scan')}",
                  f"- the ptxas numbers are read from a twin build with `--ptxas-options=-v` (sha256 `{str(e.get('ptxas_twin_sha256'))[:12]}…`): that option is recorded in the cubin's "
                  "`.note.nv.tkinfo` note and nowhere else, so the shipped cubin is built without it and the build refuses unless every other ELF section of the twin is identical",
                  f"- device record: regs/thread {e.get('regs')}, local bytes {e.get('spill_bytes')}, dynamic shared memory {e.get('smem')} B, load check {e.get('loadcheck')}"]
        if e.get("audited_sha256"):
            lines.append(f"- audited object: sha256 {e.get('audited_sha256')} — the build refuses to write, and the loader to run, any other object under this source")
        if e.get("reviewer_scan"):
            r = e["reviewer_scan"]; lines.append(f"- reviewer SASS scan: {r.get('verdict')} — {r.get('summary')} (artifact {r.get('artifact')})")
        if e.get("release_ordering_rule"):
            lines.append(f"- release-ordering rule: {e['release_ordering_rule']}")
        if e.get("register_split_rule"):
            lines.append(f"- register-split rule: {e['register_split_rule']}")
        if e.get("sass_release_coverage"):
            lines.append("- release-site coverage (ldmatrix reads scoreboard-covered before the slot's release arrive; own + via a later load of the same warp): "
                         + "; ".join(f"{c.get('site')}: {c.get('n_ldsm')} loads, own {c.get('own')} + via-later {c.get('via_later')}, uncovered {c.get('uncovered')}, fences {c.get('fences')}" for c in e["sass_release_coverage"]))
        if e.get("determinism_stress"):
            d = e["determinism_stress"]; lines.append(f"- determinism stress: {d.get('protocol')}: {d.get('launches')} launches, {d.get('differing')} differing; {d.get('note', '')}".rstrip("; "))
        if e.get("guard_proxy"):
            lines.append(f"- guard-allocation proxy: {e['guard_proxy']}")
        if e.get("e2e_identity"):
            lines.append(f"- end-to-end identity: {e['e2e_identity']}")
        lines.append("")
    with open(os.path.join(J.shipped_dir(arch), "PROVENANCE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


LEVER_OF = {}                                                            # module stem -> (word, kernel entry, source): no kit module ships a cubin in this tree
BUILD_FIELDS = ("file", "cubin_sha256", "bytes", "source_key", "source_sha256", "source_file", "module", "tag", "lever", "entry", "options", "macros", "nvrtc", "arch", "compile_secs", "smem",
                "regs_ptxas", "stack_bytes", "spill_stores", "spill_loads", "c7512", "ptxas_lines", "diagnostics", "c7517_count", "spill_free_required", "ptxas_twin_sha256", "ptxas_source", "audited_sha256")
KEEP_ON_SAME_OBJECT = ("regs", "spill_bytes", "loadcheck", "loadcheck_digest", "loadcheck_max_abs", "loadcheck_max_abs_fp32", "canary_digest", "sass_scan", "reviewer_scan", "determinism_stress",
                       "sass_release_coverage", "register_split_rule", "release_ordering_rule", "guard_proxy", "e2e_identity", "audit_artifacts",
                       "cfg", "generation", "ptxas_fingerprint", "root_cause_of_the_parked_build", "determinism_stress_package")   # per-object evidence / audit fields: kept while the
                                                                                                                                    # rebuilt object is byte-identical, cleared when it changes


def cmd_build(a):
    ok, why = J.toolchain()
    if not ok:
        sys.exit(f"build_prebuilt: the NVRTC build route is unavailable in this interpreter: {why}")
    tmp = tempfile.mkdtemp(prefix="ef2_prebuilt_")
    pairs = J.build_all(dest=tmp, arch=a.arch)
    man = load_manifest(a.arch)
    man.update(arch=a.arch, kit="esmfold2", helper="ef2_nvjit")
    man["build"] = dict(built_utc=utc(), nvrtc="%d.%d" % J.nvrtc_version(), versions=_versions(), python=sys.version.split()[0], host=f"{platform.system()}-{platform.machine()}")
    os.makedirs(J.shipped_dir(a.arch), exist_ok=True)
    for spec, cub in pairs:
        stem = J.stem_of(spec["name"])
        fn = stem + ".cubin"
        shutil.copyfile(cub.path, os.path.join(J.shipped_dir(a.arch), fn))
        lever, entry, srcfile = LEVER_OF.get(stem, ("?", "?", spec["name"]))
        prev = (man["cubins"].get(stem) or {})
        same = prev.get("cubin_sha256") == cub.sha256                  # an unchanged object keeps its device record, canary digest, review verdict and audit fields; a new object clears them
        keep = {k: v for k, v in prev.items() if same and (k in KEEP_ON_SAME_OBJECT or k not in BUILD_FIELDS)}   # the SAME object keeps every per-object evidence / audit field it carried
        if spec.get("audited_sha256") and cub.sha256 != spec["audited_sha256"]:
            sys.exit(f"build_prebuilt: {stem}: built {cub.sha256[:12]} but the module names the audited object {spec['audited_sha256'][:12]} — not shipping")
        st = cub.ptxas or {}                                            # ef2_nvjit refused the build already unless stack / spill stores / spill loads are all 0 and no C7512
        man["cubins"][stem] = dict(file=fn, cubin_sha256=cub.sha256, bytes=len(cub.data), source_key=cub.skey, source_sha256=hashlib.sha256(spec["src"].encode()).hexdigest(),
                                   source_file=srcfile, module=spec["module"], tag=spec.get("tag"), lever=lever, entry=entry, options=list(cub.options),
                                   macros=dict(spec.get("macros") or {}), nvrtc="%d.%d" % cub.nvrtc, arch=a.arch, compile_secs=round(cub.secs, 1),
                                   smem=_smem_of(spec["module"]), regs_ptxas=st.get("regs"), stack_bytes=st.get("stack_bytes"), spill_stores=st.get("spill_stores"),
                                   spill_loads=st.get("spill_loads"), c7512=bool(st.get("c7512")), ptxas_lines=[l for l in st.get("lines", []) if "info" in l or "stack frame" in l][:12],
                                   diagnostics=list(st.get("diagnostics", [])), c7517_count=sum(1 for d in st.get("diagnostics", []) if d["code"] == "C7517"),
                                   spill_free_required=bool(spec.get("spill_free", True)),   # the spec's policy: the loader holds a spill-free kernel to 0 local bytes; the other's local bytes to the device record
                                   ptxas_twin_sha256=st.get("twin_sha256"), ptxas_source=st.get("twin_note"),   # the statistics come from a -v twin, section-identical but for .note.nv.tkinfo
                                   audited_sha256=spec.get("audited_sha256"),                  # the audited object this build must (and did) reproduce, when the module names one
                                   sass_scan=keep.get("sass_scan"),                            # the reviewer's SASS scan verdict of THIS binary (filled at review; a rebuild clears it)
                                   regs=keep.get("regs"), spill_bytes=keep.get("spill_bytes"), loadcheck=keep.get("loadcheck"),
                                   loadcheck_digest=keep.get("loadcheck_digest"), loadcheck_max_abs=keep.get("loadcheck_max_abs"))
        for k, v in keep.items():                                       # review / audit / evidence fields merged by `annotate --fields` survive a rebuild of the same object
            man["cubins"][stem].setdefault(k, v)
            if man["cubins"][stem].get(k) is None:
                man["cubins"][stem][k] = v
    write_manifest(a.arch, man)
    with open(os.path.join(J.shipped_dir(a.arch), "SHA256SUMS"), "w", encoding="utf-8") as f:   # the loader (ef2_nvjit.shipped_cubin) holds every shipped cubin to its line here
        f.write("".join(f"{e['cubin_sha256']}  {e['file']}\n" for _, e in sorted(man["cubins"].items())))
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({stem: {k: e[k] for k in ("file", "cubin_sha256", "bytes", "source_key", "compile_secs")} for stem, e in man["cubins"].items()}, indent=1))
    if _gpu90():
        rec = device_check(a.arch)
        merge_check(a.arch, rec)
        print(json.dumps(rec, indent=1))
    else:
        print("build_prebuilt: no compute-capability-9.0 GPU in this process — run `build_prebuilt.py check --write` on one to record regs / spill / load-check digests", file=sys.stderr)


def _smem_of(module):
    return None


def _gpu90():
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    except Exception:  # noqa: BLE001
        return False


def device_check(arch):
    """Load every shipped cubin through the run-time path (ef2_nvjit.shipped_cubin) and run the modules' install canaries. -> record dict."""
    import torch
    J.check_shipped(arch)                                                # CPU part first: the set matches this tree
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda_bindings": _versions().get("cuda-bindings"), "utc": utc(), "cubins": {}}
    try:
        from cuda.bindings import driver as cu
        err, v = cu.cuDriverGetVersion(); out["driver"] = f"{v // 1000}.{(v % 1000) // 10}"
    except Exception:  # noqa: BLE001
        out["driver"] = None
    return out


def merge_check(arch, rec):
    man = load_manifest(arch)
    for stem, d in rec["cubins"].items():
        e = man["cubins"].get(stem)
        if not e or e.get("cubin_sha256") != d["cubin_sha256"]:
            sys.exit(f"build_prebuilt: the check record's {stem} cubin sha256 {d['cubin_sha256'][:12]} is not the manifest's (check the same tree you built)")
        e.update(regs=d["regs"], spill_bytes=d["spill_bytes"], smem=d["smem"], loadcheck=d["loadcheck"], loadcheck_digest=d["loadcheck_digest"], loadcheck_max_abs=d["loadcheck_max_abs"])
        if "canary_digest" in d:
            e["canary_digest"] = d["canary_digest"]
    man["check"] = {k: rec.get(k) for k in ("gpu", "driver", "torch", "cuda_bindings", "utc")}
    write_manifest(arch, man)


def cmd_check(a):
    if not _gpu90():
        sys.exit("build_prebuilt check: needs a compute-capability-9.0 GPU (H100 / H200) in this process")
    rec = device_check(a.arch)
    if a.write:
        merge_check(a.arch, rec)
    print(json.dumps(rec, indent=1))


def cmd_annotate(a):
    if a.src:
        with open(a.src, encoding="utf-8") as f:
            rec = json.load(f)
        merge_check(a.arch, rec)
        print(f"build_prebuilt: merged the check record of {rec.get('gpu')} ({rec.get('utc')}) into {manifest_path(a.arch)}")
    if a.fields:                                                          # {stem: {field: value}} — review verdicts / stress records / rules attached to the object NOW in the manifest
        with open(a.fields, encoding="utf-8") as f:
            fields = json.load(f)
        man = load_manifest(a.arch)
        for stem, kv in fields.items():
            if stem not in man["cubins"]:
                sys.exit(f"build_prebuilt annotate: no cubin {stem} in {manifest_path(a.arch)}")
            want = kv.pop("for_cubin_sha256", None)
            if want and man["cubins"][stem]["cubin_sha256"] != want:
                sys.exit(f"build_prebuilt annotate: {stem}: the fields are for object {want[:12]}, the manifest holds {man['cubins'][stem]['cubin_sha256'][:12]}")
            bad = [k for k in kv if k in BUILD_FIELDS]
            if bad:
                sys.exit(f"build_prebuilt annotate: {stem}: {bad} are build fields (written by `build` only)")
            man["cubins"][stem].update(kv)
        write_manifest(a.arch, man); write_provenance(a.arch, man)
        print(f"build_prebuilt: merged fields for {', '.join(fields)} into {manifest_path(a.arch)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default=J.ARCH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    c = sub.add_parser("check"); c.add_argument("--write", action="store_true")
    n = sub.add_parser("annotate"); n.add_argument("--from", dest="src", default=None); n.add_argument("--fields", default=None)
    a = ap.parse_args()
    {"build": cmd_build, "check": cmd_check, "annotate": cmd_annotate}[a.cmd](a)


if __name__ == "__main__":
    main()

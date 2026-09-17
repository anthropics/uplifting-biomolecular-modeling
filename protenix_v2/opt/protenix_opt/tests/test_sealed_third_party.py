"""The vendored / producer packages under opt/forward/flashpairformer/third_party/ are SEALED: byte-frozen at their published bytes (they are the
shared core's lift sources and other units' partners). This test fails BY NAME on any edit, addition or removal inside the tree: kit behaviour
changes go in opt/protenix_opt/, from the outside (a wrapper, a subclass, a rebinding at install time, an import at activation — see
sampler_poison_aside.py and route_preload.py for the patterns), never in the sealed files.

SOURCE OF TRUTH = the ``PINNED`` literals below: per kept file the sha256 and size of its published bytes. The json file
(tests/data/sealed_third_party.json) is a convenience listing only and must agree with the literals; nothing a kit commit can regenerate
blesses an edit.
  * RE-SEAL (a sealed file legitimately changes — a producer publishes a new package version, or a notice's wording is revised): edit the
    literal BY HAND in the same commit as the file and state the reason in a comment on that line. ``--reseal`` rewrites the json listing
    only; it cannot change what this test asserts.
  * RETIRING a package (the shared core serves the identical kernel; the kit binds it by tier word) stays allowed: delete the package AND its
    literals here in the same commit (the kit's own glue — fpf_clisampler, fpf_stackgraph, fpf_trunkgraph, ptx_lazy_init, ptx_msa_adapt —
    lives under src/: kit code, not sealed lift sources).
  * Five kept entries (protenix_fpf_ditfast/atom_fast.py, dit_fast.py, protenix_fpf_msa/__init__.py, install.py, loadcheck.py) carry the
    digests of their current bytes, which differ from the published ones in IMPORT LINES ONLY (their donors are opt_core.kernels.apb.ditfast /
    opt_core.ops.msa_fused / opt_core.kernels.apb.fpf_apb); logic unchanged. Flagged on their lines below.
"""
import hashlib
import json
import os
import sys

from protenix_opt import stack

TREE = os.path.join(os.path.abspath(stack.kit_home()), "third_party")                  # kit_home = opt/forward/flashpairformer
RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sealed_third_party.json")
GRAPHED_PY = os.path.join(os.path.dirname(TREE), "src", "infopt_graphs", "protenix", "graphed.py")   # the graphed sampler: kit code under src/ since 0.3.51 (left the sealed tree with kit112_src), still the published blob
SKIP_DIRS = ("__pycache__",)                                                            # interpreter by-products, never part of the tree
SKIP_SUFFIXES = (".pyc", ".pyo")

PINNED = {   # path under third_party/ -> (sha256, bytes) of the published bytes; edited by hand only, in the same commit as the sealed file, with the reason on the line
    "blockfuse_addon/LICENSE": ("add3b0d4449b29559d6f429d84a09a9b363d78fd516b3315a8ed946eb12ad8eb", 11788),
    "blockfuse_addon/NOTICE": ("7a93d7c0fe5979c4dc34291730fbf53b7c3d505365ae6cc48f75c2ff5bdce743", 2079),                          # notice wording revised: copyright holder line named, the LICENSE file beside it explained (text only)
    "blockfuse_addon/VENDORED.txt": ("bddca30358e76e610b267afbd730bcc2b9e3475a8b47fb815c5e6d8069e05653", 444),   # wording revised; digest re-pinned (text only, no code or binary change)
    "blockfuse_addon/blockfuse/__init__.py": ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0),
    "blockfuse_addon/blockfuse/biasln.py": ("a8f185e38769a6f105fdea2bc560ae02267686a022be3f1c9f3071d4dbe884a2", 4678),
    "blockfuse_addon/blockfuse_xl/__init__.py": ("5052a7c1ea26005052c9e440da08d944531f6051ec15028da9fca42f3e44bbc1", 10313),
    "fastln_prebuilt.py": ("5cb7f6f97ecc19d8699a7587b5d3f90342ea33037cb4b8a07911af33f1dacf35", 9702),
    "fastln_prebuilt/fast_layer_norm_cuda_v2_stream.so": ("c295028752de1efc9a714d371f9ef8b467f60943ad3183bcee0afd925290a0a4", 3092032),
    "fastln_prebuilt/SHA256SUMS": ("da652fdee7dac631720bd7b2d2a25cd1e0dbd6da186ec960946bb55a65eb7be2", 100),                    # added beside the .so: its sha256sum line, re-hashed before the extension is loaded (protenix_opt.binary_sums)
    "fastln_prebuilt/manifest.json": ("4d13388d7ef0d9e2523d436f2b9db715887eab0dc442f799b430cb7746016f77", 1928),   # wording revised; digest re-pinned (text only, no code or binary change)
    "fastln_prebuilt/patched_src/compat.h": ("24da65d5e0a58c6483abd60e7a6d6fca7e4abc9303f08c902975c77bdeacbc08", 816),
    "fastln_prebuilt/patched_src/layer_norm_cuda.cpp": ("b03d995042fede126b1e36a8e9c9e73433a598f523ad914789d7aeaeb8ac842d", 8097),
    "fastln_prebuilt/patched_src/layer_norm_cuda_kernel.cu": ("fc1540b4410728ffd8c0b761fd4c3494290c5a6256e40d84a928f608c6cbbc36", 45539),
    "fastln_prebuilt/patched_src/type_shim.h": ("824a0dfdb5bfb1316187b3a3f7c10c9b41496b97259ddf84c882152ccec66a7a", 13237),
    "fastln_prebuilt_cu130/PROVENANCE.md": ("691092b54a2bbfec0ce2a88adbe869c78c6ca9855b2862d15fb52739a820dc2d", 968),   # PROVENANCE.md wording revised; digest re-pinned (text only, no code or binary change)
    "fastln_prebuilt_cu130/fast_layer_norm_cuda_v2_stream.so": ("ab5ce7d841e9376e346131449f141f87510a3011334965cce3b9e4eb8b06ca4b", 3453104),
    "fastln_prebuilt_cu130/SHA256SUMS": ("c3c2869eba0b722c61f2be10ad624bda7b12d72ba2db891ec5c108f49eb0fc0c", 100),              # added beside the .so: its sha256sum line, re-hashed before the extension is loaded (protenix_opt.binary_sums)
    "fastln_prebuilt_cu130/manifest.json": ("dc04f0d40b306732e36e1e4c3b1abdc8a4dd6a4bc7b8bc8c1f4018a5fbcd8f58", 1928),   # wording revised; digest re-pinned (text only, no code or binary change)
    "fastln_prebuilt_cu130/patched_src/compat.h": ("24da65d5e0a58c6483abd60e7a6d6fca7e4abc9303f08c902975c77bdeacbc08", 816),
    "fastln_prebuilt_cu130/patched_src/layer_norm_cuda.cpp": ("b03d995042fede126b1e36a8e9c9e73433a598f523ad914789d7aeaeb8ac842d", 8097),
    "fastln_prebuilt_cu130/patched_src/layer_norm_cuda_kernel.cu": ("fc1540b4410728ffd8c0b761fd4c3494290c5a6256e40d84a928f608c6cbbc36", 45539),
    "fastln_prebuilt_cu130/patched_src/type_shim.h": ("824a0dfdb5bfb1316187b3a3f7c10c9b41496b97259ddf84c882152ccec66a7a", 13237),
    "fpf_cueq_pad8exact/__init__.py": ("4915ed2df6ecab8a1165b90daa5c0d37de2b6c2c3a9e29c7578917f2ff3c80a2", 13286),
    "fpf_pad8exact/__init__.py": ("26955a1c495546f156ecdaa7f3140e60e6e2d9d235d6f028831fa5ff62515059", 2255),                       # line 1 (copyright holder comment) revised; code unchanged
    "fpf_smalln/__init__.py": ("be35aefa0f1ae3e4c95c3d508946b1117dc345ff8851ccdd9db38b69951b2318", 13385),                         # line 1 (copyright holder comment) and the module docstring's first line revised; code unchanged
    "protenix_fpf_ditfast/CELLS.json": ("32c5fb59550192b4368a05ee8fdedd20e0a21128ebf10fa8cee536eaa663b7cd", 4286),
    "protenix_fpf_ditfast/NOTICE": ("42f8ef55018cb40fce16bd86a0181e796e7601d5097ecafc3aec8d405b4f351b", 1148),                        # re-sealed: the notice reworded kit-neutrally (same bytes in the shared core and every kit copy)
    "protenix_fpf_ditfast/__init__.py": ("c8c273a95e597be5aff959a30b0d88be802de1635f2f51ee506567f65d4d3cb0", 1669),
    "protenix_fpf_ditfast/_plumbing.py": ("4f29c5c538d77af5a076ea659ceaf8bbd91cb421638fb1306d95bf90964845bd", 1798),
    "protenix_fpf_ditfast/atom_fast.py": ("f2abb10179e486510d8508e9990a441d9ee72b66bdc994eab746dd1422de6184", 13662),   # digest of the current bytes: import lines re-pointed at the shared core, logic unchanged
    "protenix_fpf_ditfast/cond_dedupe.py": ("13296b54ae5b9520c57d8149b9c70b9573989cb19ef2a6cbfb6cee8795f92020", 2843),
    "protenix_fpf_ditfast/dit_fast.py": ("44b108d258ad13a342106844ffdfb9360644bf51a9cdbcab4c73a3e94255e47a", 17280),   # digest of the current bytes: import lines re-pointed at the shared core, logic unchanged
    "protenix_fpf_ditfast/vectors.json": ("6ee3cc1493a12e381a0a1fdb416f7646848c95530da6e19c17b0b2be3ffde011", 1941),
    "protenix_fpf_ditfast/vectors.py": ("6f21480e978328d825492897d8bdad919e51e156ed10b3d770d1cf67d975a913", 4940),
    "protenix_fpf_msa/CELLS.json": ("06d3661ce7dc4d953d853861af21410a2f8abfb3d1af3859a335cec98b22eb04", 4995),
    "protenix_fpf_msa/NOTICE.md": ("2720da22effe0b1f59503a81873b66877b8a5e1fdce18b4736dd0cd27fab0a99", 285),
    "protenix_fpf_msa/VECTORS.json": ("7f429fdeb52c958eb89f66016249da95bf073d6cb52d4ee2221fc83add9794f6", 1334),
    "protenix_fpf_msa/__init__.py": ("ef2950111a1ecb15524d74f1b6abbd96fd3b741593c918ad78a8ada16f34fe87", 584),   # digest of the current bytes: import lines re-pointed at the shared core, logic unchanged
    "protenix_fpf_msa/install.py": ("0c6b8e4e954d3b25c2034eddca22013619872bd40ce35927ee41f6b4cdc289a1", 11740),   # digest of the current bytes: import lines re-pointed at the shared core, logic unchanged
    "protenix_fpf_msa/loadcheck.py": ("cdbc81ef5e6a8bcbf88dbdaf89a410741fdbcdebbc52d8c155e0bfc2bf1c67bf", 3030),   # digest of the current bytes: import lines re-pointed at the shared core, logic unchanged
    "protenix_fpf_triatt_procuda/__init__.py": ("c8359d782c043818f4d3630e9a63d6a11c1db31fe29251468b16afbe7ecf4faa", 10998),
    "protenix_fpf_triatt_procuda/binding.py": ("46b9081daec059be837a8abe3dcabdd1803bcfd2936133fdb2d2def927335fe0", 6600),
    "protenix_fpf_triatt_procuda/build.py": ("89237a9ce479c5884db9f2f8b7522095fdd36ecfa890725d540338d3d8087229", 1831),
    "protenix_fpf_triatt_procuda/csrc/triatt_procuda_sm90.cu": ("fc120646461cd993a7d9fc94886a137c74873887da251fa538b1701f7880c1c1", 42281),
    "protenix_fpf_triatt_procuda/prebuilt/NOTICE.md": ("e2c6e3d0052be6c9fa0ce122fe721ae9a35aa432b918f8c08be57db8fd311517", 2120),   # NOTICE.md wording revised; digest re-pinned (text only, no code or binary change)
    "protenix_fpf_triatt_procuda/prebuilt/libtriatt_procuda_sm90a.so": ("e9009ba57e432421cf1943cbbe5c22777bc1a674ca0423448ac2f06b9baecbcf", 1025792),
    "protenix_fpf_triatt_procuda/prebuilt/SHA256SUMS": ("8eb43e4ce9b2f1ae4c3bce85d0ce8dd63a3a5d6f30e10c0028ca70dd6377424d", 93),  # added beside the .so: its sha256sum line, re-hashed before the library is loaded (protenix_opt.binary_sums)
    "protenix_fpf_triatt_procuda/prebuilt/manifest.json": ("ba04b28c0a7c0ab033ba13a16507a4f7f32e34f8fd93e4e817f41e2a1b1bcbc7", 355),
}


def _git_blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def _walk(tree: str = TREE) -> dict:
    """{relative path: {"sha256", "bytes", "git_blob"}} for every file under the tree (by-products skipped)."""
    out = {}
    for root, dirs, files in os.walk(tree):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            if name.endswith(SKIP_SUFFIXES):
                continue
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                b = f.read()
            out[os.path.relpath(path, tree)] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "git_blob": _git_blob(b)}
    return out


def _record() -> dict:
    with open(RECORD, encoding="utf-8") as f:
        return json.load(f)


def test_the_record_is_whole():
    rec = _record()
    assert rec["tree"] == "opt/forward/flashpairformer/third_party" and rec["count"] == len(rec["files"]) > 0
    tops = {p.split(os.sep)[0].split("/")[0] for p in rec["files"]}
    on_disk = {p.split(os.sep)[0] for p in _walk()}                                                  # every package (or loose file) the sealed tree holds today is recorded, and nothing else (a directory with no file in it is not a package)
    assert tops == on_disk, (sorted(tops - on_disk), sorted(on_disk - tops))
    assert "kit112_src" not in tops and os.path.isfile(GRAPHED_PY)
    for moved in ("fpf_clisampler", "fpf_stackgraph", "fpf_trunkgraph", "ptx_lazy_init", "ptx_msa_adapt", "infopt_graphs"):   # the kit's own glue: under src/ since 0.3.51, not sealed
        assert moved not in tops and os.path.isdir(os.path.join(os.path.dirname(TREE), "src", moved)), moved


def test_every_sealed_file_is_byte_identical_to_its_published_bytes():
    """Any edit inside the sealed tree fails here, naming the file (judged against the PINNED literals, not the json listing). Put the change in
    opt/protenix_opt/ instead (see sampler_poison_aside.py / route_preload.py for the patterns)."""
    pinned = {k.replace("/", os.sep): v for k, v in PINNED.items()}
    now = _walk()
    missing = sorted(set(pinned) - set(now))
    added = sorted(set(now) - set(pinned))
    changed = sorted(k for k in set(pinned) & set(now) if (now[k]["sha256"], now[k]["bytes"]) != tuple(pinned[k]))
    assert not missing, f"sealed files missing from {TREE}: {missing}"
    assert not added, f"files added inside the sealed tree {TREE} (kit code belongs in opt/protenix_opt/ or src/): {added}"
    assert not changed, "sealed files edited (restore their published bytes; kit behaviour changes go in opt/protenix_opt/ from the outside): " + ", ".join(
        f"{k} (sha256 {now[k]['sha256'][:16]} != pinned {pinned[k][0][:16]}, {now[k]['bytes']} != {pinned[k][1]} bytes)" for k in changed)


def test_the_json_listing_agrees_with_the_pinned_literals():
    """The convenience listing (tests/data/sealed_third_party.json) cannot drift from the literals: a ``--reseal`` that rewrote it to bless an
    edit fails here until the literal is edited by hand."""
    rec = _record()
    assert rec["count"] == len(rec["files"]) == len(PINNED), (rec["count"], len(rec["files"]), len(PINNED))
    drift = sorted(k for k in PINNED if k not in rec["files"] or (rec["files"][k]["sha256"], rec["files"][k]["bytes"]) != PINNED[k])
    assert not drift, f"json listing disagrees with the PINNED literals for: {drift}"


def test_the_small_n_gate_package_is_its_published_blob():
    """fpf_smalln/__init__.py is its published blob but for line 1, the copyright holder comment, and the module docstring's first line (git blob fd02b9d42eac,
    13385 bytes); its TriMul callees are named by
    env.sh and imported by protenix_opt.route_preload at activation."""
    now = _walk()[os.path.join("fpf_smalln", "__init__.py")]
    assert (now["sha256"], now["bytes"], now["git_blob"]) == ("be35aefa0f1ae3e4c95c3d508946b1117dc345ff8851ccdd9db38b69951b2318", 13385, "fd02b9d42eac47224d18ab69bed8aa5fe63bdd2c"), now


def test_graphed_py_keeps_the_poison_probe_and_no_step_aside():
    """The graphed sampler (kit code under src/ since 0.3.51; its shared audit module may be imported from the core) still raises its bias-cache
    poison self-test by name and carries no step-aside of its own: the fp16 step-aside lives in protenix_opt.sampler_poison_aside (0.3.48)."""
    with open(GRAPHED_PY, "rb") as f:
        src = f.read().decode("utf-8")
    assert "ASIDE_NONFINITE" not in src and "_step_aside" not in src and 'raise RuntimeError(f"biascache poison test FAILED' in src


def _reseal(argv) -> int:
    """Rewrite the json LISTING from the tree as it stands. This blesses nothing: the test asserts the PINNED literals (edit those by hand)."""
    files = _walk()
    doc = _record() if os.path.isfile(RECORD) else {}
    doc.update(tree="opt/forward/flashpairformer/third_party", count=len(files), files={k.replace(os.sep, "/"): v for k, v in files.items()})
    doc.setdefault("_about", "see opt/protenix_opt/tests/test_sealed_third_party.py")
    with open(RECORD, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True); f.write("\n")
    print(f"re-sealed {len(files)} files -> {RECORD}")
    return 0


if __name__ == "__main__":
    sys.exit(_reseal(sys.argv[1:]) if "--reseal" in sys.argv[1:] else 2)

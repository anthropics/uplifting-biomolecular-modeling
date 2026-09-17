"""The carried kit files are exactly the files under opt/forward/ (one fixed list); no
house-authored file sits inside a kit directory; no copy of a kernel the core ships (opt_core.kernels, stack.KERNELS) sits anywhere
in the kit."""
import hashlib
import os

from .. import modes, stack


CARRIED = (   # every file under opt/forward, by kit directory: the add-ons' code, docs, the cell table, the public input
    "forward/rf3_fpf_trimul_addon/README.md",
    "forward/rf3_fpf_trimul_addon/rf3fpf/fpf_rf3_adapter.py",
    "forward/rf3_fpf_trimul_addon/rf3fpf/fpf_rf3_msa_rows.py",
    "forward/rf3_fpf_trimul_addon/rf3fpf/fpf_rf3_triattn.py",
    "forward/rf3_fpf_trimul_addon/rf3fpf/fpf_rf3_trimul_rows.py",
    "forward/rf3_fpf_trimul_addon/rf3fpf/fpf_rf3_ln_rows.py",
    "forward/rf3_fpf_trimul_addon/rf3fpf/fpf_rf3_apb_rows.py",
    "forward/rf3_mk_dit_addon/README.md",
    "forward/rf3_mk_dit_addon/mkdit/mk2.py",
    "forward/rf3_mk_dit_addon/mkdit/mkrf3.py",
    "forward/rf3_xattempt_addon/LICENSE",
    "forward/rf3_xattempt_addon/LICENSE_NOTES.md",
    "forward/rf3_xattempt_addon/LICENSE_foundry_BSD-3-Clause.md",
    "forward/rf3_xattempt_addon/NOTICE",
    "forward/rf3_xattempt_addon/README.md",
    "forward/rf3_xattempt_addon/docs/INVARIANCE_PROOF.md",
    "forward/rf3_xattempt_addon/install.sh",
    "forward/rf3_xattempt_addon/patched/rf3/diffusion_samplers/inference_sampler.py",
    "forward/rf3_xattempt_addon/patched/rf3/graph_flags.py",
    "forward/rf3_xattempt_addon/patched/rf3/loss/loss.py",
    "forward/rf3_xattempt_addon/patched/rf3/model/RF3_structure.py",
    "forward/rf3_xattempt_addon/patched/rf3/model/layers/af3_diffusion_transformer.py",
    "forward/rf3_xattempt_addon/public_inputs/1brs_tiles.json",
)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def test_kit_directories_hold_exactly_the_carried_files():
    root = stack.opt_root()
    on_disk = sorted(os.path.relpath(os.path.join(d, f), root) for sub in ("forward",)
                     for d, _, fs in os.walk(os.path.join(root, sub)) for f in fs if "__pycache__" not in d and not f.endswith(".pyc"))
    assert on_disk == sorted(CARRIED), sorted(set(on_disk) ^ set(CARRIED))
    for rel in modes.FPF_RUNTIME_MEMBERS:                                       # every runtime member of a fast row is a carried file
        assert f"forward/rf3_fpf_trimul_addon/{rel}" in on_disk, rel


def test_no_copy_of_a_core_kernel_in_the_kit():
    """The three kernels a fast row routes to the core (stack.KERNELS) have no copy under opt/: no directory or module of a routed name, and no
    carried file whose bytes equal a file of the core's kernel copies (opt_core.kernels, hashed live off disk) — the cell table is the kit's own."""
    from opt_core import kernels as core_kernels
    root = stack.opt_root()
    core_shas = set()
    for name in stack.KERNELS:
        doc = core_kernels.sums(name)                                     # "files": live relative paths, computed now, never stored (no sha shipped)
        kroot = os.path.join(core_kernels.KERNELS_DIR, name) if doc["kind"] == "package" else core_kernels.KERNELS_DIR
        core_shas.update(sha(os.path.join(kroot, rel)) for rel in doc["files"])
    for d, dirs, fs in os.walk(root):
        if "venv" in d.split(os.sep) or "__pycache__" in d:
            continue
        for name in stack.KERNELS:
            assert name not in dirs and f"{name}.py" not in fs, (d, name)
        for f in fs:
            p = os.path.join(d, f)
            assert os.path.getsize(p) == 0 or sha(p) not in core_shas, p      # (the empty __init__.py sha is in every sums file)
    assert not os.path.exists(os.path.join(stack.fpf_home(), "fpf_trimul_v4_cells.json"))          # no kit cell table: the core's one table serves (opt_core/kernels/fpf_trimul_v4/table.json)
    assert os.environ.get("FPF_TRIMUL_V4_CELLS") in (None, stack.CORE_TRIMUL_TABLE) and os.path.isfile(stack.CORE_TRIMUL_TABLE)


def test_the_det_instrument_is_not_carried():
    for rel in ("det_segment_reduce.py", "foundry_utils_torch.py"):        # a deterministic scatter_mean instrument is not part of this tree
        for d, _, fs in os.walk(stack.opt_root()):
            assert rel not in fs or "venv" in d, (d, rel)


def test_fast_trimul_has_no_kit_side_token_ceiling():
    """The fast TriMul is admitted per call by the core's own `supported()` (its word counted when it declines): the adapter's fast path names
    no token ceiling of its own."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "..", "forward", "rf3_fpf_trimul_addon", "rf3fpf", "fpf_rf3_adapter.py")).read()
    fast = src[src.index("def _fast_forward"):src.index("def enable(")]
    assert "2048" not in fast and "N >" not in fast and "N <" not in fast and "G.supported(" in fast

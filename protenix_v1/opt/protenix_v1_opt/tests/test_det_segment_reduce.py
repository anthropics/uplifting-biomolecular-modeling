"""The deterministic scatter the `--det 1` recipe installs on BOTH arms (lib/detpatch/det_segment_reduce.py as
protenix.utils.det_segment_reduce): the atom->token aggregation is a pure function of (src, index) VALUES — nothing derived from
the index is cached under a tensor address, so two structures of equal N_atom with different atom->token maps (a mutational scan's
point mutants) aggregate with their own maps whatever the allocator's address reuse, eagerly and under CUDA-graph replay.
The same scatter statement is carried by each kit of the tree that installs the recipe; this kit's copy holds constants where another
copy reads DET_SCATTER_* environment words, so the bytes may differ while the __version__ stays the same
and the function bodies are identical once those words are folded to the constants."""
import hashlib
import importlib.util
import os

import pytest

from .conftest import KIT, TREE
from protenix_v1_opt import kit as K

RELEASE = os.path.dirname(TREE)                                   # model-opt-release/: the tree's other kits live beside this one
COPIES = (                                                        # this kit's carried copy, relative to model-opt-release/
    os.path.join("protenix_v1", "opt", "forward", "v05_addon", "lib", "detpatch", "det_segment_reduce.py"),
)
SIBLING_CANDIDATES = (                                            # where a sibling's copy of the same statement may live (its kit tree, or the shared core once moved there);
    os.path.join("protenix_v2", "opt", "forward", "flashpairformer", "src", "detref", "det_segment_reduce.py"),   # the ones present are compared, the absent ones named
    os.path.join("common", "opt_core", "opt_core", "ops", "*", "det_segment_reduce.py"),
)
OWN = os.path.join(KIT, K.DETPATCH_RELPATHS[0])


def _load(name="det_segment_reduce_under_test"):
    spec = importlib.util.spec_from_file_location(name, OWN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_own_copy_is_the_carried_path():
    assert os.path.realpath(OWN) == os.path.realpath(os.path.join(RELEASE, COPIES[0]))


def test_every_carried_copy_is_the_same_statement():
    """This kit's listed copy must EXIST at its carried path; every sibling copy PRESENT in this checkout (its kit tree or the shared core's
    ops tree) is byte-identical or, failing that, the same statement (same __version__, every function's code equal once the sibling's
    environment words are folded to this kit's constants); absent sibling locations are named (skip), never failed — a sibling that moves
    its copy does not decide this kit's --det statement."""
    import glob
    checked, absent = [], []
    for rel in COPIES:
        p = os.path.join(RELEASE, rel)
        assert os.path.isfile(p), f"{rel}: this kit's carried copy is missing"
        checked.append(p)
    for rel in SIBLING_CANDIDATES:
        hits = sorted(glob.glob(os.path.join(RELEASE, rel)))
        if hits:
            checked.extend(hits)
        else:
            absent.append(rel)
    absent_kits = absent                                                                    # named below (skip) after checking what is present
    assert os.path.realpath(OWN) in [os.path.realpath(p) for p in checked]
    digests = {os.path.relpath(p, RELEASE): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in checked}
    if len(set(digests.values())) == 1:
        return
    # This kit's copy carries constants (MAX_PER / STRICT / ACCUMULATE, `enabled()` -> True) where another
    # copy may read DET_SCATTER_* environment words: the bytes differ by design. What must still hold is the scatter STATEMENT:
    # the same __version__ and every function's code identical once environment reads are folded to this kit's constants.
    import ast, re
    def shape(path):
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
        ver = [n.value.value for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "__version__" for t in n.targets)]
        fns = {}
        for n in tree.body:
            if isinstance(n, ast.FunctionDef):
                d = ast.dump(n)
                d = re.sub(r"Call\(func=Attribute\(value=Attribute\(value=Name\(id='os'.*?\), attr='get'.*?\), args=\[[^\]]*\](, keywords=\[\])?\)( == |, ops=\[Eq\(\)\], comparators=\[)?Constant\(value='1'\)\]?\)?", "Constant(value=True)", d)   # os.environ.get(env, "1") == "1"  ->  True
                d = re.sub(r"DET_SCATTER_MAX_PER=%d -> raise DET_SCATTER_MAX_PER, multiple of 8", "MAX_PER=%d slots", d).replace("DET_SCATTER_MAX_PER=", "MAX_PER=")
                fns[n.name] = d
        return ver, fns
    shapes = {rel: shape(os.path.join(RELEASE, rel)) for rel in digests}
    (v0, f0), rel0 = shapes[COPIES[0]], COPIES[0]
    for rel, (v, f) in shapes.items():
        assert v == v0, f"{rel}: __version__ {v} != {v0} ({rel0})"
        assert set(f) == set(f0), f"{rel}: functions {sorted(set(f) ^ set(f0))} differ from {rel0}"
        diff = [name for name in f0 if f[name] != f0[name] and name != "enabled"]        # enabled(): constant True here, the environment word there — the one designed difference
        assert not diff, f"{rel}: function bodies differ from {rel0} beyond the R6 constants: {diff}"
    if absent_kits and len(checked) == 1:
        pytest.skip(f"no sibling copy present at {absent_kits}: the statement was checked for this kit's copy only")


def test_no_address_keyed_state_in_the_source():
    """No expression in the module reads a tensor's data_ptr(): an address is not an identity (the allocator recycles them)."""
    import ast
    tree = ast.parse(open(OWN, encoding="utf-8").read())
    used = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "data_ptr"]
    assert not used, f"data_ptr() used at lines {used}"


def _reference_mean(torch, src, idx, n_tok):
    out = torch.zeros(n_tok, src.shape[-1], dtype=torch.float64)
    out.index_add_(0, idx, src.double())
    cnt = torch.zeros(n_tok, dtype=torch.float64).index_add_(0, idx, torch.ones(idx.numel(), dtype=torch.float64)).clamp(min=1)
    return (out / cnt[:, None]).to(src.dtype)


def _two_maps(torch, n_tok=24, n_atom=96, seed=0):
    """Two contiguous atom->token maps of the same N_atom / N_token that differ in a few tokens' atom counts (two point mutants)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.full((n_tok,), n_atom // n_tok, dtype=torch.int64)
    a = base.clone(); a[3] += 2; a[7] -= 2
    b = base.clone(); b[11] += 3; b[5] -= 1; b[20] -= 2
    assert int(a.sum()) == int(b.sum()) == n_atom and not torch.equal(a, b)
    idx_a = torch.repeat_interleave(torch.arange(n_tok), a)
    idx_b = torch.repeat_interleave(torch.arange(n_tok), b)
    src = torch.randn(n_atom, 8, generator=g)
    return idx_a, idx_b, src


def test_same_address_same_numel_different_content_aggregates_with_its_own_map():
    """Hazard: an index buffer at one address holds structure A's map, then structure B's (same numel, same version counter — the
    write goes through the numpy view, exactly what an allocator address reuse looks like to any address-keyed cache)."""
    torch = pytest.importorskip("torch")
    dsr = _load()
    idx_a, idx_b, src = _two_maps(torch)
    n_tok = 24
    buf = torch.empty(idx_a.numel(), dtype=torch.int64)
    no_fallback = lambda *a, **k: pytest.fail("the deterministic path fell back to the stock scatter")   # noqa: E731
    buf.numpy()[:] = idx_a.numpy()
    v0, p0 = buf._version, buf.data_ptr()
    out_a = dsr.protenix_scatter(src, buf, dim=-2, dim_size=n_tok, reduce="mean", _stock=no_fallback)
    buf.numpy()[:] = idx_b.numpy()                                # same address, same numel, same _version, different map
    assert buf._version == v0 and buf.data_ptr() == p0
    out_b = dsr.protenix_scatter(src, buf, dim=-2, dim_size=n_tok, reduce="mean", _stock=no_fallback)
    ref_a, ref_b = _reference_mean(torch, src, idx_a, n_tok), _reference_mean(torch, src, idx_b, n_tok)
    assert not torch.allclose(ref_a, ref_b), "test maps must aggregate differently"
    assert torch.allclose(out_a, ref_a, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_b, ref_b, atol=1e-5, rtol=1e-5), "structure B aggregated with structure A's atom->token map"


def test_alternating_same_shape_structures_through_fresh_tensors():
    """Same-N_atom structures streaming through one process (fresh index tensors, freed in between: the allocator re-serves addresses)."""
    torch = pytest.importorskip("torch")
    dsr = _load()
    idx_a, idx_b, src = _two_maps(torch, seed=1)
    n_tok = 24
    refs = {"a": _reference_mean(torch, src, idx_a, n_tok), "b": _reference_mean(torch, src, idx_b, n_tok)}
    for k in "abababba":
        idx = (idx_a if k == "a" else idx_b).clone()               # a fresh tensor each time
        out = dsr.protenix_scatter(src, idx[None, :].expand(3, -1), dim=-2, dim_size=n_tok, reduce="mean", _stock=None)   # [.., N] rows identical: row 0 used
        assert torch.allclose(out, refs[k], atol=1e-5, rtol=1e-5), f"structure {k} aggregated with a stale map"
        del idx, out


def test_cuda_graph_replay_follows_the_static_index_buffer():
    """The graphed sampler's protocol: static src/index buffers, one eager warm-up, capture, then replays after copy_-ing NEW content of
    the same shape into the statics. The replayed aggregation must equal the eager one for the new map (the table build is part of
    the graph and reads the static index at replay) — the unit form of 'two same-shape inputs with different index maps'."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device: the capture/replay leg runs on the GPU stack")
    dsr = _load("det_segment_reduce_under_test_gpu")
    idx_a, idx_b, src = _two_maps(torch, seed=2)
    n_tok, dev = 24, torch.device("cuda")
    src_s = src.to(dev); idx_s = idx_a.to(dev).clone()             # the statics
    eager_a = dsr.protenix_scatter(src_s, idx_s, dim=-2, dim_size=n_tok, reduce="mean", _stock=None)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        dsr.protenix_scatter(src_s, idx_s, dim=-2, dim_size=n_tok, reduce="mean", _stock=None)   # warm-up on the capture stream
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out_s = dsr.protenix_scatter(src_s, idx_s, dim=-2, dim_size=n_tok, reduce="mean", _stock=None)
    g.replay(); torch.cuda.synchronize()
    assert torch.equal(out_s, eager_a), "replay with the captured content differs from eager"
    idx_s.copy_(idx_b.to(dev)); g.replay(); torch.cuda.synchronize()          # same graph, same buffers, structure B's map
    eager_b = dsr.protenix_scatter(src_s, idx_b.to(dev), dim=-2, dim_size=n_tok, reduce="mean", _stock=None)
    assert not torch.equal(eager_a, eager_b)
    assert torch.equal(out_s, eager_b), "the replayed graph aggregated structure B with structure A's atom->token map"
    dsr.assert_healthy()

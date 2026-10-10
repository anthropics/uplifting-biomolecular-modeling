"""CPU checks of captured-workspace teardown without importing the inference engine."""
import ast
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
SAMPLER = ROOT / 'protenix_v2/opt/forward/flashpairformer/src/infopt_graphs/protenix/sampler_prep.py'

class Weight:
    pass

def test_sampler_release_drops_callable_captures():
    # This helper is independent of CUDA; load its real implementation without
    # importing the sampler's torch / engine dependencies.
    tree = ast.parse(SAMPLER.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'release_statics')
    namespace = {'STATS': {'statics_released': 0}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SAMPLER), 'exec'), namespace)
    st, cond = Weight(), Weight()
    refs = weakref.ref(st), weakref.ref(cond)
    graph = object()
    entry = {'graph': graph, 'st': st, 'cond': cond, 'body': lambda st=st, cond=cond: (st, cond)}
    del st, cond
    assert namespace['release_statics'](SimpleNamespace(entries={'k': entry}), 'k') is graph
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_stack_reset_synchronizes_before_releasing_workspace_and_resets_live_budget():
    path = ROOT / 'protenix_v2/opt/forward/flashpairformer/src/fpf_stackgraph/stackgraph.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reset_cache')
    workspace = Weight()
    ref = weakref.ref(workspace)
    cache = {'graph': SimpleNamespace(payload_caches=(workspace,))}
    del workspace
    calls = []

    def synchronize():
        assert ref() is not None
        calls.append('sync')

    def empty_cache():
        assert ref() is None
        calls.append('flush')

    budget = {'pool_total': 1024, 'per_tok': 7, 'max_tok': 360}
    stats = {'sigs': 1, 'pool_gb': 1, 'pool_bytes_by_ntok': {280: 1024}, 'captures': 3}
    namespace = {
        '_CACHE': cache, '_BUDGET': budget, 'ST': stats,
        'torch': SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize, empty_cache=empty_cache)),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    namespace['reset_cache']()
    assert calls == ['sync', 'flush']
    assert cache == {}
    assert budget == {'pool_total': 0, 'per_tok': 7, 'max_tok': 360}
    assert stats == {'sigs': 0, 'pool_gb': 0, 'pool_bytes_by_ntok': {}, 'captures': 3}


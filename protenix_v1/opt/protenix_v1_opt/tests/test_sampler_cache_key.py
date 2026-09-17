"""CPU unit tests for GraphedDenoiseLoop's sampler cache key (opt/forward/v05_addon/lib/kit112_src/infopt_graphs/
protenix/graphed.py). Invariant under test: the input_feature_dict component of the key discriminates every
non-floating tensor (an integer index or bool/int mask a captured graph bakes into gather/scatter/index_select,
e.g. atom_to_token_idx) by its VALUES, not just its shape -- two items can share a tensor's shape while its index
values map differently, and a shape-only key would replay one item's captured graph against the other's buffers,
producing a silently wrong structure. Floating-point tensors key by shape/dtype alone, since their values are
always freshly copied into the static buffers before every replay.

The digest is memoized in a dict scoped to a single sample() call (created fresh each call, passed by closure to
every _chunk() invocation within it) so a chunked item -- diffusion_chunk_size splits one item's N_sample across
several _chunk() calls sharing the same input_feature_dict tensors -- pays the digest cost once per item, not
once per chunk. The cache must never be shared across DIFFERENT sample() calls: two distinct tensors can be
indistinguishable by (data_ptr, shape, dtype, autograd _version) alone if one is a freed buffer whose address a
later, unrelated tensor reuses (both default to _version=0 when never mutated in place), so a cache that outlives
its call could return a stale digest for the wrong tensor.

No GPU/CUDA needed -- both functions are plain tensor/numpy operations, torch's CPU backend is sufficient. This
test lives in the package's own suite, not inside forward/v05_addon/lib/kit112_src/infopt_graphs/tests/, because
that directory is the vendored kit's."""
import os
import sys

import pytest
import torch

from .conftest import KIT

sys.path.insert(0, os.path.join(KIT, "lib", "kit112_src"))
from infopt_graphs.protenix.graphed import _input_feature_key, _tensor_value_digest  # noqa: E402


def test_key_differs_when_an_index_tensors_values_move_at_equal_shape():
    """The exact failure mode: two atom_to_token_idx-shaped tensors with the same shape but two values swapped
    between positions -- shape alone would collide them; the content digest must not."""
    a = torch.tensor([0, 0, 1, 1, 2, 2, 2], dtype=torch.int64)
    b = a.clone(); b[2], b[4] = a[4], a[2]                           # swap two positions: same multiset of values, same shape
    assert a.shape == b.shape and not torch.equal(a, b)              # same shape, genuinely different arrangement
    assert _input_feature_key({"atom_to_token_idx": a}) != _input_feature_key({"atom_to_token_idx": b})


def test_key_matches_for_identical_values():
    a = torch.tensor([0, 0, 1, 1, 2, 2, 2], dtype=torch.int64)
    b = a.clone()
    assert _input_feature_key({"atom_to_token_idx": a}) == _input_feature_key({"atom_to_token_idx": b})


def test_feature_key_differs_when_atom_to_token_idx_values_differ_at_equal_shape():
    """Two feature dicts with identical shapes: differing atom_to_token_idx values must produce different keys,
    identical values must produce the same key. N_token=4 fixed; two different N_atom=7 mappings (mirrors two
    same-net-atom-delta point mutations that place the extra atom on a different residue)."""
    s_inputs = torch.randn(4, 32)  # (N_token, C): only .shape[-2] and .dtype are read by the real key, not here
    base = {
        "atom_to_token_idx": torch.tensor([0, 0, 1, 1, 2, 2, 2], dtype=torch.int64),
        "s_inputs": s_inputs,
    }
    same_mapping = {**base, "atom_to_token_idx": base["atom_to_token_idx"].clone()}
    different_mapping = {**base, "atom_to_token_idx": torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.int64)}

    assert base["atom_to_token_idx"].shape == different_mapping["atom_to_token_idx"].shape   # equal shape...
    assert _input_feature_key(base) == _input_feature_key(same_mapping)               # identical values -> same key
    assert _input_feature_key(base) != _input_feature_key(different_mapping)          # ...but different values -> different key


def test_feature_key_is_shape_only_for_floating_point_tensors():
    """Floating-point feature values are always freshly copied into the static buffers by copy_into() before every
    replay, so they do not need (and must not pay for) a value digest -- only the shape/dtype discriminate them."""
    a = {"s_inputs": torch.zeros(4, 32)}
    b = {"s_inputs": torch.ones(4, 32)}  # same shape/dtype, very different values
    assert a["s_inputs"].shape == b["s_inputs"].shape and a["s_inputs"].dtype == b["s_inputs"].dtype
    assert _input_feature_key(a) == _input_feature_key(b)                  # shape-only for floats: same key by design
    for name, shape, dtype, digest in _input_feature_key(a):
        assert digest is None                                              # no digest computed for a floating-point leaf


def test_key_handles_empty_and_bool_and_complex_tensors():
    assert _input_feature_key({"e": torch.zeros(0, dtype=torch.int64)}) == _input_feature_key({"e": torch.zeros(0, dtype=torch.int64)})
    a = {"mask": torch.tensor([True, False, True])}
    b = {"mask": torch.tensor([True, True, False])}
    assert _input_feature_key(a) != _input_feature_key(b)                  # bool masks are value-keyed too
    c = {"z": torch.zeros(3, dtype=torch.complex64)}
    for name, shape, dtype, digest in _input_feature_key(c):
        assert digest is None                                              # complex tensors are shape-only, like floats


def test_digest_cache_none_matches_digest_cache_scoped_for_the_same_tensor():
    """digest_cache is optional (None = no memo, always recompute) -- both paths must agree on the digest value."""
    t = torch.tensor([0, 0, 1, 1, 2, 2, 2], dtype=torch.int64)
    assert _tensor_value_digest(t, None) == _tensor_value_digest(t, {})
    assert _input_feature_key({"atom_to_token_idx": t}, None) == _input_feature_key({"atom_to_token_idx": t}, {})


def test_digest_cache_reuses_across_chunk_calls_sharing_the_same_dict_and_tensors():
    """One dict, created once per sample() call and passed to every _chunk() invocation within it
    (diffusion_chunk_size splits N_sample across several _chunk() calls, all sharing the same input_feature_dict
    tensor objects) -- a second call with the SAME tensor must hit the cache, not grow it or recompute."""
    cache: dict = {}
    d = {"atom_to_token_idx": torch.tensor([0, 0, 1, 1, 2, 2, 2], dtype=torch.int64), "s_inputs": torch.randn(4, 8)}
    first = _input_feature_key(d, cache)
    n_after_first = len(cache)
    assert n_after_first > 0
    second = _input_feature_key(d, cache)                                  # simulates a second _chunk() call in the same sample() call
    assert second == first
    assert len(cache) == n_after_first                                     # no growth: the same tensor identity hit the cache


def test_digest_cache_is_scoped_per_call_so_a_fresh_call_cannot_see_a_stale_hit():
    """A cache keyed only by tensor identity (data_ptr, shape, dtype, _version) can return a STALE digest for an
    unrelated tensor that a freed buffer's address gets reused for, if the new tensor is also never-mutated
    (_version=0 by default, same as any other fresh tensor) and happens to share shape/dtype -- exactly a same-
    total-atom-count, different-mapping item pair. Scoping the cache to one sample() call (a fresh dict every
    call, discarded when the call returns, never shared across items) makes that hazard structurally impossible:
    two DIFFERENT calls -- simulating two DIFFERENT items -- must never share a cache, so each computes its own,
    correct digest independent of whatever a same-shaped tensor from an earlier, unrelated call produced."""
    a = torch.tensor([0, 0, 1, 1, 2, 2, 2], dtype=torch.int64)             # item A's atom_to_token_idx
    b = torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.int64)             # item B's: same shape/dtype, different values, both _version=0

    cache_a: dict = {}                                                     # item A's sample() call: its own fresh cache
    key_a = _input_feature_key({"atom_to_token_idx": a}, cache_a)

    cache_b: dict = {}                                                     # item B's sample() call: a DIFFERENT fresh cache, never shares state with A's
    key_b = _input_feature_key({"atom_to_token_idx": b}, cache_b)

    assert key_a != key_b                                                  # each call computed its own correct digest, not a value borrowed from the other

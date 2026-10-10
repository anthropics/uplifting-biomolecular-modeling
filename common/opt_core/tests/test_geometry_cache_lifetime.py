"""Geometry and captured-workspace lifetimes, without a CUDA device."""
import gc
import weakref

import pytest
from opt_core.kernels.trimul import native

class Weight:
    def data_ptr(self):
        return id(self)

@pytest.fixture(autouse=True)
def clear_geometries():
    native._SHARED.clear()
    yield
    native._SHARED.clear()
    gc.collect()

def test_evicted_geometry_dies_while_weight_lives():
    owner = Weight()
    cache = native.payload_cache(0, 280, 128, 128)
    key = (native.PACK_KEY_TAG, 'pack')
    cache[key] = object()
    native.after_call(cache, {'w_ap': owner})
    ref = weakref.ref(cache)
    native.payload_cache(0, 320, 128, 128)
    native.payload_cache(0, 360, 128, 128)
    del cache
    gc.collect()
    assert ref() is None
    assert owner.data_ptr()

def test_owner_death_still_evicts_live_pack_and_memo():
    owner = Weight()
    cache = native.payload_cache(0, 280, 128, 128)
    key = (native.PACK_KEY_TAG, 'pack')
    cache[key] = object()
    cache[native.CALLS_MEMO_KEY] = {'call': {'w_ptr': owner.data_ptr()}}
    native.after_call(cache, {'w_ap': owner})
    del owner
    gc.collect()
    assert key not in cache
    assert cache[native.CALLS_MEMO_KEY] == {}

def test_capture_owns_even_mid_capture_evictions():
    with native.retain_payload_caches() as retained:
        first = native.payload_cache(0, 280, 128, 128)
        ref = weakref.ref(first)
        del first
        native.payload_cache(0, 320, 128, 128)
        native.payload_cache(0, 360, 128, 128)
    gc.collect()
    assert len(retained) == 3 and ref() is not None
    retained.clear()
    gc.collect()
    assert ref() is None

def test_capture_scopes_unwind_after_failure():
    with pytest.raises(ValueError):
        with native.retain_payload_caches():
            raise ValueError('failed capture')
    assert native._CAPTURE_CACHES.scopes == []



def test_dead_geometries_do_not_accumulate_owner_finalizers():
    owner = Weight()
    baseline = weakref.getweakrefcount(owner)
    for length in range(100):
        cache = native.payload_cache(0, length, 128, 128)
        cache[(native.PACK_KEY_TAG, 'pack')] = object()
        native.after_call(cache, {'w_ap': owner})
    del cache
    gc.collect()
    assert weakref.getweakrefcount(owner) <= baseline + native.SHARED_GEOMETRIES
    native._SHARED.clear()
    gc.collect()
    assert weakref.getweakrefcount(owner) == baseline


def test_dead_owners_remove_the_cache_death_registrations():
    cache = native.payload_cache(0, 280, 128, 128)
    baseline = weakref.getweakrefcount(cache)
    for index in range(100):
        owner = Weight()
        cache[(native.PACK_KEY_TAG, index)] = object()
        native.after_call(cache, {'w_ap': owner}, keys_before=tuple(cache)[:-1])
        del owner
    gc.collect()
    assert weakref.getweakrefcount(cache) == baseline


def test_plain_caller_cache_keeps_its_owner_death_cleanup():
    owner = Weight()
    key = (native.PACK_KEY_TAG, 'pack')
    cache = {key: object()}
    native.after_call(cache, {'w_ap': owner})
    del owner
    gc.collect()
    assert cache == {}


def test_nested_capture_scopes_record_every_lookup():
    with native.retain_payload_caches() as outer:
        first = native.payload_cache(0, 280, 128, 128)
        with native.retain_payload_caches() as inner:
            assert native.payload_cache(0, 280, 128, 128) is first
            native.payload_cache(0, 320, 128, 128)
        native.payload_cache(0, 360, 128, 128)
    assert len(outer) == 3 and len(inner) == 2


def test_pack_recreation_does_not_accumulate_live_registrations():
    owner = Weight()
    cache = native.payload_cache(0, 280, 128, 128)
    owner_refs, cache_refs = weakref.getweakrefcount(owner), weakref.getweakrefcount(cache)
    key = (native.PACK_KEY_TAG, 'pack')
    for index in range(100):
        cache[key] = object()
        native.after_call(cache, {'w_ap': owner})
        native._evict_key(cache, key, None, 'cap')
    gc.collect()
    assert weakref.getweakrefcount(owner) == owner_refs
    assert weakref.getweakrefcount(cache) == cache_refs
    assert cache._pack_finalizers == {}



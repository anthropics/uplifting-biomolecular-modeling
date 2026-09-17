"""kernels.trimul: the face's cast memo (compute_input's '_z_cast': the ONE bf16 copy of an fp32-resident z a fused row reads under autocast) is scoped
to the call that made it.  An fp32-resident caller that runs the op per sample / per item with a persistent per-module cache must not accumulate one
[N, N, c] cast copy per module across calls (measured: 8 x 361 MB retained at 1216 tokens, 26 GB at 3584, before this scope)."""
import gc
import weakref

import pytest

from opt_core.kernels import trimul as T


def _weights(torch, C, H, seed=3):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)                              # noqa: E731
    return {"ln_in_w": 1 + 0.1 * r(C), "ln_in_b": 0.1 * r(C), "w_ag": r(H, C) * C ** -0.5, "w_ap": r(H, C) * C ** -0.5, "w_bg": r(H, C) * C ** -0.5,
            "w_bp": r(H, C) * C ** -0.5, "ln_out_w": 1 + 0.1 * r(H), "ln_out_b": 0.1 * r(H), "w_o": r(C, H) * H ** -0.5, "w_og": r(C, C) * C ** -0.5}


def test_the_cast_memo_hits_within_a_call_and_is_released_by_the_face_on_return(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(T, "_autocast_dtype", lambda z: torch.bfloat16)     # as under a bf16 autocast region (device-free)
    cache = {}
    z = torch.randn(8, 8, 16)
    s0 = T.cast_memo_stats()
    a = T.compute_input(z, cache)
    b = T.compute_input(z, cache)                                            # admission then serving (or out then in of one 'outin' call): ONE cast
    assert a.dtype == torch.bfloat16 and b is a and "_z_cast" in cache
    s1 = T.cast_memo_stats()
    assert s1["stores"] == s0["stores"] + 1 and s1["hits"] == s0["hits"] + 1 and s1["live"] == s0["live"] + 1
    T.release_call_scratch(cache)
    assert "_z_cast" not in cache and T.cast_memo_stats()["live"] == s0["live"]
    wr = weakref.ref(a)
    del a, b
    gc.collect()
    assert wr() is None                                                      # the copy is gone once the call's users drop it: the memo held the only other reference


def test_no_cast_copy_survives_a_face_call_across_six_distinct_fp32_inputs(monkeypatch):
    """N = 6 calls through triangle_multiplication with DISTINCT fp32 z and ONE persistent cache (a kit's per-module dict): after every call the cache holds
    no cast memo, the memo's live count is flat, and a cast copy planted during the call (as a fused row does under autocast) is unreachable after return."""
    torch = pytest.importorskip("torch")
    C, H, N = 16, 16, 8
    W = _weights(torch, C, H)
    cache = {}
    planted = []
    real = T.compute_input

    def planting_compute_input(z, cache=None):                               # the statement row on the CPU reads z itself; plant the memo a fused row would leave
        if cache is not None:
            zc = z.to(torch.bfloat16)
            if "_z_cast" not in cache:
                T._CAST_MEMO["live"] += 1
            cache["_z_cast"] = (z, torch.bfloat16, zc)
            planted.append(weakref.ref(zc))
        return real(z, cache)

    live0 = T.cast_memo_stats()["live"]
    for i in range(6):
        z = torch.randn(1, N, N, C, generator=torch.Generator().manual_seed(10 + i))
        cache_before = dict(cache)
        planting_compute_input(z, cache)                                     # as if admission had cast inside the call ...
        try:
            out = T.triangle_multiplication(z, None, direction="outgoing", weights=W, word="torch_math", residual=False, cache=cache)
        except T.Refusal:                                                    # the statement row serves on the CPU; a refusal would still release the scratch
            out = None
        assert "_z_cast" not in cache, sorted(k for k in cache if isinstance(k, str))
        assert T.cast_memo_stats()["live"] == live0, T.cast_memo_stats()
        del out
        gc.collect()
        assert all(r() is None for r in planted), [r() is not None for r in planted]      # no cast copy of any earlier z is alive after the call returned
        assert not any(isinstance(v, tuple) and len(v) == 3 and v[0] is z for v in cache.values())   # nor is z itself held by the face's cache


def test_the_same_z_object_mutated_in_place_between_two_calls_is_recast_never_served_stale(monkeypatch):
    """Identity keying alone would hand a second call of the same module with the same z OBJECT (values changed in place) the previous call's copy.
    With the memo scoped to the call the second call reads z.to(dt) of the NEW values."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(T, "_autocast_dtype", lambda z: torch.bfloat16)
    cache = {}
    z = torch.randn(8, 8, 16)
    first = T.compute_input(z, cache)                                        # call 1 (admission / serving of one face call)
    T.release_call_scratch(cache)                                            # ... which the face releases on return
    with torch.no_grad():
        z.mul_(3.0).add_(1.0)                                                # the trunk updates z in place and calls the module again
    T.release_call_scratch(cache)                                            # (the face's entry also clears anything left before reading)
    second = T.compute_input(z, cache)                                       # call 2
    assert second is not first and torch.equal(second, z.to(torch.bfloat16)) and not torch.equal(second, first)
    with torch.no_grad():
        z.sub_(2.0)                                                          # even WITHIN a call an in-place update bumps the version: the key misses, a fresh cast
    third = T.compute_input(z, cache)
    assert third is not second and torch.equal(third, z.to(torch.bfloat16))
    T.release_call_scratch(cache)
    assert "_z_cast" not in cache
    # and through the face: a memo planted before the call (as if left over) is dropped at entry, so nothing stale can be read inside the call
    C, H, N = 16, 16, 8
    W = _weights(torch, C, H)
    z4 = torch.randn(1, N, N, C)
    stale = z4.to(torch.bfloat16); cache["_z_cast"] = (z4, torch.bfloat16, stale); T._CAST_MEMO["live"] += 1
    with torch.no_grad():
        z4.add_(5.0)
    seen = []
    real = T.compute_input
    monkeypatch.setattr(T, "compute_input", lambda z, cache=None: (seen.append(cache is None or "_z_cast" not in cache or cache["_z_cast"][2] is not stale), real(z, cache))[1])
    try:
        T.triangle_multiplication(z4, None, direction="outgoing", weights=W, word="torch_math", residual=False, cache=cache)
    except T.Refusal:
        pass
    assert "_z_cast" not in cache and all(seen)
